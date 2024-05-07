from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from pprint import pformat
from typing import Iterator

import workflows.recipe
from rich.logging import RichHandler
from workflows.services.common_service import CommonService

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"


def _setup_rich_logging(level=logging.DEBUG):
    """Setup a rich-based logging output. Using for debug running."""
    rootLogger = logging.getLogger()

    for handler in list(rootLogger.handlers):
        # We want to replace the streamhandler
        if isinstance(handler, logging.StreamHandler):
            rootLogger.handlers.remove(handler)
        # We also want to lower the output level, so pin this to the existing
        handler.setLevel(rootLogger.level)

    rootLogger.handlers.append(
        RichHandler(level=level, log_time_format="[%Y-%m-%d %H:%M:%S]")
    )


class GPUPerImageAnalysis(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"
    _spotfinder_executable: Path | None = None
    _spotfind_proc: subprocess.Popen | None = None

    def initializing(self):
        _setup_rich_logging()
        # self.log.debug("Checking Node GPU capabilities")
        # TODO: Write node sanity checks
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment.get("queue") or DEFAULT_QUEUE_NAME,
            self.gpu_per_image_analysis,
            acknowledgement=True,
            log_extender=self.extend_log,
        )
        self._spotfinder_executable = self._find_spotfinder()
        self.expected_next_index = 0

    def _find_spotfinder(self) -> Path:
        """
        Finds and sets the path to the spotfinder executable

        Returns:
            Path: The path to the spotfinder executable
        """
        # Try to get the path from the environment
        spotfinder_path = os.getenv("SPOTFINDER")

        # If environment variable is not set, check for directories
        if spotfinder_path is None:
            self.log.warn("SPOTFINDER environment variable not set")

            # Check for the spotfinder executable in the build directories
            if Path("build").exists():
                self.log.info("SPOTFINDER found in build directory")
                spotfinder_path = "build/spotfinder"
            elif Path("_build").exists():
                self.log.info("SPOTFINDER found in _build directory")
                spotfinder_path = "_build/spotfinder"
            elif Path("spotfinder").exists():
                spotfinder_path = "spotfinder"
            else:
                spotfinder_path = None
                # Failing to find the executable is handled in the main function
                # wherein we will nack the message and return.
                # Hence we leave the spotfinder_path as the default None

        # Convert to Path object
        if spotfinder_path is not None:
            spotfinder_path = Path(spotfinder_path)

        return spotfinder_path

    def gpu_per_image_analysis(
        self,
        rw: workflows.recipe.RecipeWrapper,
        header: dict,
        message: dict,
        base_path="/dev/shm/eiger",
    ):
        parameters = rw.recipe_step["parameters"]

        self.log.debug(
            f"Gotten PIA request:\nHeader:\n {pformat(header)}\nPayload:\n {pformat(rw.payload)}\n"
            f"Parameters: {pformat(rw.recipe_step['parameters'])}\n"
        )

        # Reject messages without the extra info
        if parameters.get("filename", "{filename}") == "{filename}":
            # We got a request, but didn't have required hyperion info
            self.log.debug(
                f"Rejecting PIA request for {parameters['dcid']}; no valid hyperion information"
            )
            # We just want to silently kill this message, as it wasn't for us
            rw.transport.ack(header)
            return

        # Check if dataset is being processed in order
        received_index = int(parameters["message_index"])
        # First message
        if received_index == 0:
            self.expected_next_index = 1
        # Subsequent messages
        elif received_index == self.expected_next_index:
            self.expected_next_index += 1
        # Out of order message
        elif received_index != self.expected_next_index:
            self.log.info(
                f"Expected message index {self.expected_next_index}, got {parameters['message_index']}"
            )
            # Requeue the message with a checkpoint to reorder it
            rw.checkpoint(message, header=header, delay=1)
            time.sleep(10)
            rw.transport.ack(header)
            return

        # Do sanity checks, then launch spotfinder
        if not self._spotfinder_executable.is_file():
            self.log.error(
                "Could not find spotfinder executable: %s", self._spotfinder_executable
            )
            rw.transport.nack(header)
            return
        else:
            self.log.info(f"Using SPOTFINDER: {self._spotfinder_executable}")

        # Otherwise, assume that this will work for now and nack the message
        rw.transport.ack(header)

        # Form the expected path for this dataset
        data_path = f"{base_path}/{parameters['filename']}"

        # Create a pipe for comms
        read_fd, write_fd = os.pipe()

        # Now run the spotfinder
        command = [
            self._spotfinder_executable,
            str(data_path),
            "--images",
            parameters["number_of_frames"],
            "--start-index",
            parameters["start_frame_index"],
            "--threads",
            str(40),
            "--pipe_fd",
            str(write_fd),
        ]
        self.log.info(f"Running: {' '.join(str(x) for x in command)}")
        start_time = time.monotonic()

        # Set the default channel for the result
        rw.set_default_channel("result")

        def pipe_output(read_fd: int) -> Iterator[str]:
            """
            Generator to read from the pipe and yield the output

            Args:
                read_fd: The file descriptor for the pipe

            Yields:
                str: A line of JSON output
            """
            # Reader function
            with os.fdopen(read_fd, "r") as pipe_data:
                # Process each line of JSON output
                for line in pipe_data:
                    line = line.strip()
                    yield line

        def read_and_send() -> None:
            """
            Read from the pipe and send the output to the result queue

            This function is intended to be run in a separate thread

            Returns:
                None
            """
            # Read from the pipe and send to the result queue
            for line in pipe_output(read_fd):
                self.log.info(f"Received: {line}")  # Change log level to debug?
                rw.send_to("result", line)

            self.log.info("Results finished sending")

        # Create a thread to read and send the output
        read_and_send_data_thread = threading.Thread(target=read_and_send)

        # Run the spotfinder
        self._spotfind_proc = subprocess.Popen(command, pass_fds=[write_fd])

        # Close the write end of the pipe (for this process)
        # spotfind_process will hold the write end open until it is done
        # This will allow the read end to detect the end of the output
        os.close(write_fd)

        # Start the read thread
        read_and_send_data_thread.start()

        # Wait for the process to finish
        self._spotfind_proc.wait()

        # Log the duration
        duration = time.monotonic() - start_time
        self.log.info(f"Analysis complete in {duration:.1f} s")

        # Wait for the read thread to finish
        read_and_send_data_thread.join()
