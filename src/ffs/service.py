from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Iterator, Optional

from pydantic import BaseModel, ValidationError
from rich.logging import RichHandler

import workflows.recipe
from workflows.services.common_service import CommonService

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"


class PiaRequest(BaseModel):
    dcid: int
    filename: str
    message_index: int
    number_of_frames: int
    start_frame_index: int
    startTime: Optional[datetime] = None


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


def _find_spotfinder() -> Path:
    """
    Finds and sets the path to the spotfinder executable

    Verifies that the spotfinder can run and enumerate devices.

    Returns:
        Path: The path to the spotfinder executable
    """
    # Try to get the path from the environment
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")

    # If environment variable is not set, check for directories
    if spotfinder_path is None:
        logger.warning("SPOTFINDER environment variable not set")

        for path in {"build", "_build", "."}:
            if Path(path).exists():
                spotfinder_path = path
            logger.debug("No spotfinder found at %s", path)

    # This must be set, and must exist
    if not spotfinder_path or not Path(spotfinder_path).is_file():
        logger.fatal(
            "Error: Could not find spotfinder executable. Please set SPOTFINDER environment variable."
        )
        sys.exit(1)

    spotfinder_path = Path(spotfinder_path)

    # Let's run this, to enumerate GPU and check it works
    proc = subprocess.run(
        [spotfinder_path, "--list-devices"], capture_output=True, text=True
    )
    if proc.returncode:
        logger.fatal(
            f"Error: Spotfinder at {spotfinder_path} failed to enumerate devices."
        )
        sys.exit(1)

    logger.info(f"Using spotfinder: {spotfinder_path}")
    return spotfinder_path


class GPUPerImageAnalysis(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"
    _spotfinder_executable: Path
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
        self._spotfinder_executable = _find_spotfinder()
        self.expected_next_index = 0

    def gpu_per_image_analysis(
        self,
        rw: workflows.recipe.RecipeWrapper,
        header: dict,
        message: dict,
        base_path="/dev/shm/eiger",
    ) -> None:
        try:
            parameters = PiaRequest(**rw.recipe_step["parameters"])
        except ValidationError as e:
            dcid = rw.recipe_step["parameters"].get("dcid", "(unknown DCID)")
            self.log.warning(f"Rejecting PIA request for {dcid}: \n{e}")
            rw.transport.nack(header, requeue=False)
            return

        self.log.debug(f"Got Request: {parameters!r}")

        self.log.info(
            f"Gotten PIA request for {parameters.dcid}/{parameters.message_index}: {parameters.filename}/:{parameters.start_frame_index}-{parameters.start_frame_index+parameters.number_of_frames}"
        )
        self.log.debug(
            f"Gotten PIA request:\nHeader:\n {pformat(header)}\nPayload:\n {pformat(rw.payload)}\n"
            f"Parameters: {pformat(rw.recipe_step['parameters'])}\n"
        )
        # Check if dataset is being processed in order
        received_index = int(parameters.message_index)
        # First message
        if received_index == 0:
            self.expected_next_index = 1
        # Subsequent messages
        elif received_index == self.expected_next_index:
            self.expected_next_index += 1
        elif header.get("already_requeued", False):
            # We already tried to delay this once, and it didn't appear.
            # Don't ask questions and just try to analyse this.
            self.log.info(
                f"PIA requests out-of-order; Expected {self.expected_next_index}, got {parameters.message_index}. Already Requeued once, continuing analysis."
            )
        elif received_index != self.expected_next_index:
            self.log.info(
                f"PIA requests out-of-order; Expected {self.expected_next_index}, got {parameters.message_index}. Requeueing."
            )
            rw.transport.ack(header)
            # Requeue the message with a checkpoint to reorder it
            # TODO: Should we add transactions to ack here? IIRC the AMQP transaction semantics wouldn't cover this?
            rw.checkpoint(message, header=header | {"already_requeued": True}, delay=5)
            return

        # Form the expected path for this dataset
        data_path = Path(f"{base_path}/{parameters.filename}")

        # Debugging: Reject messages that are "old", if the files are not on disk. This
        # should help avoid sitting spending hours running through all messages (meaning
        # that a manual purge is required).
        if parameters.startTime:
            age_seconds = (datetime.now() - parameters.startTime).total_seconds()
            if age_seconds > 60 and not data_path.is_dir():
                self.log.warning(
                    f"Not processing message as too old ({age_seconds:.0f} s); and no data on disk indicating retrigger"
                )
                rw.transport.ack(header)
                return

        # Otherwise, assume that this will work for now and nack the message
        rw.transport.ack(header)

        # Create a pipe for comms
        read_fd, write_fd = os.pipe()

        # Now run the spotfinder
        command = [
            str(self._spotfinder_executable),
            str(data_path),
            "--images",
            str(parameters.number_of_frames),
            "--start-index",
            str(parameters.start_frame_index),
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
                data = json.loads(line)
                data["file-seen-at"] = time.time()
                # XRC has one-based-indexing
                data["file-number"] += 1
                self.log.info(f"Sending: {data}")
                rw.send_to("result", data)

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
