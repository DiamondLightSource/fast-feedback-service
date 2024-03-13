from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from pprint import pformat

import workflows.recipe
from rich.logging import RichHandler
from workflows.services.common_service import CommonService

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"

SPOTFINDER = Path(
    "/dls/science/users/mep23677/cuda/miniapp/cuda/spotfinder/_build/spotfinder"
)


def _setup_rich_logging(level=logging.DEBUG):
    """Setup a rich-based logging output. Using for debug running."""
    rootLogger = logging.getLogger()

    for handler in list(rootLogger.handlers):
        # We want to replace the streamhandler
        if isinstance(handler, logging.StreamHandler):
            rootLogger.handlers.remove(handler)
        # We also want to lower the output level
        handler.setLevel(rootLogger.level)

    rootLogger.handlers.append(
        RichHandler(level=level, log_time_format="[%Y-%m-%d %H:%M:%S]")
    )


class GPUPerImageAnalysis(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"

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

    def gpu_per_image_analysis(
        self, rw: workflows.recipe.RecipeWrapper, header: dict, message: dict
    ):
        parameters = rw.recipe_step["parameters"]
        self.log.debug(
            f"Gotten PIA request:\nHeader:\n {pformat(header)}\nPayload:\n {pformat(rw.payload)}\n"
            f"Parameters: {pformat(rw.recipe_step['parameters'])}\n"
        )

        # Do sanity checks, then launch spotfinder
        if not SPOTFINDER.is_file():
            self.log.error("Could not find spotfinder executable: %s", SPOTFINDER)
            rw.transport.nack(header)

        # Form the expected path for this dataset
        expected_path = f"/dev/shm/eiger/{parameters['filename']}"

        # Create a pipe for comms
        # TODO: Set up pipes for communication back from process
        # (pipe_r, pipe_w) = os.pipe()

        # Now run the spotfinder
        command = [SPOTFINDER, str(expected_path)]
        self.log.info(f"Running: {' '.join(str(x) for x in command)}")
        start_time = time.monotonic()
        _result = subprocess.run(command)

        duration = time.monotonic() - start_time
        self.log.info(f"Analysis complete in {duration:.1f} s")
