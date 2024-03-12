from __future__ import annotations

import logging
from pprint import pformat

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
        # We also want to lower the output level
        handler.setLevel(rootLogger.level)

    rootLogger.handlers.append(
        RichHandler(level=level, log_time_format="[%Y-%m-%d %H:%M:%S]")
    )


class GPUPerImageAnalysis(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"

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
        self.log(
            f"Gotten PIA request:\nPayload:\n {pformat(rw.payload)}\nRecipe: {pformat(rw.recipe.recipe)}"
        )
