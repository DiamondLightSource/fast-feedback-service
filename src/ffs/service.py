from __future__ import annotations

from pprint import pformat

import workflows.recipe
from workflows.services.common_service import CommonService

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"


class GPUPerImageAnalysis(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"

    def initializing(self):
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
