from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Tuple, TypeAlias

from pydantic import BaseModel, ValidationError
from rich.logging import RichHandler

import workflows.recipe
from workflows.services.common_service import CommonService

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

DEFAULT_QUEUE_NAME = "reduce.xray_centering.gpu.compare_results"


Coordinate: TypeAlias = Tuple[int, int, int]


class XrcResult(BaseModel):
    centre_of_mass: list[float]
    max_voxel: list[int]
    max_count: int
    n_voxels: int
    total_count: int
    bounding_box: tuple[Coordinate, Coordinate]


class Parameters(BaseModel):
    dcid: int
    gpu: bool = False


class Result(BaseModel):
    timestamp: float
    result: XrcResult
    gpu: bool
    header: Any


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


class XRCResultCompare(CommonService):
    _service_name = "GPU Per-Image-Analysis"
    _logger_name = "spotfinder.service"
    _spotfinder_executable: Path
    _spotfind_proc: subprocess.Popen | None = None

    def initializing(self):
        self._result: dict[int, Result] = {}
        _setup_rich_logging()
        workflows.recipe.wrap_subscribe(
            self._transport,
            self._environment.get("queue") or DEFAULT_QUEUE_NAME,
            self.compare_xrc,
            acknowledgement=True,
            log_extender=self.extend_log,
        )

    def compare_xrc(
        self,
        rw: workflows.recipe.RecipeWrapper,
        header: dict,
        message: dict,
    ) -> None:
        try:
            result = XrcResult.model_validate(message)
            params: Parameters = Parameters.model_validate(rw.recipe_step["parameters"])
        except ValidationError as e:
            dcid = rw.recipe_step["parameters"].get("dcid", "(unknown DCID)")
            self.log.warning(f"Rejecting XRC result for {dcid}: \n{e}")
            self.log.debug(f"Contents: {rw.recipe_step['parameters']:?}")
            rw.transport.nack(header, requeue=False)
            return

        is_gpu = bool(rw.recipe_step["parameters"].get("gpu"))

        self.log.info(
            f"Gotten XRC Result for {params.dcid} ({'GPU' if is_gpu else 'CPU'})"
        )

        if params.dcid not in self._result:
            self._result[params.dcid] = Result(
                timestamp=time.time(), result=result, gpu=is_gpu, header=header
            )
            return

        other_result = self._result.pop(params.dcid)
        if other_result.gpu == result.gpu:
            self.log.error(
                f"Error: Got multiple {'GPU' if is_gpu else 'CPU'} results for {params.dcid}"
            )
            rw.transport.nack(header, requeue=False)
            rw.transport.nack(other_result.header, requeue=False)

        self.log.info(f"Compared results:\n{result=}\n{other_result=}")
        rw.transport.ack(header)
        rw.transport.ack(other_result.header)
