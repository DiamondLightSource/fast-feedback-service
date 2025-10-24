# ruff: noqa: C901

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Union, Literal
import pydantic

import gemmi
import numpy as np
import workflows.recipe
from pydantic import BaseModel, ValidationError, Field, PrivateAttr
from rich.logging import RichHandler
from workflows.services.common_service import CommonService

import ffs.index
from ffs.ssx_index import GPUIndexer

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

DEFAULT_QUEUE_NAME = "per_image_analysis.gpu"


class PiaRequest(BaseModel):
    dcid: int
    dcgid: int
    filename: Path
    message_index: int
    number_of_frames: int
    start_frame_index: int
    startTime: Optional[datetime] = None
    wavelength: float | None = None
    xBeam: float
    yBeam: float
    detector_distance: float
    d_min: float | None = None
    d_max: float | None = None
    unit_cell: tuple[float, float, float, float, float, float] | None = None
    detector: str = "Eiger16M"

    @pydantic.validator("unit_cell", pre=True)
    def check_unit_cell(cls, v):
        if not v:
            return None
        orig_v = v
        if isinstance(v, str):
            v = v.replace(",", " ").split()
        v = [float(v) for v in v]
        try:
            assert len(v) == 6
        except Exception:
            raise ValueError(f"Invalid unit_cell {orig_v}")
        return v

class Material(str, Enum):
    Si = "Si"
    CdTe = "CdTe"

class DetectorParameters(BaseModel):
    """
    Define a set of detector metadata that derived classes
    need to provide.
    This class is not to be instantiated directly.
    """
    detector_type: str
    thickness: float
    material: Material
    pixel_size_x: float
    pixel_size_y: float
    image_size_x: int
    image_size_y: int
    # mu cache not serialized to dict/json, ok as mu not needed for spotfinder
    _mu_cache: dict = PrivateAttr(default_factory=dict)

    def __init_subclass__(cls, **kwargs):
        # enforce setting of defaults for all fields in subclasses.
        super().__init_subclass__(**kwargs)
        missing_defaults = [
            name for name, field in cls.__fields__.items()
            if field.default is None and field.default_factory is None
        ]
        if missing_defaults:
            raise TypeError(
                f"{cls.__name__} must define default values for all fields: missing {missing_defaults}"
            )

    def calculate_mu(self, wavelength: float) -> float:
        if wavelength not in self._mu_cache:
            self._mu_cache[wavelength] = ffs.index.calculate_mu_for_material_at_wavelength(
                self.material, wavelength
            )
        return self._mu_cache[wavelength]


class Eiger16M(DetectorParameters):
    detector_type: Literal["Eiger16M"]
    thickness: float = 0.45
    material : Material = Material.Si
    pixel_size_x: float = 0.075
    pixel_size_y: float = 0.075
    image_size_x: int = 4148
    image_size_y: int = 4362

class Eiger4M(DetectorParameters):
    detector_type: Literal["Eiger4M"]
    thickness: float = 0.45
    material : Material = Material.Si
    pixel_size_x: float = 0.075
    pixel_size_y: float = 0.075
    image_size_x: int = 2068
    image_size_y: int = 2162


class Eiger9MCdTe(DetectorParameters):
    detector_type: Literal["Eiger9MCdTe"]
    thickness: float = 0.75
    material : Material = Material.CdTe
    pixel_size_x: float = 0.075
    pixel_size_y: float = 0.075
    image_size_x: int = 3108
    image_size_y: int = 3262


class DetectorGeometry(BaseModel):
    distance: float
    beam_center_x: float
    beam_center_y: float
    detector: Union[Eiger9MCdTe, Eiger16M, Eiger4M] = Field(..., discriminator="detector_type")

    def to_json(self):
        d = self.dict()
        d.update(self.detector.dict())
        return json.dumps(d, indent=4)


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


class MessageOrderResolver:
    """
    Handles logic over incoming message order

    Because of race conditions for two messages sent at effectively the
    same time, we can get a request for collection 1 before collection 0.
    Softly enforce ordering by waiting the first time we get a request
    for a specific message ID index.
    """

    _expected_next_message_id = 0
    _current_dcgid = 0

    def __init__(self, logger: logging.Logger):
        self.log = logger

    def should_handle_now(self, message: PiaRequest, header: dict) -> bool:
        # If we've gotten a new group, reset the counter
        if message.dcgid != self._current_dcgid:
            if self._current_dcgid:
                self.log.debug(
                    f"Got new DCGID ({message.dcgid}); resetting expected index counter."
                )
            self._current_dcgid = message.dcgid
            self._expected_next_message_id = 0

        # Subsequent messages
        if message.message_index == self._expected_next_message_id:
            self._expected_next_message_id += 1
        elif header.get("already_requeued", False):
            # We already tried to delay this once, and it didn't appear.
            # Don't ask questions and just try to analyse this.
            self.log.info(
                f"PIA requests out-of-order; Expected {self._expected_next_message_id}, got {message.message_index}. Already Requeued once, continuing analysis."
            )
        elif message.message_index != self._expected_next_message_id:
            self.log.info(
                f"PIA requests out-of-order; Expected {self._expected_next_message_id}, got {message.message_index}. Requeueing."
            )
            header["already_requeued"] = True
            return False

        return True


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
        self._order_resolver = MessageOrderResolver(self.log)
        ## Initialise the fast-feedback-indexer
        self.indexer = None
        self.output_for_index = False  # Only turn on when we have confirmed all the things we need (cell, etc)
        try:
            self.indexer = GPUIndexer()
        except ModuleNotFoundError:
            self.log.debug(
                "ffbidx not found, has the fast-feedback-indexer module been built and sourced?"
            )

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
            self.log.debug(f"Contents: {rw.recipe_step['parameters']!r}")
            rw.transport.nack(header, requeue=False)
            return
        try:
            # Create a detector geometry object
            detector_geometry = DetectorGeometry(
                distance=parameters.detector_distance,
                beam_center_x=parameters.xBeam,
                beam_center_y=parameters.yBeam,
                detector={"detector_type" : parameters.detector},
            )
            self.log.debug("{detector_geometry.to_json()=}")
        except ValidationError as e:
            self.log.warning(
                f"Rejecting PIA request for {parameters.dcgid}/{parameters.message_index}({parameters.dcid}): Invalid detector parameters \n{e}"
            )

        if self.indexer and parameters.unit_cell and parameters.wavelength:
            ## We have all we need to index, so make up to date models.
            cell = gemmi.UnitCell(*parameters.unit_cell)
            self.indexer.cell = np.reshape(
                np.array(cell.orth.mat, dtype="float32"), (3, 3)
            )  ## Cell as an orthogonalisation matrix
            ## convert beam centre to correct units (given in mm, want in px).
            px_size_x = detector_geometry.detector.pixel_size_x
            px_size_y = detector_geometry.detector.pixel_size_y
            mu = detector_geometry.detector.calculate_mu(parameters.wavelength)
            self.indexer.panel = ffs.index.make_panel(
                detector_geometry.distance,
                detector_geometry.beam_center_x / px_size_x,
                detector_geometry.beam_center_y / px_size_y,
                px_size_x,
                px_size_y,
                detector_geometry.detector.image_size_x,
                detector_geometry.detector.image_size_y,
                detector_geometry.detector.thickness,
                mu,
            )
            self.indexer.wavelength = parameters.wavelength
            self.output_for_index = (
                True  # The indexer has been configured, so can run the spotfinder
            )
            # with --output-for-index and capture the results in read_and_send.

        start_time = time.monotonic()
        self.log.info(
            f"Gotten PIA request for {parameters.dcgid}/{parameters.message_index}({parameters.dcid}): {parameters.filename}/:{parameters.start_frame_index}-{parameters.start_frame_index + parameters.number_of_frames}"
        )

        if not self._order_resolver.should_handle_now(parameters, header):
            rw.transport.ack(header)
            # Requeue the message with a checkpoint to reorder it
            # TODO: Should we add transactions to ack here? IIRC the AMQP transaction semantics wouldn't cover this?
            rw.checkpoint(message, header=header, delay=5)
            return

        if parameters.filename.is_absolute():
            data_path = parameters.filename
        else:
            # If we don't have a base path, then assume we don't have GPU mode turned on
            # Ideally this would be tested directly, but I need to work out how to get PV
            # access on the gpu-epu
            if not Path(base_path).is_dir():
                self.log.info(
                    f"Not running GPU analysis as parent dir {base_path} does not exist; Is DAQ in /dev/shm mode?"
                )
                rw.transport.ack(header)
                return

            # Form the expected path for this dataset
            data_path = base_path / parameters.filename

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
            "--detector",
            detector_geometry.to_json(),
        ]
        if parameters.wavelength is not None:
            command.extend(["--wavelength", str(parameters.wavelength)])
        if parameters.d_min:
            command.extend(["--dmin", str(parameters.d_min)])
        if parameters.d_max:
            command.extend(["--dmax", str(parameters.d_max)])
        if self.output_for_index:
            command.extend(["--output-for-index"])

        self.log.info(f"Running: {' '.join(str(x) for x in command)}")

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
                ## Do indexing
                if self.output_for_index:
                    xyzobs_px = np.array(data["spot_centers"])
                    indexing_result = self.indexer.index(xyzobs_px)
                    self.log.info(indexing_result.model_dump_json(indent=2))
                    result = indexing_result.model_dump()
                    data.update(result)
                    del data["spot_centers"]  # don't send this data array onwards.
                self.log.info(f"Sending: {data}")
                rw.set_default_channel("result")
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
