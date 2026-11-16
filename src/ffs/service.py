# ruff: noqa: C901

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

import gemmi
import numpy as np
import pydantic
import workflows.recipe
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from workflows.services.common_service import CommonService

import ffs.index
from ffs._common import (
    DeviceProbeFailed,
    ExecutableNotFound,
    find_executable,
    setup_rich_logging,
)
from ffs.ssx_index import GPUIndexer

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG

DEFAULT_QUEUE_NAME = os.getenv("FFS_QUEUE", "per_image_analysis.gpu")
# How long to wait for the first file of a CBF template to appear, in seconds
CBF_FIRST_FILE_TIMEOUT = 60.0


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
    # If unset, inferred from the file header for CBF, otherwise Eiger16M
    detector: str | None = None
    # For a CBF template: the number of the first file in the collection.
    # If unset, guessed as the lowest numbered matching file on disk.
    first_image_number: int | None = None

    @property
    def is_cbf(self) -> bool:
        return self.filename.suffix.lower() == ".cbf"

    @property
    def is_cbf_template(self) -> bool:
        return self.is_cbf and "#" in self.filename.name

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
            name
            for name, field in cls.__fields__.items()
            if field.default is None and field.default_factory is None
        ]
        if missing_defaults:
            raise TypeError(
                f"{cls.__name__} must define default values for all fields: missing {missing_defaults}"
            )

    def calculate_mu(self, wavelength: float) -> float:
        if wavelength not in self._mu_cache:
            self._mu_cache[wavelength] = (
                ffs.index.calculate_mu_for_material_at_wavelength(
                    self.material, wavelength
                )
            )
        return self._mu_cache[wavelength]


class Eiger16M(DetectorParameters):
    detector_type: Literal["Eiger16M"]
    thickness: float = 0.45
    material: Material = Material.Si
    pixel_size_x: float = 0.075
    pixel_size_y: float = 0.075
    image_size_x: int = 4148
    image_size_y: int = 4362


class Eiger4M(DetectorParameters):
    detector_type: Literal["Eiger4M"]
    thickness: float = 0.45
    material: Material = Material.Si
    pixel_size_x: float = 0.075
    pixel_size_y: float = 0.075
    image_size_x: int = 2068
    image_size_y: int = 2162


class Eiger9MCdTe(DetectorParameters):
    detector_type: Literal["Eiger9MCdTe"]
    thickness: float = 0.75
    material: Material = Material.CdTe
    pixel_size_x: float = 0.075
    pixel_size_y: float = 0.075
    image_size_x: int = 3108
    image_size_y: int = 3262


class Pilatus6M(DetectorParameters):
    detector_type: Literal["Pilatus6M"]
    thickness: float = 0.45
    material: Material = Material.Si
    pixel_size_x: float = 0.172
    pixel_size_y: float = 0.172
    image_size_x: int = 2463
    image_size_y: int = 2527


class DetectorGeometry(BaseModel):
    distance: float
    beam_center_x: float
    beam_center_y: float
    detector: Union[Eiger9MCdTe, Eiger16M, Eiger4M, Pilatus6M] = Field(
        ..., discriminator="detector_type"
    )

    def to_json(self):
        d = self.dict()
        d.update(self.detector.dict())
        return json.dumps(d, indent=4)


class CBFHeader(BaseModel):
    """The detector metadata in a (PILATUS-convention, "mini") CBF header"""

    detector_type: str | None = None
    wavelength: float | None = None  # Å
    detector_parameters: dict = Field(default_factory=dict)  # DetectorParameters


# The marker at the start of the binary section of a CBF file
_CBF_BINARY_MARKER = b"\x0c\x1a\x04\xd5"

# Map the "Detector:" header line onto our detector types. Checked in order.
_CBF_DETECTOR_PATTERNS = [
    (re.compile(r"PILATUS\S*\s+6M", re.I), "Pilatus6M"),
    (re.compile(r"EIGER\S*\s+CdTe\s+9M", re.I), "Eiger9MCdTe"),
    (re.compile(r"EIGER\S*\s+(Si\s+)?16M", re.I), "Eiger16M"),
    (re.compile(r"EIGER\S*\s+(Si\s+)?4M", re.I), "Eiger4M"),
]


def read_cbf_header(path: Path) -> CBFHeader:
    """
    Read the detector metadata out of a CBF file's text header.

    Raises:
        ValueError: The file does not look like a CBF image.
    """
    with open(path, "rb") as f:
        raw = b""
        while _CBF_BINARY_MARKER not in raw:
            chunk = f.read(16384)
            if not chunk or len(raw) > 1024 * 1024:
                raise ValueError(f"Could not find CBF binary section in {path}")
            raw += chunk
    text = raw[: raw.index(_CBF_BINARY_MARKER)].decode("ascii", errors="replace")

    def find(pattern: str) -> re.Match | None:
        return re.search(pattern, text)

    header = CBFHeader()
    params = header.detector_parameters
    if m := find(r"#\s*Detector:\s*([^\r\n]*)"):
        for pattern, detector_type in _CBF_DETECTOR_PATTERNS:
            if pattern.search(m.group(1)):
                header.detector_type = detector_type
                break
    if m := find(r"#\s*Wavelength\s+([-0-9.eE+]+)\s*A"):
        header.wavelength = float(m.group(1))
    if m := find(r"#\s*Pixel_size\s+([-0-9.eE+]+)\s*m\s*x\s*([-0-9.eE+]+)\s*m"):
        params["pixel_size_x"] = float(m.group(1)) * 1000
        params["pixel_size_y"] = float(m.group(2)) * 1000
    if m := find(r"#\s*(\w+)\s+sensor,\s*thickness\s+([-0-9.eE+]+)\s*m"):
        material = {"silicon": Material.Si, "cdte": Material.CdTe}.get(
            m.group(1).lower()
        )
        if material:
            params["material"] = material
        params["thickness"] = float(m.group(2)) * 1000
    if m := find(r"X-Binary-Size-Fastest-Dimension:\s*([0-9]+)"):
        params["image_size_x"] = int(m.group(1))
    if m := find(r"X-Binary-Size-Second-Dimension:\s*([0-9]+)"):
        params["image_size_y"] = int(m.group(1))
    return header


def expand_cbf_template(template: Path, number: int) -> Path:
    """Fill the run of '#' in a CBF filename template with an image number"""
    first = template.name.index("#")
    last = template.name.rindex("#")
    width = last - first + 1
    return template.with_name(
        f"{template.name[:first]}{number:0{width}d}{template.name[last + 1 :]}"
    )


def find_first_cbf_number(template: Path) -> int | None:
    """
    Find the lowest image number on disk matching a CBF filename template.

    The number of digits must match the template exactly, so that e.g.
    "x_#####.cbf" does not pick up "x_123456.cbf" or "x_00001_old.cbf".

    Returns:
        The lowest matching image number, or None if no files match.
    """
    first = template.name.index("#")
    last = template.name.rindex("#")
    pattern = re.compile(
        re.escape(template.name[:first])
        + f"([0-9]{{{last - first + 1}}})"
        + re.escape(template.name[last + 1 :])
    )
    try:
        names = os.listdir(template.parent)
    except FileNotFoundError:
        return None
    numbers = [int(m.group(1)) for name in names if (m := pattern.fullmatch(name))]
    return min(numbers, default=None)


def _find_spotfinder() -> tuple[Path, Path]:
    """
    Finds and sets the path to the spotfinder executable

    Verifies that the spotfinder can run and enumerate devices.

    Returns:
        Path: The path to the spotfinder executable

    Raises:
        ExecutableNotFound: Either variant is missing.
        DeviceProbeFailed: A variant ran but could not enumerate GPU
            devices.
    """
    spotfinder_path = find_executable("SPOTFINDER", "spotfinder")

    # Make sure that we have a spotfinder32 at the same location
    spotfinder_32 = spotfinder_path.parent / "spotfinder32"
    if not spotfinder_32.is_file():
        raise ExecutableNotFound("Could not find spotfinder32 variant")

    proc = subprocess.run(
        [spotfinder_32, "--list-devices"], capture_output=True, text=True
    )
    if proc.returncode:
        detail = (
            proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        )
        raise DeviceProbeFailed(
            f"Found spotfinder32 but it failed to enumerate devices: {detail}"
        )

    return (spotfinder_path, spotfinder_32)


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
    _spotfinder_executable: tuple[Path, Path]
    _spotfind_proc: subprocess.Popen | None = None

    def initializing(self):
        setup_rich_logging()
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

    def _wait_for_first_cbf_number(
        self, template: Path, timeout: float = CBF_FIRST_FILE_TIMEOUT
    ) -> int | None:
        """Guess the first image number of a CBF template, waiting for a file"""
        deadline = time.monotonic() + timeout
        while (number := find_first_cbf_number(template)) is None:
            if time.monotonic() > deadline:
                return None
            time.sleep(1)
        return number

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
        request_name = (
            f"{parameters.dcgid}/{parameters.message_index}({parameters.dcid})"
        )

        # CBF is either a single file, or a '#' template for a sequence.
        # Detector metadata comes from the file header, unless overridden by
        # the request.
        cbf_header: CBFHeader | None = None
        first_image_number = 0
        if parameters.is_cbf_template:
            if parameters.first_image_number is not None:
                first_image_number = parameters.first_image_number
            else:
                guessed = self._wait_for_first_cbf_number(parameters.filename)
                if guessed is None:
                    self.log.warning(
                        f"Rejecting PIA request for {request_name}: No files matching {parameters.filename} appeared"
                    )
                    rw.transport.nack(header, requeue=False)
                    return
                first_image_number = guessed
                self.log.info(
                    f"Guessed first image number {first_image_number} for {parameters.filename} from files on disk"
                )
            header_file = expand_cbf_template(parameters.filename, first_image_number)
        else:
            header_file = parameters.filename
        if parameters.is_cbf:
            try:
                cbf_header = read_cbf_header(header_file)
            except (OSError, ValueError) as e:
                self.log.warning(
                    f"Rejecting PIA request for {request_name}: Could not read CBF header: {e}"
                )
                rw.transport.nack(header, requeue=False)
                return
            if not parameters.is_cbf_template and parameters.number_of_frames != 1:
                self.log.warning(
                    f"CBF request {request_name} has number_of_frames={parameters.number_of_frames}; only the single file {parameters.filename} will be analysed"
                )

        detector_type = parameters.detector or (
            cbf_header.detector_type if cbf_header else "Eiger16M"
        )
        wavelength = parameters.wavelength or (
            cbf_header.wavelength if cbf_header else None
        )
        try:
            # Create a detector geometry object
            detector_geometry = DetectorGeometry(
                distance=parameters.detector_distance,
                beam_center_x=parameters.xBeam,
                beam_center_y=parameters.yBeam,
                detector={
                    "detector_type": detector_type,
                    # The header describes the actual file, so takes precedence
                    **(cbf_header.detector_parameters if cbf_header else {}),
                },
            )
            self.log.debug(f"{detector_geometry.to_json()=}")
        except ValidationError as e:
            self.log.warning(
                f"Rejecting PIA request for {request_name}: Invalid detector parameters \n{e}"
            )
            rw.transport.nack(header, requeue=False)
            return

        if self.indexer and parameters.unit_cell and wavelength:
            ## We have all we need to index, so make up to date models.
            cell = gemmi.UnitCell(*parameters.unit_cell)
            self.indexer.cell = np.reshape(
                np.array(cell.orth.mat, dtype="float32"), (3, 3)
            )  ## Cell as an orthogonalisation matrix
            ## convert beam centre to correct units (given in mm, want in px).
            px_size_x = detector_geometry.detector.pixel_size_x
            px_size_y = detector_geometry.detector.pixel_size_y
            mu = detector_geometry.detector.calculate_mu(wavelength)
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
            self.indexer.wavelength = wavelength
            self.output_for_index = (
                True  # The indexer has been configured, so can run the spotfinder
            )
            # with --output-for-index and capture the results in read_and_send.

        start_time = time.monotonic()
        self.log.info(
            f"Gotten PIA request for {parameters.dcgid}/{parameters.message_index}({parameters.dcid}): {parameters.filename}/:{parameters.start_frame_index}-{parameters.start_frame_index + parameters.number_of_frames}"
        )

        # Single CBF images are independent, so there is no ordering to
        # enforce. CBF are never in /dev/shm, so skip the age check below.
        needs_ordering = parameters.is_cbf_template or not parameters.is_cbf
        if needs_ordering and not self._order_resolver.should_handle_now(
            parameters, header
        ):
            rw.transport.ack(header)
            # Requeue the message with a checkpoint to reorder it
            # TODO: Should we add transactions to ack here? IIRC the AMQP transaction semantics wouldn't cover this?
            rw.checkpoint(message, header=header, delay=5)
            return

        if parameters.is_cbf:
            data_path = parameters.filename
        elif parameters.filename.is_absolute():
            data_path = parameters.filename
        else:
            # If we have a base path directory (e.g., /dev/shm mode), use it
            if Path(base_path).is_dir():
                # Form the expected path for this dataset
                data_path = base_path / parameters.filename
            else:
                # Otherwise, treat filename as relative to current directory or an H5 file path
                self.log.info(
                    f"Base path {base_path} does not exist; treating {parameters.filename} as direct file path"
                )
                data_path = parameters.filename

        # Debugging: Reject messages that are "old", if the files are not on disk. This
        # should help avoid sitting spending hours running through all messages (meaning
        # that a manual purge is required).
        # Only apply this check for /dev/shm directories (not for H5 files)
        if parameters.startTime and Path(base_path).is_dir() and not parameters.is_cbf:
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
        if parameters.is_cbf:
            # CBF is natively 32-bit, so go straight to the 32-bit spotfinder.
            # The spotfinder reports images from index 0, so place them in
            # the collection with the start index of the request.
            command = [str(self._spotfinder_executable[1]), str(data_path)]
            if parameters.is_cbf_template:
                command += [
                    "--images",
                    str(parameters.number_of_frames),
                    "--start-index",
                    str(first_image_number + parameters.start_frame_index),
                ]
            file_number_offset = parameters.start_frame_index
        else:
            command = [
                str(self._spotfinder_executable[0]),
                str(data_path),
                "--images",
                str(parameters.number_of_frames),
                "--start-index",
                str(parameters.start_frame_index),
            ]
            file_number_offset = 0
        command += [
            "--threads",
            str(40),
            "--pipe_fd",
            str(write_fd),
            "--detector",
            detector_geometry.to_json(),
        ]
        if wavelength is not None:
            command.extend(["--wavelength", str(wavelength)])
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
                if parameters.is_cbf_template:
                    # Report the actual file, rather than the template
                    data["file"] = str(
                        expand_cbf_template(
                            parameters.filename,
                            first_image_number
                            + parameters.start_frame_index
                            + data["file-number"],
                        )
                    )
                # XRC has one-based-indexing
                data["file-number"] += 1 + file_number_offset
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

        # Start the read thread
        read_and_send_data_thread.start()

        # Wait for the process to finish
        if self._spotfind_proc.wait() == 32:
            self.log.info("Spotfinder exited indicating data is 32-bit, relaunching")
            # We need to run the 32-bit version of the spotfinder instead....
            command[0] = str(self._spotfinder_executable[1])
            self._spotfind_proc = subprocess.Popen(command, pass_fds=[write_fd])

        # Close the write end of the pipe (for this process)
        # spotfind_process will hold the write end open until it is done
        # This will allow the read end to detect the end of the output
        os.close(write_fd)

        # Log the duration
        duration = time.monotonic() - start_time

        self.log.info(f"Analysis complete in {duration:.1f} s")

        # Wait for the read thread to finish
        read_and_send_data_thread.join()
