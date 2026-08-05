"""
Spotfinding, indexing and integration pipeline for a single dataset.

Takes the raw images and runs the spotfinder, the baseline indexer and
the GPU integrator. The stages chain by writing and reading files under
fixed names in a working directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from ffs._common import ExecutableError, find_executable, setup_rich_logging
from ffs.pipeline import PipelineResult, append_optional, run_stage, write_summary
from ffs.stages import (
    INDEXED_EXPERIMENTS,
    INDEXED_REFLECTIONS,
    PipelineRequest,
    add_tuning_arguments,
    build_indexer_command,
    build_integrator_command,
)

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "ffs_process.json"

# Written by the spotfinder under a fixed name, relative to the CWD
STRONG_REFLECTIONS = Path("results_ffs.h5")


class ProcessRequest(PipelineRequest):
    """
    Parameters for the pipeline, including the spotfinder's.

    The strong reflections are written by the first stage, so the
    inherited field names what the spotfinder produces.

    --threads, --timeout and --algorithm mean different things to the
    spotfinder and the integrator, so the spotfinder's take a prefix.
    """

    reflection: Path = STRONG_REFLECTIONS

    # Inputs
    data: Path

    # Spotfinder tuning
    spotfinder_algorithm: Optional[str] = None
    spotfinder_threads: Optional[int] = None
    spotfinder_timeout: Optional[float] = None
    images: Optional[int] = None
    start_index: Optional[int] = None
    wavelength: Optional[float] = None
    dmax: Optional[float] = None
    min_spot_size: Optional[int] = None
    min_spot_size_3d: Optional[int] = None
    max_peak_centroid_separation: Optional[float] = None
    detector: Optional[str] = None


def build_spotfinder_command(executable: Path, params: ProcessRequest) -> list[str]:
    """
    Assemble the spotfinder command line.

    --save-h5 is always passed: the table it writes is how the spots
    reach the indexer.

    Args:
        executable: Path to the spotfinder binary
        params:     The validated request

    Returns:
        list[str]: The argv to hand to subprocess
    """
    command = [
        str(executable),
        str(params.data),
        "--save-h5",
    ]
    append_optional(
        command,
        {
            "--algorithm": params.spotfinder_algorithm,
            "--threads": params.spotfinder_threads,
            "--timeout": params.spotfinder_timeout,
            "--images": params.images,
            "--start-index": params.start_index,
            "--wavelength": params.wavelength,
            "--dmin": params.dmin,
            "--dmax": params.dmax,
            "--min-spot-size": params.min_spot_size,
            "--min-spot-size-3d": params.min_spot_size_3d,
            "--max-peak-centroid-separation": params.max_peak_centroid_separation,
            "--detector": params.detector,
        },
    )
    return command


def run_pipeline(params: ProcessRequest) -> PipelineResult:
    """
    Spotfind, index and then integrate one dataset.

    Creates the working directory and changes into it, so that the
    files each stage writes under fixed names are found by the next
    one. Stops at the first stage that fails.

    Args:
        params: The validated request

    Returns:
        PipelineResult: What ran, and what it produced

    Raises:
        ExecutableError: Any of the three binaries is missing or
            unusable. Raised before the working directory is created.
    """
    spotfinder = find_executable("SPOTFINDER", "spotfinder")
    indexer = find_executable("INDEXER", "baseline_indexer", probe=False)
    integrator = find_executable("INTEGRATOR", "integrator")

    # Resolve the inputs before moving, so that relative paths on the
    # command line still mean what the caller meant. The reflection
    # table is written by the first stage, not supplied.
    params = params.model_copy(
        update={
            "data": params.data.resolve(),
            "experiment": params.experiment.resolve(),
        }
    )

    working_directory = params.working_directory.resolve()
    working_directory.mkdir(parents=True, exist_ok=True)
    os.chdir(working_directory)
    logger.info(f"Working directory: {working_directory}")

    output = params.output
    if not output.is_absolute():
        output = working_directory / output

    result = PipelineResult(
        dcid=params.dcid,
        working_directory=working_directory,
        stages=[],
    )

    result.stages.append(
        run_stage("spotfinder", build_spotfinder_command(spotfinder, params))
    )
    if result.stages[-1].exit_code:
        return result
    result.strong_reflections = working_directory / STRONG_REFLECTIONS

    result.stages.append(run_stage("indexer", build_indexer_command(indexer, params)))
    if result.stages[-1].exit_code:
        return result
    result.indexed_experiments = working_directory / INDEXED_EXPERIMENTS
    result.indexed_reflections = working_directory / INDEXED_REFLECTIONS

    result.stages.append(
        run_stage("integrator", build_integrator_command(integrator, params, output))
    )
    if result.stages[-1].exit_code:
        return result
    result.integrated_reflections = output

    return result


def run(args=None) -> int:
    """
    Command line entrypoint, mirroring the recipe parameters.

    Returns:
        int: Zero when all three stages succeeded
    """
    setup_rich_logging()

    parser = argparse.ArgumentParser(
        description="Spotfind, index and integrate a single dataset, then exit."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Raw image data to process (Nexus or CBF)",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment list from dials.import",
    )
    parser.add_argument(
        "--working-directory",
        required=True,
        help="Directory to write results into. Created if absent.",
    )
    parser.add_argument(
        "--max-cell",
        required=True,
        type=float,
        help="Maximum cell length to consider during indexing",
    )
    parser.add_argument(
        "--dmin",
        type=float,
        help="Resolution limit, applied to both spots and indexing",
    )
    parser.add_argument(
        "--dmax",
        type=float,
        help="Low resolution limit for spots",
    )
    parser.add_argument(
        "--spotfinder-algorithm",
        help="Dispersion algorithm for spotfinding",
    )
    parser.add_argument(
        "--spotfinder-threads",
        type=int,
        help="Spotfinder reader threads",
    )
    parser.add_argument(
        "--spotfinder-timeout",
        type=float,
        help="Seconds the spotfinder waits for new images",
    )
    parser.add_argument(
        "--images",
        type=int,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Index of the first image",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        help="Beam wavelength in Å",
    )
    parser.add_argument(
        "--min-spot-size",
        type=int,
        help="Pixel count below which 2D spots are dropped",
    )
    parser.add_argument(
        "--min-spot-size-3d",
        type=int,
        help="Pixel count below which 3D spots are dropped",
    )
    parser.add_argument(
        "--max-peak-centroid-separation",
        type=float,
        help="Peak to centroid distance above which spots are filtered",
    )
    parser.add_argument(
        "--detector",
        help="Detector geometry JSON, if not in the data",
    )
    add_tuning_arguments(parser)

    options = parser.parse_args(args)
    # Drop unset options so that the model defaults apply
    supplied = {k: v for k, v in vars(options).items() if v is not None}

    try:
        params = ProcessRequest(**supplied)
    except ValidationError as e:
        sys.exit(f"Invalid parameters:\n{e}")

    for description, path in (
        ("Data", params.data),
        ("Experiment", params.experiment),
    ):
        if not path.exists():
            sys.exit(f"Error: {description} not found: {path}")

    try:
        result = run_pipeline(params)
    except ExecutableError as e:
        # Nothing ran, so there is no summary to write
        logger.error("%s", e)
        return 1

    summary_path = write_summary(result, SUMMARY_FILENAME)
    logger.info(f"Wrote summary to {summary_path}")

    if not result.success:
        failed = [s.stage for s in result.stages if s.exit_code]
        logger.error(f"Pipeline failed at: {', '.join(failed)}")
        return 1

    logger.info(f"Pipeline complete, integrated to {result.integrated_reflections}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
