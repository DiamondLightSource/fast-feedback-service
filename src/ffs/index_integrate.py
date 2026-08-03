"""
Indexing and integration pipeline of a single dataset.

This is the entrypoint of the index-integrate container image. It is handed a
strong reflection table and a dials.import experiment list, runs the
baseline indexer followed by the GPU integrator, and exits. Results
are written to a working directory named by the caller.

The working directory is not a convenience: the baseline indexer
writes indexed.expt and indexed.refl under fixed names relative to
the current directory and has no output path option, so the pipeline
changes directory before running anything.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError

from ffs._common import ExecutableError, find_executable, setup_rich_logging

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "ffs_index_integrate.json"

# Written by the indexer under fixed names, relative to the CWD
INDEXED_EXPERIMENTS = Path("indexed.expt")
INDEXED_REFLECTIONS = Path("indexed.refl")


class PipelineRequest(BaseModel):
    """Every parameter the job takes, as named by the recipe."""

    # Inputs
    reflection: Path
    experiment: Path
    working_directory: Path

    max_cell: float

    # Shared, and traceability
    dmin: Optional[float] = None
    dcid: Optional[int] = None

    # Indexer tuning
    max_refine: Optional[int] = None
    macro_cycles: Optional[int] = None
    nthreads: Optional[int] = None

    # Integrator tuning
    output: Path = Path("integrated.refl")
    algorithm: Optional[str] = None
    background: Optional[str] = None
    sigma_b: Optional[float] = None
    sigma_m: Optional[float] = None
    min_zeta: Optional[float] = None
    min_bbox_depth: Optional[int] = None
    threads: Optional[int] = None
    timeout: Optional[float] = None


class StageResult(BaseModel):
    """Outcome of one executable invocation."""

    stage: str
    command: list[str]
    exit_code: int
    duration: float


class PipelineResult(BaseModel):
    """Summary of the whole job, written to the working directory."""

    dcid: Optional[int] = None
    working_directory: Path
    stages: list[StageResult]
    indexed_experiments: Optional[Path] = None
    indexed_reflections: Optional[Path] = None
    integrated_reflections: Optional[Path] = None

    @property
    def success(self) -> bool:
        return bool(self.stages) and all(s.exit_code == 0 for s in self.stages)


def _append_optional(command: list[str], options: dict[str, object]) -> None:
    """Add each flag whose value was set, leaving the rest to defaults."""
    for flag, value in options.items():
        if value is not None:
            command.extend([flag, str(value)])


def build_indexer_command(executable: Path, params: PipelineRequest) -> list[str]:
    """
    Assemble the baseline indexer command line.

    Note that the indexer spells its multi-word options with hyphens,
    unlike the integrator.

    Args:
        executable: Path to the baseline_indexer binary
        params:     The validated request

    Returns:
        list[str]: The argv to hand to subprocess
    """
    command = [
        str(executable),
        "--refl",
        str(params.reflection),
        "--expt",
        str(params.experiment),
        "--max-cell",
        str(params.max_cell),
    ]
    _append_optional(
        command,
        {
            "--dmin": params.dmin,
            "--max-refine": params.max_refine,
            "--macro-cycles": params.macro_cycles,
            "--nthreads": params.nthreads,
        },
    )
    return command


def build_integrator_command(
    executable: Path, params: PipelineRequest, output: Path
) -> list[str]:
    """
    Assemble the integrator command line.

    The integrator consumes the indexer's output directly: it predicts
    internally when the reflection flags say the table is not already
    predicted, and estimates the sigmas from the variance columns the
    indexer carries through from the strong spots.

    Args:
        executable: Path to the integrator binary
        params:     The validated request
        output:     Resolved path for the integrated reflections

    Returns:
        list[str]: The argv to hand to subprocess
    """
    command = [
        str(executable),
        "--reflection",
        str(INDEXED_REFLECTIONS),
        "--experiment",
        str(INDEXED_EXPERIMENTS),
        "--output",
        str(output),
    ]
    _append_optional(
        command,
        {
            "--algorithm": params.algorithm,
            "--background": params.background,
            "--sigma_b": params.sigma_b,
            "--sigma_m": params.sigma_m,
            "--min_zeta": params.min_zeta,
            "--min_bbox_depth": params.min_bbox_depth,
            "--threads": params.threads,
            "--timeout": params.timeout,
        },
    )
    return command


def _run_stage(stage: str, command: list[str]) -> StageResult:
    """Run one executable, logging its output and timing it."""
    logger.info(f"Running {stage}: {' '.join(command)}")

    start_time = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True)
    duration = time.monotonic() - start_time

    if proc.stdout:
        logger.info(proc.stdout.rstrip())
    if proc.returncode:
        logger.error(f"{stage} failed with exit code {proc.returncode}")
        if proc.stderr:
            logger.error(proc.stderr.rstrip())
    else:
        logger.info(f"{stage} complete in {duration:.1f} s")

    return StageResult(
        stage=stage,
        command=command,
        exit_code=proc.returncode,
        duration=duration,
    )


def run_pipeline(params: PipelineRequest) -> PipelineResult:
    """
    Index and then integrate one dataset.

    Creates the working directory and changes into it, so that the
    files the indexer writes under fixed names land alongside the
    integrated output. Stops at the first stage that fails.

    Args:
        params: The validated request

    Returns:
        PipelineResult: What ran, and what it produced

    Raises:
        ExecutableError: Either binary is missing or unusable. Raised
            before anything is created, so there is no partial result.
    """
    indexer = find_executable("INDEXER", "baseline_indexer", probe=False)
    integrator = find_executable("INTEGRATOR", "integrator")

    # Resolve the inputs before moving, so that relative paths on the
    # command line still mean what the caller meant
    params = params.model_copy(
        update={
            "reflection": params.reflection.resolve(),
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

    result.stages.append(_run_stage("indexer", build_indexer_command(indexer, params)))
    if result.stages[-1].exit_code:
        return result
    result.indexed_experiments = working_directory / INDEXED_EXPERIMENTS
    result.indexed_reflections = working_directory / INDEXED_REFLECTIONS

    result.stages.append(
        _run_stage("integrator", build_integrator_command(integrator, params, output))
    )
    if result.stages[-1].exit_code:
        return result
    result.integrated_reflections = output

    return result


def write_summary(result: PipelineResult) -> Path:
    """Write the machine-readable job summary next to the results."""
    summary_path = result.working_directory / SUMMARY_FILENAME
    summary_path.write_text(result.model_dump_json(indent=2))
    return summary_path


def run(args=None) -> int:
    """
    Command line entrypoint, mirroring the recipe parameters.

    Returns:
        int: Zero when both stages succeeded
    """
    setup_rich_logging()

    parser = argparse.ArgumentParser(
        description="Index and integrate a single dataset, then exit."
    )
    parser.add_argument(
        "--reflection", required=True, help="Strong reflection table (HDF5)"
    )
    parser.add_argument(
        "--experiment", required=True, help="Experiment list from dials.import"
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
    parser.add_argument("--dmin", type=float, help="Resolution limit")
    parser.add_argument("--dcid", type=int, help="Data collection ID, for logging")
    parser.add_argument("--max-refine", type=int, help="Candidate lattices to refine")
    parser.add_argument("--macro-cycles", type=int, help="Post-indexing macrocycles")
    parser.add_argument("--nthreads", type=int, help="Threads for the indexer FFT")
    parser.add_argument("--output", help="Integrated reflection table filename")
    parser.add_argument("--algorithm", help="Foreground algorithm: dials or ellipsoid")
    parser.add_argument("--background", help="Background model: constant, tukey or glm")
    parser.add_argument("--sigma-b", dest="sigma_b", type=float, help="σ_b in degrees")
    parser.add_argument("--sigma-m", dest="sigma_m", type=float, help="σ_m in degrees")
    parser.add_argument("--min-zeta", dest="min_zeta", type=float, help="Minimum zeta")
    parser.add_argument(
        "--min-bbox-depth",
        dest="min_bbox_depth",
        type=int,
        help="Images a reflection must span to contribute to sigma estimation",
    )
    parser.add_argument("--threads", type=int, help="Integrator reader threads")
    parser.add_argument("--timeout", type=float, help="Seconds to wait for images")

    options = parser.parse_args(args)
    # Drop unset options so that the model defaults apply
    supplied = {k: v for k, v in vars(options).items() if v is not None}

    try:
        params = PipelineRequest(**supplied)
    except ValidationError as e:
        sys.exit(f"Invalid parameters:\n{e}")

    for description, path in (
        ("Reflection", params.reflection),
        ("Experiment", params.experiment),
    ):
        if not path.is_file():
            sys.exit(f"Error: {description} file not found: {path}")

    try:
        result = run_pipeline(params)
    except ExecutableError as e:
        # Nothing ran, so there is no summary to write
        logger.error("%s", e)
        return 1

    summary_path = write_summary(result)
    logger.info(f"Wrote summary to {summary_path}")

    if not result.success:
        failed = [s.stage for s in result.stages if s.exit_code]
        logger.error(f"Pipeline failed at: {', '.join(failed)}")
        return 1

    logger.info(f"Pipeline complete, integrated to {result.integrated_reflections}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
