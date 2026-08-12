"""
Indexing and integration pipeline of a single dataset.

Takes a strong reflection table and an experiment list and then chains
the baseline indexer and the GPU integrator.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from pydantic import ValidationError

from ffs._common import ExecutableError, find_executable, setup_rich_logging
from ffs.pipeline import PipelineResult, StageResult, run_stage, write_summary
from ffs.stages import (
    INDEXED_EXPERIMENTS,
    INDEXED_REFLECTIONS,
    PipelineRequest,
    add_tuning_arguments,
    build_indexer_command,
    build_integrator_command,
)

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "ffs_index_integrate.json"


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
            before the working directory is created.
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

    def record(stage: str, command: list[str]) -> StageResult:
        stage_result = run_stage(stage, command)
        result.stages.append(stage_result)
        return stage_result

    if record("indexer", build_indexer_command(indexer, params)).exit_code:
        return result
    result.indexed_experiments = working_directory / INDEXED_EXPERIMENTS
    result.indexed_reflections = working_directory / INDEXED_REFLECTIONS

    if record(
        "integrator", build_integrator_command(integrator, params, output)
    ).exit_code:
        return result
    result.integrated_reflections = output

    return result


def build_parser() -> argparse.ArgumentParser:
    """
    Assemble the command line.

    Every field of the request model has a flag here, named by
    replacing its underscores with hyphens. Whatever runs this as a
    subprocess relies on that, so it is locked by a test rather than
    left as a convention.

    Returns:
        argparse.ArgumentParser: The parser the entrypoint runs
    """
    parser = argparse.ArgumentParser(
        description="Index and integrate a single dataset, then exit."
    )
    parser.add_argument(
        "--reflection",
        required=True,
        help="Strong reflection table (HDF5)",
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
        help="Resolution limit",
    )
    add_tuning_arguments(parser)
    return parser


def run(args=None) -> int:
    """
    Command line entrypoint.

    Returns:
        int: Zero when both stages succeeded
    """
    setup_rich_logging()

    options = build_parser().parse_args(args)
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
