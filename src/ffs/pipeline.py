"""
Stage running and result recording.

Runs compiled executables in order, stopping at the first failure, and
writes a machine-readable summary for whatever collected the job.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


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


def append_optional(command: list[str], options: dict[str, object]) -> None:
    """Add each flag whose value was set, leaving the rest to defaults."""
    for flag, value in options.items():
        if value is not None:
            command.extend([flag, str(value)])


def run_stage(stage: str, command: list[str]) -> StageResult:
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


def write_summary(result: PipelineResult, filename: str) -> Path:
    """
    Write the machine-readable job summary next to the results.

    Args:
        result:   What ran, and what it produced
        filename: Name to write under, relative to the working directory

    Returns:
        Path: The summary that was written
    """
    summary_path = result.working_directory / filename
    summary_path.write_text(result.model_dump_json(indent=2))
    return summary_path
