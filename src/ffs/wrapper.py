"""
Zocalo wrappers, so a recipe can run the pipelines on the cluster.

A recipe dispatches through the cluster submission service, whose Slurm
job loads the FFS module and runs zocalo.wrap against a name registered
under zocalo.wrappers. The wrapper reads the recipe's job_parameters,
runs the pipeline, publishes what it produced and offers those files
for attachment.

The runner sends the starting, success and failure messages itself,
from the boolean run() returns, so nothing here reports its own
outcome.

Deliberately built on zocalo's own BaseWrapper rather than the dlstbx
subclass, so that the GPU job needs nothing but this package. The
metrics dlstbx would have contributed are not reproduced here.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import ValidationError
from zocalo.wrapper import BaseWrapper

from ffs._common import ExecutableError, create_parent_symlink
from ffs.index_integrate import SUMMARY_FILENAME as INDEX_INTEGRATE_SUMMARY
from ffs.index_integrate import run_pipeline as run_index_integrate
from ffs.pipeline import PipelineResult, StageResult, write_summary
from ffs.spotfind_index_integrate import (
    SUMMARY_FILENAME as SPOTFIND_INDEX_INTEGRATE_SUMMARY,
)
from ffs.spotfind_index_integrate import SpotfindIndexIntegrateRequest
from ffs.spotfind_index_integrate import run_pipeline as run_spotfind_index_integrate
from ffs.stages import PipelineRequest

logger = logging.getLogger(__name__)

# Results carried by every pipeline, paired with the ISPyB attachment
# type and the rank that orders them in SynchWeb. Absent entries are
# the stages that did not run.
REPORTED_RESULTS = (
    ("integrated_reflections", "Result", 1),
    ("indexed_experiments", "Result", 2),
    ("strong_reflections", "Result", 2),
)

RequestT = TypeVar("RequestT", bound=PipelineRequest)


class PipelineWrapper(BaseWrapper, Generic[RequestT]):
    """
    Parameters in, pipeline run, results published.

    Subclasses name the pipeline. Nothing here knows what the stages
    are or what the binaries take.
    """

    # Key in job_parameters holding this pipeline's own parameters,
    # matching the name the recipe passes to zocalo.wrap
    parameter_key: str
    summary_filename: str
    request_model: type[RequestT]

    def build_request(self, job_parameters: dict[str, Any]) -> RequestT:
        """
        Build the request from the recipe's job_parameters.

        The pipeline's own parameters sit under parameter_key and are
        named as the model names them, so the recipe needs no
        translation. The working directory and the data collection ID
        are common to every wrapped program and sit alongside it.

        Args:
            job_parameters: The job_parameters block of the recipe step

        Returns:
            RequestT: The validated request
        """
        supplied = dict(job_parameters.get(self.parameter_key) or {})
        for key in ("dcid", "working_directory"):
            if job_parameters.get(key) is not None:
                supplied.setdefault(key, job_parameters[key])
        return self.request_model(**supplied)

    def run_pipeline(
        self,
        params: RequestT,
        on_stage: Callable[[StageResult], None],
    ) -> PipelineResult:
        raise NotImplementedError

    def report_stage(self, stage: StageResult) -> None:
        """Announce a finished stage on the recipe's updates channel."""
        self.update(f"{stage.stage} finished in {stage.duration:.1f} s")

    def publish_directory(self, path: Path, symlink_name: str | None) -> Path:
        """
        Create a directory the recipe names, and link to it.

        Args:
            path:         Directory to create
            symlink_name: Short name to reach it by from up the visit
                          tree, or None to leave no link

        Returns:
            Path: The directory, now existing
        """
        path.mkdir(parents=True, exist_ok=True)
        if symlink_name:
            try:
                create_parent_symlink(path, symlink_name)
            except (OSError, ValueError):
                # A convenience link is not worth failing a run over
                self.log.warning(
                    "Could not link to %s as %s", path, symlink_name, exc_info=True
                )
        return path

    def report_results(
        self,
        result: PipelineResult,
        summary: Path,
        job_parameters: dict[str, Any],
    ) -> None:
        """
        Copy the results where the recipe asks, and offer them up.

        Attachment fields follow the ISPyB AutoProcProgramAttachment
        columns. A result the pipeline did not reach is None and is
        skipped, as is one whose stage claimed success without leaving
        the file behind.

        Args:
            result:         What ran, and what it produced
            summary:        The machine-readable summary just written
            job_parameters: The job_parameters block of the recipe step
        """
        produced = [(summary, "Log", 2)]
        produced += [
            (path, file_type, rank)
            for field, file_type, rank in REPORTED_RESULTS
            if (path := getattr(result, field)) is not None
        ]

        symlink_name = job_parameters.get("create_symlink")
        if symlink_name:
            self.publish_directory(result.working_directory, symlink_name)

        results_directory = job_parameters.get("results_directory")
        if results_directory:
            results_directory = self.publish_directory(
                Path(results_directory), symlink_name
            )

        final = job_parameters.get("pipeline-final") or {}
        final_directory = final.get("path")
        final_patterns = final.get("patterns") or []
        if final_directory:
            final_directory = self.publish_directory(
                Path(final_directory), symlink_name
            )

        attached = []
        for path, file_type, rank in produced:
            if not path.is_file():
                self.log.warning("Not attaching %s, which was not written", path)
                continue

            if results_directory:
                path = Path(shutil.copy(path, results_directory / path.name))
            if final_directory and any(
                fnmatch(path.name, pattern) for pattern in final_patterns
            ):
                shutil.copy(path, final_directory / path.name)

            self.record_result_individual_file(
                {
                    "file_path": str(path.parent),
                    "file_name": path.name,
                    "file_type": file_type,
                    "importance_rank": rank,
                }
            )
            attached.append(str(path))

        if attached:
            self.record_result_all_files({"filelist": attached})

    def run(self) -> bool:
        """
        Run the pipeline named by this wrapper.

        Returns:
            bool: True when every stage succeeded. A bad parameter or a
                missing binary is a False rather than an exception, so
                the recipe reports a processing failure rather than a
                crashed job.
        """
        job_parameters = self.recwrap.recipe_step.get("job_parameters") or {}

        try:
            params = self.build_request(job_parameters)
        except (ValidationError, TypeError) as e:
            self.log.error("Invalid job parameters for %s: %s", self.parameter_key, e)
            return False

        try:
            result = self.run_pipeline(params, self.report_stage)
        except ExecutableError as e:
            self.log.error("%s", e)
            return False

        summary = write_summary(result, self.summary_filename)
        self.log.info("Wrote summary to %s", summary)
        self.report_results(result, summary, job_parameters)

        if not result.success:
            failed = [s.stage for s in result.stages if s.exit_code]
            self.log.error("Pipeline failed at: %s", ", ".join(failed))
            return False

        return True


class IndexIntegrateWrapper(PipelineWrapper[PipelineRequest]):
    """Index and integrate a dataset the spotfinder service has seen."""

    _logger_name = "ffs.wrapper.index_integrate"

    parameter_key = "ffs_index_integrate"
    summary_filename = INDEX_INTEGRATE_SUMMARY
    request_model = PipelineRequest

    def run_pipeline(
        self,
        params: PipelineRequest,
        on_stage: Callable[[StageResult], None],
    ) -> PipelineResult:
        return run_index_integrate(params, on_stage=on_stage)


class SpotfindIndexIntegrateWrapper(PipelineWrapper[SpotfindIndexIntegrateRequest]):
    """Spotfind, index and integrate a dataset from its raw images."""

    _logger_name = "ffs.wrapper.spotfind_index_integrate"

    parameter_key = "ffs_spotfind_index_integrate"
    summary_filename = SPOTFIND_INDEX_INTEGRATE_SUMMARY
    request_model = SpotfindIndexIntegrateRequest

    def run_pipeline(
        self,
        params: SpotfindIndexIntegrateRequest,
        on_stage: Callable[[StageResult], None],
    ) -> PipelineResult:
        return run_spotfind_index_integrate(params, on_stage=on_stage)
