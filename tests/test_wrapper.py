import json
import os
import stat
from pathlib import Path

import pytest

from ffs.wrapper import IndexIntegrateWrapper, SpotfindIndexIntegrateWrapper


class FakeRecipeWrapper:
    """
    Stands in for the RecipeWrapper zocalo.wrap builds from the message.

    Records what was sent instead of needing a transport. Channels the
    recipe does not wire are dropped by the real one, so the test
    accepts every channel and asserts on what arrived.
    """

    def __init__(self, job_parameters: dict):
        self.recipe_step = {"job_parameters": job_parameters}
        self.environment = {"ID": "test-recipe"}
        self.sent: list[tuple[str, object]] = []

    def send_to(self, channel: str, payload: object = "") -> None:
        self.sent.append((channel, payload))

    def channel(self, name: str) -> list[object]:
        return [payload for channel, payload in self.sent if channel == name]


def write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def stub_binaries(tmp_path, monkeypatch):
    """Point the three binaries at stubs, as the pipeline's own tests do."""
    probe_guard = 'if [ "$1" = "--list-devices" ]; then exit 0; fi'

    def install(
        spotfinder_body: str = "touch results_ffs.h5",
        indexer_body: str = "touch indexed.expt indexed.refl",
        integrator_body: str = "touch integrated.refl",
    ):
        monkeypatch.setenv(
            "SPOTFINDER",
            os.fspath(
                write_script(
                    tmp_path / "stub_spotfinder", f"{probe_guard}\n{spotfinder_body}"
                )
            ),
        )
        monkeypatch.setenv(
            "INDEXER", os.fspath(write_script(tmp_path / "stub_indexer", indexer_body))
        )
        monkeypatch.setenv(
            "INTEGRATOR",
            os.fspath(
                write_script(
                    tmp_path / "stub_integrator", f"{probe_guard}\n{integrator_body}"
                )
            ),
        )

    return install


@pytest.fixture
def visit(tmp_path):
    """
    A visit tree deep enough for the parent symlinks to have somewhere
    to go, laid out as the beamline lays one out.
    """
    return tmp_path / "visit"


@pytest.fixture
def job_parameters(tmp_path, visit):
    """
    The job_parameters block a recipe step would carry.

    Shaped as the xia2 and fast_dp recipes shape theirs: the program's
    own parameters under its wrapper name, with the working directory
    and the DCID alongside.
    """
    data = tmp_path / "images.nxs"
    experiment = tmp_path / "imported.expt"
    data.touch()
    experiment.touch()

    def make(key: str, **overrides):
        pipeline = {
            "experiment": os.fspath(experiment),
            "max_cell": 100.0,
        }
        pipeline.update(
            {"data": os.fspath(data)}
            if key == "ffs_spotfind_index_integrate"
            else {"reflection": os.fspath(data)}
        )
        pipeline.update(overrides)
        return {
            key: pipeline,
            "dcid": 12345,
            "working_directory": os.fspath(visit / "tmp" / "12345" / "ffs"),
        }

    return make


@pytest.fixture
def publishing(visit):
    """The directories a recipe asks for the results to be published to."""
    return {
        "results_directory": os.fspath(visit / "processed" / "12345" / "ffs"),
        "create_symlink": "ffs",
        "pipeline-final": {
            "path": os.fspath(visit / "final" / "12345" / "ffs"),
            "patterns": ["integrated.refl", "ffs_spotfind_index_integrate.json"],
        },
    }


def run_wrapper(wrapper_class, parameters):
    """Drive a wrapper the way zocalo.wrap drives it, minus the transport."""
    wrapper = wrapper_class()
    recwrap = FakeRecipeWrapper(parameters)
    wrapper.set_recipe_wrapper(recwrap)
    return wrapper.run(), recwrap


def attachments(recwrap) -> dict[str, dict]:
    """The individual-file payloads, keyed by the file they name."""
    return {p["file_name"]: p for p in recwrap.channel("result-individual-file")}


def test_the_recipe_parameters_reach_the_request_model(job_parameters):
    wrapper = SpotfindIndexIntegrateWrapper()
    parameters = job_parameters("ffs_spotfind_index_integrate", dmin=1.81)

    params = wrapper.build_request(parameters)

    assert params.dcid == 12345, "the DCID must come from the common block"
    assert params.max_cell == 100.0, "the pipeline block must reach the model"
    assert params.dmin == 1.81, "optional parameters must reach the model"
    assert params.working_directory.name == "ffs", (
        "the working directory is common to every wrapped program, not per-pipeline"
    )


def test_a_successful_run_reports_every_file_it_produced(
    tmp_path, job_parameters, stub_binaries, monkeypatch
):
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    succeeded, recwrap = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    assert succeeded is True, "three succeeding stages must report success"
    attached = attachments(recwrap)
    assert "ffs_spotfind_index_integrate.json" in attached, (
        "the summary must be offered for attachment"
    )
    assert "integrated.refl" in attached, "the integrated reflections must be attached"
    assert "results_ffs.h5" in attached, "the spotfinder output must be attached"
    assert all(p["file_path"] for p in attached.values()), (
        "every attachment needs a directory, since ISPyB stores path and name apart"
    )


def test_each_finished_stage_is_announced(
    tmp_path, job_parameters, stub_binaries, monkeypatch
):
    """The recipe routes updates to update_processing_status."""
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    _, recwrap = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    updates = recwrap.channel("updates")
    assert len(updates) == 3, "one update per stage, so a stalled run is visible"
    assert "spotfinder" in updates[0], "updates must name the stage that finished"


def test_a_failed_stage_reports_failure_without_raising(
    tmp_path, job_parameters, stub_binaries, monkeypatch
):
    stub_binaries(indexer_body="exit 3")
    monkeypatch.chdir(tmp_path)

    succeeded, recwrap = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    assert succeeded is False, "a failed stage must return False, not raise"
    assert recwrap.channel("result-individual-file"), (
        "the summary must still be attached, since it records where it failed"
    )


def test_the_wrapper_does_not_report_its_own_outcome(
    tmp_path, job_parameters, stub_binaries, monkeypatch
):
    """zocalo.wrap sends success and failure itself, from the returned bool."""
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    _, recwrap = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    assert not recwrap.channel("success"), "double-reporting would confuse ISPyB"
    assert not recwrap.channel("failure"), "the runner owns the outcome channels"
    assert not recwrap.channel("starting"), "the runner sends starting before run()"


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"ffs_spotfind_index_integrate": {}},
        {"ffs_spotfind_index_integrate": {"data": ""}, "working_directory": "/x"},
    ],
    ids=["no job_parameters", "empty block", "empty data path"],
)
def test_unusable_parameters_fail_rather_than_raise(parameters, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    succeeded, _ = run_wrapper(SpotfindIndexIntegrateWrapper, parameters)

    assert succeeded is False, (
        "a bad recipe must land as a processing failure, not a crashed job"
    )


def test_a_missing_binary_fails_rather_than_raising(
    tmp_path, job_parameters, monkeypatch
):
    monkeypatch.setenv("SPOTFINDER", os.fspath(tmp_path / "absent_spotfinder"))
    monkeypatch.chdir(tmp_path)

    succeeded, _ = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    assert succeeded is False, "a packaging fault must not escape as an exception"


def test_index_integrate_reads_its_own_parameter_block(
    tmp_path, job_parameters, stub_binaries, monkeypatch
):
    """Each wrapper takes the block named for the wrapper the recipe calls."""
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    succeeded, recwrap = run_wrapper(
        IndexIntegrateWrapper, job_parameters("ffs_index_integrate")
    )

    assert succeeded is True, "the index-integrate block must reach its own wrapper"
    assert "ffs_index_integrate.json" in attachments(recwrap), (
        "each pipeline writes its own summary, so both can share a directory"
    )


def test_the_two_wrappers_do_not_share_a_parameter_key():
    assert (
        SpotfindIndexIntegrateWrapper.parameter_key
        != IndexIntegrateWrapper.parameter_key
    ), "one recipe must be able to carry parameters for both"


def test_results_are_copied_to_the_results_directory_and_attached_from_there(
    tmp_path, job_parameters, publishing, stub_binaries, monkeypatch
):
    """The working directory is scratch, so nothing is attached from it."""
    stub_binaries()
    monkeypatch.chdir(tmp_path)
    parameters = job_parameters("ffs_spotfind_index_integrate") | publishing
    results_directory = Path(publishing["results_directory"])

    succeeded, recwrap = run_wrapper(SpotfindIndexIntegrateWrapper, parameters)

    assert succeeded is True, "publishing must not disturb the outcome"
    attached = attachments(recwrap)
    assert (results_directory / "integrated.refl").is_file(), (
        "the results must be copied out of the working directory"
    )
    assert all(
        p["file_path"] == os.fspath(results_directory) for p in attached.values()
    ), "attachments must point at the published copy, not the scratch one"


def test_only_the_named_patterns_reach_the_final_directory(
    tmp_path, job_parameters, publishing, stub_binaries, monkeypatch
):
    stub_binaries()
    monkeypatch.chdir(tmp_path)
    final_directory = Path(publishing["pipeline-final"]["path"])

    run_wrapper(
        SpotfindIndexIntegrateWrapper,
        job_parameters("ffs_spotfind_index_integrate") | publishing,
    )

    assert {p.name for p in final_directory.iterdir()} == {
        "integrated.refl",
        "ffs_spotfind_index_integrate.json",
    }, "the final directory carries the named patterns and nothing else"


def test_each_published_directory_is_linked_from_the_visit(
    tmp_path, visit, job_parameters, publishing, stub_binaries, monkeypatch
):
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    run_wrapper(
        SpotfindIndexIntegrateWrapper,
        job_parameters("ffs_spotfind_index_integrate") | publishing,
    )

    for parent in ("tmp", "processed", "final"):
        link = visit / parent / "ffs"
        assert link.is_symlink(), f"{parent} must carry a link to the run"
        assert link.resolve().name == "ffs", f"the {parent} link must reach the results"


def test_publishing_is_skipped_when_the_recipe_asks_for_none(
    tmp_path, visit, job_parameters, stub_binaries, monkeypatch
):
    """Running the pipeline by hand does not create beamline directories."""
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    _, recwrap = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    assert not (visit / "processed").exists(), (
        "no results directory means nothing is published"
    )
    assert all(p["file_path"].endswith("ffs") for p in attachments(recwrap).values()), (
        "attachments fall back to the working directory"
    )


def test_a_stage_that_succeeds_without_writing_its_file_is_not_attached(
    tmp_path, job_parameters, stub_binaries, monkeypatch
):
    """A zero exit code is not proof the output landed."""
    stub_binaries(integrator_body="true")
    monkeypatch.chdir(tmp_path)

    succeeded, recwrap = run_wrapper(
        SpotfindIndexIntegrateWrapper, job_parameters("ffs_spotfind_index_integrate")
    )

    assert succeeded is True, "the exit code is what decides the outcome"
    assert "integrated.refl" not in attachments(recwrap), (
        "attaching a path to a file that is not there would break SynchWeb"
    )


def test_the_published_summary_records_the_outcome(
    tmp_path, job_parameters, publishing, stub_binaries, monkeypatch
):
    stub_binaries()
    monkeypatch.chdir(tmp_path)

    run_wrapper(
        SpotfindIndexIntegrateWrapper,
        job_parameters("ffs_spotfind_index_integrate") | publishing,
    )

    summary = json.loads(
        (
            Path(publishing["results_directory"]) / "ffs_spotfind_index_integrate.json"
        ).read_text()
    )
    assert summary["success"] is True, (
        "the summary must stand alone once it is attached"
    )
    assert [s["stage"] for s in summary["stages"]] == [
        "spotfinder",
        "indexer",
        "integrator",
    ], "the summary must record the stages in the order they ran"
