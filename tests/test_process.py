import json
import os
import stat
from pathlib import Path

import pytest

from ffs._common import ExecutableNotFound
from ffs.process import (
    SUMMARY_FILENAME,
    ProcessRequest,
    build_spotfinder_command,
    run_pipeline,
)


@pytest.fixture
def inputs(tmp_path):
    """A data file and experiment list that exist, but are empty."""
    data = tmp_path / "images.nxs"
    experiment = tmp_path / "imported.expt"
    data.touch()
    experiment.touch()
    return data, experiment


@pytest.fixture
def request_for(tmp_path, inputs):
    """Build a minimal request, with room to override fields."""
    data, experiment = inputs

    def make(**overrides):
        return ProcessRequest(
            data=data,
            experiment=experiment,
            working_directory=tmp_path / "work",
            max_cell=100.0,
            **overrides,
        )

    return make


def write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def stub_binaries(tmp_path, monkeypatch):
    """
    Point SPOTFINDER, INDEXER and INTEGRATOR at stub scripts.

    Each writes the file the real binary would, so that run_pipeline's
    bookkeeping can be checked without a GPU.
    """

    # find_executable probes the CUDA tools with --list-devices before
    # using them, so those stubs have to answer that first. The indexer
    # is not probed, so it does not need the guard.
    probe_guard = 'if [ "$1" = "--list-devices" ]; then exit 0; fi'

    def install(spotfinder_body: str, indexer_body: str, integrator_body: str):
        spotfinder = write_script(
            tmp_path / "stub_spotfinder", f"{probe_guard}\n{spotfinder_body}"
        )
        indexer = write_script(tmp_path / "stub_indexer", indexer_body)
        integrator = write_script(
            tmp_path / "stub_integrator", f"{probe_guard}\n{integrator_body}"
        )
        monkeypatch.setenv("SPOTFINDER", os.fspath(spotfinder))
        monkeypatch.setenv("INDEXER", os.fspath(indexer))
        monkeypatch.setenv("INTEGRATOR", os.fspath(integrator))
        return spotfinder, indexer, integrator

    return install


@pytest.fixture
def working_stubs(stub_binaries):
    """Stubs that all succeed, writing what the real binaries write."""
    return stub_binaries(
        spotfinder_body="touch results_ffs.h5",
        indexer_body="touch indexed.expt indexed.refl",
        integrator_body="touch integrated.refl",
    )


def test_spotfinder_command_takes_the_data_positionally(request_for):
    params = request_for()
    command = build_spotfinder_command(Path("spotfinder"), params)

    assert command[:2] == ["spotfinder", str(params.data)], (
        "the data path is a positional argument, not a flag"
    )
    assert "--save-h5" in command, (
        "the HDF5 table is how the spots reach the indexer, so it is not optional"
    )
    assert "--threads" not in command, "unset options must be left to the binary"


def test_spotfinder_command_uses_the_unprefixed_flags(request_for):
    """The prefixed request fields exist to avoid clashing with later stages."""
    params = request_for(
        spotfinder_algorithm="dispersion",
        spotfinder_threads=40,
        spotfinder_timeout=30.0,
        images=900,
        start_index=1,
        wavelength=0.9795,
        dmin=1.81,
        dmax=40.0,
        min_spot_size=3,
        min_spot_size_3d=3,
        max_peak_centroid_separation=2.0,
        detector='{"pixel_size": 0.075}',
    )
    command = build_spotfinder_command(Path("spotfinder"), params)

    for flag, value in [
        ("--algorithm", "dispersion"),
        ("--threads", "40"),
        ("--timeout", "30.0"),
        ("--images", "900"),
        ("--start-index", "1"),
        ("--wavelength", "0.9795"),
        ("--dmin", "1.81"),
        ("--dmax", "40.0"),
        ("--min-spot-size", "3"),
        ("--min-spot-size-3d", "3"),
        ("--max-peak-centroid-separation", "2.0"),
        ("--detector", '{"pixel_size": 0.075}'),
    ]:
        assert command[command.index(flag) + 1] == value, (
            f"{flag} must reach the spotfinder unprefixed"
        )


def test_the_prefixed_and_unprefixed_tuning_flags_reach_different_stages(request_for):
    """--threads, --timeout and --algorithm mean different things per binary."""
    from ffs.stages import build_integrator_command

    params = request_for(
        spotfinder_algorithm="dispersion",
        spotfinder_threads=40,
        spotfinder_timeout=30.0,
        algorithm="dials",
        threads=8,
        timeout=60.0,
    )
    spotfinder = build_spotfinder_command(Path("spotfinder"), params)
    integrator = build_integrator_command(
        Path("integrator"), params, Path("integrated.refl")
    )

    assert spotfinder[spotfinder.index("--threads") + 1] == "40"
    assert integrator[integrator.index("--threads") + 1] == "8"
    assert spotfinder[spotfinder.index("--timeout") + 1] == "30.0"
    assert integrator[integrator.index("--timeout") + 1] == "60.0"
    assert spotfinder[spotfinder.index("--algorithm") + 1] == "dispersion"
    assert integrator[integrator.index("--algorithm") + 1] == "dials"


def test_the_indexer_reads_what_the_spotfinder_wrote(request_for):
    """The strong reflections are produced by stage one, not supplied."""
    from ffs.stages import build_indexer_command

    command = build_indexer_command(Path("baseline_indexer"), request_for())

    assert command[command.index("--refl") + 1] == "results_ffs.h5", (
        "the indexer must read the spotfinder's output, relative to the working "
        "directory both stages run in"
    )


def test_pipeline_runs_all_three_stages_in_order(
    tmp_path, request_for, working_stubs, monkeypatch
):
    # Start somewhere else, so the chdir is what puts files in place
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(request_for())

    assert result.success
    assert [s.stage for s in result.stages] == ["spotfinder", "indexer", "integrator"]
    work = tmp_path / "work"
    assert (work / "results_ffs.h5").is_file()
    assert (work / "indexed.refl").is_file()
    assert (work / "integrated.refl").is_file()
    assert result.strong_reflections == work / "results_ffs.h5"
    assert result.integrated_reflections == work / "integrated.refl"


def test_pipeline_stops_when_the_spotfinder_fails(
    tmp_path, request_for, stub_binaries, monkeypatch
):
    stub_binaries(
        spotfinder_body="exit 2",
        indexer_body="touch indexer_should_not_run",
        integrator_body="touch integrator_should_not_run",
    )
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(request_for())

    assert not result.success
    assert [s.stage for s in result.stages] == ["spotfinder"], (
        "a failed spotfinder leaves the later stages nothing to read"
    )
    assert result.stages[0].exit_code == 2
    assert result.strong_reflections is None
    assert not (tmp_path / "work" / "indexer_should_not_run").exists()


def test_pipeline_stops_when_the_indexer_fails(
    tmp_path, request_for, stub_binaries, monkeypatch
):
    stub_binaries(
        spotfinder_body="touch results_ffs.h5",
        indexer_body="exit 3",
        integrator_body="touch integrator_should_not_run",
    )
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(request_for())

    assert not result.success
    assert [s.stage for s in result.stages] == ["spotfinder", "indexer"]
    # The spotfinder did succeed, so its output is still recorded
    assert result.strong_reflections is not None
    assert result.indexed_reflections is None


def test_entrypoint_writes_a_summary(tmp_path, inputs, working_stubs, monkeypatch):
    from ffs.process import run

    data, experiment = inputs
    monkeypatch.chdir(tmp_path)

    exit_code = run(
        [
            "--data",
            os.fspath(data),
            "--experiment",
            os.fspath(experiment),
            "--working-directory",
            os.fspath(tmp_path / "work"),
            "--max-cell",
            "100",
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "work" / SUMMARY_FILENAME).read_text())
    assert [s["stage"] for s in summary["stages"]] == [
        "spotfinder",
        "indexer",
        "integrator",
    ]
    assert all(s["exit_code"] == 0 for s in summary["stages"])
    assert summary["strong_reflections"].endswith("results_ffs.h5")


def test_the_two_pipelines_do_not_share_a_summary_filename():
    """Both may be run into the same working directory."""
    from ffs.index_integrate import SUMMARY_FILENAME as INDEX_INTEGRATE_SUMMARY

    assert SUMMARY_FILENAME != INDEX_INTEGRATE_SUMMARY, (
        "one summary must not overwrite the other"
    )


def test_entrypoint_rejects_missing_inputs(tmp_path, monkeypatch):
    from ffs.process import run

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run(
            [
                "--data",
                os.fspath(tmp_path / "absent.nxs"),
                "--experiment",
                os.fspath(tmp_path / "absent.expt"),
                "--working-directory",
                os.fspath(tmp_path / "work"),
                "--max-cell",
                "100",
            ]
        )


def test_entrypoint_accepts_a_directory_of_images(
    tmp_path, inputs, working_stubs, monkeypatch
):
    """Live processing points the spotfinder at /dev/shm, not at a file."""
    from ffs.process import run

    _, experiment = inputs
    data_directory = tmp_path / "shm"
    data_directory.mkdir()
    monkeypatch.chdir(tmp_path)

    exit_code = run(
        [
            "--data",
            os.fspath(data_directory),
            "--experiment",
            os.fspath(experiment),
            "--working-directory",
            os.fspath(tmp_path / "work"),
            "--max-cell",
            "100",
        ]
    )

    assert exit_code == 0, "a directory is a valid data input, not a missing file"


def test_run_pipeline_raises_when_the_spotfinder_is_missing(
    request_for, tmp_path, monkeypatch
):
    monkeypatch.setenv("SPOTFINDER", os.fspath(tmp_path / "absent_spotfinder"))

    with pytest.raises(ExecutableNotFound, match="SPOTFINDER"):
        run_pipeline(request_for())

    assert not (tmp_path / "work").exists(), (
        "the working directory must not be created when a binary is missing"
    )
