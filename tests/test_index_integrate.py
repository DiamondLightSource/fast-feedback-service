import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from ffs._common import DeviceProbeFailed, ExecutableNotFound
from ffs.index_integrate import SUMMARY_FILENAME, run_pipeline
from ffs.stages import PipelineRequest


@pytest.fixture
def inputs(tmp_path):
    """A reflection table and experiment list that exist, but are empty."""
    reflection = tmp_path / "strong.refl"
    experiment = tmp_path / "imported.expt"
    reflection.touch()
    experiment.touch()
    return reflection, experiment


@pytest.fixture
def request_for(tmp_path, inputs):
    """Build a minimal request, with room to override fields."""
    reflection, experiment = inputs

    def make(**overrides):
        return PipelineRequest(
            reflection=reflection,
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
    Point INDEXER and INTEGRATOR at scripts we control.

    Each writes the file the real binary would, so that run_pipeline's
    bookkeeping can be checked without a GPU.
    """

    # find_executable probes the CUDA tools with --list-devices before
    # using them, so the stub integrator has to answer that first
    probe_guard = 'if [ "$1" = "--list-devices" ]; then exit 0; fi'

    def install(indexer_body: str, integrator_body: str):
        indexer = write_script(tmp_path / "stub_indexer", indexer_body)
        integrator = write_script(
            tmp_path / "stub_integrator", f"{probe_guard}\n{integrator_body}"
        )
        monkeypatch.setenv("INDEXER", os.fspath(indexer))
        monkeypatch.setenv("INTEGRATOR", os.fspath(integrator))
        return indexer, integrator

    return install


def test_pipeline_creates_and_uses_the_working_directory(
    tmp_path, request_for, stub_binaries, monkeypatch
):
    stub_binaries(
        indexer_body="touch indexed.expt indexed.refl",
        integrator_body="touch integrated.refl",
    )
    # Start somewhere else, so the chdir is what puts files in place
    monkeypatch.chdir(tmp_path)
    params = request_for()

    result = run_pipeline(params)

    assert result.success
    work = tmp_path / "work"
    assert work.is_dir()
    assert (work / "indexed.expt").is_file()
    assert (work / "indexed.refl").is_file()
    assert (work / "integrated.refl").is_file()
    assert result.integrated_reflections == work / "integrated.refl"
    assert [s.stage for s in result.stages] == ["indexer", "integrator"]


def test_pipeline_stops_when_the_indexer_fails(
    tmp_path, request_for, stub_binaries, monkeypatch
):
    stub_binaries(
        indexer_body="exit 3",
        integrator_body="touch integrator_should_not_run",
    )
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(request_for())

    assert not result.success
    assert [s.stage for s in result.stages] == ["indexer"]
    assert result.stages[0].exit_code == 3
    assert result.indexed_reflections is None
    assert not (tmp_path / "work" / "integrator_should_not_run").exists()


def test_pipeline_reports_integrator_failure(
    tmp_path, request_for, stub_binaries, monkeypatch
):
    stub_binaries(
        indexer_body="touch indexed.expt indexed.refl",
        integrator_body="exit 1",
    )
    monkeypatch.chdir(tmp_path)

    result = run_pipeline(request_for())

    assert not result.success
    assert [s.stage for s in result.stages] == ["indexer", "integrator"]
    # The indexer did succeed, so its outputs are still recorded
    assert result.indexed_reflections is not None
    assert result.integrated_reflections is None


def test_entrypoint_writes_a_summary(tmp_path, inputs, stub_binaries, monkeypatch):
    from ffs.index_integrate import run

    reflection, experiment = inputs
    stub_binaries(
        indexer_body="touch indexed.expt indexed.refl",
        integrator_body="touch integrated.refl",
    )
    monkeypatch.chdir(tmp_path)

    exit_code = run(
        [
            "--reflection",
            os.fspath(reflection),
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
    assert [s["stage"] for s in summary["stages"]] == ["indexer", "integrator"]
    assert all(s["exit_code"] == 0 for s in summary["stages"])
    assert summary["integrated_reflections"].endswith("integrated.refl")


def test_entrypoint_rejects_missing_inputs(tmp_path, monkeypatch):
    from ffs.index_integrate import run

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run(
            [
                "--reflection",
                os.fspath(tmp_path / "absent.refl"),
                "--experiment",
                os.fspath(tmp_path / "absent.expt"),
                "--working-directory",
                os.fspath(tmp_path / "work"),
                "--max-cell",
                "100",
            ]
        )


def test_run_pipeline_raises_when_an_executable_is_missing(
    request_for, tmp_path, monkeypatch
):
    monkeypatch.setenv("INDEXER", os.fspath(tmp_path / "absent_indexer"))

    with pytest.raises(ExecutableNotFound, match="INDEXER"):
        run_pipeline(request_for())

    assert not (tmp_path / "work").exists(), (
        "the working directory must not be created when a binary is missing"
    )


def test_run_pipeline_reports_the_probe_output_when_the_device_check_fails(
    request_for, tmp_path, monkeypatch
):
    """The probe's stderr separates a driver mismatch from an absent device."""
    # The indexer is looked up first, so it has to survive for the
    # integrator's probe to be the thing that fails
    monkeypatch.setenv(
        "INDEXER", os.fspath(write_script(tmp_path / "stub_indexer", "exit 0"))
    )
    monkeypatch.setenv(
        "INTEGRATOR",
        os.fspath(
            write_script(
                tmp_path / "sick_integrator",
                'echo "CUDA driver version is insufficient" >&2\nexit 1',
            )
        ),
    )

    with pytest.raises(DeviceProbeFailed) as excinfo:
        run_pipeline(request_for())

    assert "CUDA driver version is insufficient" in str(excinfo.value), (
        "the probe's own output must reach the exception message"
    )


def test_entrypoint_reports_a_missing_executable(inputs, tmp_path, monkeypatch):
    from ffs.index_integrate import run

    reflection, experiment = inputs
    monkeypatch.setenv("INDEXER", os.fspath(tmp_path / "absent_indexer"))

    exit_code = run(
        [
            "--reflection",
            os.fspath(reflection),
            "--experiment",
            os.fspath(experiment),
            "--working-directory",
            os.fspath(tmp_path / "work"),
            "--max-cell",
            "100",
        ]
    )

    assert exit_code == 1, "a missing binary must exit non-zero"
    assert not (tmp_path / "work" / SUMMARY_FILENAME).exists(), (
        "no summary should be written when neither stage ran"
    )


@pytest.mark.parametrize("name", ["INDEXER", "INTEGRATOR"])
def test_executables_run(name):
    """Smoke test that the built binaries exist and start."""
    path = os.getenv(name)
    assert path is not None
    if not Path(path).is_file():
        pytest.skip(f"{name} has not been built")

    # Neither accepts --help without arguments cleanly, so use --version
    # for the CUDA tool and a bare run for the indexer, which reports
    # its own usage error rather than crashing.
    flag = "--version" if name == "INTEGRATOR" else "--help"
    result = subprocess.run([path, flag], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
