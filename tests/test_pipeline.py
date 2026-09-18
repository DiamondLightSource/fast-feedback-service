import json
import stat
from pathlib import Path

import pytest

from ffs.pipeline import (
    PipelineResult,
    StageResult,
    append_optional,
    run_stage,
    write_summary,
)


def write_script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_append_optional_drops_the_flags_that_were_never_set():
    command = ["binary"]
    append_optional(command, {"--set": 4, "--unset": None, "--zero": 0})

    assert command == ["binary", "--set", "4", "--zero", "0"], (
        "only None means unset; a falsy value is still a value the caller chose"
    )


def test_run_stage_records_what_it_ran(tmp_path):
    script = write_script(tmp_path / "ok", "echo hello")

    result = run_stage("thing", [str(script)])

    assert result.stage == "thing", "the stage name must survive into the result"
    assert result.exit_code == 0, "a successful script must record a zero exit code"
    assert result.command == [str(script)], "the command must be recorded verbatim"
    assert result.duration >= 0, "the duration must be recorded"


def test_run_stage_records_a_failure_rather_than_raising(tmp_path):
    script = write_script(tmp_path / "bad", "echo trouble >&2\nexit 7")

    result = run_stage("thing", [str(script)])

    assert result.exit_code == 7, (
        "a failing stage must be reported, not raised, so the caller can summarise it"
    )


@pytest.mark.parametrize(
    "exit_codes,expected",
    [([], False), ([0], True), ([0, 0], True), ([0, 1], False), ([1], False)],
)
def test_success_requires_every_stage_to_have_run_and_passed(
    tmp_path, exit_codes, expected
):
    result = PipelineResult(
        working_directory=tmp_path,
        stages=[
            StageResult(stage=f"s{i}", command=["x"], exit_code=code, duration=0.0)
            for i, code in enumerate(exit_codes)
        ],
    )

    assert result.success is expected, (
        f"stages exiting {exit_codes} should report success={expected}"
    )


def test_write_summary_uses_the_filename_it_is_given(tmp_path):
    result = PipelineResult(working_directory=tmp_path, stages=[])

    first = write_summary(result, "one.json")
    second = write_summary(result, "two.json")

    assert first != second, (
        "two pipelines sharing a working directory must not overwrite each other"
    )
    assert json.loads(first.read_text())["stages"] == [], (
        "the summary must be valid JSON of the result model"
    )
