from pathlib import Path

import pytest
from pydantic import ValidationError

from ffs.stages import PipelineRequest, build_indexer_command, build_integrator_command


@pytest.fixture
def request_for(tmp_path):
    """Build a minimal request, with room to override fields."""

    def make(**overrides):
        params = {
            "reflection": tmp_path / "strong.refl",
            "experiment": tmp_path / "imported.expt",
            "working_directory": tmp_path / "work",
            "max_cell": 100.0,
        }
        return PipelineRequest(**{**params, **overrides})

    return make


@pytest.mark.parametrize(
    "field", ["reflection", "experiment", "working_directory", "output"]
)
def test_an_empty_path_is_rejected_rather_than_becoming_the_current_directory(
    request_for, field
):
    """Path("") is Path("."), which exists, so an unset value must not reach it."""
    with pytest.raises(ValidationError, match="must not be empty"):
        request_for(**{field: ""})


def test_indexer_command_omits_unset_options(request_for):
    params = request_for()
    command = build_indexer_command(Path("baseline_indexer"), params)

    assert command == [
        "baseline_indexer",
        "--refl",
        str(params.reflection),
        "--expt",
        str(params.experiment),
        "--max-cell",
        "100.0",
    ], "unset options must be left to the binary's own defaults"


def test_indexer_command_uses_hyphenated_flags(request_for):
    params = request_for(dmin=1.81, max_refine=5, macro_cycles=3, nthreads=8)
    command = build_indexer_command(Path("baseline_indexer"), params)

    for flag, value in [
        ("--dmin", "1.81"),
        ("--max-refine", "5"),
        ("--macro-cycles", "3"),
        ("--nthreads", "8"),
    ]:
        assert command[command.index(flag) + 1] == value, (
            f"{flag} must reach the indexer hyphenated"
        )


def test_integrator_command_reads_the_indexer_output(request_for):
    params = request_for()
    command = build_integrator_command(
        Path("integrator"), params, Path("/work/integrated.refl")
    )

    assert command[command.index("--reflection") + 1] == "indexed.refl", (
        "the integrator must read the indexed table, not the strong spots"
    )
    assert command[command.index("--experiment") + 1] == "indexed.expt", (
        "the integrator must read the indexed experiments"
    )
    assert command[command.index("--output") + 1] == "/work/integrated.refl", (
        "the resolved output path must reach the integrator"
    )
    assert "--algorithm" not in command, "unset options must be left to the binary"


def test_integrator_command_uses_underscored_flags(request_for):
    params = request_for(
        algorithm="dials",
        background="glm",
        sigma_b=0.03,
        sigma_m=0.1,
        min_zeta=0.02,
        min_bbox_depth=4,
        threads=8,
        timeout=60.0,
    )
    command = build_integrator_command(Path("integrator"), params, Path("out.refl"))

    for flag, value in [
        ("--algorithm", "dials"),
        ("--background", "glm"),
        ("--sigma_b", "0.03"),
        ("--sigma_m", "0.1"),
        ("--min_zeta", "0.02"),
        ("--min_bbox_depth", "4"),
        ("--threads", "8"),
        ("--timeout", "60.0"),
    ]:
        assert command[command.index(flag) + 1] == value, (
            f"{flag} must reach the integrator underscored"
        )
