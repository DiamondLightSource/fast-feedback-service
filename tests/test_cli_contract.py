"""
The command line is what an external caller drives these pipelines by.

Anything wrapping them builds an argv from the same parameter names the
request models use, so the mapping between a model field and its flag is
an interface, not an internal detail. These tests fail here rather than
in the wrapper's repository when the two drift apart.
"""

import pytest

from ffs import index_integrate, spotfind_index_integrate
from ffs.stages import PipelineRequest

# The spotfinder writes the reflection table, so the caller of the
# three-stage pipeline never supplies one.
WRITTEN_BY_AN_EARLIER_STAGE = {"spotfind_index_integrate": {"reflection"}}

PIPELINES = [
    pytest.param(index_integrate, PipelineRequest, id="index_integrate"),
    pytest.param(
        spotfind_index_integrate,
        spotfind_index_integrate.SpotfindIndexIntegrateRequest,
        id="spotfind_index_integrate",
    ),
]


def flags(parser) -> set[str]:
    """Every long option the parser accepts."""
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


@pytest.mark.parametrize("module, model", PIPELINES)
def test_every_model_field_has_a_flag_named_after_it(module, model):
    accepted = flags(module.build_parser())
    exempt = WRITTEN_BY_AN_EARLIER_STAGE.get(module.__name__.rpartition(".")[2], set())

    missing = {
        field
        for field in model.model_fields
        if field not in exempt and f"--{field.replace('_', '-')}" not in accepted
    }

    assert not missing, (
        f"{sorted(missing)} have no command line flag, so a caller building "
        "an argv from the model field names cannot pass them"
    )


@pytest.mark.parametrize("module, model", PIPELINES)
def test_the_parser_accepts_nothing_the_model_cannot_hold(module, model):
    """A flag with no field behind it would be silently discarded."""
    fields = {field.replace("_", "-") for field in model.model_fields}
    stray = {
        option
        for option in flags(module.build_parser())
        if option != "--help" and option.removeprefix("--") not in fields
    }

    assert not stray, f"{sorted(stray)} are parsed but reach no model field"


@pytest.mark.parametrize("module, model", PIPELINES)
def test_the_exemptions_are_the_ones_documented(module, model):
    """A new field escaping the mapping must be a deliberate decision."""
    accepted = flags(module.build_parser())
    without_a_flag = {
        field
        for field in model.model_fields
        if f"--{field.replace('_', '-')}" not in accepted
    }
    expected = WRITTEN_BY_AN_EARLIER_STAGE.get(
        module.__name__.rpartition(".")[2], set()
    )

    assert without_a_flag == expected, (
        "the fields with no flag must be exactly the ones an earlier stage writes"
    )


def test_the_two_pipelines_do_not_share_a_summary_filename():
    assert index_integrate.SUMMARY_FILENAME != (
        spotfind_index_integrate.SUMMARY_FILENAME
    ), "a caller reading the summary back must know which pipeline wrote it"
