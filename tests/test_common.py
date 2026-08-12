import pytest

from ffs._common import create_parent_symlink


@pytest.fixture
def results(tmp_path):
    """A results directory buried the usual two levels down a visit."""
    path = tmp_path / "processed" / "12345" / "ffs"
    path.mkdir(parents=True)
    return path


def test_the_link_lands_the_requested_number_of_levels_up(tmp_path, results):
    assert create_parent_symlink(results, "ffs") is True, "a fresh name must be linked"

    link = tmp_path / "processed" / "ffs"
    assert link.is_symlink(), "the link belongs two levels above the destination"
    assert link.resolve() == results, "the link must reach the results"


def test_the_link_target_is_relative(tmp_path, results):
    """An absolute target would break when the visit is mounted elsewhere."""
    create_parent_symlink(results, "ffs")

    target = (tmp_path / "processed" / "ffs").readlink()

    assert not target.is_absolute(), "a relative target survives a remount"
    assert target.parts == ("12345", "ffs"), (
        "the target spans exactly the levels the link was raised by"
    )


def test_a_real_directory_of_that_name_is_left_alone(tmp_path, results):
    occupied = tmp_path / "processed" / "ffs"
    occupied.mkdir()

    assert create_parent_symlink(results, "ffs") is False, (
        "someone else's directory is not ours to replace"
    )
    assert not occupied.is_symlink(), "the real directory must survive"


def test_an_existing_link_is_kept_unless_overwriting_is_asked_for(tmp_path, results):
    older = tmp_path / "processed" / "99999" / "ffs"
    older.mkdir(parents=True)
    create_parent_symlink(older, "ffs")

    assert create_parent_symlink(results, "ffs") is False, (
        "the first run to finish keeps the name by default"
    )
    assert create_parent_symlink(results, "ffs", overwrite=True) is True, (
        "a caller that wants the latest run must be able to say so"
    )
    assert (tmp_path / "processed" / "ffs").resolve() == results, (
        "overwriting must repoint the link, not stack a second one"
    )


def test_no_temporary_link_is_left_behind(tmp_path, results):
    create_parent_symlink(results, "ffs")

    assert {p.name for p in (tmp_path / "processed").iterdir()} == {"12345", "ffs"}, (
        "the staging link must be renamed into place, not left as litter"
    )


@pytest.mark.parametrize(
    "levels",
    [1, 0],
    ids=["one level cannot carry a relative target", "zero levels self-links"],
)
def test_too_few_levels_is_rejected(results, levels):
    with pytest.raises(ValueError):
        create_parent_symlink(results, "ffs", levels=levels)


def test_a_destination_too_shallow_to_raise_a_link_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        create_parent_symlink("/", "ffs")
