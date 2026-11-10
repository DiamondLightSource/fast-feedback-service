"""Move the development pins to the newest versions that solve together.

Each exact pin is loosened to a lower bound at its current version, and
micromamba solves the result for one platform without installing it. The
versions it picks are written back as exact pins, so the file only ever
holds a set the solver has accepted as a whole. Bumping each pin on its
own, as Dependabot does, misses the constraints packages place on each
other.

Wildcard pins reach the solver unchanged, which makes them the way to
hold a package back. A lower bound also admits development snapshots,
which conda orders above the last release before them; when the solver
picks one for a pinned package, the solve runs again with that package
held at its current pin. Each change is printed as ``name old -> new``,
and checking the result on other platforms is left to the caller.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from derive import SOURCE

# An exact pin, the only kind that moves; a trailing comment is kept
EXACT = re.compile(
    r"^  - (?P<name>[A-Za-z0-9_.-]+)=(?P<version>[0-9][A-Za-z0-9_.+]*)"
    r"(?P<rest>\s*(?:#.*)?)$"
)

# Marks a development snapshot or a release candidate in a version
PRERELEASE = re.compile(r"dev|rc|alpha|beta|pre|\d[ab]\d", re.IGNORECASE)


def loosen(source: str, held: set[str]) -> str:
    """Return the source with every exact pin not in ``held`` made a lower bound."""
    lines = []
    for line in source.splitlines():
        # An exact pin becomes a floor the solver may move up from; wildcards,
        # held pins and the header pass through as written. The trailing
        # comment is dropped, as the solver never sees this file again.
        match = EXACT.match(line)
        if match is not None and match["name"] not in held:
            line = f"  - {match['name']}>={match['version']}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def pinned(source: str) -> set[str]:
    """Return the names of the exact pins in the source."""
    return {m["name"] for line in source.splitlines() if (m := EXACT.match(line))}


def solve(environment: str, platform: str | None) -> dict[str, str]:
    """
    Solve an environment file without installing it.

    Returns the version chosen for every package in the solution, keyed by
    name. Raises RuntimeError with the solver's output when it fails.
    """
    # CI has micromamba and a developer machine usually mamba; both share
    # the same solver and the same JSON output
    solver = shutil.which("micromamba") or shutil.which("mamba")
    if solver is None:
        raise RuntimeError("neither micromamba nor mamba is on PATH")

    with tempfile.TemporaryDirectory() as scratch:
        # The solver reads the file type from the extension
        spec = Path(scratch) / "environment.yml"
        spec.write_text(environment)

        # A dry run solves and reports without downloading or installing; the
        # prefix only has to be a path that does not exist yet
        command = [solver, "create", "--dry-run", "--json", "-y"]
        command += ["-p", str(Path(scratch) / "prefix"), "-f", str(spec)]
        if platform is not None:
            command += ["--platform", platform]
        result = subprocess.run(command, capture_output=True, text=True)

    # An unsatisfiable set fails here, with the solver's own explanation
    if result.returncode != 0:
        raise RuntimeError(f"the solve failed:\n{result.stdout}{result.stderr}")

    # LINK lists every package the environment would contain, dependencies
    # included, each with the version the solver chose
    linked = json.loads(result.stdout)["actions"]["LINK"]
    return {package["name"]: package["version"] for package in linked}


def apply(source: str, solved: dict[str, str]) -> tuple[str, list[str]]:
    """
    Write the solved versions into the exact pins of the source.

    Returns the new source and one ``name old -> new`` line per change.
    Raises ValueError for a pinned package the solution does not contain.
    """
    lines, changes = [], []
    for line in source.splitlines():
        # Only exact pins are rewritten; every other line is copied as it is
        match = EXACT.match(line)
        if match is not None:
            name, old = match["name"], match["version"]

            # Every pinned package is a direct request, so a solution without
            # it means the solve and the file disagree
            if name not in solved:
                raise ValueError(f"{name} is pinned but not in the solution")

            # Replace the version alone, keeping any trailing comment
            if solved[name] != old:
                line = f"  - {name}={solved[name]}{match['rest']}"
                changes.append(f"{name} {old} -> {solved[name]}")
        lines.append(line)
    return "\n".join(lines) + "\n", changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", help="solve for this platform, e.g. linux-64")
    args = parser.parse_args()

    source = SOURCE.read_text()
    held: set[str] = set()
    try:
        # Solve with every unheld pin as a floor. A pinned package that lands
        # on a pre-release is held at its current version and the solve runs
        # again; each pass holds at least one more, so the loop ends.
        while True:
            solved = solve(loosen(source, held), args.platform)
            snapshots = {
                name
                for name in pinned(source) - held
                if PRERELEASE.search(solved.get(name, ""))
            }
            if not snapshots:
                break
            held |= snapshots

        # Write the final solution back as exact pins
        updated, changes = apply(source, solved)
    except (RuntimeError, ValueError) as error:
        sys.exit(f"{SOURCE}: {error}")

    # The file is only touched once every step has succeeded, and the
    # changes go to stdout for the pull request body
    SOURCE.write_text(updated)
    print("\n".join(changes))


if __name__ == "__main__":
    main()
