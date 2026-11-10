"""Move the development pins to the newest versions that solve together.

Each exact pin is loosened to a lower bound at its current version, and
micromamba solves the result, without installing it, for every platform
the environments install on. The versions it picks are written back as
exact pins, so the file only ever holds a set the solver has accepted as
a whole. Bumping each pin on its own, as Dependabot does, misses the
constraints packages place on each other.

Each platform is solved separately, and conda-forge builds reach each
platform at different times, so the platforms can choose different
versions. Where they disagree, the package is capped at each version
chosen and every platform solves again. Caps only ever fall and the
current pins always satisfy them, so the solves settle on one set that
installs everywhere, at worst the set already in the file.

Wildcard pins reach the solver unchanged, which makes them the way to
hold a package back. A lower bound also admits development snapshots,
which conda orders above the last release before them; when the solver
picks one for a pinned package, the solve runs again with that package
capped below it. Each change is printed as ``name old -> new``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from derive import SOURCE, derive, read_names

# An exact pin, the only kind that moves; a trailing comment is kept
EXACT = re.compile(
    r"^  - (?P<name>[A-Za-z0-9_.-]+)=(?P<version>[0-9][A-Za-z0-9_.+]*)"
    r"(?P<rest>\s*(?:#.*)?)$"
)

# Marks a development snapshot or a release candidate in a version
PRERELEASE = re.compile(r"dev|rc|alpha|beta|pre|\d[ab]\d", re.IGNORECASE)

# Every platform an environment installs on: the two image architectures,
# and the macOS test job, which drops the packages in its list
PLATFORMS = {
    "linux-64": None,
    "linux-aarch64": None,
    "osx-arm64": Path(__file__).parent / "macos" / "packages",
}

# The macOS of the test runner. A Linux host reports none, and without one
# the solver rejects every build that needs a minimum macOS.
MACOS_VERSION = "26.0"


def loosen(source: str, bounds: dict[str, set[str]]) -> str:
    """
    Return the source with every exact pin made a lower bound.

    A package in ``bounds`` also carries each upper bound it maps to, such
    as ``<=12.1.0``; the solver reads several as the tightest of them.
    """
    lines = []
    for line in source.splitlines():
        # An exact pin becomes a floor the solver may move up from, plus the
        # caps earlier passes added; wildcards and the header pass through as
        # written. The trailing comment is dropped, as the solver never sees
        # this file again.
        match = EXACT.match(line)
        if match is not None:
            line = f"  - {match['name']}>={match['version']}"
            line += "".join(
                f",{bound}" for bound in sorted(bounds.get(match["name"], ()))
            )
        lines.append(line)
    return "\n".join(lines) + "\n"


def pinned(source: str) -> dict[str, str]:
    """Return the exact pins in the source, each name mapped to its version."""
    pins = (EXACT.match(line) for line in source.splitlines())
    return {m["name"]: m["version"] for m in pins if m is not None}


def solve(environment: str, platform: str) -> dict[str, str]:
    """
    Solve an environment file for a platform without installing it.

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
        command += ["--platform", platform]

        # A macOS solve needs a macOS version to check builds against
        env = os.environ.copy()
        if platform.startswith("osx"):
            env["CONDA_OVERRIDE_OSX"] = MACOS_VERSION
        result = subprocess.run(command, capture_output=True, text=True, env=env)

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


def tighten(
    solutions: dict[str, dict[str, str]], current: dict[str, str]
) -> set[tuple[str, str]]:
    """
    Return the upper bounds that bring the platforms towards one solution.

    Each bound is a ``(name, bound)`` pair. An empty result means no pinned
    package is on a new pre-release and every platform chose the same
    versions.
    """
    # A pinned package the solver moved to a pre-release, on any platform, is
    # capped just below it. A pin that is already a pre-release is left alone,
    # as a cap at its own version could never be met.
    snapshots = {
        (name, f"<{solution[name]}")
        for solution in solutions.values()
        for name, version in current.items()
        if name in solution
        and solution[name] != version
        and PRERELEASE.search(solution[name])
    }
    if snapshots:
        return snapshots

    # A pinned package the platforms disagree on is capped at every version
    # chosen, which holds each platform to the lowest of them. A package a
    # platform does not install, such as the Linux compilers on macOS, is
    # judged on the platforms that do.
    caps = set()
    for name in current:
        chosen = {solution[name] for solution in solutions.values() if name in solution}
        if len(chosen) > 1:
            caps |= {(name, f"<={version}") for version in chosen}
    return caps


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

    source = SOURCE.read_text()
    current = pinned(source)
    bounds: dict[str, set[str]] = {}
    try:
        while True:
            # Solve every platform from the same loosened file, the macOS one
            # without the packages its list drops
            loosened = loosen(source, bounds)
            solutions = {}
            for platform, dropped in PLATFORMS.items():
                environment = loosened
                if dropped is not None:
                    environment = derive(
                        loosened, read_names(dropped.read_text()), keep=False
                    )
                solutions[platform] = solve(environment, platform)

            # Stop once the platforms agree. Every new bound excludes a version
            # some platform just chose, so each pass makes progress and the
            # loop ends; a bound already in place would mean it cannot.
            tighter = tighten(solutions, current)
            if not tighter:
                break
            if all(bound in bounds.get(name, ()) for name, bound in tighter):
                raise RuntimeError("the platforms cannot agree on one set of pins")
            for name, bound in tighter:
                bounds.setdefault(name, set()).add(bound)

        # The platforms agree, so any one of them gives each pinned version
        solved: dict[str, str] = {}
        for solution in solutions.values():
            solved |= solution
        updated, changes = apply(source, solved)
    except (RuntimeError, ValueError) as error:
        sys.exit(f"{SOURCE}: {error}")

    # The file is only touched once every step has succeeded, and the
    # changes go to stdout for the pull request body
    SOURCE.write_text(updated)
    print("\n".join(changes))


if __name__ == "__main__":
    main()
