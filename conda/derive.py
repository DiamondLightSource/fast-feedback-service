"""Derive an environment file from the development environment.

The development environment is the only file that carries versions. The
runtime environment keeps the packages it ships, and the macOS one drops
the Linux compilers, so a version moves everywhere at once when the
development pin moves and the two cannot drift apart.

    python conda/derive.py --keep conda/runtime/packages > runtime.yml
    python conda/derive.py --drop conda/macos/packages > macos.yml

A list file names one package per line, and ``#`` starts a comment. Every
listed package has to appear in the development environment, so a typo or
a rename fails here instead of quietly changing what is installed.

The file is filtered line by line, not parsed as YAML. It runs before any
conda environment exists, on whatever Python the machine has, and PyYAML
is not part of the standard library. Only flat ``  - name=version``
entries are understood; comments in the dependency list are left out of
the output, and a nested section such as ``- pip:`` is rejected rather
than half copied.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The single file that carries versions
SOURCE = Path(__file__).parent / "development" / "environment.yml"

# A flat dependency entry; the name runs up to the first version operator
ENTRY = re.compile(r"^  - ([A-Za-z0-9_.-]+)")


def read_names(text: str) -> set[str]:
    """Return the package names in a list file, without comments."""
    lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return {line for line in lines if line}


def derive(source: str, names: set[str], keep: bool) -> str:
    """
    Filter the dependency list of an environment file.

    The header above ``dependencies:`` passes through unchanged. With
    ``keep`` only the named packages remain; without it they are removed.
    Raises ValueError for an entry that is not flat, or for a named package
    the source does not contain.
    """
    header, _, entries = source.partition("dependencies:\n")
    lines = [header + "dependencies:"]
    seen = set()
    for line in entries.splitlines():
        # Comments and blank lines stay behind in the development file
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = ENTRY.match(line)
        if match is None:
            raise ValueError(f"unsupported entry: {line!r}")
        listed = match[1] in names
        if listed:
            seen.add(match[1])
        if listed == keep:
            lines.append(line)

    # A name that matched nothing is a typo or a renamed package
    if missing := names - seen:
        raise ValueError(f"not in the development environment: {sorted(missing)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--keep", type=Path, help="keep only the listed packages")
    mode.add_argument("--drop", type=Path, help="remove the listed packages")
    args = parser.parse_args()

    names = read_names((args.keep or args.drop).read_text())
    try:
        sys.stdout.write(derive(SOURCE.read_text(), names, keep=bool(args.keep)))
    except ValueError as error:
        sys.exit(f"{SOURCE}: {error}")


if __name__ == "__main__":
    main()
