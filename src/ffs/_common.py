"""Helpers shared between the entrypoints in this package."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.logging import RichHandler

logger = logging.getLogger(__name__)


class ExecutableError(RuntimeError):
    """A compiled FFS executable cannot be used."""


class ExecutableNotFound(ExecutableError):
    """The executable is missing. A packaging or configuration fault."""


class DeviceProbeFailed(ExecutableError):
    """The executable runs but cannot enumerate GPU devices. A node fault."""


def setup_rich_logging(level=logging.DEBUG):
    """Setup a rich-based logging output. Using for debug running."""
    rootLogger = logging.getLogger()

    for handler in list(rootLogger.handlers):
        # We want to replace the streamhandler
        if isinstance(handler, logging.StreamHandler):
            rootLogger.handlers.remove(handler)
        # We also want to lower the output level, so pin this to the existing
        handler.setLevel(rootLogger.level)

    # Check if we're in a TTY (interactive) or not (k8s container)
    is_tty = sys.stdout.isatty()

    if is_tty:
        # Interactive mode: use RichHandler with formatting
        rootLogger.handlers.append(
            RichHandler(level=level, log_time_format="[%Y-%m-%d %H:%M:%S]")
        )
    else:
        # Container mode: simple output for Graylog
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        # Simple format: just the message
        handler.setFormatter(logging.Formatter("%(message)s"))
        rootLogger.handlers.append(handler)


def create_parent_symlink(
    destination: Path, name: str, levels: int = 2, overwrite: bool = False
) -> bool:
    """
    Link to a results directory from further up the visit tree.

    Processing output is buried under a per-run directory, so the
    beamline convention gives it a short, stable name at a fixed depth
    above: <visit>/processing/ffs points at whichever run produced it.
    The link target is relative, so it survives the visit tree being
    mounted at a different root.

    A real file or directory already holding the name is left alone.
    Only a symlink is replaced, and only when asked.

    Args:
        destination: Directory the link should point at
        name:        Name of the link
        levels:      How far above the destination to place it
        overwrite:   Replace an existing symlink of that name

    Returns:
        bool: Whether a link was created

    Raises:
        ValueError: The destination is too shallow to sit that many
            levels below a link, or levels is below the two the
            relative target needs.
    """
    if levels < 2:
        raise ValueError(f"levels must be at least 2, got {levels}")

    destination = Path(destination)
    parts = destination.parts
    if len(parts) <= levels:
        raise ValueError(f"{destination} is not {levels} levels below anything")

    link = Path(*parts[:-levels]) / name
    target = Path(*parts[-levels:])

    if link.exists(follow_symlinks=False) and not (overwrite and link.is_symlink()):
        return False

    # symlink() will not replace an existing name, so build the link
    # beside its final position and rename over it. Staging in the same
    # directory keeps the rename within one filesystem.
    staging = link.with_name(f".tmp.{name}")
    staging.unlink(missing_ok=True)
    staging.symlink_to(target)
    staging.replace(link)
    return True


def find_executable(env_var: str, name: str, probe: bool = True) -> Path:
    """
    Find one of the compiled FFS executables and check that it runs.

    The environment variable takes precedence over PATH, so that a
    development build can be pointed at without reordering PATH.

    Args:
        env_var: Environment variable holding an explicit path
        name:    Executable name to fall back to searching PATH for
        probe:   Run the executable to enumerate GPU devices. Only the
                 CUDA executables accept --list-devices; the baseline
                 tools do not.

    Returns:
        Path: The path to the executable

    Raises:
        ExecutableNotFound: The executable is missing.
        DeviceProbeFailed: The executable ran but could not enumerate
            GPU devices.
    """
    path: str | Path | None = os.getenv(env_var)

    if not path:
        path = shutil.which(name)

    if not path or not Path(path).is_file():
        raise ExecutableNotFound(
            f"Could not find {name} executable. Please set the {env_var} environment variable."
        )

    path = Path(path)

    if probe:
        # Run this, to enumerate GPUs and check it works
        proc = subprocess.run([path, "--list-devices"], capture_output=True, text=True)
        if proc.returncode:
            detail = (
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"exit code {proc.returncode}"
            )
            raise DeviceProbeFailed(
                f"{name} at {path} failed to enumerate devices: {detail}"
            )

    logger.info(f"Using {name}: {path}")

    return path
