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


def find_executable(env_var: str, name: str) -> Path:
    """
    Find one of the compiled FFS executables and check that it runs.

    The environment variable takes precedence over PATH, so that a
    development build can be pointed at without reordering PATH.

    Args:
        env_var: Environment variable holding an explicit path
        name:    Executable name to fall back to searching PATH for

    Returns:
        Path: The path to the executable

    Raises:
        SystemExit: The executable is missing, or failed to enumerate
            GPU devices.
    """
    path: str | Path | None = os.getenv(env_var)

    if not path:
        path = shutil.which(name)

    if not path or not Path(path).is_file():
        sys.exit(
            f"Error: Could not find {name} executable. Please set the {env_var} environment variable."
        )

    path = Path(path)

    # Run this, to enumerate GPUs and check it works
    proc = subprocess.run([path, "--list-devices"], capture_output=True, text=True)
    if proc.returncode:
        sys.exit(f"Error: {name} at {path} failed to enumerate devices.")

    logger.info(f"Using {name}: {path}")

    return path
