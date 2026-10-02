from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent
SRC_DIR = ROOT_DIR / "src"

# Ahead of site-packages, so an editable install elsewhere cannot win.
sys.path.insert(0, os.fspath(SRC_DIR))

# Several tests run the CLI entry points as subprocesses, and a fresh
# interpreter inherits sys.path from nothing. Prepended rather than
# replaced, since the caller may be pointing at something deliberately.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [os.fspath(SRC_DIR), *filter(None, [os.environ.get("PYTHONPATH")])]
)


# The suite drives the compiled binaries as subprocesses rather than
# importing them, and finds them through these variables. Anchored on this
# file, not the working directory, so a run from anywhere still locates the
# build; the binaries themselves come from a plain `build/` configured in
# the repository root.
@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["INDEXER"] = os.fspath(ROOT_DIR / "build/bin/baseline_indexer")
    os.environ["SPOTFINDER"] = os.fspath(ROOT_DIR / "build/bin/spotfinder")
    os.environ["PREDICTOR"] = os.fspath(ROOT_DIR / "build/bin/baseline_predictor")
    os.environ["SPOTFINDER_32BIT"] = os.fspath(ROOT_DIR / "build/bin/spotfinder32")
    os.environ["INTEGRATOR"] = os.fspath(ROOT_DIR / "build/bin/integrator")
    os.environ["BASELINE_INTEGRATOR"] = os.fspath(
        ROOT_DIR / "build/bin/baseline_integrator"
    )
    os.environ["FFS_ROOT_DIR"] = os.fspath(ROOT_DIR)
