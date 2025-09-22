import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["INDEXER"] = os.fspath(Path.cwd() / "build/bin/baseline_indexer")
    os.environ["SPOTFINDER"] = os.fspath(Path.cwd() / "build/bin/spotfinder")
    os.environ["SPOTFINDER_32BIT"] = os.fspath(
        Path.cwd() / "build_32bit/bin/spotfinder"
    )
    os.environ["FFS_ROOT_DIR"] = os.fspath(Path.cwd())
