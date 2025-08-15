import os
import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def set_env_vars():
    os.environ["INDEXER"] = os.fspath(Path.cwd() / "build/bin/baseline_indexer")