import json
import os
import subprocess
from pathlib import Path

import h5py
import pytest


def test_gpu_ssx_index(dials_data, tmp_path):
    """
    Test the GPU SSX indexing code.

    First runs spotfinding and then ssx_index.py script.
    Skips if ffbidx is not sourced/available.
    """
    try:
        import ffbidx  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("ffbidx not installed")
    try:
        ffbidx.Indexer()
        ffbidx.runtime_check()
    except RuntimeError:
        pytest.skip("ffbidx installed but not functional on this system")

    # FIXME need to generate imported.expt for ssx_index
    # The relevant parts needed from an imported.expt generated from
    # dials.import $(dials.data get -q lysozyme_ssx_25keV)/lysozyme_25keV.nxs \
    #   distance=420.33 fast_slow_beam_centre=1597.74,1692.41
    imported_expt_stub = {
        "beam": [{"wavelength": 0.4959}],
        "detector": [
            {
                "panels": [
                    {
                        "origin": [-119.625, 126.90, -420.33],
                        "image_size": [3108, 3262],
                        "pixel_size": [0.075, 0.075],
                        "thickness": 0.75,
                        "mu": 7.285849919020163,
                    }
                ],
                "hierarchy": {
                    "origin": [-0.2055, 0.03075, 0.0],
                },
            }
        ],
    }
    with open(tmp_path / "tmp.expt", "w") as f:
        json.dump(imported_expt_stub, f)

    # First do spotfinding
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER_32BIT")
    ssx_index = Path(os.getenv("FFS_ROOT_DIR")) / "src" / "ffs" / "ssx_index.py"

    assert spotfinder_path
    d = dials_data("lysozyme_ssx_25keV", pathlib=True)
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "lysozyme_25keV.nxs",
            "--threads",
            "10",
            "--save-h5",
            "--images",
            "100",
            "--algorithm",
            "dispersion_extended",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr

    # Now run the indexer

    proc = subprocess.run(
        [
            "python",
            ssx_index,
            "-r",
            "results_ffs.h5",
            "-e",
            tmp_path / "tmp.expt",
            "-c",
            "79",
            "79",
            "38",
            "90",
            "90",
            "90",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    assert tmp_path / "indexed.refl"
    assert tmp_path / "indexed_crystals.json"
    with h5py.File(tmp_path / "indexed.refl", "r") as file:
        data = file["/dials/processing/group_0/id"]
        assert set(data) == set(range(0, 34))  # i.e. 34 / 100 images were indexed
