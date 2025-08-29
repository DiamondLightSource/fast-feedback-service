"""
Tests for the fast-feedback spotfinder. Output values are compared to the equivalent
processing from dials v3.25.0, which should be exactly the same as the GPU spotfinder.

Equivalent dials command:
dials.import $(dials.data get -q thaumatin_i03_rotation)/thau_2_1.nxs image_range=1,10
dials.find_spots imported.expt algorithm=dispersion d_min=3.0 disable_parallax_correction=True
"""

import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest


def strip_ansi(text: str) -> str:
    # Strip out colour escape sequences from the log output.
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m|\033\[[0-9;]*m")
    return ansi_escape.sub("", text)


pixels_match_regex = r"image\s+(\d+).*?(\d+)\s+strong pixels"
spots_match_regex = r"Calculated\s+(\d+)\s+spots"
min_spot_size_regex = r"Filtered\s+(\d+)\s+spots with size < 3 pixels"
max_separation_regex = r"Filtered\s+(\d+)\s+spots with peak-centroid distance > 2"


def test_dispersion(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "thau_2_1.nxs",
            "--images",
            "10",
            "--threads",
            "10",
            "--save-h5",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 2505  # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 1468  # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 33  # number removed by filtering.
    expected_strong_pixels = {
        0: 1399,
        1: 1372,
        2: 1343,
        3: 1296,
        4: 1191,
        5: 1250,
        6: 1211,
        7: 1189,
        8: 1250,
        9: 1246,
    }

    for line in loglines:
        if "strong pixels" in line:
            match = re.search(pixels_match_regex, line)
            if match:
                found_strong_pixels[int(match.group(1))] = int(match.group(2))
        elif "Calculated" in line:
            match = re.search(spots_match_regex, line)
            if match:
                n_spots_found = int(match.group(1))
        elif "Filtered" in line:
            match = re.search(min_spot_size_regex, line)
            if match:
                n_filtered_min_size = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels

    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....
    with h5py.File(tmp_path / "results_ffs.h5", "r") as file:
        data = file["/dials/processing/group_0/xyzobs.px.value"]
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        assert minimum.tolist() == pytest.approx([388.14, 208.50, 0.50], abs=5e-3)
        assert maximum.tolist() == pytest.approx([4071.50, 4297.79, 9.50], abs=5e-3)
        assert mean.tolist() == pytest.approx([2074.33, 2117.60, 4.79], abs=5e-3)


def test_dispersion_dmin(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "thau_2_1.nxs",
            "--images",
            "10",
            "--threads",
            "10",
            "--dmin",
            "3.0",
            "--save-h5",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 994  # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 504  # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 14  # number removed by filtering.
    expected_strong_pixels = {
        0: 755,
        1: 743,
        2: 725,
        3: 709,
        4: 624,
        5: 660,
        6: 678,
        7: 666,
        8: 705,
        9: 741,
    }

    for line in loglines:
        if "strong pixels" in line:
            match = re.search(pixels_match_regex, line)
            if match:
                found_strong_pixels[int(match.group(1))] = int(match.group(2))
        elif "Calculated" in line:
            match = re.search(spots_match_regex, line)
            if match:
                n_spots_found = int(match.group(1))
        elif "Filtered" in line:
            match = re.search(min_spot_size_regex, line)
            if match:
                n_filtered_min_size = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels

    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....
    with h5py.File(tmp_path / "results_ffs.h5", "r") as file:
        data = file["/dials/processing/group_0/xyzobs.px.value"]
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        assert minimum.tolist() == pytest.approx([1191.80, 1336.02, 0.50], abs=5e-3)
        assert maximum.tolist() == pytest.approx([2853.02, 3077.50, 9.50], abs=5e-3)
        assert mean.tolist() == pytest.approx([2043.29, 2214.73, 4.84], abs=5e-3)


def test_dispersion_extended(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "thau_2_1.nxs",
            "--images",
            "10",
            "--threads",
            "10",
            "--algorithm",
            "dispersion_extended",
            "--save-h5",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 1669  # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 526  # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 35  # number removed by filtering.
    expected_strong_pixels = {
        0: 2753,
        1: 2650,
        2: 2686,
        3: 2440,
        4: 2355,
        5: 2350,
        6: 2353,
        7: 2412,
        8: 2519,
        9: 2457,
    }

    for line in loglines:
        if "strong pixels" in line:
            match = re.search(pixels_match_regex, line)
            if match:
                found_strong_pixels[int(match.group(1))] = int(match.group(2))
        elif "Calculated" in line:
            match = re.search(spots_match_regex, line)
            if match:
                n_spots_found = int(match.group(1))
        elif "Filtered" in line:
            match = re.search(min_spot_size_regex, line)
            if match:
                n_filtered_min_size = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels

    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....
    with h5py.File(tmp_path / "results_ffs.h5", "r") as file:
        data = file["/dials/processing/group_0/xyzobs.px.value"]
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        assert minimum.tolist() == pytest.approx([388.26, 147.63, 0.50], abs=5e-3)
        assert maximum.tolist() == pytest.approx([4071.50, 4296.19, 9.50], abs=5e-3)
        assert mean.tolist() == pytest.approx([2080.53, 2130.00, 4.80], abs=5e-3)


def test_dispersion_extended_dmin(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "thau_2_1.nxs",
            "--images",
            "10",
            "--threads",
            "10",
            "--algorithm",
            "dispersion_extended",
            "--dmin",
            "3.0",
            "--save-h5",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 758  # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 242  # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 14  # number removed by filtering.
    expected_strong_pixels = {
        0: 1493,
        1: 1451,
        2: 1405,
        3: 1315,
        4: 1242,
        5: 1252,
        6: 1308,
        7: 1320,
        8: 1390,
        9: 1442,
    }

    for line in loglines:
        if "strong pixels" in line:
            match = re.search(pixels_match_regex, line)
            if match:
                found_strong_pixels[int(match.group(1))] = int(match.group(2))
        elif "Calculated" in line:
            match = re.search(spots_match_regex, line)
            if match:
                n_spots_found = int(match.group(1))
        elif "Filtered" in line:
            match = re.search(min_spot_size_regex, line)
            if match:
                n_filtered_min_size = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels

    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....
    with h5py.File(tmp_path / "results_ffs.h5", "r") as file:
        data = file["/dials/processing/group_0/xyzobs.px.value"]
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        assert minimum.tolist() == pytest.approx([1192.19, 1335.99, 0.50], abs=5e-3)
        assert maximum.tolist() == pytest.approx([2920.70, 3077.46, 9.50], abs=5e-3)
        assert mean.tolist() == pytest.approx([2047.54, 2216.19, 4.86], abs=5e-3)


def test_dispersion_gridscan(dials_data, tmp_path):
    """
    An extended test to test the 2d connected components analysis only.

    Runs the spotfinder on 420 images, and doesn't apply any filtering,
    so just tests that the 2D strong pixel maps have the same number
    of strong pixels to dials.
    """
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_grid_scans", pathlib=True)
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "thau_3_113.nxs",
            "--threads",
            "10",
            "--save-h5",
            "--min-spot-size",
            "1",  # i.e. don't filter on spot size
            "--max-peak-centroid-separation",
            "20",  # i.e. don't filter on peak-centroid separation
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 154824

    expected_strong_pixels = {}
    root_dir = Path(os.getenv("FFS_ROOT_DIR"))
    assert root_dir
    dials_output_regex = r"Found\s+(\d+)\s+strong pixels on image\s+(\d+)\s+"
    ## This file is the output of dials.find_spots with the options 
    ## disable_parallax_correction=True max_separation=20 min_spot_size=1
    with open(root_dir / "tests/dials_2d_spotfinding_output.txt", "r") as f:
        for line in f.readlines():
            match = re.search(dials_output_regex, line)
            if match:
                expected_strong_pixels[int(match.group(2)) - 1] = int(match.group(1))

    spots_match_regex_2d = r"Succesfully wrote\s+(\d+)\s+2D reflections to HDF5 file"

    for line in loglines:
        if "strong pixels" in line:
            match = re.search(pixels_match_regex, line)
            if match:
                found_strong_pixels[int(match.group(1))] = int(match.group(2))
        elif "Succesfully" in line:
            match = re.search(spots_match_regex_2d, line)
            if match:
                n_spots_found = int(match.group(1))

    assert n_spots_found == n_spots_expected
    assert found_strong_pixels == expected_strong_pixels

    ## Now load the file and evaluate the calculated centroids....
    with h5py.File(tmp_path / "results_ffs.h5", "r") as file:
        data = file["/dials/processing/group_0/xyzobs.px.value"]
        assert data.shape == (n_spots_expected, 3)
        minimum = np.min(data, axis=0)
        maximum = np.max(data, axis=0)
        mean = np.mean(data, axis=0)
        assert minimum.tolist() == pytest.approx([0.50, 0.50, 0.50], abs=5e-3)
        assert maximum.tolist() == pytest.approx([4147.50, 4361.50, 0.50], abs=5e-3)
        assert mean.tolist() == pytest.approx([2070.02, 2141.43, 0.50], abs=5e-3)
