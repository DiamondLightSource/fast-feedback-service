"""
Tests for the fast-feedback spotfinder. Output values are compared to the equivalent
processing from dials v3.25.0, which should be exactly the same as the GPU spotfinder.
"""

import os
import re
import subprocess

def strip_ansi(text: str) -> str:
    # Strip out colour escape sequences from the log output.
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m|\033\[[0-9;]*m')
    return ansi_escape.sub('', text)

pixels_match_regex = r'image\s+(\d+).*?(\d+)\s+strong pixels'
spots_match_regex = r'Calculated\s+(\d+)\s+spots'
min_spot_size_regex = r'Filtered\s+(\d+)\s+spots with size < 3 pixels'
max_separation_regex = r'Filtered\s+(\d+)\s+spots with peak-centroid distance > 2'

def test_dispersion(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [spotfinder_path, d / "thau_2_1.nxs", "--images", "10", "--threads", "10"], capture_output=True, cwd=tmp_path)
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 2505 # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 1468 # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 33 # number removed by filtering.
    expected_strong_pixels = {0:1399, 1:1372, 2:1343, 3:1296, 4:1191, 5:1250, 6:1211, 7:1189, 8:1250, 9:1246}
    
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
                n_filtered_out = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels
    
    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....

def test_dispersion_dmin(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [spotfinder_path, d / "thau_2_1.nxs", "--images", "10", "--threads", "10", "--dmin", "3.0"], capture_output=True, cwd=tmp_path)
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 995 # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 505 # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 14 # number removed by filtering.
    expected_strong_pixels = {0:755, 1:744, 2:725, 3:709, 4:624, 5:660, 6:678, 7:668, 8:705, 9:741}
    
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
                n_filtered_out = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels
    
    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....

def test_dispersion_extended(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [spotfinder_path, d / "thau_2_1.nxs", "--images", "10", "--threads", "10", "--algorithm", "dispersion_extended"], capture_output=True, cwd=tmp_path)
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 1669 # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 526 # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 35 # number removed by filtering.
    expected_strong_pixels = {0:2753, 1:2650, 2:2686, 3:2440, 4:2355, 5:2350, 6:2353, 7:2412, 8:2519, 9:2457}
    
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
                n_filtered_out = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels
    
    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....

def test_dispersion_extended_dmin(dials_data, tmp_path):
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER")
    assert spotfinder_path
    d = dials_data("thaumatin_i03_rotation", pathlib=True)
    proc = subprocess.run(
        [spotfinder_path, d / "thau_2_1.nxs", "--images", "10", "--threads", "10",
        "--algorithm", "dispersion_extended", "--dmin", "3.0"], capture_output=True, cwd=tmp_path)
    assert not proc.stderr
    log = proc.stdout
    loglines = strip_ansi(log.decode()).split("\n")

    ## First check that the expected number of strong pixels per image have been found
    found_strong_pixels = {}
    n_spots_found = None
    n_spots_expected = 758 # before filtering.
    n_filtered_min_size = None
    n_filtered_min_size_expected = 243 # number removed by filtering.
    n_filtered_max_sep = None
    n_filtered_max_sep_expected = 14 # number removed by filtering.
    expected_strong_pixels = {0:1487, 1:1451, 2:1405, 3:1315, 4:1242, 5:1252, 6:1305, 7:1325, 8:1392, 9:1443}
    
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
                n_filtered_out = int(match.group(1))
            match = re.search(max_separation_regex, line)
            if match:
                n_filtered_max_sep = int(match.group(1))

    assert found_strong_pixels == expected_strong_pixels
    
    ## Check that the 3DCC analysis finds the expected number of strong spots
    assert n_spots_found == n_spots_expected
    assert n_filtered_min_size == n_filtered_min_size_expected
    assert n_filtered_max_sep == n_filtered_max_sep_expected

    ## Now load the file and evaluate the calculated centroids....