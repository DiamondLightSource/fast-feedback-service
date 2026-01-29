import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

# First define a test indexed experiment - the first single rotation image from
# the dials_i03_thau dataset.

expt_json = {
    "__id__": "ExperimentList",
    "experiment": [
        {
            "__id__": "Experiment",
            "identifier": "855e4e32-a69b-4d3f-5c71-7da43a7f2bfc",
            "beam": 0,
            "detector": 0,
            "goniometer": 0,
            "scan": 0,
            "crystal": 0,
        }
    ],
    "imageset": [
        {
            "__id__": "ImageSequence",
            "template": "thau_2_1.nxs",
            "single_file_indices": [0],
            "mask": None,
            "gain": None,
            "pedestal": None,
            "dx": None,
            "dy": None,
            "params": {"dynamic_shadowing": "Auto", "multi_panel": False},
        }
    ],
    "beam": [
        {
            "__id__": "monochromatic",
            "direction": [-0.0034229667619783467, -0.0, 0.999994141632113],
            "wavelength": 0.9762458439949315,
            "divergence": 0.0,
            "sigma_divergence": 0.0,
            "polarization_normal": [0.0, 1.0, 0.0],
            "polarization_fraction": 0.999,
            "flux": 0.0,
            "transmission": 1.0,
            "probe": "x-ray",
            "sample_to_source_distance": 0.0,
        }
    ],
    "detector": [
        {
            "panels": [
                {
                    "name": "/entry/instrument/detector/module",
                    "type": "SENSOR_PAD",
                    "fast_axis": [
                        0.9999343565703017,
                        0.008972928699089156,
                        0.007125243918478585,
                    ],
                    "slow_axis": [
                        0.009011312015482328,
                        -0.9999449607587793,
                        -0.005373240073119384,
                    ],
                    "origin": [
                        -154.36992588196173,
                        164.3947680648576,
                        -198.15267891180412,
                    ],
                    "raw_image_offset": [0, 0],
                    "image_size": [4148, 4362],
                    "pixel_size": [0.075, 0.075],
                    "trusted_range": [0.0, 46051.0],
                    "thickness": 0.45000000000000007,
                    "material": "Si",
                    "mu": 3.9219876752936167,
                    "identifier": "",
                    "mask": [],
                    "gain": 1.0,
                    "pedestal": 0.0,
                    "px_mm_strategy": {"type": "ParallaxCorrectedPxMmStrategy"},
                }
            ],
            "hierarchy": {
                "name": "",
                "type": "",
                "fast_axis": [1.0, 0.0, 0.0],
                "slow_axis": [0.0, 1.0, 0.0],
                "origin": [0.0, 0.0, 0.0],
                "raw_image_offset": [0, 0],
                "image_size": [0, 0],
                "pixel_size": [0, 0],
                "trusted_range": [0, 0],
                "thickness": 0.0,
                "material": "",
                "mu": 0.0,
                "identifier": "",
                "mask": [],
                "gain": 1.0,
                "pedestal": 0.0,
                "px_mm_strategy": {"type": "SimplePxMmStrategy"},
                "children": [{"panel": 0}],
            },
        }
    ],
    "goniometer": [
        {
            "axes": [
                [1.0, -0.0025, 0.0056],
                [-0.006, -0.0264, -0.9996],
                [1.0, 0.0, 0.0],
            ],
            "angles": [0.0, 0.0, 0.0],
            "names": ["phi", "chi", "omega"],
            "scan_axis": 2,
        }
    ],
    "scan": [
        {
            "image_range": [1, 1],
            "batch_offset": 0,
            "properties": {
                "epochs": [0.0],
                "exposure_time": [0.0],
                "oscillation": [0.0, 0.09999999999999964],
            },
        }
    ],
    "crystal": [
        {
            "__id__": "crystal",
            "real_space_a": [-18.3617922001806, -1.3985262449809124, -54.866705726818],
            "real_space_b": [-32.09880690850544, 47.2184015325075, 9.47764074028691],
            "real_space_c": [115.77476247140888, 86.89078795711316, -40.81230881268615],
            "space_group_hall_symbol": " P 1",
        }
    ],
    "profile": [],
    "scaling_model": [],
}


def test_predict_static(tmp_path):
    predictor_path: str | Path | None = os.getenv("PREDICTOR")

    with open(tmp_path / "test.expt", "w") as f:
        json.dump(expt_json, f)

    expts = tmp_path / "test.expt"
    subprocess.run(
        [predictor_path, "-e", expts],
        capture_output=True,
        cwd=tmp_path,
    )
    # assert not proc.stderr - H5Tclose error?
    assert (tmp_path / "predicted.refl").exists()
    with h5py.File(tmp_path / "predicted.refl", "r") as file:
        data = file["/dials/processing/group_0/id"]
        assert data.size == 464
        ## check a few random reflections by comparison to dials.
        miller_index = file["/dials/processing/group_0/miller_index"][()].reshape(-1, 3)
        xyzcal_px = file["/dials/processing/group_0/xyzcal.px"][()].reshape(-1, 3)
        expected_hkl = [[-28, 14, 93], [-26, 14, 90], [-14, -30, -11]]
        expected_xyzcal = [
            [3937.314, 91.352, 0.649],
            [3769.996, 214.530, 0.341],
            [2979.930, 3628.805, 0.877],
        ]
        for hkl, xyz in zip(expected_hkl, expected_xyzcal):
            sel = np.all(miller_index == hkl, axis=1)
            assert xyzcal_px[sel].flatten() == pytest.approx(xyz, abs=1e-2)


def test_predict_varying(tmp_path):
    predictor_path: str | Path | None = os.getenv("PREDICTOR")

    # Add a scan varying crystal model.
    sv_expt_json = expt_json
    sv_expt_json["crystal"][0]["A_at_scan_points"] = [
        [
            -0.005459727201059019,
            -0.00957881993314865,
            0.005117309749372354,
            -0.00042498931737436986,
            0.014098506559502566,
            0.0038416886950916533,
            -0.016390331545472165,
            0.002844169681824414,
            -0.0018112274387013555,
        ],
        [
            -0.005459734253117143,
            -0.009578815538092726,
            0.005117308300437957,
            -0.00042498758360674213,
            0.014098504268823347,
            0.003841688575830816,
            -0.01639033413246863,
            0.002844172984156842,
            -0.001811225938788269,
        ],
    ]

    with open(tmp_path / "test.expt", "w") as f:
        json.dump(sv_expt_json, f)

    expts = tmp_path / "test.expt"
    subprocess.run(
        [predictor_path, "-e", expts],
        capture_output=True,
        cwd=tmp_path,
    )
    # assert not proc.stderr - H5Tclose error?
    assert (tmp_path / "predicted.refl").exists()
    with h5py.File(tmp_path / "predicted.refl", "r") as file:
        data = file["/dials/processing/group_0/id"]
        assert data.size == 451
        ## check a few random reflections by comparison to dials.
        miller_index = file["/dials/processing/group_0/miller_index"][()].reshape(-1, 3)
        xyzcal_px = file["/dials/processing/group_0/xyzcal.px"][()].reshape(-1, 3)
        expected_hkl = [[-28, 14, 93], [-26, 14, 90], [-14, -30, -11]]
        expected_xyzcal = [
            [3937.727, 90.932, 0.782],
            [3770.352, 214.153, 0.473],
            [2980.114, 3628.977, 0.943],
        ]
        for hkl, xyz in zip(expected_hkl, expected_xyzcal):
            sel = np.all(miller_index == hkl, axis=1)
            assert xyzcal_px[sel].flatten() == pytest.approx(xyz, abs=1e-2)
