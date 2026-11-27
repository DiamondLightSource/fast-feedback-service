import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

expected_output_crystals = [
    {
        "__id__": "crystal",
        "real_space_a": [-73.3657913208008, -0.9755071997642522, -29.385274887084968],
        "real_space_b": [-1.410078525543213, -78.96023559570315, 4.7539267539978045],
        "real_space_c": [-14.323430061340332, 3.0948548316955575, 35.46766662597657],
        "space_group_hall_symbol": "P 1",
    },
    {
        "__id__": "crystal",
        "real_space_a": [-48.313030242919915, -38.71310806274415, -47.89557266235352],
        "real_space_b": [56.415370941162124, -54.083133697509766, -14.067997932434073],
        "real_space_c": [-13.226760864257818, -20.67352104187012, 29.483562469482422],
        "space_group_hall_symbol": "P 1",
    },
    {
        "__id__": "crystal",
        "real_space_a": [56.055877685546854, -12.154879570007328, 53.15177536010743],
        "real_space_b": [-15.600127220153817, 70.9559555053711, 32.53805160522461],
        "real_space_c": [-25.86274528503418, -16.022525787353516, 23.559381484985348],
        "space_group_hall_symbol": "P 1",
    },
    {
        "__id__": "crystal",
        "real_space_a": [26.21974182128906, 73.18634033203125, 14.546281814575199],
        "real_space_b": [-13.889213562011712, 20.628921508789062, -74.7467575073242],
        "real_space_c": [-35.432971954345696, 10.647741317749022, 10.097808837890623],
        "space_group_hall_symbol": "P 1",
    },
    {
        "__id__": "crystal",
        "real_space_a": [33.06575012207031, -63.21855545043945, 33.94499206542969],
        "real_space_b": [-36.75036239624024, 17.65143585205078, 68.35107421875001],
        "real_space_c": [-29.752025604248047, -21.311590194702152, -10.15173053741455],
        "space_group_hall_symbol": "P 1",
    },
    {
        "__id__": "crystal",
        "real_space_a": [-56.92699432373046, 26.57844924926757, 49.20372390747069],
        "real_space_b": [-1.6878218650817849, 68.55046844482425, -39.88540649414063],
        "real_space_c": [-26.45606422424316, -14.229729652404787, -22.834901809692386],
        "space_group_hall_symbol": "P 1",
    },
    {
        "__id__": "crystal",
        "real_space_a": [-20.259445190429684, 9.534605979919434, -75.90452575683594],
        "real_space_b": [-74.32333374023438, -21.109268188476562, 16.741840362548825],
        "real_space_c": [-8.774566650390625, 36.43461227416992, 7.592679500579832],
        "space_group_hall_symbol": "P 1",
    },
]


def make_imported_expt(dials_data, tmp_path):
    data = {"__id__": "ExperimentList"}
    data["experiment"] = [
        {
            "__id__": "Experiment",
            "identifier": f"{i}",
            "beam": 0,
            "detector": 0,
            "goniometer": 0,
            "scan": i,
            "imageset": 0,
        }
        for i in range(0, 20)
    ]
    data["scan"] = [
        {
            "image_range": [i, i],
            "batch_offset": 0,
            "properties": {
                "epochs": [0.0],
                "exposure_time": [0.0],
                "oscillation": [0.0],
                "oscillation_width": [0.0],
            },
            "valid_image_ranges": {},
        }
        for i in range(1, 21)
    ]
    data["imageset"] = [
        {
            "__id__": "ImageSequence",
            "template": str(dials_data("lysozyme_ssx_25keV") / "lysozyme_25keV.nxs"),
            "single_file_indices": list(range(0, 20)),
            "mask": "",
            "gain": "",
            "pedestal": "",
            "dx": "",
            "dy": "",
            "params": {"dynamic_shadowing": "Auto", "multi_panel": False},
        }
    ]
    data["beam"] = [
        {
            "__id__": "monochromatic",
            "direction": [0.0, 0.0, 1.0],
            "wavelength": 0.4959,
            "divergence": 0.0,
            "sigma_divergence": 0.0,
            "polarization_normal": [0.0, 1.0, 0.0],
            "polarization_fraction": 0.999,
            "flux": 0.0,
            "transmission": 1.0,
            "probe": "x-ray",
            "sample_to_source_distance": 0.0,
        }
    ]
    data["detector"] = [
        {
            "panels": [
                {
                    "name": "/entry/instrument/detector/module",
                    "type": "SENSOR_PAD",
                    "fast_axis": [1.0, 0.0, 0.0],
                    "slow_axis": [0.0, -1.0, 0.0],
                    "origin": [-119.625, 126.90, -420.33],
                    "raw_image_offset": [0, 0],
                    "image_size": [3108, 3262],
                    "pixel_size": [0.075, 0.075],
                    "trusted_range": [-1.0, 50649.0],
                    "thickness": 0.75,
                    "material": "CdTe",
                    "mu": 7.285849919020173,
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
                "origin": [-0.20550000000000068, 0.030750000000011823, 0.0],
                "raw_image_offset": [0, 0],
                "image_size": [0, 0],
                "pixel_size": [0.0, 0.0],
                "trusted_range": [0.0, 0.0],
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
    ]
    data["goniometer"] = [
        {
            "rotation_axis": [1.0, 0.0, 0.0],
            "fixed_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "setting_rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    ]
    data["crystal"] = []
    data["profile"] = []
    data["scaling_model"] = []
    with open(tmp_path / "imported.expt", "w") as f:
        json.dump(data, f)


def test_gpu_ssx_index_integrate(dials_data, tmp_path):
    """
    Test the GPU SSX indexing and integration code.

    First runs spotfinding and then ssx_index.py script, then ssx_integrate.
    Skips if ffbidx is not sourced/available.
    """
    try:
        import ffbidx  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("ffbidx not installed")
    try:
        ffbidx.Indexer()
        #ffbidx.runtime_check()
    except RuntimeError:
        pytest.skip("ffbidx installed but not functional on this system")

    # dials.import $(dials.data get -q lysozyme_ssx_25keV)/lysozyme_25keV.nxs \
    #   distance=420.33 fast_slow_beam_centre=1597.74,1692.41 image_range=1,20
    make_imported_expt(dials_data, tmp_path)
    imported_expt = tmp_path / "imported.expt"
    assert imported_expt.exists()

    # First do spotfinding
    spotfinder_path: str | Path | None = os.getenv("SPOTFINDER_32BIT")
    ssx_index = Path(os.getenv("FFS_ROOT_DIR")) / "src" / "ffs" / "ssx_index.py"
    integrator_path: str | Path | None = os.getenv("SSX_INTEGRATOR")

    assert spotfinder_path
    d = dials_data("lysozyme_ssx_25keV")
    proc = subprocess.run(
        [
            spotfinder_path,
            d / "lysozyme_25keV.nxs",
            "--threads",
            "1",
            "--save-h5",
            "--images",
            "20",
            "--algorithm",
            "dispersion_extended",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr

    # check that the ids are in sorted order, as this is assumed in the indexer
    r = h5py.File(tmp_path / "results_ffs.h5")
    ids = r["dials"]["processing"]["group_0"]["id"]
    is_sorted = np.all(ids[:-1] <= ids[1:])
    assert is_sorted

    # Now run the indexer
    # Note, if you run dials equivalent procesing using dials.find_spots, the z component of
    # xyzobs.px increments with image number, which results in a different orientation matrix
    # and miller indices equivalent by symmetry but in a different indexing choice.
    proc = subprocess.run(
        [
            "python",
            ssx_index,
            "-r",
            "results_ffs.h5",
            "-e",
            "imported.expt",
            "-c",
            "79",
            "79",
            "38",
            "90",
            "90",
            "90",
            "--output-summary",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    assert (tmp_path / "indexed.refl").exists()
    assert (tmp_path / "indexed.expt").exists()
    with open(tmp_path / "indexed.expt", "r") as f:
        expts = json.load(f)
        crystals = expts["crystal"]
        for i, (crystal, expected) in enumerate(
            zip(crystals, expected_output_crystals)
        ):
            print(f"Comparing crystal {i + i}")
            assert crystal["real_space_a"] == pytest.approx(
                expected["real_space_a"], abs=1e-9
            )
            assert crystal["real_space_b"] == pytest.approx(
                expected["real_space_b"], abs=1e-9
            )
            assert crystal["real_space_c"] == pytest.approx(
                expected["real_space_c"], abs=1e-9
            )
    assert (tmp_path / "results_summary.json").exists()
    with open(tmp_path / "results_summary.json", "r") as f:
        results_summary = json.load(f)
        assert len(results_summary) == 20

    with h5py.File(tmp_path / "indexed.refl", "r") as file:
        ids = file["/dials/processing/group_0/id"]
        xyzcal = file["/dials/processing/group_0/xyzcal.px"]
        xyzobs = file["/dials/processing/group_0/xyzobs.px.value"]
        miller_index = file["/dials/processing/group_0/miller_index"]
        delpsi = file["/dials/processing/group_0/delpsical.rad"]
        assert len(ids) == 83
        assert set(ids) == set(range(0, 7))  # i.e. 7 / 20 images were indexed
        # test values against expected / dials equivalent
        minimum = np.min(xyzcal, axis=0)
        maximum = np.max(xyzcal, axis=0)
        mean = np.mean(xyzcal, axis=0)
        assert minimum.tolist() == pytest.approx([734.62, 736.70, 0.00], abs=5e-3)
        assert maximum.tolist() == pytest.approx([2787.80, 2673.81, 0.00], abs=5e-3)
        assert mean.tolist() == pytest.approx([1684.52, 1714.55, 0.00], abs=5e-3)

        minimum = np.min(xyzobs, axis=0)
        maximum = np.max(xyzobs, axis=0)
        mean = np.mean(xyzobs, axis=0)
        assert minimum.tolist() == pytest.approx([735.07, 736.18, 0.50], abs=5e-3)
        assert maximum.tolist() == pytest.approx([2788.09, 2672.50, 18.50], abs=5e-3)
        assert mean.tolist() == pytest.approx([1684.60, 1714.56, 12.6566], abs=5e-3)

        assert np.min(miller_index, axis=0).tolist() == [-19, -30, -17]
        assert np.max(miller_index, axis=0).tolist() == [25, 21, 10]
        assert [int(i) for i in np.mean(miller_index, axis=0).tolist()] == [1, -1, 0]

        assert np.min(delpsi) == pytest.approx(-0.0018911, abs=1e-6)
        assert np.max(delpsi) == pytest.approx(0.0138742, abs=1e-6)
        assert np.mean(delpsi) == pytest.approx(0.0001618, abs=1e-6)

    ## Now run gpu integrator
    proc = subprocess.run(
        [
            integrator_path,
            "-r",
            "indexed.refl",
            "-e",
            "indexed.expt",
        ],
        capture_output=True,
        cwd=tmp_path,
    )
    assert not proc.stderr
    assert (tmp_path / "integrated.refl").exists()
    assert (tmp_path / "integrated.expt").exists()

    # Expected n refls are not ground truth values - just use to catch unexpected changes
    # from current state
    expected_n_refls = [1626, 270, 118, 450, 214, 116, 182]
    expected_mean_intensity = [0.76, 3.43, 6.61, 0.46, 12.88, 11.0, 6.25]
    with h5py.File(tmp_path / "integrated.refl", "r") as file:
        ids = file["/dials/processing/group_0/id"][()]
        intensities = file["/dials/processing/group_0/intensity.sum.value"][()]
        assert ids.size == 2976
        assert set(ids) == set(range(0, 7))  # 7 indexed images are all integrated.
        for id_, n_expected, I_expected in zip(
            range(0, 7), expected_n_refls, expected_mean_intensity
        ):
            n_this = ids[ids == id_].size
            assert n_this == n_expected
            intensities_this = intensities[ids == id_]
            assert np.mean(intensities_this) == pytest.approx(I_expected, 1e-2)
