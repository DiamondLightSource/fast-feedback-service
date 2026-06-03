import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

data_dir = Path("/scratch/ffs_integrate_test_data/")
dials_integrated = data_dir / "integrated.refl"


@pytest.mark.skipif(not data_dir.is_dir(), reason="Data directory not available")
def test_baseline_integrator_dials_equivalence(tmp_path, dials_data):
    integrator_path: str | Path | None = os.getenv("BASELINE_INTEGRATOR")
    assert integrator_path is not None
    input_refls = data_dir / "predicted.refl"
    input_expts = data_dir / "indexed.expt"

    result = subprocess.run(
        [
            integrator_path,
            input_refls,
            input_expts,
            "-a",
            "dials",
            "--sigma_b",
            "0.03",
            "--sigma_m",
            "0.1",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert not result.returncode

    # Note, we are not expecting identical results to dials due to differences in
    # pixel parallax correction calculation, which slightly affects foreground and background
    # assignment. Instead test that the intensities are very close.

    with h5py.File(dials_integrated) as dials_refl:
        n_dials = len(dials_refl["/dials/processing/group_0/flags"])
        intensity_dials = dials_refl["/dials/processing/group_0/intensity.sum.value"]
        nfg_dials = dials_refl["/dials/processing/group_0/num_pixels.foreground"]
        midx_dials = dials_refl["/dials/processing/group_0/miller_index"]
        s1_dials = dials_refl["/dials/processing/group_0/s1"]

        with h5py.File(tmp_path / "integrated.refl") as ffs_refl:
            flags = ffs_refl["/dials/processing/group_0/flags"]
            assert len(flags) == n_dials
            nfg_ffs = ffs_refl["/dials/processing/group_0/num_pixels.foreground"]
            midx_ffs = ffs_refl["/dials/processing/group_0/miller_index"]
            intensity_ffs = ffs_refl["/dials/processing/group_0/intensity.sum.value"]
            s1_ffs = ffs_refl["/dials/processing/group_0/s1"]

            # Sort primarily based on miller index, then additionally on s1 to handle
            # the rare case of two reflections having the same miller index.
            keys_dials = (
                s1_dials[:, 2],
                s1_dials[:, 1],
                s1_dials[:, 0],
                midx_dials[:, 2],
                midx_dials[:, 1],
                midx_dials[:, 0],
            )
            keys_ffs = (
                s1_ffs[:, 2],
                s1_ffs[:, 1],
                s1_ffs[:, 0],
                midx_ffs[:, 2],
                midx_ffs[:, 1],
                midx_ffs[:, 0],
            )

            order_dials = np.lexsort(keys_dials)
            order_ffs = np.lexsort(keys_ffs)

            # Apply ordering
            nfg_dials_sorted = nfg_dials[order_dials]
            nfg_ffs_sorted = nfg_ffs[order_ffs]

            I_dials_sorted = intensity_dials[order_dials]
            I_ffs_sorted = intensity_ffs[order_ffs]

            midx_dials_sorted = midx_dials[order_dials]
            midx_ffs_sorted = midx_ffs[order_ffs]

            # check sort was correct
            assert np.all(midx_dials_sorted == midx_ffs_sorted)

            # compute deltas
            nfg_diff = nfg_dials_sorted - nfg_ffs_sorted
            I_diff = I_dials_sorted - I_ffs_sorted

            # filter nonzero differences
            nfg_deltas = nfg_diff[nfg_diff != 0]
            I_deltas = I_diff[I_diff != 0]

            ## Check that the current state of the output matches our expected level of difference.
            ## Total number of integrated reflections is 49579.
            assert len(nfg_deltas) == 870
            assert len(I_deltas) == 123
            assert np.max(np.absolute(np.array(I_deltas))) == 4


@pytest.mark.skipif(not data_dir.is_dir(), reason="Data directory not available")
def test_baseline_integrator_ellipsoid(tmp_path, dials_data):
    integrator_path: str | Path | None = os.getenv("BASELINE_INTEGRATOR")
    assert integrator_path is not None
    input_refls = data_dir / "predicted.refl"
    input_expts = data_dir / "indexed.expt"

    result = subprocess.run(
        [
            integrator_path,
            input_refls,
            input_expts,
            "-a",
            "ellipsoid",
            "--sigma_b",
            "0.03",
            "--sigma_m",
            "0.1",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert not result.returncode

    # Define two helper functions for sorting the data in the presence of
    # unequal numbers of reflections in the two files.
    def make_key_dtype(midx, s1):
        return np.dtype(
            [
                ("h", midx.dtype),
                ("k", midx.dtype),
                ("l", midx.dtype),
                ("s1x", s1.dtype),
                ("s1y", s1.dtype),
                ("s1z", s1.dtype),
            ]
        )

    def make_keys(midx, s1, dtype):
        keys = np.empty(len(midx), dtype=dtype)
        keys["h"], keys["k"], keys["l"] = midx.T
        keys["s1x"], keys["s1y"], keys["s1z"] = s1.T
        return keys

    with h5py.File(dials_integrated) as dials_refl:
        intensity_dials = dials_refl["/dials/processing/group_0/intensity.sum.value"][
            ()
        ]
        midx_dials = dials_refl["/dials/processing/group_0/miller_index"][()]
        s1_dials = dials_refl["/dials/processing/group_0/s1"][()]

        with h5py.File(tmp_path / "integrated.refl") as ffs_refl:
            midx_ffs = ffs_refl["/dials/processing/group_0/miller_index"][()]
            intensity_ffs = ffs_refl["/dials/processing/group_0/intensity.sum.value"][
                ()
            ]
            s1_ffs = ffs_refl["/dials/processing/group_0/s1"][()]
            assert len(intensity_ffs) == 50147

            dtype = make_key_dtype(midx_dials, s1_dials)
            keys_dials = make_keys(midx_dials, s1_dials, dtype)
            keys_ffs = make_keys(midx_ffs, s1_ffs, dtype)

            common_keys, idx_dials, idx_ffs = np.intersect1d(
                keys_dials, keys_ffs, return_indices=True
            )
            I_diff = intensity_dials[idx_dials] - intensity_ffs[idx_ffs]
            I_deltas = I_diff[I_diff != 0]

            ## Check that the current state of the output matches our expected level of difference.
            ## Most are different, but only a small number are >20 pixels different.
            assert len(I_deltas) == 47852
            absolute_differences = np.absolute(np.array(I_deltas))
            assert np.max(absolute_differences) == 166
            assert len(absolute_differences[absolute_differences > 30]) == 457
