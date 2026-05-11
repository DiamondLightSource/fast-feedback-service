"""
Tests for the fast-feedback integrator.

Test command:
bin/integrator
../integrator/tests/comp/filtered.refl
../integrator/tests/comp/indexed.expt --sigma_m 0.002 --sigma_b 0.0002
"""

import os
import subprocess
from pathlib import Path

import h5py


def test_integrator_basic(dials_data, tmp_path):
    """Test basic integrator functionality with reflection and experiment files."""

    integrator_path: str | Path | None = os.getenv("INTEGRATOR")
    assert integrator_path, "INTEGRATOR environment variable not set"

    # /dls/science/groups/scisoft/DIALS/dials_data/thaumatin_i03_rotation/
    dials_data_path = dials_data("thaumatin_i03_rotation", pathlib=True)
    assert dials_data_path

    integration_data_path = Path("tests/thaumatin_i03_rotation.data").resolve()
    assert integration_data_path.exists()

    # sigma b: 0.022098 degrees = 0.0003856828581 radians
    # sigma m: 0.105343 degrees = 0.0018385821939 radians
    command = [
        integrator_path,
        "--reflection",
        str(integration_data_path / "indexed.refl"),
        "--experiment",
        str(integration_data_path / "indexed.expt"),
        "--images",
        dials_data_path / "thau_2_1.nxs",
        "--sigma_m",
        "0.00183",
        "--sigma_b",
        "0.00038",
    ]

    proc = subprocess.run(
        command,
        capture_output=True,
        cwd=tmp_path,
    )
    # assert not proc.stderr, f"Error: {proc.stderr.decode()}"
    assert proc.returncode == 0, f"Non-zero return code: {proc.returncode}"

    # Validate output files exist
    output_reflections = tmp_path / "output_reflections.h5"
    voxel_kabsch_data = tmp_path / "voxel_kabsch_data.h5"

    assert output_reflections.exists(), "output_reflections.h5 not created"
    assert voxel_kabsch_data.exists(), "voxel_kabsch_data.h5 not created"

    # Validate reflection data structure
    with h5py.File(output_reflections, "r") as f:
        # Navigate to the processing group
        processing_group = f["dials/processing/group_0"]

        # Check that computed_bbox column was added
        assert "computed_bbox" in processing_group, (
            "computed_bbox column not found in output"
        )

        # Check original columns are preserved
        assert "s1" in processing_group, "s1 column missing from output"
        assert "xyzcal.mm" in processing_group, "xyzcal.mm column missing from output"
        assert "bbox" in processing_group, "bbox column missing from output"

        # Validate computed_bbox has correct shape
        bbox_data = processing_group["bbox"][:]
        computed_bbox_data = processing_group["computed_bbox"][:]

        assert bbox_data.shape == computed_bbox_data.shape, (
            f"Shape mismatch: bbox {bbox_data.shape} vs computed_bbox {computed_bbox_data.shape}"
        )

        # Basic sanity check: bounding boxes should have reasonable values
        assert computed_bbox_data.shape[1] == 6, "Computed bbox should have 6 columns"

        # Check that x_min < x_max and y_min < y_max for computed bboxes
        x_min, x_max = computed_bbox_data[:, 0], computed_bbox_data[:, 1]
        y_min, y_max = computed_bbox_data[:, 2], computed_bbox_data[:, 3]
        z_min, z_max = computed_bbox_data[:, 4], computed_bbox_data[:, 5]

        # Debug: Print some computed bbox values
        print("\nComputed bbox statistics:")
        print(f"Number of reflections: {len(computed_bbox_data)}")
        print(f"x_min range: {x_min.min():.6e} to {x_min.max():.6e}")
        print(f"x_max range: {x_max.min():.6e} to {x_max.max():.6e}")
        print(f"y_min range: {y_min.min():.6e} to {y_min.max():.6e}")
        print(f"y_max range: {y_max.min():.6e} to {y_max.max():.6e}")
        print(f"z_min range: {z_min.min():.6e} to {z_min.max():.6e}")
        print(f"z_max range: {z_max.min():.6e} to {z_max.max():.6e}")

        # Show first few computed bboxes
        print("\nFirst 5 computed bboxes:")
        for i in range(min(5, len(computed_bbox_data))):
            print(
                f"  [{i}]: x=({x_min[i]:.6e}, {x_max[i]:.6e}), y=({y_min[i]:.6e}, {y_max[i]:.6e}), z=({z_min[i]:.6e}, {z_max[i]:.6e})"
            )

        # Compare with original bbox for context
        orig_bbox = processing_group["bbox"][:]
        orig_x_min, orig_x_max = orig_bbox[:, 0], orig_bbox[:, 1]
        print(
            f"\nOriginal bbox x_min range: {orig_x_min.min():.6e} to {orig_x_min.max():.6e}"
        )
        print(
            f"Original bbox x_max range: {orig_x_max.min():.6e} to {orig_x_max.max():.6e}"
        )

        # Count invalid bboxes
        invalid_x = ~(x_min < x_max)
        invalid_y = ~(y_min < y_max)
        invalid_z = ~(z_min < z_max)
        print("\nInvalid bbox counts:")
        print(f"Invalid x bounds: {invalid_x.sum()} / {len(computed_bbox_data)}")
        print(f"Invalid y bounds: {invalid_y.sum()} / {len(computed_bbox_data)}")
        print(f"Invalid z bounds: {invalid_z.sum()} / {len(computed_bbox_data)}")

        assert (x_min < x_max).all(), "Invalid x bounds in computed bbox"
        assert (y_min < y_max).all(), "Invalid y bounds in computed bbox"
        assert (z_min < z_max).all(), "Invalid z bounds in computed bbox"

    # Validate voxel data structure
    with h5py.File(voxel_kabsch_data, "r") as f:
        # Navigate to the processing group
        processing_group = f["dials/processing/group_0"]

        # Check expected columns exist
        expected_columns = [
            "kabsch_coordinates",
            "reflection_id",
            "pixel_coordinates",
            "voxel_s1_length",
        ]
        for col in expected_columns:
            assert col in processing_group, f"Column {col} not found in voxel data"

        # Check data consistency
        kabsch_coords = processing_group["kabsch_coordinates"][:]
        reflection_ids = processing_group["reflection_id"][:]
        pixel_coords = processing_group["pixel_coordinates"][:]
        s1_lengths = processing_group["voxel_s1_length"][:]

        num_voxels = len(reflection_ids)
        assert kabsch_coords.shape == (num_voxels, 3), (
            f"Kabsch coordinates shape mismatch: {kabsch_coords.shape}"
        )
        assert pixel_coords.shape == (num_voxels, 3), (
            f"Pixel coordinates shape mismatch: {pixel_coords.shape}"
        )
        assert len(s1_lengths) == num_voxels, (
            f"S1 lengths count mismatch: {len(s1_lengths)} vs {num_voxels}"
        )

        # Validate that s1_lengths are positive
        assert (s1_lengths > 0).all(), "All s1 lengths should be positive"

        # Validate that reflection IDs are non-negative integers
        assert (reflection_ids >= 0).all(), "Reflection IDs should be non-negative"

        print(
            f"Successfully validated {num_voxels} voxels across {len(set(reflection_ids.flatten()))} reflections"
        )
