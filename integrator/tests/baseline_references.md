# Generate baseline references for KabschTransformTest

`KabschTransformTest` (`test_kabsch.cc`) runs the GPU Kabsch foreground/background
pixel classifier and compares per-reflection pixel counts against a CPU baseline,
matched by row index.

## Data

Lives in `$FFS_INTEGRATE_TEST_DATA` (default `/scratch/ffs_integrate_test_data`).
The test is skipped if the directory is missing.

| File | Provides |
|------|----------|
| `integrated_1_10.refl` | predictions: `s1`, `xyzcal.mm`, `bbox`, `miller_index` |
| `indexed_1_10.expt` | beam, detector panel, goniometer, scan |
| `baseline_<algo>_1_10.refl` | reference `num_pixels.foreground` / `.background` |

The baseline references are generated. Regenerate them whenever the baseline algorithm, its parameters, or the input data change. `$FFS_KABSCH_BASELINE_REFERENCE` overrides the reference path.

## Generating the baseline references

Run the CPU baseline integrator once per algorithm, with the same `sigma_b` /
`sigma_m` the test uses (`N_SIGMA = 3`; changing these without updating the test
causes spurious mismatches):

```bash
DATA=/scratch/ffs_integrate_test_data   # or $FFS_INTEGRATE_TEST_DATA

bin/baseline_integrator "$DATA/integrated_1_10.refl" "$DATA/indexed_1_10.expt" \
    "$DATA/baseline_dials_1_10.refl"     --algorithm dials     --sigma_b 0.03 --sigma_m 0.1
bin/baseline_integrator "$DATA/integrated_1_10.refl" "$DATA/indexed_1_10.expt" \
    "$DATA/baseline_ellipsoid_1_10.refl" --algorithm ellipsoid --sigma_b 0.03 --sigma_m 0.1
```

