# Baseline Integrator

An unoptimised CPU-only implementation of the DIALS integration algorithm for testing and comparison.

## Overview

The baseline integrator provides a reference implementation of integration algorithms that:
- Reads HDF5 reflection tables and experiment files
- Computes bounding boxes using the Kabsch coordinate system
- Transforms pixel coordinates to Kabsch space
- Uses only CPU algorithms (no CUDA/GPU dependencies)

## Files

- `integrator.cc` - Main application entry point
- `kabsch.cc` - Kabsch coordinate transformation algorithms
- `extent./cc` - Bounding box computation algorithms

## Usage

```bash
./baseline_integrator reflections.refl experiments.expt [options]
```

### Required Arguments
- `reflections.refl` - Input reflection table (HDF5 format)
- `experiments.expt` - Input experiment list (JSON format)

### Optional Arguments
- `--sigma_m` - Mosaicity standard deviation (default: 0.0001)
- `--sigma_b` - Beam divergence standard deviation (default: 0.0001)
- `--output` - Output HDF5 file (default: baseline_integration_results.h5)

### Example

```bash
./baseline_integrator strong.refl experiments.expt \
  --sigma_m 0.0002 \
  --sigma_b 0.0001 \
  --output my_results.h5
```

## Output

The application produces:
1. Main results file containing:
   - Original reflection data
   - Computed bounding boxes in `baseline_bbox` column
2. Voxel data file (if voxels processed) containing:
   - Kabsch coordinates for sampled voxel centers
   - Reflection IDs mapping voxels to reflections
   - Pixel coordinates and s1 lengths

## Algorithm Details

### Kabsch Coordinate System
The Kabsch coordinate system provides a geometry-invariant framework for integration:
- ε₁: displacement perpendicular to scattering plane
- ε₂: displacement within scattering plane  
- ε₃: displacement along rotation axis

### Bounding Box Computation
Bounding boxes are computed by:
1. Calculating angular divergence parameters (Δb, Δm)
2. Projecting divergences onto Kabsch basis vectors
3. Finding corners of integration region in reciprocal space
4. Transforming back to detector pixel coordinates