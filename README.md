<h1 align="center">Fast Feedback Service</h1>

<div align="center">

![FAST, drawn in diffraction spots found by the spotfinder][banner]

<sub><sub>*FAST* logo detected through the GPU spotfinder  |  [`Generated with synthetic data`][banner_svg]</sub></sub>

</div>

<p align="center">
  <a href="https://diamondlightsource.github.io/fast-feedback-service/">Documentation</a> ·
  <a href="https://github.com/DiamondLightSource/fast-feedback-service/pkgs/container/fast-feedback-service">Container image</a> ·
  <a href="https://dials.github.io/">DIALS</a>
</p>

<p align="center">
  <a href="https://github.com/DiamondLightSource/fast-feedback-service/actions/workflows/docker-image.yml"><img alt="Build" src="https://img.shields.io/github/actions/workflow/status/DiamondLightSource/fast-feedback-service/docker-image.yml?branch=main&style=flat-square&label=build&logo=githubactions&logoColor=white"></a>
  <a href="https://diamondlightsource.github.io/fast-feedback-service/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/DiamondLightSource/fast-feedback-service/docs.yml?branch=main&style=flat-square&label=docs&logo=readthedocs&logoColor=white"></a>
  <a href="https://github.com/DiamondLightSource/fast-feedback-service/pkgs/container/fast-feedback-service"><img alt="Container image" src="https://img.shields.io/badge/container-ghcr.io-1F6FBF?style=flat-square&logo=docker&logoColor=white"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img alt="CUDA 13" src="https://img.shields.io/badge/CUDA-13-76B900?style=flat-square&logo=nvidia&logoColor=white"></a>
  <a href="https://github.com/DiamondLightSource/fast-feedback-service/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/DiamondLightSource/fast-feedback-service?style=flat-square&color=1F6FBF"></a>
</p>

GPU-accelerated processing of X-ray diffraction data, following the
[DIALS](https://dials.github.io/) algorithms. It serves both as a
real-time feedback service on the beamline, where GPU spotfinding
drives X-ray centring, and as an accelerated spotfind-index-integrate
pipeline for fast feedback on complete datasets.

Spotfinding and integration run on the GPU. Rotation data is indexed
by the C++ baseline indexer; stills and SSX data are indexed on the
GPU by [ffbidx](https://github.com/paulscherrerinstitute/fast-feedback-indexer).

```text
images -> spotfinder (GPU) -> indexer -> integrator (GPU) -> intensities
               |              rotation: C++ baseline
               v              stills/SSX: ffbidx (GPU)
        X-ray centring
```

## Quickstart

You need a machine with an NVIDIA GPU and the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 13.3, with
`nvcc` on `PATH`. Everything else comes from [`environment.yml`]:

```bash
mamba env create -f environment.yml -p ./ENV
mamba activate ENV/
./build.sh
./build/bin/spotfinder image.h5
```

## Documentation

The full documentation, including the API reference generated from the
source, is published at
<https://diamondlightsource.github.io/fast-feedback-service/>.

- [Building][building]: dependencies, `build.sh`, CUDA versions and
  choosing a GPU architecture.
- [Running the service][service]: environment variables and launching
  under Zocalo.
- [Testing][testing]: the pytest suite and the regression data.

<!-- PERFORMANCE: a measured throughput table (detector, GPU, images/s) goes here -->

## Repository Structure
| Folder              | Contents                                                    |
| ------------------- | ----------------------------------------------------------- |
| [`spotfinder/`]     | CUDA implementation of the DIALS dispersion spotfinder      |
| [`integrator/`]     | CUDA integrator, for rotation and SSX data                  |
| [`baseline/`]       | CPU implementations of the spotfinder, indexer, predictor, refiner and integrator, used for comparison and for indexing rotation data |
| [`h5read/`]         | A small C/C++ library to read HDF5 files in a standard way  |
| [`include/`]        | Headers shared across the programs                          |
| [`src/ffs/`]        | The Python service and pipelines, and the shared C++ sources |
| [`dx2/`]            | Submodule of dx2, a C++ implementation of the dxtbx data model |
| [`scripts/`]        | Developer scripts                                           |
| [`tests/`]          | Python tests for the programs and pipelines                 |
| [`docs/`]           | Sphinx documentation                                        |

[banner]: docs/images/banner.svg
[banner_svg]: scripts/banner_svg.py
[building]: docs/building.md
[service]: docs/service.md
[testing]: docs/testing.md
[`environment.yml`]: environment.yml
[`spotfinder/`]: spotfinder/
[`integrator/`]: integrator/
[`baseline/`]: baseline/
[`h5read/`]: h5read/
[`include/`]: include/
[`src/ffs/`]: src/ffs/
[`dx2/`]: dx2/
[`scripts/`]: scripts/
[`tests/`]: tests/
[`docs/`]: docs/
