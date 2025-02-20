# Fast Feedback Service
This is a GPU implementation of the [DIALS](https://dials.github.io/) spotfinder algorithm. It is designed to be used in a beamline for real-time feedback and x-ray centring during data collections.

The service, a python script in [`src/`], watches a queue for requests to process images. It then launches the compiled C++ CUDA executable to process the images. This executable is compiled from the source code in [`spotfinder/`].

## Setup
In order to create a development environment and compile the service, you need to have the following:

### Dependencies
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Boost](https://www.boost.org/)
- benchmark
- gtest
- cmake
- hdf5
- hdf5-external-filter-plugins

For example, you can create a conda/mamba environment with the following command:
```bash
mamba create -c conda-forge -p ENV boost-cpp benchmark gtest cmake hdf5 hdf5-external-filter-plugins compilers bitshuffle spdlog
```

### Initialising submodules
This repository uses submodules for the `dx2` dependency. To initialise the submodules, run the following in the root of the repository:
```bash
git submodule update --init --recursive
```

### Compiling the CUDA code
To compile the CUDA code, you need to run the following:
```bash
mamba activate ENV/         # Activate your environment
cd fast-feedback-service/   # Go to the root of the repository
mkdir build                 # Create a build directory
cd build                    # Go to the build directory
cmake ..                    # Run cmake to generate the makefile
make                        # Compile the code
```
This will create the executable `spotfinder` in the [`build/bin/`] directory.

## Usage
### Environment Variables
The service uses the following environment variables:
- `SPOTFINDER`: The path to the compiled spotfinder executable.
  - If not set, the service will look for the executable in the `build/bin/` or `_build/bin` directories.
- `LOG_LEVEL`: The level of logging to use provided by `spdlog`. Not setting this will default to `info`.
  - Other levels are: `trace`, `debug`, `info`, `warn`, `error`, `critical`, `off`.

### Running the service
To run the service, you need to be on a machine with an NVIDIA GPU and the CUDA toolkit installed.

Set up the environment variables:
```bash
export SPOTFINDER=/path/to/spotfinder
export ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml
```
Then launch the service through Zocalo:
```bash
zocalo.service -s GPUPerImageAnalysis
```

## Goals
- [x] 500 Hz throughput for Eiger real-time data collection
- [ ] 2500 Hz throughput for Jungfrau real-time data collection

## Repository Structure
| Folder Name       | Implementation                                             |
| -------------     | ---------------------------------------------------------- |
| [`baseline/spotfinder`]     | A standalone implementation of the standard DIALS dispersion spotfinder that can be used for comparison. |
| [`h5read/`]       | A small C/C++ library to read hdf5 files in a standard way |
| [`include/`]      | Common utility code, like coloring output, or image comparison, is stored here. |
| [`src/`]          | Service to run the spotfinder |
| [`spotfinder/`]   | CUDA implementation of the spotfinder algorithm |
| [`tests/`]        | Tests for the spotfinder code |

[`src/`]: src/
[`spotfinder/`]: spotfinder/
[`build/bin/`]: build/bin/
[`baseline/spotfinder`]: baseline/spotfinder
[`h5read/`]: h5read/
[`include/`]: include/
[`tests/`]: tests/
