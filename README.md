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
- gemmi
- pytest
- dials-data

You can create a conda/mamba environment using the provided `environment.yml` file:
```bash
mamba env create -f environment.yml -p ./ENV
```

### Building the project

#### Using the build script (recommended)
This repository includes a convenient build script that handles submodule initialization, build configuration, and compilation. The build script supports both 16/32-bit pixel data formats and provides both development and production build modes.

**Quick start:**
```bash
mamba activate ENV/         # Activate your environment
cd fast-feedback-service/   # Go to the root of the repository
./build.sh                  # Build both 16-bit and 32-bit versions for development
```

**Build script options:**
```bash
./build.sh [OPTIONS]

OPTIONS:
    -p, --production       Build for production (single build directory)
    -3, --32bit            Use 32-bit pixel data (only with --production)
    -c, --clean            Clean build directories before building
    -j, --jobs N           Number of parallel jobs (default: auto-detected)
    -h, --help             Show help message
```

**Build modes:**

*Development Build (default):*
- Creates both `build/` (16-bit) and `build_32bit/` (32-bit) directories for 16/32-bit pixel data
- Uses `RelWithDebInfo` configuration for debugging with optimizations
- Builds both configurations for comprehensive testing

*Production Build:*
- Creates only `build/` directory with specified configuration
- Uses `Release` configuration for maximum performance
- Removes `build_32bit/` if it exists to avoid confusion

**Examples:**
```bash
./build.sh                              # Development build (both 16-bit and 32-bit)
./build.sh --production                 # Production build with 16-bit
./build.sh --production --32bit         # Production build with 32-bit
./build.sh --clean                      # Clean and rebuild development builds
./build.sh --production --clean --32bit # Clean production build with 32-bit
```

The build script automatically:
- Initializes git submodules if needed
- Detects and uses Ninja build system if available (faster than Make)
- Creates the `spotfinder` executable in `build/bin/` (and `build_32bit/bin/` for development builds)

#### Manual building
If you prefer to build manually or need more control over the build process:

**Initialising submodules:**
```bash
git submodule update --init --recursive
```

**Compiling the CUDA code:**
```bash
mamba activate ENV/         # Activate your environment
cd fast-feedback-service/   # Go to the root of the repository
mkdir build                 # Create a build directory
cd build                    # Go to the build directory
cmake ..                    # Run cmake to generate the makefile
make                        # Compile the code
```
This will create the executable `spotfinder` in the [`build/bin/`] directory.

**Configuring pixel data precision:**
By default, the service is compiled to handle 16-bit pixel data. For detectors that produce 32-bit pixel data, you can enable 32-bit support using the `PIXEL_DATA_32BIT` option.

*Using ccmake (recommended):*
```bash
cd build
ccmake ..                   # Opens an interactive configuration interface
# Navigate to PIXEL_DATA_32BIT and toggle it to ON
# Press 'c' to configure, then 'g' to generate
make                        # Compile with the new settings
```

*Using cmake command line:*
```bash
cd build
cmake -DPIXEL_DATA_32BIT=ON ..  # Enable 32-bit pixel data support
make                            # Compile with the new settings
```

### Installing the python module (for indexing)
This project defines a small python module, to provide functionality for indexing.
To run indexing code, this needs to be installed into the python environment by
running this command in the root directory:
```bash
pip install .
```

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

## Running the program tests
To run the tests, you need to have pytest and dials-data available in your environment and be on a machine with an NVIDIA GPU and the CUDA toolkit installed.
(These tests assume you have built the spotfinder in a folder called `build`. For the 32bit data tests, it is assumed that there is also
a build folder called `build_32bit` which was built with the `PIXEL_DATA_32BIT` cmake flag.)
Run:
```bash
python -m pytest tests/ -v --regression
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
