# Fast Feedback Service
This is a GPU implementation of the [DIALS](https://dials.github.io/) spotfinder algorithm. It is designed to be used in a beamline for real-time feedback and x-ray centring during data collections.

The service, a python script in [`src/`], watches a queue for requests to process images. It then launches the compiled C++ CUDA executable to process the images. This executable is compiled from the source code in [`spotfinder/`].

## Setup
In order to create a development environment and compile the service, you need to have the following:

### Dependencies
Everything the build needs is in [`environment.yml`], which is the
authoritative list: compilers, cmake, ninja, the HDF5 and Boost stack, and
the `scikit-build-core` backend that pip builds through.

```bash
mamba env create -f environment.yml -p ./ENV
```

The [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is the
exception, since it comes from a module rather than from conda. Which
version is covered under [Building the project](#building-the-project).

### Building the project

```bash
module load cuda/13.3.1     # what the deployments build against; see below
mamba activate ENV/
./build.sh
```

The first run configures, compiles, and installs the python package into
the active environment, which takes a couple of minutes. Every run after
that is just a compile, and costs nothing when nothing has changed.

Programs are run straight out of the build directory and are always what
the last compile produced:

```bash
./build/bin/spotfinder image.h5
```

`build.sh` is a convenience wrapper. Underneath it there are two commands:

```bash
pip install --no-build-isolation -e .   # configure, compile, install
ninja -C build                          # compile
```

pip is what drives the build. It runs cmake, which compiles everything,
and then installs the python package and the compiled programs into the
environment. After that a plain `ninja -C build` is enough for a C++
change, which is what `build.sh` does on later runs.

One thing a compile does not cover: `import ffs.index` and
`import ffs.integrate` read from the environment's site-packages, not from
the build directory, because that is the only place python looks for a
submodule of an installed package. If you change the C++ behind either
extension module, use `./build.sh --install` to refresh them.

**Options:**

```bash
./build.sh              compile; install first if the environment needs it
./build.sh --install    force the install, refreshing the environment
./build.sh --clean      discard the build directory and start over
./build.sh -j N         limit parallelism
./build.sh --cuda MOD   suggest a different CUDA module
```

The CUDA version is read from `ARG CUDA_VERSION` in the [`Dockerfile`],
because the container base image is what limits which version can be
used. `build.sh` reports the CUDA it found and warns when it differs from
that one, since binaries built against one CUDA need the same one on the
library path to run.

#### Choosing a GPU architecture
CUDA code is compiled for specific compute capabilities, and a binary
will not load on a GPU it was not compiled for. The `CUDA_ARCH` cmake
option controls this:

```bash
./build.sh                                                 # native: the GPU in this machine (default)
CMAKE_ARGS=-DCUDA_ARCH=86 ./build.sh --install             # a specific compute capability
CMAKE_ARGS="-DCUDA_ARCH=80;90" ./build.sh --install        # several
CMAKE_ARGS=-DCUDA_ARCH=all-supported ./build.sh --install  # everything the container ships
```

The architecture is fixed when cmake configures, so changing it needs
`--install`, which reconfigures. `CMAKE_ARGS` is how any cmake option
reaches the build now that pip drives it.

The default, `native`, reads the compute capability from `nvidia-smi`
and builds a single cubin for it, which is the fastest option for local
development. If no GPU is present it falls back to `sm_75`.

`all-supported` is what the published container image is built with. It
produces a fat binary carrying native code for every architecture from
Turing to Blackwell, plus PTX so that a card newer than any of them is
JIT-compiled by the driver rather than rejected. It takes considerably
longer to compile.

Note that CUDA 13 dropped Maxwell, Pascal and Volta, so `sm_75` (Turing)
is the oldest architecture the container image can target.

For what a fat binary actually contains and how the driver picks out of
it, see [nvcc GPU Compilation][nvcc-gpu-compilation] on virtual versus
real architectures, cubins and PTX, and the [Blackwell Compatibility
Guide][blackwell-compat] for the rules on which cubin runs on which
card.

[nvcc-gpu-compilation]: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-compilation
[blackwell-compat]: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/

## Usage
### Environment Variables
The service uses the following environment variables:
- `SPOTFINDER`: The path to the compiled spotfinder executable.
  - If not set, it is looked up on `PATH`, where an install puts it. Point
    this at `build/bin/spotfinder` to use a development build instead.
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
(These tests assume you have built in a folder called `build`. The 32-bit
data tests use `build/bin/spotfinder32`, which the same build produces.)
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
[`Dockerfile`]: Dockerfile
[`environment.yml`]: environment.yml
[`baseline/spotfinder`]: baseline/spotfinder
[`h5read/`]: h5read/
[`include/`]: include/
[`tests/`]: tests/
