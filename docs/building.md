# Building

## Dependencies
Everything the build needs is in [`environment.yml`], which is the
authoritative list: compilers, cmake, ninja, the HDF5 and Boost stack, and
the `scikit-build-core` backend that pip builds through.

```bash
mamba env create -f environment.yml -p ./ENV
```

The [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) is the
exception, and is installed separately. The build only needs `nvcc` on
`PATH`, so any installation will do; NVIDIA's installers put it under
`/usr/local`, versioned by release:

```bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
```

Which version to install is covered under
[Building the project](#building-the-project).

At Diamond, CUDA comes from the module system instead:

```bash
module load cuda/13.3.1
```

## Building the project

With CUDA on `PATH`:

```bash
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
./build.sh --cuda MOD   suggest a different CUDA module (Diamond only)
```

The CUDA version is read from `ARG CUDA_VERSION` in the [`Dockerfile`],
because the container base image is what limits which version can be
used. `build.sh` reports the CUDA it found and warns when it differs from
that one, since binaries built against one CUDA need the same one on the
library path to run. Its messages suggest the matching Diamond module;
elsewhere, they mean putting that version's `nvcc` on `PATH`.

### Choosing a GPU architecture
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

## Building the documentation

API documentation is generated from the Doxygen comments in the source
by Doxygen + Sphinx (via Breathe and Exhale) and published to GitHub
Pages on every push to `main`.

To build it locally you need the `doxygen` and `graphviz` binaries
(`mamba install -c conda-forge doxygen=1.18.0 graphviz`, the version CI
uses) plus the Python packages:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

[`environment.yml`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/environment.yml
[`Dockerfile`]: https://github.com/DiamondLightSource/fast-feedback-service/tree/main/Dockerfile
