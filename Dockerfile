ARG CUDA_VERSION=13.3.1

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates curl

# Install micromamba. Prefer install script so we don't have to determine platform here
RUN curl -sL micro.mamba.pm/install.sh | BIN_FOLDER=/opt/bin INIT_YES=no bash

# Setup paths we use
ENV PATH=/opt/ffs/bin:/opt/build_env/bin:/opt/bin:$PATH

# Copy environment files
COPY environment.yml runtime-environment.yml /opt/

# Create build environment from environment.yml
RUN micromamba create -y -f /opt/environment.yml -p /opt/build_env ninja

# Create the runtime environment from runtime-environment.yml
RUN micromamba create -y -f /opt/runtime-environment.yml -p /opt/ffs

# Copy source
COPY . /opt/ffs_src
ENV CMAKE_GENERATOR=Ninja

# The source is copied without its git history, so the version cannot be
# resolved here and the build system passes one in. Both the python metadata
# and the compiled binaries take it from this.
ARG FFS_VERSION=0.0.0.dev0

# Build and install in one step. pip configures and builds through cmake, then
# installs the result. pip is the runtime environment's own, so the extension
# modules are built for its interpreter; cmake, ninja and the compilers resolve
# to the build environment, which has them where the runtime one does not.
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS="${FFS_VERSION}" \
    CMAKE_ARGS="-DCUDA_ARCH=all-supported \
                -DHDF5_ROOT=/opt/ffs \
                -DCMAKE_INSTALL_RPATH=/opt/ffs/lib \
                -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
                -DUSE_REDUCED_PRECISION=OFF" \
    /opt/ffs/bin/pip3 install --no-build-isolation --root-user-action=ignore /opt/ffs_src

# The extension modules arrive with the wheel rather than with cmake, so
# prove they import before the runtime stage copies the prefix
RUN /opt/ffs/bin/python3 -c "import ffs.index, ffs.integrate, ffs.pipeline, ffs.ssx_index"

# Now copy this into an isolated runtime container. Both published
# images derive from this stage, so the build above runs once.
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime_base

LABEL org.opencontainers.image.authors="Nicholas Devenish <nicholas.devenish@diamond.ac.uk>, Dimitrios Vlachos <dimitrios.vlachos@diamond.ac.uk>, James Beilsten-Edmands <james.beilsten-edmands@diamond.ac.uk>" \
      org.opencontainers.image.source="https://github.com/DiamondLightSource/fast-feedback-service" \
      org.opencontainers.image.licenses="BSD-3-Clause"

COPY --from=build /opt/ffs /opt/ffs

# Set environment variables for the executables
ENV PATH=/opt/ffs/bin:$PATH
ENV SPOTFINDER=/opt/ffs/bin/spotfinder
ENV INDEXER=/opt/ffs/bin/baseline_indexer
ENV INTEGRATOR=/opt/ffs/bin/integrator
ENV LD_LIBRARY_PATH=/opt/ffs/lib:$LD_LIBRARY_PATH
# ENV ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml

# Batch image: run a pipeline over a single dataset, then exit. It
# carries every batch entrypoint rather than one each, since they differ
# only in which of the same three binaries they call. The caller selects
# one by overriding the command.
FROM runtime_base AS batch

LABEL org.opencontainers.image.title="fast-feedback-service-batch" \
      org.opencontainers.image.description="Batch GPU processing of a single dataset"

CMD ["/opt/ffs/bin/ffs_index_integrate"]

# Long-running service image. Kept last so that a bare `docker build`
# with no --target still produces the service image.
FROM runtime_base AS service

LABEL org.opencontainers.image.title="fast-feedback-service" \
      org.opencontainers.image.description="GPU-accelerated fast-feedback X-ray diffraction analysis service"

# # Start the service
CMD ["/opt/ffs/bin/zocalo.service", "-s", "GPUPerImageAnalysis"]
