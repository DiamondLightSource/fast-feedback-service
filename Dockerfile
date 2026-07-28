ARG CUDA_VERSION=13.0.2

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

# Placeholder for the version string to be passed from the build system.
ARG FFS_VERSION_DESCRIBE=

# Build the C++/CUDA backend
WORKDIR /opt/build
RUN cmake /opt/ffs_src \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/ffs \
    -DHDF5_ROOT=/opt/ffs \
    -DPython3_ROOT_DIR=/opt/ffs \
    -DCUDA_ARCH=all-supported \
    -DFFS_VERSION_DESCRIBE="${FFS_VERSION_DESCRIBE}" \
    -DCMAKE_INSTALL_RPATH=/opt/ffs/lib \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DUSE_REDUCED_PRECISION=OFF

RUN cmake --build .

RUN cmake --install .

# Install Python package. setuptools_scm has no git history to read
# here, so hand it the version cmake already resolved.
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS="$(cat /opt/build/FFS_VERSION)" \
    /opt/ffs/bin/pip3 install --root-user-action=ignore /opt/ffs_src

# Now copy this into an isolated runtime container
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

LABEL org.opencontainers.image.title="fast-feedback-service" \
      org.opencontainers.image.description="GPU-accelerated fast-feedback X-ray diffraction analysis service" \
      org.opencontainers.image.authors="Nicholas Devenish <nicholas.devenish@diamond.ac.uk>, Dimitrios Vlachos <dimitrios.vlachos@diamond.ac.uk>, James Beilsten-Edmands <james.beilsten-edmands@diamond.ac.uk>" \
      org.opencontainers.image.source="https://github.com/DiamondLightSource/fast-feedback-service" \
      org.opencontainers.image.licenses="BSD-3-Clause"

COPY --from=build /opt/ffs /opt/ffs

# Set environment variables for the service
ENV PATH=/opt/ffs/bin:$PATH
ENV SPOTFINDER=/opt/ffs/bin/spotfinder
ENV LD_LIBRARY_PATH=/opt/ffs/lib:$LD_LIBRARY_PATH
# ENV ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml

# # Start the service
CMD ["/opt/ffs/bin/zocalo.service", "-s", "GPUPerImageAnalysis"]
