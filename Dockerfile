LABEL org.opencontainers.image.title="fast-feedback-service" \
      org.opencontainers.image.description="GPU-accelerated fast-feedback X-ray diffraction analysis service" \
      org.opencontainers.image.authors="Nicholas Devenish <nicholas.devenish@diamond.ac.uk>, Dimitrios Vlachos <dimitrios.vlachos@diamond.ac.uk>" \
      org.opencontainers.image.source="https://github.com/DiamondLightSource/fast-feedback-service" \
      org.opencontainers.image.licenses="BSD-3-Clause"

ARG CUDA_VERSION=12.6.3

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

# Explicitly set CUDA environment variables
#ENV CUDA_HOME=/usr/local/cuda
#ENV PATH=${CUDA_HOME}/bin:/opt/bin:${PATH}
#ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates curl

# Install micromamba. Prefer install script so we don't have to determine platform here
RUN curl -sL micro.mamba.pm/install.sh | BIN_FOLDER=/opt/bin INIT_YES=no bash

# Setup paths we use
ENV PATH=/opt/env/bin:/opt/bin:$PATH


# Create build environment (compilers + C++ deps)
RUN micromamba create -y -c conda-forge -p /opt/env \
    gcc=11 \
    gxx=11 \
    boost-cpp \
    benchmark \
    gtest \
    cmake \
    hdf5 \
    hdf5-external-filter-plugins \
    bitshuffle \
    ninja \
    gemmi

# Copy source
COPY . /opt/ffs_src

# Build the C++/CUDA backend
WORKDIR /opt/build
RUN cmake /opt/ffs_src -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ffs
RUN cmake --build .

# Create a python environment to install into
RUN micromamba create -y -c conda-forge -p /opt/ffs \
    python \
    pip \
    hdf5 \
    hdf5-external-filter-plugins \
    gemmi \
    zocalo \
    workflows \
    pydantic \
    rich

RUN cmake --install --component Runtime .
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS=1.0 /opt/ffs/bin/pip3 install /opt/ffs_src

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
COPY --from=build /opt/ffs /opt/ffs

# Set environment variables for the service
# ENV SPOTFINDER=/service/docker_build/spotfinder
# ENV ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml

# # Start the service
CMD ["/opt/ffs/bin/zocalo.service", "-s", "GPUPerImageAnalysis"]
