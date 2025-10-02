ARG CUDA_VERSION=12.6.3

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

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

# Build the C++/CUDA backend
WORKDIR /opt/build
RUN cmake /opt/ffs_src -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ffs -DHDF5_ROOT=/opt/ffs
RUN cmake --build . --target spotfinder

RUN cmake --install . --component Runtime
RUN SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS=1.0 /opt/ffs/bin/pip3 install /opt/ffs_src

# Now copy this into an isolated runtime container
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

LABEL org.opencontainers.image.title="fast-feedback-service" \
      org.opencontainers.image.description="GPU-accelerated fast-feedback X-ray diffraction analysis service" \
      org.opencontainers.image.authors="Nicholas Devenish <nicholas.devenish@diamond.ac.uk>, Dimitrios Vlachos <dimitrios.vlachos@diamond.ac.uk>" \
      org.opencontainers.image.source="https://github.com/DiamondLightSource/fast-feedback-service" \
      org.opencontainers.image.licenses="BSD-3-Clause"

COPY --from=build /opt/ffs /opt/ffs

# Set environment variables for the service
ENV PATH=/opt/ffs/bin:$PATH
ENV SPOTFINDER=/opt/ffs/bin/spotfinder
# ENV ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml

# # Start the service
CMD ["/opt/ffs/bin/zocalo.service", "-s", "GPUPerImageAnalysis"]
