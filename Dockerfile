LABEL org.opencontainers.image.title="fast-feedback-service" \
      org.opencontainers.image.description="GPU-accelerated fast-feedback X-ray diffraction analysis service" \
      org.opencontainers.image.authors="Nicholas Devenish <nicholas.devenish@diamond.ac.uk>, Dimitrios Vlachos <dimitrios.vlachos@diamond.ac.uk>" \
      org.opencontainers.image.source="https://github.com/DiamondLightSource/fast-feedback-service" \
      org.opencontainers.image.licenses="BSD-3-Clause"

# Official NVIDIA CUDA image as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Remove need for interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Explicitly set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /service

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Clear apt cache

# Install Mambaforge
ENV MAMBA_ROOT_PREFIX=/opt/mambaforge
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
ARG MAMBA_VERSION=25.3.0-3
RUN wget --no-verbose --no-cache https://github.com/conda-forge/miniforge/releases/download/${MAMBA_VERSION}/Miniforge3-${MAMBA_VERSION}-Linux-x86_64.sh -O mambaforge.sh && \
    bash mambaforge.sh -b -p $MAMBA_ROOT_PREFIX && \
    rm mambaforge.sh
# Remove installer

# Create base environment (Python + runtime)
RUN mamba create -y -c conda-forge -p /opt/env \
    python=3.11 \
    pip \
    zocalo \
    workflows \
    pydantic \
    rich && \
    mamba clean -afy

# Create build environment (compilers + C++ deps)
RUN mamba install -y -c conda-forge -p /opt/env \
    gcc_linux-64=11 \
    gxx_linux-64=11 \
    boost-cpp \
    benchmark \
    gtest \
    cmake \
    hdf5 \
    hdf5-external-filter-plugins \
    bitshuffle \
    gemmi && \
    mamba clean -afy

# Add conda environment to PATH
ENV PATH=/opt/env/bin:$PATH
ENV CONDA_PREFIX=/opt/env

# Copy source
COPY . .

# Set environment variables for the build
ENV CC=/opt/env/bin/x86_64-conda-linux-gnu-cc
ENV CXX=/opt/env/bin/x86_64-conda-linux-gnu-c++
ENV CUDAHOSTCXX=/opt/env/bin/x86_64-conda-linux-gnu-g++

# Build the C++/CUDA backend
RUN mkdir -p docker_build && \
    cd docker_build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Set environment variables for the service
ENV SPOTFINDER=/service/docker_build/spotfinder
ENV ZOCALO_CONFIG=/dls_sw/apps/zocalo/live/configuration.yaml

# Start the service
CMD ["zocalo.service", "-s", "GPUPerImageAnalysis"]