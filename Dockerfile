# syntax=docker/dockerfile:1.27.1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
ARG CUDA_VERSION=13.3.1

# Source of the micromamba binary. The image is multi-arch, so each
# platform build takes its own binary from it.
FROM mambaorg/micromamba:2.10.0@sha256:e1476182cd40eb1f7c0cfae4adec9bfd78f3d7f05c61368306abfdc8939c1198 AS micromamba

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install -y --no-install-recommends git ca-certificates

# Install micromamba. It is statically linked, so it runs on the CUDA
# base unchanged.
COPY --from=micromamba /bin/micromamba /opt/bin/micromamba

# Setup paths we use
ENV PATH=/opt/ffs/bin:/opt/build_env/bin:/opt/bin:$PATH

# Copy environment files
COPY environment.yml runtime-environment.yml /opt/

# Create build environment from environment.yml
RUN micromamba create -y -f /opt/environment.yml -p /opt/build_env

# Create the runtime environment from runtime-environment.yml
RUN micromamba create -y -f /opt/runtime-environment.yml -p /opt/ffs

# Copy source
COPY . /opt/ffs_src
ENV CMAKE_GENERATOR=Ninja

# The source is copied without its git history, so the version cannot be
# resolved here and the build system passes one in. Both the python metadata
# and the compiled binaries take it from this.
ARG FFS_VERSION=0.0.0.dev0

# ccache comes from the build environment and cmake picks it up on its own. The
# default compiler check compares the compiler's mtime, which a rebuilt image
# changes, so compare contents instead or every run is a miss.
ENV CCACHE_DIR=/ccache
ENV CCACHE_COMPILERCHECK=content

# Build and install in one step. pip configures and builds through cmake, then
# installs the result. pip is the runtime environment's own, so the extension
# modules are built for its interpreter; cmake, ninja and the compilers resolve
# to the build environment, which has them where the runtime one does not.
#
# Each cached source is passed to cmake as an override, since FetchContent
# keeps its download record in the build tree and would re-clone it. git
# must trust the sources, which belong to the CI runner's user, not root.
#
# ccache keeps its statistics in the cache directory, which persists across
# runs, so they are zeroed first to report this build alone.
RUN --mount=type=cache,target=/ccache \
    --mount=type=cache,target=/deps \
    ccache --zero-stats \
    && deps_args="" \
    && for src in /deps/*-src; do \
        [ -d "$src" ] || continue; \
        name=$(basename "$src" -src | tr a-z A-Z); \
        deps_args="$deps_args -DFETCHCONTENT_SOURCE_DIR_${name}=$src"; \
    done \
    && GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*' \
    CCACHE_LOGFILE=/tmp/ccache.log \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FFS="${FFS_VERSION}" \
    CMAKE_ARGS="-DCUDA_ARCH=all-supported \
                -DHDF5_ROOT=/opt/ffs \
                -DCMAKE_INSTALL_RPATH=/opt/ffs/lib \
                -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
                -DUSE_REDUCED_PRECISION=OFF \
                -DFETCHCONTENT_BASE_DIR=/deps \
                ${deps_args}" \
    /opt/ffs/bin/pip3 install --no-build-isolation --root-user-action=ignore /opt/ffs_src \
    && ccache --show-stats --verbose \
    && { grep -i unsupported /tmp/ccache.log | sed 's/^\[[^]]*\] //' | sort | uniq -c; \
         pid=$(grep -m1 -i unsupported /tmp/ccache.log | sed -E 's/^\[[^ ]+ ([0-9]+)\].*/\1/'); \
         [ -n "$pid" ] && grep -m1 " $pid\] Command line" /tmp/ccache.log; true; }

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

# Run as an unprivileged UID with no passwd entry by default. It is
# numeric so the kubelet can verify runAsNonRoot; a deployment sets the
# real beamline UID over it. A later stage that installs packages needs
# USER root first.
USER 65532:65532

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
