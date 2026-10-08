#!/usr/bin/env bash
# Build the project for development.
#
# pip owns the build. It configures cmake, compiles, and installs the python
# package into the active environment, which is what makes `import ffs` work.
# Once that has happened a plain `ninja -C build` is enough for a C++ change,
# and that is what this script does on every subsequent run.
#
#   ./build.sh              compile; install first if the environment needs it
#   ./build.sh --install    force the install, refreshing the environment
#
# A compile updates the build directory only. If you changed C++ that goes into
# `ffs.index` or `ffs.integrate`, use --install, because site-packages is the
# only place python can read an extension module from.
#   ./build.sh --clean      discard the build directory and start over
#   ./build.sh -j N         limit parallelism
#
# The programs are run straight out of the build directory, as
# ./build/bin/spotfinder, and are always what the last compile produced. The
# copies pip puts in the environment are for the deployments.

set -euo pipefail

# The container base image limits CUDA version updates, so it is the source.
_cuda_version=$(sed -n 's/^ARG CUDA_VERSION=//p' "$(dirname "$0")/Dockerfile")
[[ -n "$_cuda_version" ]] || { echo "Could not read ARG CUDA_VERSION from the Dockerfile" >&2; exit 1; }
CUDA_MODULE=${CUDA_MODULE:-cuda/$_cuda_version}

CLEAN=false
INSTALL=false
JOBS=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
fail()    { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--install) INSTALL=true; shift ;;
        -c|--clean)   CLEAN=true; shift ;;
        -j|--jobs)    JOBS="$2"; shift 2 ;;
        --cuda)       CUDA_MODULE="$2"; shift 2 ;;
        -h|--help)    sed -n '2,17p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *)            fail "Unknown option: $1" ;;
    esac
done

cd "$(dirname "$0")"

[[ -f CMakeLists.txt ]] || fail "Run this from the repository, not elsewhere"
[[ -n "${CONDA_PREFIX:-}" ]] || fail "No environment is active. Activate one first, e.g. mamba activate ENV/"

status "Environment: $CONDA_PREFIX"

if [[ ! -f dx2/CMakeLists.txt ]]; then
    status "Initialising git submodules"
    git submodule update --init --recursive
fi

dx2_at=$(git -C dx2 rev-parse --short HEAD 2>/dev/null)
dx2_want=$(git rev-parse --short "HEAD:dx2" 2>/dev/null)
if [[ -n "$dx2_at" && -n "$dx2_want" && "$dx2_at" != "$dx2_want" ]]; then
    warn "dx2 is at $dx2_at but this branch records $dx2_want."
    warn "Compile errors against dx2 usually mean this. To match:"
    warn "    git submodule update --init --recursive"
fi

command -v nvcc >/dev/null || fail "No nvcc on PATH. Load a CUDA module: module load $CUDA_MODULE"

cuda_found=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
cuda_wanted=${CUDA_MODULE#*/}
status "CUDA $cuda_found from $(command -v nvcc)"
if [[ "$cuda_found" != "${cuda_wanted%.*}" ]]; then
    warn "Both deployments build against $CUDA_MODULE, so this does not match them."
    warn "Binaries built here also need this same CUDA on the library path to run."
    warn "To match: module load $CUDA_MODULE"
else
    status "Matches what the deployments build against ($CUDA_MODULE)"
fi

# Deliberately not loading a CUDA module here. Building against one this shell
# does not have would produce binaries that cannot find their own runtime
# library, so the shell stays authoritative and this only checks and reports.
# nvcc also refuses a host compiler newer than it supports, which is the usual
# way this goes wrong after a fresh environment.
check_nvcc_works() {
    local probe="${TMPDIR:-/tmp}/ffs-cuda-probe-$$"
    printf '__global__ void k(){}\nint main(){return 0;}\n' > "$probe.cu"
    if ! nvcc -o "$probe.out" "$probe.cu" > "$probe.log" 2>&1; then
        warn "nvcc cannot compile with the compiler in this environment:"
        sed 's/^/    /' "$probe.log" | head -3
        rm -f "$probe.cu" "$probe.out" "$probe.log"
        fail "Load a CUDA that supports it, e.g. module load $CUDA_MODULE, then retry"
    fi
    rm -f "$probe.cu" "$probe.out" "$probe.log"
}

if [[ "$CLEAN" == "true" ]]; then
    status "Removing the build directory"
    rm -rf build
fi

# The build directory is pip's: it configures cmake there. Without it, or
# without the package installed here, there is nothing for ninja to build.
if [[ ! -f build/CMakeCache.txt ]] \
   || ! python -c "import ffs, pathlib, sys; sys.exit(0 if pathlib.Path(ffs.__file__).resolve().parent == pathlib.Path('src/ffs').resolve() else 1)" 2>/dev/null; then
    INSTALL=true
fi

[[ -n "$JOBS" ]] && export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"

if [[ "$INSTALL" == "true" ]]; then
    check_nvcc_works
    status "Installing (configures, compiles and installs into the environment)"
    # Dependencies are resolved here, unlike the deployments: environment.yml
    # carries what the C++ build needs, not the package's python requirements.
    python -m pip install --no-build-isolation -e .
else
    status "Compiling"
    ninja -C build ${JOBS:+-j "$JOBS"}
fi

success "Build complete"
