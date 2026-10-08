#!/usr/bin/env bash
# filepath: module/deploy.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status()  { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
VERSION=dev

INSTALL_ROOT=/dls_sw/apps/fast-feedback-service
MODULE_ROOT=/dls_sw/apps/Modules/modulefiles/fast-feedback-service
MODULE_NAME=
PREFIX=
BUILD_ENV=/tmp/ffs-build-env
# The container base image limits CUDA version updates, so it is the source.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_MODULE=cuda/$(sed -n 's/^ARG CUDA_VERSION=//p' "$SRC/Dockerfile")
[[ "$CUDA_MODULE" != "cuda/" ]] || { echo "Could not read ARG CUDA_VERSION from $SRC/Dockerfile" >&2; exit 1; }
BUILD_DIR=build_module
RECREATE=false
INCREMENTAL=false
INSTALL_MODULEFILE=false
INSTALL_LATEST=false
JOBS=

# Function to show usage
show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build and install the fast feedback service into a module prefix.

OPTIONS:
    -v, --version NAME     Release name; names the install directory and
                           the module (default: $VERSION)
    -p, --prefix PATH      Install prefix, overriding
                           <install-root>/<version>
        --install-root DIR Base of the install tree (default: $INSTALL_ROOT)
        --module-root DIR  Where modulefiles live (default: $MODULE_ROOT)
        --module-name NAME Module name, overriding <version>
    -b, --build-env PATH   Build environment prefix (default: $BUILD_ENV)
    -c, --cuda MODULE      CUDA module to build against (default: $CUDA_MODULE)
    -j, --jobs N           Cap parallel build jobs
    -r, --recreate         Delete and recreate both conda environments
    -i, --incremental      Reuse the build directory instead of starting
                           clean. Faster while iterating, but a cached
                           path from an earlier prefix will be believed.
    -m, --modulefile       Also install the modulefile
    -l, --latest           Point the latest symlink at this module.
                           Requires --modulefile.
    -h, --help             Show this help

The modulefile is only written with --modulefile, since that makes the
build visible to everyone on the machine. The latest symlink is what an
unversioned recipe resolves to, so moving it is a second deliberate step.

EXAMPLES:
    $0                                       # build .../ffs/dev, no module
    $0 --modulefile                          # ... and publish it as dev
    $0 --version 1.0.0 --modulefile          # a release
    $0 --version 1.0.0 --modulefile --latest # ... that recipes pick up
    $0 --prefix /scratch/ffs --module-root /scratch/modules --modulefile
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)    VERSION="$2"; shift 2 ;;
        -p|--prefix)     PREFIX="$2"; shift 2 ;;
        --install-root)  INSTALL_ROOT="$2"; shift 2 ;;
        --module-root)   MODULE_ROOT="$2"; shift 2 ;;
        --module-name)   MODULE_NAME="$2"; shift 2 ;;
        -b|--build-env)  BUILD_ENV="$2"; shift 2 ;;
        -c|--cuda)       CUDA_MODULE="$2"; shift 2 ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -r|--recreate)   RECREATE=true; shift ;;
        -i|--incremental) INCREMENTAL=true; shift ;;
        -m|--modulefile) INSTALL_MODULEFILE=true; shift ;;
        -l|--latest)     INSTALL_LATEST=true; shift ;;
        -h|--help)       show_help; exit 0 ;;
        *) print_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Derive the paths the version implies
: "${PREFIX:=$INSTALL_ROOT/$VERSION}"
: "${MODULE_NAME:=$VERSION}"

# Validate arguments
[[ -f "$SRC/CMakeLists.txt" ]] || { print_error "Not a source tree: $SRC"; exit 1; }
command -v mamba >/dev/null || { print_error "mamba not found"; exit 1; }
if [[ "$INSTALL_LATEST" == "true" ]]; then
    [[ "$INSTALL_MODULEFILE" == "true" ]] \
        || { print_error "--latest needs --modulefile"; exit 1; }
    [[ "$MODULE_NAME" != "latest" ]] \
        || { print_error "The module is already named latest"; exit 1; }
fi

# Show what will be built and where
print_status "Source:      $SRC"
print_status "Version:     $VERSION"
print_status "Prefix:      $PREFIX"
print_status "Build env:   $BUILD_ENV"
print_status "CUDA module: $CUDA_MODULE"
if [[ "$INSTALL_MODULEFILE" == "true" ]]; then
    print_status "Module:      $MODULE_ROOT/$MODULE_NAME"
else
    print_status "Module:      not published (pass --modulefile)"
fi
if [[ "$INSTALL_LATEST" == "true" ]]; then
    print_status "Latest:      $MODULE_ROOT/latest -> $MODULE_NAME"
fi

# Create the build and runtime environments
if [[ "$RECREATE" == "true" ]]; then
    print_warning "Removing existing environments"
    rm -rf "$BUILD_ENV" "$PREFIX"
fi

if [[ -d "$BUILD_ENV" ]]; then
    print_status "Build environment already present, reusing it"
else
    print_status "Creating build environment"
    mamba create -y -f "$SRC/environment.yml" -p "$BUILD_ENV" ninja
fi

if [[ -d "$PREFIX" ]]; then
    print_status "Runtime environment already present, reusing it"
else
    print_status "Creating runtime environment"
    mamba create -y -f "$SRC/runtime-environment.yml" -p "$PREFIX"
fi

if [[ ! -x "$BUILD_ENV/bin/patchelf" ]]; then
    print_status "Adding patchelf to the build environment"
    mamba install -y -p "$BUILD_ENV" patchelf
fi

# Configure and build
module load "$CUDA_MODULE"

for tool in cmake ninja patchelf; do
    [[ -x "$BUILD_ENV/bin/$tool" ]] || { print_error "No $tool in $BUILD_ENV"; exit 1; }
done

if [[ "$INCREMENTAL" == "true" ]]; then
    print_warning "Reusing the existing build directory"
else
    rm -rf "${SRC:?}/${BUILD_DIR:?}"
fi

# pip drives the build, so what cmake needs is passed through CMAKE_ARGS. nvcc
# defaults its host compiler to the g++ on PATH and nothing activates the build
# environment, so the compilers are named explicitly. The pip is the prefix's
# own, so the extension modules are built for the interpreter that will import
# them, while cmake, ninja and the compilers come from the build environment.
print_status "Building and installing"
[[ -n "$JOBS" ]] && export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
PATH="$BUILD_ENV/bin:$PATH" \
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$BUILD_ENV \
            -DCMAKE_C_COMPILER=$BUILD_ENV/bin/cc \
            -DCMAKE_CXX_COMPILER=$BUILD_ENV/bin/c++ \
            -DCMAKE_CUDA_HOST_COMPILER=$BUILD_ENV/bin/c++ \
            -DHDF5_ROOT=$PREFIX \
            -DPython_ROOT_DIR=$PREFIX \
            -DCUDA_ARCH=all-supported \
            -DCMAKE_INSTALL_RPATH=$PREFIX/lib;$PREFIX/lib64 \
            -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
            -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF \
            -DUSE_REDUCED_PRECISION=OFF" \
    "$PREFIX/bin/pip" install --no-build-isolation --no-deps \
        -C build-dir="$SRC/$BUILD_DIR" "$SRC"

# Drop the build environment from the RPATHs; conda's gcc specs file adds it
rpath="$PREFIX/lib:$PREFIX/lib64"
patch_rpaths() {
    local file
    for file in "$@"; do
        [[ -f "$file" ]] || continue
        case "$(file -b "$file" 2>/dev/null)" in
            ELF*) "$BUILD_ENV/bin/patchelf" --set-rpath "$rpath" "$file" ;;
        esac
    done
}

# pip installs rather than cmake, so there is no install manifest to walk.
# The set is named explicitly: everything else in the prefix belongs to conda
# and its RPATHs are not ours to rewrite.
print_status "Rewriting RPATHs"
platlib="$("$PREFIX/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')"
shopt -s nullglob
installed=("$PREFIX"/bin/spotfinder "$PREFIX"/bin/spotfinder32 \
           "$PREFIX"/bin/baseline_indexer "$PREFIX"/bin/integrator \
           "$platlib"/ffs/*.so "$platlib"/ffbidx/*.so "$platlib"/lib64/*.so*)
shopt -u nullglob
if [[ ${#installed[@]} -lt 4 ]]; then
    print_error "The install produced fewer artifacts than expected in $PREFIX."
    print_error "Found: ${installed[*]:-nothing}"
    exit 1
fi
patch_rpaths "${installed[@]}"

ffs_version="$("$PREFIX/bin/python" -c 'import ffs; print(ffs.__version__)')"
print_status "Installed version $ffs_version"

# Verify the install
print_status "Verifying the install"
failed=false

check_elf() {
    local path="$1" name="${1##*/}" missing
    if missing=$(ldd "$path" 2>/dev/null | grep "not found"); then
        print_error "$name has unresolved libraries:"
        echo "$missing"
        failed=true
    fi
    if readelf -d "$path" 2>/dev/null | grep -qE 'R(UN)?PATH.*(/tmp/|/home/)'; then
        print_error "$name has a build-time path in its RPATH:"
        readelf -d "$path" | grep -E 'R(UN)?PATH'
        failed=true
    fi
}

for binary in spotfinder spotfinder32 baseline_indexer integrator; do
    path="$PREFIX/bin/$binary"
    if [[ ! -x "$path" ]]; then
        print_error "$binary was not installed"
        failed=true
        continue
    fi
    check_elf "$path"
done

for extension in "${extensions[@]}"; do
    check_elf "$extension"
done

for script in ffs_spotfind_index_integrate ffs_index_integrate ssx_index; do
    [[ -x "$PREFIX/bin/$script" ]] || { print_error "$script missing"; failed=true; }
done

# ldd cannot see a missing Python extension, so import them
modules="ffs.index ffs.pipeline ffs.spotfind_index_integrate ffs.index_integrate ffs.ssx_index"
if ! "$PREFIX/bin/python" -c "import ${modules// /, }" >/dev/null 2>&1; then
    print_error "The deployed package cannot import its own modules:"
    "$PREFIX/bin/python" -c "import ${modules// /, }" 2>&1 | tail -3
    failed=true
fi

archs=$(cuobjdump --list-elf "$PREFIX/bin/spotfinder" 2>/dev/null \
        | grep -oE 'sm_[0-9]+' | sort -uV | tr '\n' ' ')
print_status "Architectures: ${archs:-none found}"
[[ -n "$archs" ]] || { print_error "No cubins in the spotfinder"; failed=true; }

if [[ "$failed" == "true" ]]; then
    print_error "Verification failed; not installing a modulefile"
    exit 1
fi
print_success "Install verified"

# Install the modulefile
if [[ "$INSTALL_MODULEFILE" == "true" ]]; then
    target="$MODULE_ROOT/$MODULE_NAME"
    [[ -e "$target" ]] && print_warning "Replacing existing $target"

    staged=$(mktemp)
    trap 'rm -f "$staged"' EXIT
    sed -e "s#@PREFIX@#$PREFIX#g" \
        -e "s#@CUDA_MODULE@#$CUDA_MODULE#g" \
        -e "s#@FFS_VERSION@#$ffs_version#g" \
        "$SRC/module/modulefile" > "$staged"

    # Refuse to publish a module still carrying a placeholder
    if grep -q '@[A-Z_]\+@' "$staged"; then
        print_error "Unsubstituted placeholders remain in the modulefile:"
        grep -n '@[A-Z_]\+@' "$staged"
        exit 1
    fi

    mkdir -p "$MODULE_ROOT"
    cp "$staged" "$target"
    chmod 664 "$target"
    print_success "Published fast-feedback-service/$MODULE_NAME -> $PREFIX"

    if [[ "$INSTALL_LATEST" == "true" ]]; then
        # A relative target keeps the link valid if the module root moves.
        # -n stops an existing link being followed into its own target.
        ln -sfn "$MODULE_NAME" "$MODULE_ROOT/latest"
        print_success "fast-feedback-service/latest -> $MODULE_NAME"
    fi
else
    print_status "Skipping the modulefile. Rerun with --modulefile to publish it."
fi

print_success "Done"
