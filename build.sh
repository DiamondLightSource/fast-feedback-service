#!/usr/bin/env bash
# filepath: build.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PRODUCTION=false
PIXEL_32BIT=false
CLEAN=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build script for Fast Feedback Service to manage development and production builds.

OPTIONS:
    -p, --production       Build for production (single build directory)
    -3, --32bit            Use 32-bit pixel data (only with --production)
    -c, --clean            Clean build directories before building
    -j, --jobs N           Number of parallel jobs (default: $(nproc 2>/dev/null || echo 4))
    -h, --help             Show this help message

DEVELOPMENT BUILD (default):
    Creates both 'build' (16-bit) and 'build_32bit' (32-bit) directories for 16/32-bit pixel data.
    Runs cmake and builds both configurations for testing.

PRODUCTION BUILD:
    Creates only 'build' directory with specified configuration.
    Removes 'build_32bit' if it exists.
    Use --32bit flag to build with 32-bit pixel data support.

EXAMPLES:
    $0                              # Development build (both 16-bit and 32-bit)
    $0 --production                 # Production build with 16-bit pixels
    $0 --production --32bit         # Production build with 32-bit pixels
    $0 --clean                      # Clean and rebuild development builds
    $0 --production --clean --32bit # Clean production build with 32-bit

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--production)
            PRODUCTION=true
            shift
            ;;
        -3|--32bit)
            PIXEL_32BIT=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$PIXEL_32BIT" == "true" && "$PRODUCTION" == "false" ]]; then
    print_error "--32bit flag can only be used with --production"
    exit 1
fi

# Detect build system preference
BUILD_SYSTEM="Unix Makefiles"
BUILD_CMD="make"
if command -v ninja >/dev/null 2>&1; then
    BUILD_SYSTEM="Ninja"
    BUILD_CMD="ninja"
    print_status "Using Ninja build system"
else
    print_status "Using Make build system"
fi

# Function to initialize git submodules
init_submodules() {
    print_status "Checking git submodules..."
    if [[ -f .gitmodules ]]; then
        if [[ ! -f dx2/CMakeLists.txt ]]; then
            print_status "Initializing git submodules..."
            git submodule update --init --recursive
        else
            print_status "Git submodules already initialized"
        fi
    fi
}

# Function to configure and build a directory
build_directory() {
    local build_dir="$1"
    local cmake_args="$2"
    local description="$3"
    
    print_status "Building $description in $build_dir..."
    
    # Clean if requested
    if [[ "$CLEAN" == "true" && -d "$build_dir" ]]; then
        print_status "Cleaning $build_dir..."
        rm -rf "$build_dir"
    fi
    
    # Create build directory
    mkdir -p "$build_dir"
    
    # Configure with cmake
    print_status "Configuring $description..."
    (
        cd "$build_dir"
        cmake .. -G "$BUILD_SYSTEM" $cmake_args
    )
    
    # Build
    print_status "Compiling $description..."
    (
        cd "$build_dir"
        if [[ "$BUILD_CMD" == "ninja" ]]; then
            ninja
        else
            make -j"$JOBS"
        fi
    )

    # Install
    print_status "Installing $description..."
    (
        cd "$build_dir"
        if [[ "$BUILD_CMD" == "ninja" ]]; then
            ninja "install"
        else
            make "install" -j"$JOBS"
        fi
    )
    
    print_success "Successfully built $description"
}

# Main build logic
main() {
    print_status "Fast Feedback Service Build Script"
    print_status "=================================="
    
    # Check if we're in the right directory
    if [[ ! -f CMakeLists.txt ]]; then
        print_error "CMakeLists.txt not found. Please run this script from the project root."
        exit 1
    fi
    
    # Initialize submodules
    init_submodules
    
    if [[ "$PRODUCTION" == "true" ]]; then
        print_status "Production build mode"
        
        # Remove build_32bit if it exists
        if [[ -d build_32bit ]]; then
            print_status "Removing build_32bit directory for production build"
            rm -rf build_32bit
        fi
        
        # Build production version
        if [[ "$PIXEL_32BIT" == "true" ]]; then
            build_directory "build" "-DPIXEL_DATA_32BIT=ON -DCMAKE_BUILD_TYPE=Release" "production (32-bit)"
        else
            build_directory "build" "-DCMAKE_BUILD_TYPE=Release" "production (16-bit)"
        fi
    else
        print_status "Development build mode"
        
        # Build both 16-bit and 32-bit versions
        build_directory "build" "-DCMAKE_BUILD_TYPE=RelWithDebInfo" "development (16-bit)"
        build_directory "build_32bit" "-DPIXEL_DATA_32BIT=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo" "development (32-bit)"
    fi
    
    print_success "Build completed successfully!"
    
    # Show build artifacts
    print_status "Build artifacts:"
    if [[ -f build/bin/spotfinder ]]; then
        echo "  - build/bin/spotfinder"
    fi
    if [[ -f build_32bit/bin/spotfinder ]]; then
        echo "  - build_32bit/bin/spotfinder"
    fi
    if [[ -f build/bin/baseline_indexer ]]; then
        echo "  - build/bin/baseline_indexer"
    fi
    if [[ -f build_32bit/bin/baseline_indexer ]]; then
        echo "  - build_32bit/bin/baseline_indexer"
    fi
}

# Run main function
main "$@"