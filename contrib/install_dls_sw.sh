#!/bin/bash
DIR=/dls_sw/apps/fast-feedback-service

# Handle cases where we've updated this script and want to refresh fully
if ! shasum -c $DIR/refresh.sha >/dev/null; then
    echo "Install script has changed or fresh install; resetting"
    (
        set -x
        rm -rf $DIR/ENV $DIR/build
        shasum $DIR/source/contrib/install_dls_sw.sh > $DIR/refresh.sha
    )
fi

module load mamba
if [[ ! -d $DIR/ENV ]]; then
    (
    set -x
    mamba create -c conda-forge -yp $DIR/ENV \
        python=3.11 \
        cmake \
        ninja \
        boost-cpp \
        benchmark \
        gtest \
        cmake \
        hdf5 \
        hdf5-external-filter-plugins \
        c-compiler \
        cxx-compiler \
        "gxx=12" \
        bitshuffle
    )
fi

mamba activate $DIR/ENV
module load cuda

set -euo pipefail
set -x

$DIR/ENV/bin/pip3 install -e $DIR/source
if [[ ! -d "$DIR/build" ]]; then
    mkdir -p $DIR/build
    cd $DIR/build
    cmake ../source -GNinja -DCMAKE_INSTALL_PREFIX=$DIR
fi
cd "$DIR/build"

cmake --build .
