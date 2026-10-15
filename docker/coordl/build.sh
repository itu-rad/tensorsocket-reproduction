#!/usr/bin/env bash
# Build the CoorDL image (tsrepro:coordl):
#   1. obtain CoorDL's DALI wheel (Python 3.6, CUDA 10)
#   2. install it, with DS-Analyzer, into a Python 3.6 / PyTorch 1.8.1 image
#
# By default the wheel is downloaded: CoorDL's own build can no longer run
# unattended because NVIDIA withdrew the CUDA 10.0 patch 1 (nvJPEG) installer it
# fetches. The published wheel is the one CoorDL was run with for the paper,
# built from CoorDL @ bcde72d with patches/coordl.patch; the checksum
# pins it.
#
#   COORDL_WHEEL_URL          where to download the wheel from
#   COORDL_BUILD_FROM_SOURCE  1: compile the wheel with CoorDL's docker build
#                             instead (needs the CUDA 10.0 nvJPEG patch)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WHEEL="nvidia_dali-0.20.0.dev0-12345-cp36-cp36m-manylinux2014_x86_64.whl"
WHEEL_SHA256="594bce74fe9f157a267b6907c275b78b7793cc2d5bd6f7054d7d44253bdc69a0"
ZENODO_RECORD=23400672  # doi:10.5281/zenodo.23400672
COORDL_WHEEL_URL="${COORDL_WHEEL_URL:-https://zenodo.org/records/$ZENODO_RECORD/files/$WHEEL?download=1}"
COORDL_COMMIT="${COORDL_COMMIT:-bcde72d}"
WORK="${COORDL_WORKDIR:-$HERE/.build}"

wheel_ok() {
    [ -f "$HERE/$WHEEL" ] && echo "$WHEEL_SHA256  $HERE/$WHEEL" | sha256sum -c --status
}

build_from_source() {
    mkdir -p "$WORK"
    if [ ! -d "$WORK/CoorDL" ]; then
        git clone https://github.com/msr-fiddle/CoorDL.git "$WORK/CoorDL"
    fi
    git -C "$WORK/CoorDL" checkout "$COORDL_COMMIT"
    git -C "$WORK/CoorDL" submodule update --init --recursive
    # Plugin import fix and a still-available CUDA 10 build image (patches/coordl.patch).
    if git -C "$WORK/CoorDL" apply --check "$HERE/patches/coordl.patch" 2>/dev/null; then
        git -C "$WORK/CoorDL" apply "$HERE/patches/coordl.patch"
    fi
    if ! ls "$WORK"/CoorDL/build-docker-Release-36-10_x86_64/nvidia_dali-*.whl >/dev/null 2>&1; then
        (cd "$WORK/CoorDL/docker" && PYVER=3.6 CUDA_VERSION=10 CREATE_RUNNER=NO ./build.sh)
    fi
    cp "$WORK"/CoorDL/build-docker-Release-36-10_x86_64/nvidia_dali-*.whl "$HERE/"
}

if [ "${COORDL_BUILD_FROM_SOURCE:-0}" = 1 ]; then
    rm -f "$HERE"/nvidia_dali-*.whl
    build_from_source
elif ! wheel_ok; then
    rm -f "$HERE/$WHEEL"
    curl -fL --retry 3 -o "$HERE/$WHEEL.part" "$COORDL_WHEEL_URL"
    mv "$HERE/$WHEEL.part" "$HERE/$WHEEL"
    if ! wheel_ok; then
        echo "error: $WHEEL from $COORDL_WHEEL_URL does not match sha256 $WHEEL_SHA256" >&2
        exit 1
    fi
fi

docker build -t tsrepro:coordl "$HERE"
