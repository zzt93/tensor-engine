#!/bin/bash
set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR=build

cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -G Ninja
cmake --build ${BUILD_DIR} -j$(nproc)