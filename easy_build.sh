#!/bin/bash
set -e

source "$(dirname "$0")/setup.sh"

mkdir -p build
cd build
cmake ..
make