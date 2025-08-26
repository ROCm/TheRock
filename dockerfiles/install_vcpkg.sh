#!/bin/bash
# Copyright 2025 Advanced Micro Devices, Inc.

set -euo pipefail

VCPKG_HASH="$1"

git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
git -c advice.detachedHead=false checkout ${VCPKG_HASH}
./bootstrap-vcpkg.sh
