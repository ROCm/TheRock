#!/bin/bash
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

PATCHELF_GIT_REF="${1:?usage: $0 <NixOS/patchelf git ref>}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"
SOURCE_URL="https://github.com/NixOS/patchelf/archive/${PATCHELF_GIT_REF}.tar.gz"
SHORT_GIT_REF="${PATCHELF_GIT_REF:0:12}"

# The PyPA manylinux base image installs a pipx patchelf at /usr/local/bin.
# Remove that known install before installing our pinned source build so PATH
# cannot silently resolve back to the base image copy.
PIPX_PATCHELF_VENV="/opt/_internal/pipx/venvs/patchelf"
PIPX_PATCHELF_BIN="${PIPX_PATCHELF_VENV}/bin/patchelf"
if [ "$(readlink /usr/local/bin/patchelf || true)" = "${PIPX_PATCHELF_BIN}" ]; then
    rm -f /usr/local/bin/patchelf
fi
rm -rf "${PIPX_PATCHELF_VENV}"

curl --silent --fail --show-error --location \
    "${SOURCE_URL}" \
    --output patchelf.tar.gz

mkdir -p src
tar -xzf patchelf.tar.gz --strip-components=1 -C src

cd src
BASE_VERSION="$(cat version)"
LOCAL_VERSION="${BASE_VERSION}+therock.${SHORT_GIT_REF}"
printf "%s\n" "${LOCAL_VERSION}" > version
./bootstrap.sh
./configure --prefix="${INSTALL_PREFIX}"
make -j"$(nproc)"
make install

hash -r
test "$(command -v patchelf)" = "${INSTALL_PREFIX}/bin/patchelf"
patchelf --version
