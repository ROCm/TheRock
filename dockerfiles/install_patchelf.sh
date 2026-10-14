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
case "${PATCHELF_GIT_REF}" in
    d0f70eea5397606c486857e0a105e53ec123904a)
        PATCHELF_SHA256="0bda9fc5f4e233e655f591eff7c43ab9334f95ac04e39e26d6c1f85458daa3a7"
        ;;
    *)
        echo "Unsupported patchelf git ref: ${PATCHELF_GIT_REF}" >&2
        exit 1
        ;;
esac
SOURCE_ARCHIVE="patchelf-${PATCHELF_GIT_REF}.tar.gz"

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
    --output "${SOURCE_ARCHIVE}"

printf '%s  %s\n' "${PATCHELF_SHA256}" "${SOURCE_ARCHIVE}" | sha256sum --check --strict

mkdir -p src
tar -xzf "${SOURCE_ARCHIVE}" --strip-components=1 -C src

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
