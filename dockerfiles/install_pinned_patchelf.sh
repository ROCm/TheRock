#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install the NixOS/patchelf git ref pinned by the manylinux Dockerfile.
# Honors INSTALL_PREFIX (passed through to install_patchelf.sh).

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
DOCKERFILE="${SCRIPT_DIR}/build_manylinux_x86_64.Dockerfile"

PATCHELF_GIT_REF="$(sed -n 's/^ENV PATCHELF_GIT_REF="\(.*\)"/\1/p' "${DOCKERFILE}")"
if [ -z "${PATCHELF_GIT_REF}" ]; then
    echo "error: could not extract PATCHELF_GIT_REF from ${DOCKERFILE}" >&2
    exit 1
fi

exec "${SCRIPT_DIR}/install_patchelf.sh" "${PATCHELF_GIT_REF}"
