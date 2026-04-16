#!/bin/bash
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Installs the NVIDIA CUDA Toolkit from the official NVIDIA RHEL8 repository.
# Downloads and checksum-verifies the NVIDIA GPG key and .repo file before use,
# then installs a pinned, versioned toolkit package via dnf.
#
# Usage:   install_cuda.sh <cuda_version>
# Example: install_cuda.sh 12.9
#          install_cuda.sh 13.2

set -euo pipefail

CUDA_VERSION="${1:-}"

if [[ -z "${CUDA_VERSION}" ]]; then
  echo "ERROR: CUDA version argument required." >&2
  echo "Usage: $0 <version>  (e.g. $0 12.9  or  $0 13.2)" >&2
  exit 1
fi

CUDA_REPO_BASE="https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64"

GPG_KEY_URL="${CUDA_REPO_BASE}/D42D0685.pub"
GPG_KEY_SHA256="27e46a2d43e125859fb8a62c3b75bf798aeb95fa6f7d9bf790c1167ed9a0b39c"

REPO_FILE_URL="${CUDA_REPO_BASE}/cuda-rhel8.repo"
REPO_FILE_SHA256="8d5bbb8dc62e0f0701a27355659248c3a11477e80a1b3c93a63ff116d705c06f"

declare -A CUDA_PACKAGE_SPECS=(
  ["12.9"]="cuda-toolkit-12-9-12.9.0-1"
  ["13.2"]="cuda-toolkit-13-2-13.2.0-1"
)

if [[ -z "${CUDA_PACKAGE_SPECS[${CUDA_VERSION}]+x}" ]]; then
  echo "ERROR: Unknown CUDA version '${CUDA_VERSION}'." >&2
  echo "Supported versions: ${!CUDA_PACKAGE_SPECS[*]}" >&2
  exit 1
fi

ARCH="$(uname -m)"
PACKAGE_SPEC="${CUDA_PACKAGE_SPECS[${CUDA_VERSION}]}.${ARCH}"

echo "Downloading NVIDIA GPG key"
curl --silent --fail --show-error --location \
  "${GPG_KEY_URL}" \
  --output nvidia.pub

echo "Verifying GPG key checksum"
echo "${GPG_KEY_SHA256}  nvidia.pub" | sha256sum --check --strict

rpm --import nvidia.pub

echo "Downloading CUDA repo file"
curl --silent --fail --show-error --location \
  "${REPO_FILE_URL}" \
  --output cuda-rhel8.repo

echo "Verifying repo file checksum"
echo "${REPO_FILE_SHA256}  cuda-rhel8.repo" | sha256sum --check --strict
cp cuda-rhel8.repo /etc/yum.repos.d/cuda-rhel8.repo

dnf config-manager --set-enabled powertools 2>/dev/null ||
  dnf config-manager --set-enabled crb 2>/dev/null ||
  true

echo "Installing ${PACKAGE_SPEC}"
dnf install -y "${PACKAGE_SPEC}"
dnf clean all
rm -rf /var/cache/dnf

echo "Verifying CUDA installation"
/usr/local/cuda/bin/nvcc --version

echo "=== CUDA Toolkit ${CUDA_VERSION} installed successfully ==="
