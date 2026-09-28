#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Installs Python interpreters with shared library support.
# These are required for embedding Python (e.g., in rocgdb).
#
# Usage: ./install_shared_pythons.sh <build_dir>
#   build_dir: Directory for downloading and building (caller handles cleanup)
#
# Installs to /opt/python-shared/cp3XX-cp3XX/ to match manylinux conventions.

set -euo pipefail

BUILD_ROOT="${1:?Build directory required}"
INSTALL_ROOT="/opt/python-shared"

# Python versions to build (major.minor.patch)
PYTHON_VERSIONS=(
  "3.10.16"
  "3.11.11"
  "3.12.9"
  "3.13.2"
  "3.14.0"
)
declare -Ar PYTHON_SHA256S=(
  ["3.10.16"]="f2e22ed965a93cfeb642378ed6e6cdbc127682664b24123679f3d013fafe9cd0"
  ["3.11.11"]="883bddee3c92fcb91cf9c09c5343196953cbb9ced826213545849693970868ed"
  ["3.12.9"]="45313e4c5f0e8acdec9580161d565cf5fea578e3eabf25df7cc6355bf4afa1ee"
  ["3.13.2"]="b8d79530e3b7c96a5cb2d40d431ddb512af4a563e863728d8713039aa50203f9"
  ["3.14.0"]="88d2da4eed42fa9a5f42ff58a8bc8988881bd6c547e297e46682c2687638a851"
)

mkdir -p "${BUILD_ROOT}"

download_python() {
  local version="$1"
  local url="https://www.python.org/ftp/python/${version}/Python-${version}.tgz"
  local tarball="${BUILD_ROOT}/Python-${version}.tgz"
  local expected_sha256="${PYTHON_SHA256S[${version}]:-}"
  if [[ -z "${expected_sha256}" ]]; then
    echo "[error] No SHA256 configured for Python ${version}" >&2
    return 1
  fi
  echo "[download] Python ${version}"
  curl --silent --fail --show-error --location "${url}" \
    --output "${tarball}"
  printf '%s  %s\n' "${expected_sha256}" "${tarball}" | sha256sum --check --strict
}

build_python() {
  local version="$1"
  local major_minor="${version%.*}"
  # Handle alpha/beta/rc versions: 3.14.0a4 -> 3.14
  major_minor="${major_minor%a*}"
  major_minor="${major_minor%b*}"
  major_minor="${major_minor%rc*}"
  local short_version="${major_minor//./}"
  local install_dir="${INSTALL_ROOT}/cp${short_version}-cp${short_version}"
  local src_dir="${BUILD_ROOT}/Python-${version}"
  local build_dir="${BUILD_ROOT}/build-${version}"

  echo "[build] Python ${version} -> ${install_dir}"

  # Extract
  tar xzf "${BUILD_ROOT}/Python-${version}.tgz" -C "${BUILD_ROOT}"

  # Configure and build out-of-tree
  mkdir -p "${build_dir}"
  cd "${build_dir}"

  "${src_dir}/configure" \
    --prefix="${install_dir}" \
    --enable-shared \
    LDFLAGS="-Wl,-rpath,${install_dir}/lib" \
    > configure.log 2>&1

  make -j"$(nproc)" > build.log 2>&1
  make install > install.log 2>&1

  # Verify the shared library exists
  if [[ ! -f "${install_dir}/lib/libpython${major_minor}.so.1.0" ]]; then
    echo "[error] Shared library not found for Python ${version}"
    cat configure.log build.log install.log
    return 1
  fi

  echo "[done] Python ${version}"
}

echo "=== Downloading Python sources ==="
for version in "${PYTHON_VERSIONS[@]}"; do
  download_python "${version}" &
done
wait

echo "=== Building Python interpreters ==="
for version in "${PYTHON_VERSIONS[@]}"; do
  build_python "${version}" &
done
wait

echo "=== Installed Python interpreters ==="
for dir in "${INSTALL_ROOT}"/cp*; do
  if [[ -d "${dir}" ]]; then
    "${dir}/bin/python3" --version
  fi
done
