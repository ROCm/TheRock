#!/bin/bash
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

mkdir -p "${BUILD_ROOT}"

download_python() {
  local version="$1"
  local url="https://www.python.org/ftp/python/${version}/Python-${version}.tgz"
  echo "[download] Python ${version}"
  curl --silent --fail --show-error --location "${url}" \
    --output "${BUILD_ROOT}/Python-${version}.tgz"
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
