#!/usr/bin/env bash
set -euo pipefail

# Uses env from Dockerfile:
#   OMPI_VER (e.g., 5.0.8)
#   OMPI_PREFIX (e.g., /opt/openmpi-5.0.8)

builddir="/tmp/build-openmpi"
mkdir -p "$builddir"
cd "$builddir"

curl "https://download.open-mpi.org/release/open-mpi/v${OMPI_VER%.*}/openmpi-${OMPI_VER}.tar.gz" -o openmpi-${OMPI_VER}.tar.gz
tar xzf "openmpi-${OMPI_VER}.tar.gz"
cd "openmpi-${OMPI_VER}"

./configure --prefix="${OMPI_PREFIX}" --enable-wrapper-rpath
make -j"$(nproc)"
make install

# cleanup
cd /
rm -rf "$builddir"
