#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install sccache from official GitHub releases.
#
# Usage: ./install_sccache.sh <VERSION>
# Example: ./install_sccache.sh "0.13.0"

set -euo pipefail

SCCACHE_VERSION="$1"

ARCH="$(uname -m)"
if [ "${ARCH}" != "x86_64" ]; then
    echo "Unsupported architecture: ${ARCH}. Only x86_64 is supported."
    exit 1
fi
SCCACHE_ARCH="x86_64-unknown-linux-musl"

SCCACHE_TARBALL="sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}.tar.gz"
SCCACHE_URL="https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/${SCCACHE_TARBALL}"
case "${SCCACHE_VERSION}:${SCCACHE_ARCH}" in
    0.14.0:x86_64-unknown-linux-musl)
        SCCACHE_SHA256="8424b38cda4ecce616a1557d81328f3d7c96503a171eab79942fad618b42af44"
        ;;
    *)
        echo "Unsupported sccache version/architecture: ${SCCACHE_VERSION}/${SCCACHE_ARCH}" >&2
        exit 1
        ;;
esac

echo "Downloading sccache ${SCCACHE_VERSION} for ${ARCH}..."
curl --silent --fail --show-error --location \
    "${SCCACHE_URL}" \
    --output "${SCCACHE_TARBALL}"

printf '%s  %s\n' "${SCCACHE_SHA256}" "${SCCACHE_TARBALL}" | sha256sum --check --strict

tar xf "${SCCACHE_TARBALL}"
cp "sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}/sccache" /usr/local/bin/
chmod +x /usr/local/bin/sccache

echo "sccache installed successfully:"
sccache --version
