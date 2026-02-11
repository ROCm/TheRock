#!/bin/bash
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

echo "Downloading sccache ${SCCACHE_VERSION} for ${ARCH}..."
curl --silent --fail --show-error --location \
    "${SCCACHE_URL}" \
    --output sccache.tar.gz

tar xf sccache.tar.gz
cp "sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}/sccache" /usr/local/bin/
chmod +x /usr/local/bin/sccache

echo "sccache installed successfully:"
sccache --version
