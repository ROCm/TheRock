#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install the Rust toolchain via rustup.
#
# Usage: ./install_rust.sh <TOOLCHAIN_VERSION>
# Example: ./install_rust.sh "1.88.0"
#          ./install_rust.sh "stable"
#
# Installs rustup/cargo to a shared location (RUSTUP_HOME/CARGO_HOME) and
# symlinks the toolchain binaries into /usr/local/bin so they are on PATH
# for all users.

set -euo pipefail

RUST_VERSION="${1:-stable}"
RUSTUP_INIT_VERSION="1.29.0"
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)
        RUSTUP_TARGET="x86_64-unknown-linux-gnu"
        RUSTUP_SHA256="4acc9acc76d5079515b46346a485974457b5a79893cfb01112423c89aeb5aa10"
        ;;
    aarch64)
        RUSTUP_TARGET="aarch64-unknown-linux-gnu"
        RUSTUP_SHA256="9732d6c5e2a098d3521fca8145d826ae0aaa067ef2385ead08e6feac88fa5792"
        ;;
    *)
        echo "Unsupported rustup-init architecture: ${ARCH}" >&2
        exit 1
        ;;
esac
RUSTUP_BOOTSTRAP="rustup-init"

export RUSTUP_HOME="/usr/local/rustup"
export CARGO_HOME="/usr/local/cargo"

echo "Installing Rust toolchain '${RUST_VERSION}' via rustup-init ${RUSTUP_INIT_VERSION}..."
curl --silent --fail --show-error --location \
    "https://static.rust-lang.org/rustup/archive/${RUSTUP_INIT_VERSION}/${RUSTUP_TARGET}/rustup-init" \
    --output "${RUSTUP_BOOTSTRAP}"

printf '%s  %s\n' "${RUSTUP_SHA256}" "${RUSTUP_BOOTSTRAP}" | sha256sum --check --strict

chmod +x "${RUSTUP_BOOTSTRAP}"
"./${RUSTUP_BOOTSTRAP}" -y \
    --no-modify-path \
    --profile minimal \
    --default-toolchain "${RUST_VERSION}"

rm -f "${RUSTUP_BOOTSTRAP}"

# Make the toolchain available system-wide.
chmod -R a+rwX "${RUSTUP_HOME}" "${CARGO_HOME}"
for bin in "${CARGO_HOME}/bin/"*; do
    ln -sf "${bin}" "/usr/local/bin/$(basename "${bin}")"
done

echo "rust installed successfully:"
rustc --version
cargo --version
