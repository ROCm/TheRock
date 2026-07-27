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

export RUSTUP_HOME="/usr/local/rustup"
export CARGO_HOME="/usr/local/cargo"

echo "Installing Rust toolchain '${RUST_VERSION}' via rustup..."
curl --silent --fail --show-error --location \
    "https://sh.rustup.rs" \
    --output rustup-init.sh

sh rustup-init.sh -y \
    --no-modify-path \
    --profile minimal \
    --default-toolchain "${RUST_VERSION}"

rm -f rustup-init.sh

# Make the toolchain available system-wide.
chmod -R a+rwX "${RUSTUP_HOME}" "${CARGO_HOME}"
for bin in "${CARGO_HOME}/bin/"*; do
    ln -sf "${bin}" "/usr/local/bin/$(basename "${bin}")"
done

echo "rust installed successfully:"
rustc --version
cargo --version
