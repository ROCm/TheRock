#!/bin/bash
set -euo pipefail

# Path to the directory containing ROCm tarballs
# TODO find the correct path for the artifact tarballs
TARBALL_DIR="${1:-/rocm-tarballs}"
INSTALL_PREFIX="/opt/rocm"

echo "Installing ROCm from tarballs in $TARBALL_DIR"

# 1. Create installation directory
sudo mkdir -p "$INSTALL_PREFIX"

# 2. Extract all tarballs to /opt/rocm
for tarball in "$TARBALL_DIR"/rocm-*.tar.gz; do
  echo "Extracting: $tarball"
  sudo tar -xvzf "$tarball" -C "$INSTALL_PREFIX"
done

# 3. Set up environment variables
ROCM_ENV_FILE="/etc/profile.d/rocm.sh"
echo "Writing environment variables to $ROCM_ENV_FILE"

sudo tee "$ROCM_ENV_FILE" > /dev/null <<EOF
export PATH=$INSTALL_PREFIX/bin:\$PATH
export LD_LIBRARY_PATH=$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib64:\$LD_LIBRARY_PATH
export ROCM_PATH=$INSTALL_PREFIX
EOF

# 4. Configure dynamic linker
ROCM_LDCONF_FILE="/etc/ld.so.conf.d/rocm.conf"
echo "Writing dynamic linker config to $ROCM_LDCONF_FILE"

sudo tee "$ROCM_LDCONF_FILE" > /dev/null <<EOF
$INSTALL_PREFIX/lib
$INSTALL_PREFIX/lib64
EOF

echo "Running ldconfig"
sudo ldconfig

# 5. Confirm installation
echo "ROCm installed to $INSTALL_PREFIX"
echo "ROCm tools:"
which hipcc || echo "hipcc not found (ROCm toolchain missing?)"
which rocminfo || echo "rocminfo not found (might not be in your PATH yet)"
