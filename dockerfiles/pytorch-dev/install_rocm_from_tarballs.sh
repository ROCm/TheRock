#!/bin/bash
set -euo pipefail

# === Configuration ===
RELEASE_TAG="${RELEASE_TAG:-nightly-release}"
ROCM_VERSION_DATE="${ROCM_VERSION_DATE:-$(date +'%Y%m%d')}"
ROCM_VERSION_PREFIX="6.4.0rc"
AMDGPU_FAMILIES="${AMDGPU_FAMILIES:-gfx94X gfx110X gfx1201}"
OUTPUT_ARTIFACTS_DIR="${OUTPUT_ARTIFACTS_DIR:-/rocm-tarballs}"
INSTALL_PREFIX="/opt/rocm"
GITHUB_RELEASE_BASE_URL="https://github.com/ROCm/TheRock/releases/download"

echo "[INFO] Installing ROCm artifacts for: $AMDGPU_FAMILIES"
echo "[INFO] Using ROCm version date: $ROCM_VERSION_DATE"
echo "[INFO] Output directory: $OUTPUT_ARTIFACTS_DIR"

mkdir -p "$OUTPUT_ARTIFACTS_DIR"

# === Step 1: Download and Extract for each GPU target ===
for target in $AMDGPU_FAMILIES; do
  TARGET_DIR="${OUTPUT_ARTIFACTS_DIR}/${target}"
  echo "[INFO] Fetching tarball for target: $target"
  mkdir -p "$TARGET_DIR"

  TARBALL_NAME="therock-dist-linux-${target}-dgpu-${ROCM_VERSION_PREFIX}${ROCM_VERSION_DATE}.tar.gz"
  TARBALL_URL="${GITHUB_RELEASE_BASE_URL}/${RELEASE_TAG}/${TARBALL_NAME}"
  TARBALL_PATH="${TARGET_DIR}/${TARBALL_NAME}"

  echo "[INFO] Downloading from: $TARBALL_URL"
  wget -q --show-progress -O "$TARBALL_PATH" "$TARBALL_URL"

  echo "[INFO] Extracting $TARBALL_PATH to $INSTALL_PREFIX"
  sudo tar -xvzf "$TARBALL_PATH" -C "$INSTALL_PREFIX"
done

# === Step 2: Setup Environment Variables ===
ROCM_ENV_FILE="/etc/profile.d/rocm.sh"
echo "[INFO] Writing environment config to $ROCM_ENV_FILE"
sudo tee "$ROCM_ENV_FILE" > /dev/null <<EOF
export PATH=$INSTALL_PREFIX/bin:\$PATH
export ROCM_PATH=$INSTALL_PREFIX
EOF

# === Step 3: Configure Dynamic Linker ===
ROCM_LDCONF_FILE="/etc/ld.so.conf.d/rocm.conf"
echo "[INFO] Writing dynamic linker config to $ROCM_LDCONF_FILE"
sudo tee "$ROCM_LDCONF_FILE" > /dev/null <<EOF
$INSTALL_PREFIX/lib
$INSTALL_PREFIX/lib64
EOF

echo "[INFO] Running ldconfig..."
sudo ldconfig

# === Step 4: Validation ===
echo "[INFO] ROCm installed to $INSTALL_PREFIX"
which hipcc || echo "[WARN] hipcc not found"
which rocminfo || echo "[WARN] rocminfo not found"
