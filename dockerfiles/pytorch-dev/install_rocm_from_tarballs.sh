#!/bin/bash
set -euo pipefail

# Ensure wget is installed
if ! command -v wget >/dev/null 2>&1; then
  echo "[INFO] wget not found. Installing..."
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y wget
  else
    echo "[ERROR] wget installation not supported on this OS. Please install it manually."
    exit 1
  fi
fi

# Configuration
RELEASE_TAG="${RELEASE_TAG:-nightly-release}"
ROCM_VERSION_DATE="${ROCM_VERSION_DATE:-$(date -d '3 days ago' +'%Y%m%d')}"
ROCM_VERSION_PREFIX="6.4.0rc"
INSTALL_PREFIX="${INSTALL_PREFIX:-/therock/build/dist/rocm}"
OUTPUT_ARTIFACTS_DIR="${OUTPUT_ARTIFACTS_DIR:-/rocm-tarballs}"
GITHUB_RELEASE_BASE_URL="https://github.com/ROCm/TheRock/releases/download"

# Parse AMDGPU targets from input
if [[ $# -ge 1 ]]; then
  AMDGPU_TARGETS="$1"
else
  AMDGPU_TARGETS="gfx942 gfx1100 gfx1201"
fi

mkdir -p "$OUTPUT_ARTIFACTS_DIR"
echo "[INFO] Installing ROCm for targets: $AMDGPU_TARGETS"
echo "[INFO] Date: $ROCM_VERSION_DATE | Output Dir: $OUTPUT_ARTIFACTS_DIR"

# Fallback encoding map
fallback_target_name() {
  case "$1" in
    gfx942) echo "gfx94X-dcgpu" ;;
    gfx1100) echo "gfx110X-dgpu" ;;
    gfx1201) echo "gfx120X-dgpu" ;;
    *) echo "" ;;
  esac
}

# Step 1: Download and Extract
for target in $AMDGPU_TARGETS; do
  TARGET_DIR="${OUTPUT_ARTIFACTS_DIR}/${target}"
  mkdir -p "$TARGET_DIR"

  # Try primary format
  TARBALL_NAME="therock-dist-linux-${target}-${ROCM_VERSION_PREFIX}${ROCM_VERSION_DATE}.tar.gz"
  TARBALL_URL="${GITHUB_RELEASE_BASE_URL}/${RELEASE_TAG}/${TARBALL_NAME}"
  TARBALL_PATH="${TARGET_DIR}/${TARBALL_NAME}"

  echo "[INFO] Trying to download: $TARBALL_URL"
  if ! wget -q -O "$TARBALL_PATH" "$TARBALL_URL"; then
    echo "[WARN] Primary tarball not found for $target. Trying fallback encoding..."

    fallback=$(fallback_target_name "$target")
    if [[ -z "$fallback" ]]; then
      echo "[ERROR] No fallback rule for $target"
      exit 1
    fi

    TARBALL_NAME="therock-dist-linux-${fallback}-${ROCM_VERSION_PREFIX}${ROCM_VERSION_DATE}.tar.gz"
    TARBALL_URL="${GITHUB_RELEASE_BASE_URL}/${RELEASE_TAG}/${TARBALL_NAME}"
    TARBALL_PATH="${TARGET_DIR}/${TARBALL_NAME}"

    echo "[INFO] Trying fallback: $TARBALL_URL"
    if ! wget -q --show-progress -O "$TARBALL_PATH" "$TARBALL_URL"; then
      echo "[ERROR] Could not download tarball for $target (fallback: $fallback)"
      exit 1
    fi
  fi

  mkdir -p "$INSTALL_PREFIX"
  echo "[INFO] Extracting $TARBALL_PATH to $INSTALL_PREFIX"
  tar -xzf "$TARBALL_PATH" -C "$INSTALL_PREFIX"
done

# Step 2: Setup Environment Variables
ROCM_ENV_FILE="/etc/profile.d/rocm.sh"
echo "[INFO] Writing environment config to $ROCM_ENV_FILE"
tee "$ROCM_ENV_FILE" > /dev/null <<EOF
export PATH=$INSTALL_PREFIX/bin:\$PATH
export ROCM_PATH=$INSTALL_PREFIX
EOF

# Step 3: Dynamic Linker
ROCM_LDCONF_FILE="/etc/ld.so.conf.d/rocm.conf"
echo "[INFO] Writing dynamic linker config to $ROCM_LDCONF_FILE"
tee "$ROCM_LDCONF_FILE" > /dev/null <<EOF
$INSTALL_PREFIX/lib
$INSTALL_PREFIX/lib64
EOF

echo "[INFO] Running ldconfig..."
ldconfig

# Step 4: Validate
echo "[INFO] ROCm installed to $INSTALL_PREFIX"
which hipcc || echo "[WARN] hipcc not found"
which rocminfo || echo "[WARN] rocminfo not found"
