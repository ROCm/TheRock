#!/bin/bash
set -xeuo pipefail

# -----------------------------------------------------------------------------
# install_rocm_from_tarballs.sh
#
# Download and install ROCm tarballs for specified AMDGPU targets from
# TheRock's nightly GitHub releases.
#
# Usage:
#   ./install_rocm_from_tarballs.sh "gfx942 gfx1100"
#
# Environment Variables (optional):
#   RELEASE_TAG, ROCM_VERSION_DATE, ROCM_VERSION_PREFIX, INSTALL_PREFIX,
#   OUTPUT_ARTIFACTS_DIR
#
# Requirements:
#   curl (auto-installed if missing),
#   jq (auto-installed if missing),
#   bash
# -----------------------------------------------------------------------------

for tool in curl jq; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[INFO] $tool not found. Installing..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update
      apt-get install -y "$tool"
    else
      echo "[ERROR] $tool installation not supported on this OS. Please install it manually."
      exit 1
    fi
  fi
done

# Configuration
RELEASE_TAG="${RELEASE_TAG:-nightly-release}"
ROCM_VERSION_DATE="${ROCM_VERSION_DATE:-$(date -d '3 days ago' +'%Y%m%d')}"

# Determine current working directory
WORKING_DIR="$(pwd)"
echo "[INFO] Running from directory: $WORKING_DIR"

# Default: version.json relative to working directory
: "${VERSION_JSON_PATH:=/therock/src/version.json}"

if [[ ! -f "$VERSION_JSON_PATH" ]]; then
  echo "[ERROR] version.json not found at $VERSION_JSON_PATH"
  exit 1
fi

ROCM_VERSION=$(jq -r '.["rocm-version"]' "$VERSION_JSON_PATH")
ROCM_VERSION_PREFIX="${ROCM_VERSION}rc"


INSTALL_PREFIX="${INSTALL_PREFIX:-/therock/build/dist/rocm}"
OUTPUT_ARTIFACTS_DIR="${OUTPUT_ARTIFACTS_DIR:-/rocm-tarballs}"
GITHUB_RELEASE_BASE_URL="https://github.com/ROCm/TheRock/releases/download"

# Parse AMDGPU targets from input
if [[ $# -ge 1 ]]; then
  AMDGPU_TARGETS="$1"
else
  AMDGPU_TARGETS="gfx942"
fi

mkdir -p "${OUTPUT_ARTIFACTS_DIR}"
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
  mkdir -p "${TARGET_DIR}"

  # Try primary format
  TARBALL_NAME="therock-dist-linux-${target}-${ROCM_VERSION_PREFIX}${ROCM_VERSION_DATE}.tar.gz"
  TARBALL_URL="${GITHUB_RELEASE_BASE_URL}/${RELEASE_TAG}/${TARBALL_NAME}"
  TARBALL_PATH="${TARGET_DIR}/${TARBALL_NAME}"

  echo "[INFO] Trying to download: $TARBALL_URL"
  if ! curl -sSL --fail -o "$TARBALL_PATH" "$TARBALL_URL"; then
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
    if ! curl -sSL --fail -o "$TARBALL_PATH" "$TARBALL_URL"; then
      echo "[ERROR] Could not download tarball for $target (fallback: $fallback)"
      exit 1
    fi
  fi

  mkdir -p "${INSTALL_PREFIX}"
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

# Step 3: Validate
echo "[INFO] ROCm installed to $INSTALL_PREFIX"
which hipcc || echo "[WARN] hipcc not found"
which rocminfo || echo "[WARN] rocminfo not found"
