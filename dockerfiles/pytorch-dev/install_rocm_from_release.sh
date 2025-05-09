#!/bin/bash
set -xeuo pipefail

# -----------------------------------------------------------------------------
# install_rocm_from_release.sh
#
# Download and install ROCm tarballs for specified AMDGPU targets from
# TheRock's nightly GitHub releases (or a forked repository).
#
# Usage:
#   INSTALL_PREFIX=/pathToInstall RELEASE_VERSION=6.4.0rc20250424 ./install_rocm_from_release.sh "gfx942 gfx1100"
#
# Environment Variables (optional):
#   RELEASE_VERSION       - Full version string like 6.4.0rc20250424 (required if no version.json)
#   ROCM_VERSION          - Base ROCm version like 6.4.0 (optional, auto-extracted from RELEASE_VERSION)
#   RELEASE_TAG           - GitHub release tag to pull from (default: nightly-tarball)
#   ROCM_VERSION_DATE     - Build date (default: 1 days ago)
#   INSTALL_PREFIX        - Installation path (default: /therock/build/dist/rocm)
#   OUTPUT_ARTIFACTS_DIR  - Directory to store downloaded tarballs (default: /rocm-tarballs)
#   GITHUB_REPO           - GitHub repository name (default: ROCm/TheRock)
#
# Requirements:
#   curl (auto-installed if missing)
#   jq (auto-installed if missing)
#   bash
#
# Notes:
#   - Setting GITHUB_REPO allows installing from forks or custom repositories.
#   - RELEASE_VERSION controls tarball naming. ROCM_VERSION is used for internal environment setup.
#
# Example:
#   GITHUB_REPO="myorg/myrock" RELEASE_VERSION=6.4.0rc20250425 ./install_rocm_from_release.sh gfx942
#
# -----------------------------------------------------------------------------


# Ensure required tools
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
RELEASE_TAG="${RELEASE_TAG:-nightly-tarball}"
ROCM_VERSION_DATE="${ROCM_VERSION_DATE:-$(date -d '1 days ago' +'%Y%m%d')}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/therock/build/dist/rocm}"
OUTPUT_ARTIFACTS_DIR="${OUTPUT_ARTIFACTS_DIR:-/rocm-tarballs}"

GITHUB_REPO="${GITHUB_REPO:-ROCm/TheRock}"
GITHUB_RELEASE_BASE_URL="https://github.com/${GITHUB_REPO}/releases/download"

# Determine current working directory
WORKING_DIR="$(pwd)"
echo "[INFO] Running from directory: $WORKING_DIR"

# Determine RELEASE_VERSION and ROCM_VERSION
if [[ -z "${RELEASE_VERSION:-}" ]]; then
  echo "[INFO] RELEASE_VERSION not set. Reading version.json..."
  : "${VERSION_JSON_PATH:=/therock/src/version.json}"

  if [[ ! -f "$VERSION_JSON_PATH" ]]; then
    echo "[ERROR] version.json not found at $VERSION_JSON_PATH"
    exit 1
  fi

  ROCM_VERSION=$(jq -r '.["rocm-version"]' "$VERSION_JSON_PATH")
  RELEASE_VERSION="${ROCM_VERSION}rc${ROCM_VERSION_DATE}"
  echo "[INFO] Constructed RELEASE_VERSION from version.json: $RELEASE_VERSION"
  echo "[INFO] Using ROCM_VERSION from version.json: $ROCM_VERSION"
else
  echo "[INFO] Using user-provided RELEASE_VERSION: $RELEASE_VERSION"

  if [[ -z "${ROCM_VERSION:-}" ]]; then
    ROCM_VERSION="$(echo "$RELEASE_VERSION" | sed -E 's/rc[0-9]+$//')"
    echo "[INFO] Auto-derived ROCM_VERSION: $ROCM_VERSION"
  else
    echo "[INFO] Using user-provided ROCM_VERSION: $ROCM_VERSION"
  fi
fi

# Parse AMDGPU targets
if [[ $# -ge 1 ]]; then
  AMDGPU_TARGETS="$1"
else
  AMDGPU_TARGETS="gfx942"
fi

mkdir -p "${OUTPUT_ARTIFACTS_DIR}"
echo "[INFO] Installing ROCm for targets: $AMDGPU_TARGETS"
echo "[INFO] Build date: $ROCM_VERSION_DATE"
echo "[INFO] Output dir: $OUTPUT_ARTIFACTS_DIR"

# === Fallback encoding map ===
fallback_target_name() {
  case "$1" in
    gfx942) echo "gfx94X-dcgpu" ;;
    gfx1100) echo "gfx110X-dgpu" ;;
    gfx1201) echo "gfx120X-all" ;;
    *) echo "" ;;
  esac
}

# Step 1: Download and Extract
for target in $AMDGPU_TARGETS; do
  TARGET_DIR="${OUTPUT_ARTIFACTS_DIR}/${target}"
  mkdir -p "${TARGET_DIR}"

  # Primary attempt
  TARBALL_NAME="therock-dist-linux-${target}-${RELEASE_VERSION}.tar.gz"
  TARBALL_URL="${GITHUB_RELEASE_BASE_URL}/${RELEASE_TAG}/${TARBALL_NAME}"
  TARBALL_PATH="${TARGET_DIR}/${TARBALL_NAME}"

  echo "[INFO] Trying primary tarball: $TARBALL_URL"
  if ! curl -sSL --fail -o "$TARBALL_PATH" "$TARBALL_URL"; then
    echo "[WARN] Primary tarball not found for $target. Trying fallback encoding..."

    fallback=$(fallback_target_name "$target")
    if [[ -z "$fallback" ]]; then
      echo "[ERROR] No fallback rule for target $target"
      exit 1
    fi

    TARBALL_NAME="therock-dist-linux-${fallback}-${RELEASE_VERSION}.tar.gz"
    TARBALL_URL="${GITHUB_RELEASE_BASE_URL}/${RELEASE_TAG}/${TARBALL_NAME}"
    TARBALL_PATH="${TARGET_DIR}/${TARBALL_NAME}"

    echo "[INFO] Trying fallback tarball: $TARBALL_URL"
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
