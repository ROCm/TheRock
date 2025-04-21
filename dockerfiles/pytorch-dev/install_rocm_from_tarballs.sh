#!/bin/bash
set -euo pipefail

# === Configuration ===
ARTIFACT_RUN_ID="${ARTIFACT_RUN_ID:-}"
AMDGPU_FAMILIES="${AMDGPU_FAMILIES:-gfx942 gfx1100 gfx1201}"
OUTPUT_ARTIFACTS_DIR="${OUTPUT_ARTIFACTS_DIR:-/rocm-tarballs}"
FETCH_ARTIFACT_ARGS="${FETCH_ARTIFACT_ARGS:-"--all"}"
INSTALL_PREFIX="/opt/rocm"

if [[ -z "$ARTIFACT_RUN_ID" ]]; then
  echo "[ERROR] ARTIFACT_RUN_ID must be set"
  exit 1
fi

echo "[INFO] Installing ROCm artifacts for: $AMDGPU_FAMILIES"
echo "[INFO] Artifact run ID: $ARTIFACT_RUN_ID"
echo "[INFO] Destination directory: $OUTPUT_ARTIFACTS_DIR"

# === Step 1: Fetch and Extract for each GPU target ===
for target in $AMDGPU_FAMILIES; do
  TARGET_DIR="${OUTPUT_ARTIFACTS_DIR}/${target}"
  echo "[INFO] Fetching artifacts for target: $target -> $TARGET_DIR"
  mkdir -p "$TARGET_DIR"

  python3 ./build_tools/fetch_artifacts.py \
    --run-id="$ARTIFACT_RUN_ID" \
    --target="$target" \
    --build-dir="$TARGET_DIR" \
    $FETCH_ARTIFACT_ARGS

  echo "[INFO] Installing ROCm from tarballs for $target"
  for tarball in "$TARGET_DIR"/*.tar.*; do
    echo "[INFO] Extracting $tarball"
    case "$tarball" in
      *.tar.gz) sudo tar -xvzf "$tarball" -C "$INSTALL_PREFIX" ;;
      *.tar.xz) sudo tar -xvJf "$tarball" -C "$INSTALL_PREFIX" ;;
      *) echo "[WARN] Skipping unknown archive format: $tarball" ;;
    esac
  done
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
