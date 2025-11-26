#!/bin/bash
set -euo pipefail

# gRPC source tarball generation script

GRPC_VERSION="${GRPC_VERSION:-v1.67.1}"
GRPC_TAG="${GRPC_VERSION}"
OUTPUT_DIR="${PWD}/tarballs"
TEMP_DIR=$(mktemp -d)

echo "===================================="
echo "gRPC Tarball Generation"
echo "===================================="
echo "Version: ${GRPC_VERSION}"
echo "Tag: ${GRPC_TAG}"
echo "Output: ${OUTPUT_DIR}"
echo "Temp: ${TEMP_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Clone gRPC repository
echo "Step 1: Cloning gRPC repository..."
cd "${TEMP_DIR}"
git clone --depth 1 --branch "${GRPC_TAG}" --recurse-submodules \
  https://github.com/grpc/grpc.git grpc-${GRPC_VERSION}

# Enter the cloned directory
cd "grpc-${GRPC_VERSION}"

# Ensure all submodules are initialized and updated
echo ""
echo "Step 2: Syncing submodules..."
git submodule sync --recursive
git submodule update --init --recursive --depth 1

# Remove all .git directories and files
echo ""
echo "Step 3: Removing .git directories..."
find . -name ".git" -exec rm -rf {} + 2>/dev/null || true
find . -name ".gitignore" -delete
find . -name ".gitmodules" -delete
find . -name ".gitattributes" -delete

# Return to temp directory
cd "${TEMP_DIR}"

# Create tarball
echo ""
echo "Step 4: Creating tarball..."
TARBALL_NAME="grpc-${GRPC_VERSION}.tar.gz"
tar -czf "${OUTPUT_DIR}/${TARBALL_NAME}" "grpc-${GRPC_VERSION}"

# Calculate SHA256 hash
echo ""
echo "Step 5: Calculating SHA256 hash..."
cd "${OUTPUT_DIR}"
SHA256=$(sha256sum "${TARBALL_NAME}" | awk '{print $1}')

# Display results
echo ""
echo "===================================="
echo "Tarball created successfully!"
echo "===================================="
echo "File: ${OUTPUT_DIR}/${TARBALL_NAME}"
echo "Size: $(du -h "${TARBALL_NAME}" | cut -f1)"
echo "SHA256: ${SHA256}"
echo ""
