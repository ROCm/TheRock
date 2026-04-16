#!/bin/bash
# Build the ROCmGPU DriverKit extension and host app.
#
# Usage:
#   ./build.sh              # Compile check (Debug, ad-hoc signed)
#   ./build.sh release      # Compile check (Release, ad-hoc signed)
#   ./build.sh clean        # Clean build artifacts
#
# Requirements:
#   - Xcode 15+ with DriverKit SDK
#   - macOS 12.1+ SDK
#   - Apple Silicon Mac (for arm64 target)
#
# Ad-hoc signing is sufficient for validating that the sources compile
# against the DriverKit SDK. Ad-hoc DEXTs CANNOT be installed via
# OSSystemExtensionRequest — the framework silently rejects them with
# code 4 (.extensionNotFound) regardless of SIP / dev-mode settings.
#
# To build an installable DEXT you need:
#   1. Paid Apple Developer Program membership.
#   2. Apple-approved entitlements:
#        - com.apple.developer.driverkit.transport.pci (managed — request
#          at developer.apple.com/system-extensions/ for your Team ID +
#          bundle id ai.rocm.gpu.driver + match 0x00001002&0x0000FFFF)
#        - com.apple.developer.driverkit.userclient-access (managed —
#          request for ai.rocm.gpu.app referring to ai.rocm.gpu.driver)
#   3. Build with automatic signing + device registration, e.g.:
#        xcodebuild build -project ROCmGPU/ROCmGPU.xcodeproj \
#            -scheme ROCmGPUApp -configuration Debug \
#            -allowProvisioningUpdates -allowProvisioningDeviceRegistration \
#            ONLY_ACTIVE_ARCH=YES ARCHS=arm64 VALID_ARCHS=arm64 \
#            CODE_SIGN_STYLE=Automatic DEVELOPMENT_TEAM=<YOUR_TEAM_ID>
#   4. On the target Mac: csrutil disable + `sudo systemextensionsctl
#      developer on` + reboot.
#   5. Copy built .app to /Applications and run `ROCmGPUApp install`.
#
# See userspace_driver/macos_driver/README.md for background.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../ROCmGPU"
BUILD_DIR="$SCRIPT_DIR/../build"
CONFIG="${1:-Debug}"

case "$CONFIG" in
    clean)
        echo "Cleaning build artifacts..."
        rm -rf "$BUILD_DIR"
        echo "Done."
        exit 0
        ;;
    release|Release)
        CONFIG="Release"
        ;;
    debug|Debug|"")
        CONFIG="Debug"
        ;;
    *)
        echo "Usage: $0 [debug|release|clean]"
        exit 1
        ;;
esac

echo "Building ROCmGPU ($CONFIG)..."
echo "  Project: $PROJECT_DIR"
echo "  Output:  $BUILD_DIR"

# Build the DEXT
echo ""
echo "=== Building ROCmGPUDriver.dext ==="
xcodebuild build \
    -project "$PROJECT_DIR/ROCmGPU.xcodeproj" \
    -scheme "ROCmGPUDriver" \
    -configuration "$CONFIG" \
    -derivedDataPath "$BUILD_DIR" \
    ONLY_ACTIVE_ARCH=NO \
    2>&1 | tail -20

# Build the host app
echo ""
echo "=== Building ROCmGPUApp ==="
xcodebuild build \
    -project "$PROJECT_DIR/ROCmGPU.xcodeproj" \
    -scheme "ROCmGPUApp" \
    -configuration "$CONFIG" \
    -derivedDataPath "$BUILD_DIR" \
    ONLY_ACTIVE_ARCH=NO \
    2>&1 | tail -20

# Find the built products
DEXT_PATH=$(find "$BUILD_DIR" -name "ROCmGPUDriver.dext" -type d 2>/dev/null | head -1)
APP_PATH=$(find "$BUILD_DIR" -name "ROCmGPUApp.app" -type d 2>/dev/null | head -1)

echo ""
echo "=== Build complete ==="
if [ -n "$DEXT_PATH" ]; then
    echo "  DEXT: $DEXT_PATH"
else
    echo "  DEXT: NOT FOUND (check build logs)"
fi
if [ -n "$APP_PATH" ]; then
    echo "  App:  $APP_PATH"
else
    echo "  App:  NOT FOUND (check build logs)"
fi

# For development: self-sign if no identity
echo ""
echo "=== Self-signing for development ==="
if [ -n "$DEXT_PATH" ]; then
    codesign --force --deep --sign - "$DEXT_PATH" 2>/dev/null && \
        echo "  DEXT signed (ad-hoc)" || \
        echo "  DEXT signing skipped"
fi
if [ -n "$APP_PATH" ]; then
    codesign --force --deep --sign - "$APP_PATH" 2>/dev/null && \
        echo "  App signed (ad-hoc)" || \
        echo "  App signing skipped"
fi

echo ""
echo "Next: Run ./install.sh to install the DEXT"
