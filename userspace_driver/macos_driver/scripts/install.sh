#!/bin/bash
# Install and manage the ROCmGPU DriverKit extension.
#
# Usage:
#   ./install.sh              # Install the DEXT
#   ./install.sh uninstall    # Remove the DEXT
#   ./install.sh status       # Check DEXT status
#   ./install.sh dev-setup    # Full development setup (SIP must be disabled)
#
# Prerequisites for development (without Apple entitlements):
#   1. Boot into Recovery Mode (hold Power button on Apple Silicon)
#   2. Open Terminal and run: csrutil disable
#   3. Reboot normally
#   4. Run: sudo systemextensionsctl developer on

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
ACTION="${1:-install}"

# Find built app
APP_PATH=$(find "$BUILD_DIR" -name "ROCmGPUApp.app" -type d 2>/dev/null | head -1)

case "$ACTION" in
    install)
        if [ -z "$APP_PATH" ]; then
            echo "ERROR: ROCmGPUApp.app not found. Run build.sh first."
            exit 1
        fi

        echo "Installing ROCmGPU DEXT..."
        echo "  App: $APP_PATH"

        # Copy app to /Applications for SystemExtensions to find it
        echo "  Copying to /Applications..."
        sudo cp -R "$APP_PATH" /Applications/ROCmGPU.app

        # Run the app to trigger DEXT installation
        echo "  Requesting DEXT activation..."
        /Applications/ROCmGPU.app/Contents/MacOS/ROCmGPUApp install

        echo ""
        echo "If prompted, enable the extension in:"
        echo "  System Settings > General > Login Items & Extensions > Driver Extensions"
        ;;

    uninstall)
        echo "Uninstalling ROCmGPU DEXT..."
        if [ -f "/Applications/ROCmGPU.app/Contents/MacOS/ROCmGPUApp" ]; then
            /Applications/ROCmGPU.app/Contents/MacOS/ROCmGPUApp uninstall
        fi
        sudo rm -rf /Applications/ROCmGPU.app
        echo "Done."
        ;;

    status)
        echo "ROCmGPU DEXT status:"
        echo ""
        systemextensionsctl list 2>/dev/null | grep -i rocm || echo "  Not registered"
        echo ""
        # Check IOKit registry
        echo "IOKit services:"
        ioreg -l | grep -i ROCmGPU || echo "  No ROCmGPU services found"
        ;;

    dev-setup)
        echo "=== ROCmGPU Development Setup ==="
        echo ""

        # Check SIP status
        SIP_STATUS=$(csrutil status 2>/dev/null || echo "unknown")
        if echo "$SIP_STATUS" | grep -q "disabled"; then
            echo "[OK] SIP is disabled"
        else
            echo "[!!] SIP is ENABLED. Development requires SIP disabled."
            echo "     Boot into Recovery Mode and run: csrutil disable"
            exit 1
        fi

        # Enable developer mode for system extensions
        echo "Enabling developer mode for system extensions..."
        sudo systemextensionsctl developer on

        # Build
        echo ""
        echo "Building..."
        "$SCRIPT_DIR/build.sh" debug

        # Install
        echo ""
        "$0" install

        echo ""
        echo "=== Development setup complete ==="
        echo ""
        echo "Test with:"
        echo "  python3 -c \"from amd_gpu_driver.backends.macos import MacOSDevice; d = MacOSDevice(); d.open(); print(d)\""
        ;;

    *)
        echo "Usage: $0 [install|uninstall|status|dev-setup]"
        exit 1
        ;;
esac
