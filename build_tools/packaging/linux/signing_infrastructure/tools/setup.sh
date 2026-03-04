#!/bin/bash
# Setup script for gpg shim

set -e

INSTALL_DIR="${1:-$HOME/.local/bin}"

echo "Setting up GPG shim for remote RPM signing..."

# Make gpgshim executable
chmod +x gpgshim

# Create installation directory if it doesn't exist
mkdir -p "$INSTALL_DIR"

# Copy gpgshim to installation directory
cp gpgshim "$INSTALL_DIR/gpgshim"

echo ""
echo "✓ GPG shim installed to: $INSTALL_DIR/gpgshim"
echo ""
echo "Configuration:"
echo "  Set the following environment variables:"
echo ""
echo "  export GPG_SIGNING_SERVER='http://your-signing-server:8080/sign'"
echo "  export GPG_KEY_ID='your-key-id'  # optional, can be specified in rpmsign command"
echo "  export GPG_TIMEOUT='30'  # optional, default is 30 seconds"
echo ""
echo "Usage:"
echo "  To use with rpmsign, either:"
echo "  1. Add $INSTALL_DIR to your PATH before system directories"
echo "  2. Or use: rpmsign --define '_gpg_path $INSTALL_DIR/gpgshim' ..."
echo "  3. Or create an alias: alias gpg='$INSTALL_DIR/gpgshim'"
echo ""
