#!/bin/bash
# Symbol versioning for libcap 2.69 (similar to elfutils)
set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBCAP_MAP="${SCRIPT_DIR}/libcap.map"
MAKEFILE="${SOURCE_DIR}/libcap/Makefile"

if [ ! -f "$LIBCAP_MAP" ]; then
  echo "ERROR: libcap.map not found at $LIBCAP_MAP" >&2
  exit 1
fi

if [ ! -f "$MAKEFILE" ]; then
  echo "ERROR: libcap Makefile not found at $MAKEFILE" >&2
  exit 1
fi

echo "==> Applying symbol versioning patches to libcap"

# Copy version script
echo "    Copying libcap.map to source directory"
cp "$LIBCAP_MAP" "${SOURCE_DIR}/libcap/libcap.map"

# Prefix version tag
echo "    Prefixing version tag with AMDROCM_SYSDEPS_1.0_"
sed -i 's/\bLIBCAP_2\.69\b/AMDROCM_SYSDEPS_1.0_LIBCAP_2.69/g' \
  "${SOURCE_DIR}/libcap/libcap.map"

# Add --version-script to Makefile
echo "    Patching Makefile to use version script"
sed -i '/\$(LD).*-Wl,-soname,\$(MAJCAPLIBNAME)/ {
  /--version-script/! s|-o \$(MINCAPLIBNAME)|-Wl,--version-script,libcap.map -o $(MINCAPLIBNAME)|
}' "$MAKEFILE"

if ! grep -q -- "--version-script,libcap.map" "$MAKEFILE"; then
  echo "ERROR: Failed to patch Makefile with --version-script" >&2
  exit 1
fi

echo "==> Symbol versioning patches applied successfully"
