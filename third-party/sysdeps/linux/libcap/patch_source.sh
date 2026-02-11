#!/bin/bash
# Symbol versioning for libcap 2.69 (similar to elfutils)
set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBCAP_MAP="${SCRIPT_DIR}/libcap.map"
LIBPSX_MAP="${SCRIPT_DIR}/libpsx.map"
MAKEFILE="${SOURCE_DIR}/libcap/Makefile"
PROGS_MAKEFILE="${SOURCE_DIR}/progs/Makefile"

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
echo "    Copying libcap.map and libpsx.map to source directory"
cp "$LIBCAP_MAP" "${SOURCE_DIR}/libcap/libcap.map"
cp "$LIBPSX_MAP" "${SOURCE_DIR}/libcap/libpsx.map"

# Prefix version tag
echo "    Prefixing version tag with AMDROCM_SYSDEPS_1.0_"
sed -i 's/\bLIBCAP_2\.69\b/AMDROCM_SYSDEPS_1.0_LIBCAP_2.69/g' \
  "${SOURCE_DIR}/libcap/libcap.map"
# libpsx.map is having AMDROCM_SYSDEPS_1.0, no need to edit again

# Add --version-script to Makefile
echo "    Patching Makefile to use version script"
sed -i '/\$(LD).*-Wl,-soname,\$(MAJCAPLIBNAME)/ {
  /--version-script/! s|-o \$(MINCAPLIBNAME)|-Wl,--version-script,libcap.map -o $(MINCAPLIBNAME)|
}' "$MAKEFILE"

sed -i '/\$(LD).*-Wl,-soname,\$(MAJPSXLIBNAME)/ {
  /--version-script/! s|-o \$(MINPSXLIBNAME)|-Wl,--version-script,libpsx.map -o $(MINPSXLIBNAME)|
}' "$MAKEFILE"


if ! grep -q -- "--version-script,libcap.map" "$MAKEFILE"; then
  echo "ERROR: Failed to patch Makefile with --version-script" >&2
  exit 1
fi

if ! grep -q -- "--version-script,libpsx.map" "$MAKEFILE"; then
  echo "ERROR: Failed to patch Makefile with --version-script" >&2
  exit 1
fi

echo "==> Symbol versioning patches applied successfully"


echo "==> Applying Library renaming patches to libcap and psx"

# Create library with librocm_sysdeps_cap.so and librocm_sysdeps_psx.so
sed -i 's|CAPLIBNAME=$(LIBTITLE).so|CAPLIBNAME=librocm_sysdeps_cap.so|' "$MAKEFILE"
sed -i 's|PSXLIBNAME=$(PSXTITLE).so|PSXLIBNAME=librocm_sysdeps_psx.so|' "$MAKEFILE"
# Change libcap.so with librocm_sysdeps_cap.so in progs/Makefile
sed -i 's|libcap.so|librocm_sysdeps_cap.so|' "$PROGS_MAKEFILE"

echo "==> Library renaming patches applied successfully"
