#!/bin/bash
# Symbol versioning for libatomic from GCC
set -e

SOURCE_DIR="${1:?Source directory must be given}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_LDS="${SCRIPT_DIR}/version.lds"
LIBATOMIC_MAKEFILE="${SOURCE_DIR}/libatomic/Makefile.am"

if [ ! -f "$VERSION_LDS" ]; then
  echo "ERROR: version.lds not found at $VERSION_LDS" >&2
  exit 1
fi

if [ ! -f "$LIBATOMIC_MAKEFILE" ]; then
  echo "ERROR: libatomic Makefile.am not found at $LIBATOMIC_MAKEFILE" >&2
  exit 1
fi

echo "==> Applying symbol versioning patches to libatomic"

# Copy version script to libatomic source directory
echo "    Copying version.lds to libatomic source directory"
cp "$VERSION_LDS" "${SOURCE_DIR}/libatomic/version.lds"

# Patch Makefile.am to add version script to LDFLAGS
echo "    Patching Makefile.am to use version script"
if ! grep -q "version.lds" "$LIBATOMIC_MAKEFILE"; then
  sed -i '/^libatomic_la_LDFLAGS =/a\libatomic_la_LDFLAGS += -Wl,--version-script=$(srcdir)/version.lds' "$LIBATOMIC_MAKEFILE"
fi

echo "==> Symbol versioning patches applied successfully"

