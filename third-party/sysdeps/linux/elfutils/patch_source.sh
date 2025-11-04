#!/bin/bash

set -e

SOURCE_DIR="${1:?Source directory must be given}"
MAP_FILES="$(find $SOURCE_DIR -name '*.map')"
C_FILES="$(find $SOURCE_DIR -name '*.c')"
CONFIGURE_FILE="$SOURCE_DIR/configure"

# Prefix symbol version file symbols.
echo "Patching map files: $MAP_FILES"
sed -i -E 's|\b(ELFUTILS_[0-9\.]+)|AMDROCM_SYSDEPS_1.0_\1|' $MAP_FILES

# Prefix symbol version definitions in source files (configure contains a test
# too).
echo "Patching source files: $C_FILES"
sed -i -E 's@^((OLD_VERSION|NEW_VERSION|COMPAT_VERSION_NEWPROTO|COMPAT_VERSION).+)(ELFUTILS_[0-9\.]+.*)$@\1AMDROCM_SYSDEPS_1.0_\3@' \
    $C_FILES $CONFIGURE_FILE

echo "Patching Makefile.in"
LIBELF_MAKEFILE_IN="$SOURCE_DIR/libelf/Makefile.in"
LIBASM_MAKEFILE_IN="$SOURCE_DIR/libasm/Makefile.in"
LIBDW_MAKEFILE_IN="$SOURCE_DIR/libdw/Makefile.in"
DBGINFOD_MAKEFILE_IN="$SOURCE_DIR/debuginfod/Makefile.in"
SOURCE_MAKEFILE_IN="$SOURCE_DIR/src/Makefile.in"

# Replace libelf_pic.a with librocm_sysdeps_elf_pic.a
sed -i -E 's/\blibelf_pic\.a\b/librocm_sysdeps_elf_pic.a/g' "$LIBELF_MAKEFILE_IN"
# Replace libelf- with librocm_sysdeps_elf-
sed -i -E 's/libelf-/librocm_sysdeps_elf-/g' "$LIBELF_MAKEFILE_IN"
# Replace libelf.so with librocm_sysdeps_elf.so
sed -i -E 's/\blibelf\.so\b/librocm_sysdeps_elf.so/g' "$DBGINFOD_MAKEFILE_IN" "$LIBASM_MAKEFILE_IN" "$LIBDW_MAKEFILE_IN" "$LIBELF_MAKEFILE_IN" "$SOURCE_MAKEFILE_IN"

#Similar replacement done for libdw and libasm
sed -i -E 's/\blibdw_pic\.a\b/librocm_sysdeps_dw_pic.a/g' "$LIBDW_MAKEFILE_IN"
sed -i -E 's/libdw-/librocm_sysdeps_dw-/g' "$LIBDW_MAKEFILE_IN"
sed -i -E 's/\blibdw\.so\b/librocm_sysdeps_dw.so/g' "$DBGINFOD_MAKEFILE_IN" "$LIBASM_MAKEFILE_IN" "$LIBDW_MAKEFILE_IN" "$SOURCE_MAKEFILE_IN"

sed -i -E 's/\blibasm_pic\.a\b/librocm_sysdeps_asm_pic.a/g' "$LIBASM_MAKEFILE_IN"
sed -i -E 's/libasm-/librocm_sysdeps_asm-/g' "$LIBASM_MAKEFILE_IN"
sed -i -E 's/\blibasm\.so\b/librocm_sysdeps_asm.so/g' "$DBGINFOD_MAKEFILE_IN" "$LIBASM_MAKEFILE_IN" "$SOURCE_MAKEFILE_IN"
