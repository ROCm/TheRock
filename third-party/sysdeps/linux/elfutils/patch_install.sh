#!/bin/bash
set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"

# update_library_links
# ---------------------
# Purpose:
#   Normalize a shared library so that its real file is named exactly as its ELF SONAME,
#   and rename the input file (usually a symlink) to a desired linker name (e.g., libfoo.so).
#
# Synopsis:
#   update_library_links <libfile> <linker_name>
#
# Arguments:
#   libfile      Path to the library file or symlink.
#                Example: $PREFIX/lib/librocm_sysdeps_elf.so
#   linker_name  Desired linker-name filename to exist in the same directory.
#                Example: libelf.so
update_library_links() {
    local libfile="$1"        # e.g. $PREFIX/lib/librocm_sysdeps_elf.so
    local linker_name="$2"    # e.g. libelf.so

    if [ ! -e "$libfile" ]; then
        echo "Error: File '$libfile' not found" >&2
        return 1
    fi

    local dir="$(dirname -- "$libfile")"
    # Get the soname and realname
    local lib_soname="$("$PATCHELF" --print-soname "$libfile" 2>/dev/null || true)"
    local realname="$(readlink -f -- "$libfile" 2>/dev/null || true)"

    if [[ -z "$lib_soname" || -z "$realname" ]]; then
        [[ -z "$lib_soname" ]] && echo "Error: No SONAME found in '$libfile'" >&2
        [[ -z "$realname" ]] && echo "Error: readlink -f failed for '$libfile'" >&2
        return 1
    fi

    if [[ "$realname" != "$dir/$lib_soname" ]]; then
        # Move the real file to $dir/$lib_soname
        mv -v -- "$realname" "$dir/$lib_soname"
    fi
    # Rename symlink in the same directory
    mv "$libfile" "$dir/$linker_name"
}

update_library_links "$PREFIX/lib/librocm_sysdeps_elf.so" "libelf.so"
update_library_links "$PREFIX/lib/librocm_sysdeps_asm.so" "libasm.so"
update_library_links "$PREFIX/lib/librocm_sysdeps_dw.so" "libdw.so"

# pc files are not output with a relative prefix. Sed it to relative.
sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc

# We don't need old v0 compat libs.
rm -f $PREFIX/lib/*-0.so

# We don't want the static libs.
rm $PREFIX/lib/*.a

# We don't need the programs.
rm -Rf $PREFIX/bin
