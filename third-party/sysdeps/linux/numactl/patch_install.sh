#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set -e

PREFIX="${1:?Expected install prefix argument}"
PATCHELF="${PATCHELF:-patchelf}"
NM="${NM:-nm}"

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
#                Example: $PREFIX/lib/librocm_sysdeps_numa.so
#   linker_name  Desired linker-name filename to exist in the same directory.
#                Example: libnuma.so
update_library_links() {
    local libfile="$1"        # e.g. $PREFIX/lib/librocm_sysdeps_numa.so
    local linker_name="$2"    # e.g. libnuma.so

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
 pushd "$dir" > /dev/null
 ln -sf "$lib_soname" "$linker_name"
 popd > /dev/null
 rm "$libfile"
    else
    # Rename symlink in the same directory
        mv "$libfile" "$dir/$linker_name"
    fi
}

update_library_links "$PREFIX/lib/librocm_sysdeps_numa.so" "libnuma.so"

# numactl's configure unconditionally probes for and links -latomic (upstream
# uses AC_SEARCH_LIBS([__atomic_fetch_and_1], [atomic])). On targets that lower
# libnuma's small atomics inline (e.g. x86_64) this leaves a DT_NEEDED
# libatomic.so.1 entry that resolves no symbols, which makes tools like rocminfo
# fail to load on systems without libatomic installed. Drop the entry only when it is
# unused; on targets where the atomics are emitted as out-of-line calls (e.g. sparc, riscv) the
# dependency is genuine and is left untouched so the build keeps working.
strip_unused_libatomic_dependency() {
    local libfile="$1"
    local needed_libs
    local undefined_symbols

    if ! needed_libs="$("$PATCHELF" --print-needed "$libfile")"; then
        echo "Error: could not read NEEDED entries from $libfile" >&2
        return 1
    fi
    if ! printf "%s\n" "$needed_libs" | grep -Fxq "libatomic.so.1"; then
        # No libatomic dependency to begin with; nothing to do.
        return 0
    fi

    # A genuine dependency shows up as an unresolved dynamic symbol that libatomic
    # provides. Match both the __atomic_* helper names (which may be unversioned,
    # e.g. weak references that nm reports as undefined) and the @LIBATOMIC version
    # tag (which also covers the non-__atomic_-prefixed C11 helpers libatomic
    # exports, such as atomic_thread_fence and atomic_flag_*).
    if ! undefined_symbols="$("$NM" -D --undefined-only "$libfile")"; then
        echo "Error: could not read dynamic undefined symbols from $libfile" >&2
        return 1
    fi
    if printf "%s\n" "$undefined_symbols" | grep -Eq '(^|[[:space:]])__atomic_|@LIBATOMIC'; then
        echo "Note: $libfile imports libatomic symbols; keeping libatomic.so.1 dependency"
        return 0
    fi

    echo "Removing unused libatomic.so.1 dependency from $libfile"
    "$PATCHELF" --remove-needed libatomic.so.1 "$libfile"
}

# Resolve the SONAME from the dev symlink created above rather than hardcoding the
# version, so a future numactl soname bump does not silently target a missing file.
numa_soname="$("$PATCHELF" --print-soname "$PREFIX/lib/libnuma.so" 2>/dev/null || true)"
if [[ -z "$numa_soname" ]]; then
    echo "Error: could not determine libnuma SONAME from $PREFIX/lib/libnuma.so" >&2
    exit 1
fi
strip_unused_libatomic_dependency "$PREFIX/lib/$numa_soname"

# pc files are not output with a relative prefix. Sed it to relative.
sed -i -E 's|^prefix=.+|prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc
sed -i -E 's|^exec_prefix=.+|exec_prefix=${pcfiledir}/../..|' $PREFIX/lib/pkgconfig/*.pc
sed -i -E 's|^libdir=.+|libdir=${prefix}/lib|' $PREFIX/lib/pkgconfig/*.pc
sed -i -E 's|^includedir=.+|includedir=${prefix}/include|' $PREFIX/lib/pkgconfig/*.pc

# We don't want library descriptors or binaries.
rm $PREFIX/lib/*.la
rm -Rf $PREFIX/bin
