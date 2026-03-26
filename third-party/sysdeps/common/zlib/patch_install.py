# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import platform
import shutil
import sys

PREFIX = sys.argv[1]

if platform.system() == "Linux":
    lib_dir = Path(PREFIX) / "lib"
    # Create libz.so as a symlink to the soname. zlib 1.3.1 installed a
    # librocm_sysdeps_z.so namelink; 1.3.2 does not, so create the symlink
    # explicitly rather than moving a namelink that may not be present.
    (lib_dir / "libz.so").symlink_to("librocm_sysdeps_z.so.1")
    namelink = lib_dir / "librocm_sysdeps_z.so"
    if namelink.is_symlink() or namelink.exists():
        namelink.unlink()
    # We don't want the static lib on Linux.
    (lib_dir / "librocm_sysdeps_z.a").unlink()

# Remove zlib's auto-generated cmake config; TheRock provides its own in
# lib/cmake/ZLIB/. On Windows the case-insensitive filesystem makes
# lib/cmake/zlib the same directory, causing conflicts if both exist.
cmake_dir = Path(PREFIX) / "lib" / "cmake" / "zlib"
if cmake_dir.is_dir():
    shutil.rmtree(cmake_dir)
