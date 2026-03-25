# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import platform
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
elif platform.system() == "Windows":
    # We don't want the libz.dll on Windows.
    (Path(PREFIX) / "bin" / "zlib.dll").unlink()
    (Path(PREFIX) / "lib" / "zlib.lib").unlink()
