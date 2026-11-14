# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import platform
import shutil
import sys

PREFIX = sys.argv[1]

if platform.system() == "Linux":
    source = str(Path(PREFIX) / "lib" / "librocm_sysdeps_liblzma.so")
    destination = str(Path(PREFIX) / "lib" / "liblzma.so")
    shutil.move(source, destination)
    # We don't want the static lib on Linux - delete it if it is there
    static_lib = Path(PREFIX) / "lib" / "librocm_sysdeps_liblzma.a"
    if static_lib.exists():
        static_lib.unlink()
elif platform.system() == "Windows":
    # Keep the DLL and its import lib: rocgdb (built with mingw gcc) links
    # -llzma against the import lib and loads the DLL at runtime.  The DLL
    # carries the private rocm_sysdeps_liblzma name (see liblzma_include.cmake).
    pass
