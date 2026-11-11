# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import shutil
import sys

therock_source_dir = Path(os.environ["THEROCK_SOURCE_DIR"])
sys.path.insert(0, str(therock_source_dir / "build_tools"))
from patch_linux_so import relativize_pc_file

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
    # We don't want the .dll on Windows.
    (Path(PREFIX) / "bin" / "liblzma.dll").unlink()
    (Path(PREFIX) / "lib" / "liblzma.lib").unlink()

# xz's auto-generated pkg-config files hardcode the absolute build-time prefix.
# Rewrite them in place to be relocatable.
for pc_file in (Path(PREFIX) / "lib" / "pkgconfig").glob("*.pc"):
    relativize_pc_file(pc_file)
