# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Post-install fixups for the popt sysdep: drop static and libtool files, give the
# prefixed library its SONAME-named file and a libpopt.so dev symlink, and make the
# pkg-config file relocatable.

from pathlib import Path
import os
import platform
import shutil
import sys

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / "build_tools"))
from patch_linux_so import update_library_links, relativize_pc_file

prefix = sys.argv[1] if len(sys.argv) > 1 else None
if not prefix:
    sys.exit("Error: Expected install prefix argument")
patchelf_exe = os.environ.get("PATCHELF")
if not patchelf_exe:
    sys.exit("Error: PATCHELF not defined")

if platform.system() == "Linux":
    lib_dir = Path(prefix) / "lib"
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    update_library_links(
        lib_dir / "librocm_sysdeps_popt.so", "libpopt.so", patchelf_exe
    )

    for pc_file in (lib_dir / "pkgconfig").glob("*.pc"):
        relativize_pc_file(pc_file)

    # Documentation and translations are not shipped.
    for unused in ("share/man", "share/locale", "share/doc"):
        shutil.rmtree(Path(prefix) / unused, ignore_errors=True)
