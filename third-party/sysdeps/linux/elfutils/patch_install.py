# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "build_tools"))
from patch_linux_so import update_sysdeps_library_links, relativize_pc_file

# Validate the install prefix argument.
if len(sys.argv) < 2:
    print("Error: Expected install prefix argument")
    sys.exit(1)

prefix = Path(sys.argv[1])
lib_dir = prefix / "lib"

# libelf.so -> librocm_sysdeps_elf.so.1, libasm.so -> librocm_sysdeps_asm.so.1,
# libdw.so -> librocm_sysdeps_dw.so.1
update_sysdeps_library_links(lib_dir, "libelf.so")
update_sysdeps_library_links(lib_dir, "libasm.so")
update_sysdeps_library_links(lib_dir, "libdw.so")

# pc files are not output with a relative prefix, and libdir is absolute
# because configure is given an absolute --libdir. Make them relocatable.
for pc_file in (lib_dir / "pkgconfig").glob("*.pc"):
    relativize_pc_file(pc_file)

# We don't want the static libs.
for file_path in lib_dir.glob("*.a"):
    file_path.unlink()

# We don't need the programs.
shutil.rmtree(prefix / "bin", ignore_errors=True)
