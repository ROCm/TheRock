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

# libnuma.so -> librocm_sysdeps_numa.so.1
update_sysdeps_library_links(lib_dir, "libnuma.so")

# Fix .pc files to use relocatable paths.
for pc_file in (lib_dir / "pkgconfig").glob("*.pc"):
    relativize_pc_file(pc_file)

# We don't want library descriptors or binaries.
for la_file in lib_dir.glob("*.la"):
    la_file.unlink()
shutil.rmtree(prefix / "bin", ignore_errors=True)
