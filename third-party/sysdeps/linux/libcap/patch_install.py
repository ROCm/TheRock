# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "build_tools"))
from patch_linux_so import update_sysdeps_library_links, relativize_pc_file

# Validate the install prefix argument.
if len(sys.argv) < 2:
    print("Error: Expected install prefix argument")
    sys.exit(1)

lib_dir = Path(sys.argv[1]) / "lib"

# libcap.so -> librocm_sysdeps_cap.so.2, libpsx.so -> librocm_sysdeps_psx.so.2
update_sysdeps_library_links(lib_dir, "libcap.so")
update_sysdeps_library_links(lib_dir, "libpsx.so")

# Fix .pc files to use relocatable paths.
for pc_file in (lib_dir / "pkgconfig").glob("*.pc"):
    relativize_pc_file(pc_file)
