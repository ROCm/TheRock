# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "build_tools"))
from patch_linux_so import update_sysdeps_library_links

# Validate the install prefix argument.
if len(sys.argv) < 2:
    print("Error: Expected install prefix argument")
    sys.exit(1)

lib_dir = Path(sys.argv[1]) / "lib"

# libdrm.so -> librocm_sysdeps_drm.so.2,
# libdrm_amdgpu.so -> librocm_sysdeps_drm_amdgpu.so.1
update_sysdeps_library_links(lib_dir, "libdrm.so")
update_sysdeps_library_links(lib_dir, "libdrm_amdgpu.so")

# No .pc file fixup needed: meson already generates relocatable .pc files
# (-Dpkgconfig.relocatable=true in CMakeLists.txt).
