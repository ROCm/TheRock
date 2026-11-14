# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import platform
import sys

repo_root = Path(__file__).resolve().parents[4]
build_tools_path = repo_root / "build_tools"
sys.path.insert(0, str(build_tools_path))
from relativize_pc_files import relativize_pc_files


# Validate the install prefix argument.
if len(sys.argv) < 2:
    print("Error: Expected install prefix argument")
    sys.exit(1)

# 1st argument is the installation prefix.
install_prefix = Path(sys.argv[1])

if platform.system() == "Linux":
    # Rename librocm_sysdeps_zstd.so to libzstd.so.
    lib_dir = install_prefix / "lib"
    (lib_dir / "librocm_sysdeps_zstd.so").replace(lib_dir / "libzstd.so")

# zstd does not write its .pc files with a relative prefix.
relativize_pc_files(install_prefix)
