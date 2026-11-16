# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import shutil
import sys

repo_root = Path(__file__).resolve().parents[4]
build_tools_path = repo_root / "build_tools"
sys.path.insert(0, str(build_tools_path))
from patch_linux_so import update_library_links, relativize_pc_file


# Fetch an environment variable or exit if it is not found.
def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


# Validate the install prefix argument.
prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not prefix:
    print("Error: Expected install prefix argument")
    sys.exit(1)

# 1st argument is the installation prefix.
install_prefix = sys.argv[1]

# Required environment variables.
patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    # Specify the directory containing the libraries.
    lib_dir = Path(install_prefix) / "lib"

    # Rename the real files to their SONAME and create linker-name symlinks
    # (e.g. libelf.so -> librocm_sysdeps_elf.so.1).
    libraries = [
        ("librocm_sysdeps_elf.so", "libelf.so"),
        ("librocm_sysdeps_asm.so", "libasm.so"),
        ("librocm_sysdeps_dw.so", "libdw.so"),
    ]
    for source_name, linker_name in libraries:
        update_library_links(lib_dir / source_name, linker_name, patchelf_exe)

    # pc files are not output with a relative prefix, and libdir is absolute
    # because configure is given an absolute --libdir. Make them relocatable.
    for pc_file in (lib_dir / "pkgconfig").glob("*.pc"):
        relativize_pc_file(pc_file)

    # We don't want the static libs.
    for file_path in lib_dir.glob("*.a"):
        file_path.unlink()

    # We don't need the programs.
    shutil.rmtree(Path(install_prefix) / "bin", ignore_errors=True)
