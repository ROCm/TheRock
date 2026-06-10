# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import subprocess
import sys


# Fetch an environment variable or exit if it is not found.
def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


# Validate the install prefix argument.
if len(sys.argv) < 2:
    print("Error: Expected install prefix argument")
    sys.exit(1)

install_prefix = Path(sys.argv[1])

# Required environment variables.
therock_source_dir = Path(get_env_or_exit("THEROCK_SOURCE_DIR"))
patchelf_exe = get_env_or_exit("PATCHELF")

# Import common utilities from build_tools using THEROCK_SOURCE_DIR
script_path = therock_source_dir / "build_tools" / "patch_linux_so.py"
sys.path.insert(0, str(script_path.parent))
from patch_linux_so import relativize_pc_file

if platform.system() == "Linux":
    lib_dir = install_prefix / "lib"

    # Remove static libs (*.a) and libtool descriptors (*.la). oneTBB is built
    # shared-only; only the *.so files are kept.
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Set an $ORIGIN RPATH on each shared library so the co-installed TBB libs
    # (e.g. tbbmalloc_proxy -> tbbmalloc) resolve each other at runtime.
    for lib_path in lib_dir.glob("*.so*"):
        if lib_path.is_symlink():
            continue
        try:
            subprocess.run(
                [patchelf_exe, "--set-rpath", "$ORIGIN", str(lib_path)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"Error: Failed to set RPATH on {lib_path.name} (Exit: {e.returncode})"
            )
            sys.exit(e.returncode)

    # Fix .pc files to use relocatable paths.
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)
