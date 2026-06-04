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
prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not prefix:
    print("Error: Expected install prefix argument")
    sys.exit(1)

install_prefix = sys.argv[1]

# Required environment variables.
therock_source_dir = Path(get_env_or_exit("THEROCK_SOURCE_DIR"))
patchelf_exe = get_env_or_exit("PATCHELF")

# Import common utilities from build_tools using THEROCK_SOURCE_DIR
script_path = therock_source_dir / "build_tools" / "patch_linux_so.py"
sys.path.insert(0, str(script_path.parent))
from patch_linux_so import update_library_links, relativize_pc_file

if platform.system() == "Linux":
    lib_dir = Path(install_prefix) / "lib"

    # Remove static libs (*.a) and descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Set RPATH on all prefixed shared libraries.
    for lib_path in lib_dir.glob("librocm_sysdeps_*.so*"):
        if lib_path.is_symlink():
            continue
        try:
            subprocess.run(
                [
                    patchelf_exe,
                    "--set-rpath",
                    "$ORIGIN",
                    str(lib_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"Error: Failed to set RPATH on {lib_path.name} (Exit: {e.returncode})"
            )
            sys.exit(e.returncode)

    # Create linker-name symlinks (e.g. libtbb.so -> librocm_sysdeps_tbb.so)
    libraries = [
        ("librocm_sysdeps_tbb.so", "libtbb.so"),
        ("librocm_sysdeps_tbbmalloc.so", "libtbbmalloc.so"),
        ("librocm_sysdeps_tbbmalloc_proxy.so", "libtbbmalloc_proxy.so"),
    ]
    for source_name, linker_name in libraries:
        source = lib_dir / source_name
        if source.exists():
            update_library_links(source, linker_name, patchelf_exe)

    # Fix .pc files to use relocatable paths.
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)
