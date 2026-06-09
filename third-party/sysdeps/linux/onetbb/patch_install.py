# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
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

# Import common utilities from build_tools using THEROCK_SOURCE_DIR
script_path = therock_source_dir / "build_tools" / "patch_linux_so.py"
sys.path.insert(0, str(script_path.parent))
from patch_linux_so import relativize_pc_file

if platform.system() == "Linux":
    lib_dir = Path(install_prefix) / "lib"

    # Remove libtool descriptors (*.la). Shared libs (*.so) are not built since
    # oneTBB is configured static-only; only the *.a archives are kept.
    for file_path in lib_dir.iterdir():
        if file_path.suffix == ".la":
            file_path.unlink(missing_ok=True)

    # Create linker-name symlinks (e.g. libtbb.a -> librocm_sysdeps_tbb.a) so
    # consumers can link with the conventional names.
    libraries = [
        ("librocm_sysdeps_tbb.a", "libtbb.a"),
        ("librocm_sysdeps_tbbmalloc.a", "libtbbmalloc.a"),
        ("librocm_sysdeps_tbbmalloc_proxy.a", "libtbbmalloc_proxy.a"),
    ]
    for source_name, linker_name in libraries:
        source = lib_dir / source_name
        if source.exists():
            link_path = lib_dir / linker_name
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(source_name)

    # Fix .pc files to use relocatable paths.
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)
