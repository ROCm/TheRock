# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import subprocess
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

patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    # Specify the directory containing the libraries.
    lib_dir = Path(install_prefix) / "lib"
    pkgconfig_dir = lib_dir / "pkgconfig"

    # Remove static libs (*.a) and descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # We only ship the 8-bit library that GLib links against; drop the POSIX
    # wrapper library and its pkg-config file.
    for posix_file in lib_dir.glob("libpcre2-posix.*"):
        posix_file.unlink(missing_ok=True)
    (pkgconfig_dir / "libpcre2-posix.pc").unlink(missing_ok=True)

    # Update library linking
    source = lib_dir / "librocm_sysdeps_pcre2-8.so"
    update_library_links(source, "libpcre2-8.so")

    # Clean up RUNPATH to only contain $ORIGIN
    target_lib = lib_dir / "libpcre2-8.so"
    if target_lib.exists():
        try:
            subprocess.run(
                [patchelf_exe, "--set-rpath", "$ORIGIN", str(target_lib)], check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to set RPATH on {target_lib}: {e}", flush=True)

    # Make .pc file relocatable
    pcre2_pc = pkgconfig_dir / "libpcre2-8.pc"
    if pcre2_pc.exists():
        relativize_pc_file(pcre2_pc)
