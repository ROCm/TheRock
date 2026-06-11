# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import shutil
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

    # Update library linking for each libnl library
    libraries = [
        ("librocm_sysdeps_nl_3.so", "libnl-3.so"),
        ("librocm_sysdeps_nl_genl_3.so", "libnl-genl-3.so"),
        ("librocm_sysdeps_nl_route_3.so", "libnl-route-3.so"),
        ("librocm_sysdeps_nl_idiag_3.so", "libnl-idiag-3.so"),
        ("librocm_sysdeps_nl_nf_3.so", "libnl-nf-3.so"),
        ("librocm_sysdeps_nl_xfrm_3.so", "libnl-xfrm-3.so"),
        ("librocm_sysdeps_nl_cli_3.so", "libnl-cli-3.so"),
    ]

    for source_name, linker_name in libraries:
        source = lib_dir / source_name
        if source.exists():
            update_library_links(source, linker_name)

            # Clean up RUNPATH to only contain $ORIGIN
            target_lib = lib_dir / linker_name
            if target_lib.exists():
                try:
                    subprocess.run(
                        [patchelf_exe, "--set-rpath", "$ORIGIN", str(target_lib)],
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(
                        f"Warning: Failed to set RPATH on {target_lib}: {e}", flush=True
                    )

    # Make .pc files relocatable
    pc_files = [
        "libnl-3.0.pc",
        "libnl-genl-3.0.pc",
        "libnl-route-3.0.pc",
        "libnl-idiag-3.0.pc",
        "libnl-nf-3.0.pc",
        "libnl-xfrm-3.0.pc",
        "libnl-cli-3.0.pc",
    ]

    for pc_name in pc_files:
        pc_file = pkgconfig_dir / pc_name
        if pc_file.exists():
            relativize_pc_file(pc_file)

    # Create header symlinks for test compatibility
    # Headers are installed in libnl3/netlink/, but tests expect netlink/
    include_dir = Path(install_prefix) / "include"
    libnl3_dir = include_dir / "libnl3"

    if libnl3_dir.exists():
        # Create symlink from include/netlink to include/libnl3/netlink
        netlink_symlink = include_dir / "netlink"
        if netlink_symlink.exists() or netlink_symlink.is_symlink():
            netlink_symlink.unlink()

        libnl3_netlink = libnl3_dir / "netlink"
        if libnl3_netlink.exists():
            netlink_symlink.symlink_to("libnl3/netlink", target_is_directory=True)
