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


def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


prefix = sys.argv[1] if len(sys.argv) > 1 else None
if not prefix:
    print("Error: Expected install prefix argument")
    sys.exit(1)

patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    lib_dir = Path(prefix) / "lib"
    pkgconfig_dir = lib_dir / "pkgconfig"

    # Remove static archives and libtool descriptors.
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # (prefixed real name, restored dev-symlink name) for all default-built libs.
    libraries = [
        ("librocm_sysdeps_lttng_ust.so", "liblttng-ust.so"),
        ("librocm_sysdeps_lttng_ust_common.so", "liblttng-ust-common.so"),
        ("librocm_sysdeps_lttng_ust_tracepoint.so", "liblttng-ust-tracepoint.so"),
        ("librocm_sysdeps_lttng_ust_ctl.so", "liblttng-ust-ctl.so"),
        ("librocm_sysdeps_lttng_ust_dl.so", "liblttng-ust-dl.so"),
        ("librocm_sysdeps_lttng_ust_fd.so", "liblttng-ust-fd.so"),
        ("librocm_sysdeps_lttng_ust_fork.so", "liblttng-ust-fork.so"),
        ("librocm_sysdeps_lttng_ust_libc_wrapper.so", "liblttng-ust-libc-wrapper.so"),
        (
            "librocm_sysdeps_lttng_ust_pthread_wrapper.so",
            "liblttng-ust-pthread-wrapper.so",
        ),
        ("librocm_sysdeps_lttng_ust_cyg_profile.so", "liblttng-ust-cyg-profile.so"),
        (
            "librocm_sysdeps_lttng_ust_cyg_profile_fast.so",
            "liblttng-ust-cyg-profile-fast.so",
        ),
    ]

    for source_name, linker_name in libraries:
        source = lib_dir / source_name
        if source.exists():
            update_library_links(source, linker_name, patchelf_exe)
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

    # Make the shipped .pc files (lttng-ust.pc, lttng-ust-ctl.pc) relocatable.
    # Their `Libs:` reference the unprefixed -llttng-ust* names, which resolve
    # through the dev symlinks above.
    if pkgconfig_dir.is_dir():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)

    # We do not ship tools/binaries for this sysdep.
    bin_dir = Path(prefix) / "bin"
    if bin_dir.is_dir():
        for f in bin_dir.iterdir():
            f.unlink()
