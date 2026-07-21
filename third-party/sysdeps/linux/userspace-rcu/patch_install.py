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

    # (prefixed real name, restored dev-symlink name) for all eight flavors.
    libraries = [
        ("librocm_sysdeps_urcu.so", "liburcu.so"),
        ("librocm_sysdeps_urcu_common.so", "liburcu-common.so"),
        ("librocm_sysdeps_urcu_bp.so", "liburcu-bp.so"),
        ("librocm_sysdeps_urcu_cds.so", "liburcu-cds.so"),
        ("librocm_sysdeps_urcu_mb.so", "liburcu-mb.so"),
        ("librocm_sysdeps_urcu_memb.so", "liburcu-memb.so"),
        ("librocm_sysdeps_urcu_qsbr.so", "liburcu-qsbr.so"),
        ("librocm_sysdeps_urcu_signal.so", "liburcu-signal.so"),
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

    # Make the shipped .pc files relocatable. Their `Libs:` reference the
    # unprefixed -lurcu* names, which resolve through the dev symlinks above.
    if pkgconfig_dir.is_dir():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)

    # We do not ship tools/binaries for this sysdep.
    bin_dir = Path(prefix) / "bin"
    if bin_dir.is_dir():
        for f in bin_dir.iterdir():
            f.unlink()
