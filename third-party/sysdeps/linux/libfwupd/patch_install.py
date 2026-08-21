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
from patch_linux_so import relativize_pc_file

# libfwupd's SONAME (soversion 3). Its GLib/libcurl dependencies are already the
# renamed sysdep SONAMEs (it links them from the bundle), so no NEEDED fixup is
# required here.
_SONAME = "librocm_sysdeps_fwupd.so.3"


def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


def sh(*args):
    subprocess.run([str(a) for a in args], check=True)


prefix = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if not prefix:
    print("Error: Expected install prefix argument")
    sys.exit(1)

install_prefix = sys.argv[1]
patchelf_exe = get_env_or_exit("PATCHELF")

if platform.system() == "Linux":
    lib_dir = Path(install_prefix) / "lib"
    pkgconfig_dir = lib_dir / "pkgconfig"

    # Remove static libs (*.a) and libtool descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Rename the SONAME/real file and keep only the canonical linker symlink.
    dev = lib_dir / "libfwupd.so"
    if dev.exists() or dev.is_symlink():
        real = dev.resolve()
        sh(patchelf_exe, "--set-soname", _SONAME, real)
        for link in lib_dir.glob("libfwupd.so*"):
            if link.is_symlink():
                link.unlink()
        target = lib_dir / _SONAME
        if real != target:
            shutil.move(str(real), str(target))
        (lib_dir / "libfwupd.so").symlink_to(_SONAME)
        sh(patchelf_exe, "--set-rpath", "$ORIGIN", target)

    # Make the pkg-config file relocatable / free of absolute -L paths.
    fwupd_pc = pkgconfig_dir / "fwupd.pc"
    if fwupd_pc.exists():
        relativize_pc_file(fwupd_pc)
