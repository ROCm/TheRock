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

# GLib installs several interdependent shared libraries. We rename each to a
# librocm_sysdeps_* SONAME so the bundle can coexist with a system GLib, then fix
# up the inter-library NEEDED entries to point at the renamed siblings.
GLIB_LIBS = ["glib-2.0", "gobject-2.0", "gmodule-2.0", "gthread-2.0", "gio-2.0"]


def get_env_or_exit(var_name):
    value = os.environ.get(var_name)
    if value is None:
        print(f"Error: {var_name} not defined")
        sys.exit(1)
    return value


def sh(*args):
    subprocess.run([str(a) for a in args], check=True)


def needed(patchelf_exe, path):
    out = subprocess.check_output(
        [patchelf_exe, "--print-needed", str(path)], text=True
    )
    return out.split()


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

    # Step 1: rename each library's SONAME and real file, and keep only the
    # canonical linker symlink (lib<name>.so).
    for name in GLIB_LIBS:
        dev = lib_dir / f"lib{name}.so"
        if not (dev.exists() or dev.is_symlink()):
            continue
        real = dev.resolve()
        new_soname = f"librocm_sysdeps_{name}.so.0"
        sh(patchelf_exe, "--set-soname", new_soname, real)
        for link in lib_dir.glob(f"lib{name}.so*"):
            if link.is_symlink():
                link.unlink()
        target = lib_dir / new_soname
        if real != target:
            shutil.move(str(real), str(target))
        (lib_dir / f"lib{name}.so").symlink_to(new_soname)

    # Step 2: repoint the inter-GLib NEEDED entries at the renamed siblings and
    # normalise RUNPATH. Leaf deps (pcre2/libffi/zlib) are already the renamed
    # sysdep SONAMEs because GLib links them from the bundle.
    replacements = {f"lib{n}.so.0": f"librocm_sysdeps_{n}.so.0" for n in GLIB_LIBS}
    for name in GLIB_LIBS:
        target = lib_dir / f"librocm_sysdeps_{name}.so.0"
        if not target.exists():
            continue
        current = needed(patchelf_exe, target)
        for old, new in replacements.items():
            if old in current:
                sh(patchelf_exe, "--replace-needed", old, new, target)
        sh(patchelf_exe, "--set-rpath", "$ORIGIN", target)

    # Step 3: make the pkg-config files relocatable / free of absolute -L paths.
    for name in GLIB_LIBS:
        pc = pkgconfig_dir / f"{name}.pc"
        if pc.exists():
            relativize_pc_file(pc)
