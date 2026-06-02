# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import platform
import shutil
import subprocess
import sys

build_tools_path = Path(__file__).resolve().parent.parent.parent / "build_tools"
sys.path.insert(0, str(build_tools_path))
from patch_linux_so import update_library_links, relativize_pc_file


def rename_pc_files(pc_file: Path) -> None:
    """Rename a .pc file by removing the 'rocm_sysdeps_' prefix if present.

    This function is used when ROCm‑patched pkg‑config files need to be
    restored to their original upstream names. If the filename begins with the
    prefix 'rocm_sysdeps_', the prefix is stripped and the file is renamed in place.

    Parameters
    ----------
    pc_file : Path Path object pointing to the .pc file to be examined and possibly renamed.

    Returns
    -------
    None The function performs an in‑place rename and does not return a value.
    """
    prefix = "rocm_sysdeps_"
    if pc_file.name.startswith(prefix):
        new_name = pc_file.name[len(prefix) :]
        new_path = pc_file.with_name(new_name)
        pc_file.rename(new_path)


def symlink_or_copy(existing_path, new_link):
    """Create symlink if the destination filesystem supports it. Create a copy otherwise.
    Exists to support Windows, where only modern systems might support symlinks.
    """
    existing_path = Path(existing_path)
    new_link = Path(new_link)
    new_link.parent.mkdir(parents=True, exist_ok=True)

    if new_link.exists() or new_link.is_symlink():
        if new_link.is_dir() and not new_link.is_symlink():
            shutil.rmtree(new_link)
        else:
            new_link.unlink()

    try:
        rel_target = os.path.relpath(existing_path, start=new_link.parent)
        new_link.symlink_to(rel_target, target_is_directory=existing_path.is_dir())
        return
    except OSError:
        pass

    if existing_path.is_dir():
        shutil.copytree(existing_path, new_link)
    else:
        shutil.copy2(existing_path, new_link)


def link_header_files_under_dir(source_dir, dest_dir):
    """Support applications referencing ncurses header through
    both `<ncurses.h>` and `<ncursesw/ncurses.h>` by making

    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    if not source_dir.exists():
        return
    dest_dir.mkdir(parents=True, exist_ok=True)

    for header_path in source_dir.iterdir():
        if header_path.is_file() and header_path.suffix == ".h":
            symlink_or_copy(header_path, dest_dir / header_path.name)


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

# 1st argument is the installation prefix.
install_prefix = Path(sys.argv[1])

# Required environment variables.
patchelf_exe = get_env_or_exit("PATCHELF")

# Make headers available under <ncursesw/> e.g.
# `<ncurses.h>` and `<ncursesw/ncurses.h>`
# This follows Ubuntu and Fedora packaging
include_dir = Path(install_prefix) / "include"
ncursesw_dir = include_dir / "ncursesw"
link_header_files_under_dir(include_dir, ncursesw_dir)

if platform.system() == "Linux":
    # Specify the directory
    lib_dir = Path(install_prefix) / "lib"

    # Remove static libs (*.a) and descriptors (*.la).
    for file_path in lib_dir.iterdir():
        if file_path.suffix in (".a", ".la"):
            file_path.unlink(missing_ok=True)

    # Update library linking
    for so_file in lib_dir.glob("librocm_sysdeps_*.so"):
        update_library_links(so_file)

    # Fix .pc files to use relocatable paths.
    pkgconfig_dir = lib_dir / "pkgconfig"
    if pkgconfig_dir.exists():
        for pc_file in pkgconfig_dir.glob("*.pc"):
            relativize_pc_file(pc_file)
            rename_pc_files(pc_file)

elif platform.system() == "Windows":
    # Do nothing for now.
    sys.exit(0)
