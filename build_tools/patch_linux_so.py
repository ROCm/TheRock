#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Performs various surgical operations on linux shared libraries."""

import argparse
import glob
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


def run_command(args: list[str | Path], cwd: Path):
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def capture(args: list[str | Path], cwd: Path) -> str:
    args = [str(arg) for arg in args]
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    return subprocess.check_output(
        args, cwd=str(cwd), stdin=subprocess.DEVNULL
    ).decode()


def resolve_symlinks(lib_path: Path):
    all_paths: list[Path] = [lib_path]
    all_paths.extend([Path(p) for p in glob.glob(f"{str(lib_path)}.*")])
    return all_paths


def update_library_links(
    libfile: Path, linker_name: str, patchelf: str | None = None
) -> None:
    """
    Normalize a shared library so that its real file is named exactly as its ELF SONAME,
    and ensure a canonical linker-visible symlink exists.

    Deprecated: new callers should use `update_sysdeps_library_links`. Once all
    sysdeps patch_install scripts have moved to it, this function will be folded
    into `update_sysdeps_library_links`, which will be the only entry point.

    This function is used when a library has been installed under a prefixed or
    non-standard filename (e.g., librocm_sysdeps_elf.so).
    It performs the following operations:
    - Extracts the library's SONAME using `patchelf --print-soname`.
    - Resolves the underlying real file (following symlinks).
    - Renames the real file to match its SONAME if it does not already.
    - Creates or updates a symlink named `linker_name` pointing to the SONAME file.
    - Removes or renames the original file or symlink as appropriate.

    Final layout example:
    - libhwloc.so → librocm_sysdeps_hwloc.so.5
    - librocm_sysdeps_hwloc.so.5 (real file)

    Parameters
    ----------
    libfile : Path
        Path to the library file or symlink to normalize.
        Example: /prefix/lib/librocm_sysdeps_elf.so

    linker_name : str
        The desired linker-visible filename to create in the same directory.
        Example: libelf.so

    patchelf : str, optional
        Path to the `patchelf` executable used to extract the SONAME.
        Defaults to the PATCHELF environment variable (set by the build), or
        "patchelf" from PATH.
    """
    if patchelf is None:
        patchelf = os.environ.get("PATCHELF", "patchelf")
    if not libfile.exists():
        raise FileNotFoundError(f"File '{libfile}' not found")

    dir_path = libfile.parent

    # Get SONAME
    try:
        lib_soname = subprocess.check_output(
            [patchelf, "--print-soname", str(libfile)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"patchelf --print-soname failed for '{libfile}'") from e

    if not lib_soname:
        raise RuntimeError(f"Empty SONAME returned by patchelf for '{libfile}'")

    # Resolve real file path
    try:
        realname = libfile.resolve(strict=True)
    except FileNotFoundError:
        print(f"Error: resolve() failed for '{libfile}'", flush=True)
        return

    target_real = dir_path / lib_soname
    symlink_path = dir_path / linker_name

    if realname != target_real:
        # Move real file to $dir/$soname
        shutil.move(str(realname), str(target_real))

        # Create/update linker symlink
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(lib_soname)

        # Remove the original symlink or file
        if libfile.is_symlink() or libfile.exists():
            libfile.unlink()
    else:
        # Rename symlink in the same directory
        if symlink_path.exists():
            symlink_path.unlink()
        libfile.rename(symlink_path)


def update_sysdeps_library_links(
    lib_dir: Path, linker_name: str, sysdeps_name: str | None = None
) -> None:
    """Normalizes a sysdeps library given its original linker name.

    This is the entry point for the sysdeps patch_install scripts. It currently
    wraps `update_library_links`, which is deprecated and will be folded into
    this function once all callers have moved here.

    The installed (prefixed) library file is derived from the linker name, e.g.
    `libcap.so` -> `librocm_sysdeps_cap.so`. Results in:
    - libcap.so -> librocm_sysdeps_cap.so.2
    - librocm_sysdeps_cap.so.2 (real file)

    Parameters
    ----------
    lib_dir : Path
        Directory containing the installed library.

    linker_name : str
        The original linker-visible filename to create (e.g. libcap.so).

    sysdeps_name : str, optional
        The installed library filename, if it is not the linker name with the
        `lib` prefix replaced by `librocm_sysdeps_` (e.g. libnl-genl-3.so is
        installed as librocm_sysdeps_nl_genl_3.so).
    """
    if sysdeps_name is None:
        sysdeps_name = "librocm_sysdeps_" + linker_name.removeprefix("lib")
    update_library_links(lib_dir / sysdeps_name, linker_name)


def relativize_pc_file(pc_file: Path) -> None:
    """Make a .pc file relocatable by using pcfiledir-relative paths.

    Replaces the absolute prefix= line with a pcfiledir-relative path,
    then replaces all other occurrences of the absolute prefix with ${prefix}.
    Also removes absolute -L flags from Libs.private that may have leaked from
    build-time LDFLAGS.
    Assumes the .pc file is located at $PREFIX/lib/pkgconfig/.

    Parameters
    ----------
    pc_file : Path
        Path to the .pc file to make relocatable.
    """

    content = pc_file.read_text()

    # Find the original absolute prefix value.
    original_prefix = None
    for line in content.splitlines():
        if line.startswith("prefix="):
            original_prefix = line[len("prefix=") :]
            break

    if not original_prefix:
        return

    # Replace the prefix line with pcfiledir-relative path.
    # .pc files are in $PREFIX/lib/pkgconfig, so go up 2 levels.
    content = content.replace(f"prefix={original_prefix}", "prefix=${pcfiledir}/../..")
    # Replace all other occurrences of the absolute path with ${prefix}.
    # Use trailing / to avoid partial matches.
    content = content.replace(f"{original_prefix}/", "${prefix}/")

    # Remove absolute -L paths from Libs.private (these leak from build-time LDFLAGS)
    # Match patterns like: -L/absolute/path/to/lib
    content = re.sub(r"-L/[^\s]+", "", content)

    pc_file.write_text(content)


def add_prefix(args: argparse.Namespace):
    all_libs: list[Path] = args.so_files
    updated_libs: list[Path] = []
    soname_updates: dict[str, str] = {}

    # First update the SONAME of all requested libraries. This presumes that
    # the libraries are in typical symlink form.
    for lib_path in all_libs:
        orig_paths = resolve_symlinks(lib_path)
        lib_path_canon = lib_path.resolve()
        orig_soname = capture(
            [args.patchelf, "--print-soname", str(lib_path)], cwd=Path.cwd()
        ).strip()
        soname_prefix = ""
        soname_stem = orig_soname
        if orig_soname.startswith("lib"):
            soname_prefix = "lib"
            soname_stem = orig_soname[len("lib") :]
        new_soname = f"{soname_prefix}{args.add_prefix}{soname_stem}"
        new_lib_path = lib_path.parent / f"{new_soname}"
        if new_lib_path.exists():
            new_lib_path.unlink()
        print(f"Prefixing SONAME {orig_soname} -> {new_soname} for {lib_path_canon}")
        lib_path_canon.rename(new_lib_path)
        run_command(
            [
                args.patchelf,
                "--set-soname",
                new_soname,
                new_lib_path,
            ],
            cwd=Path.cwd(),
        )
        updated_libs.append(new_lib_path)
        soname_updates[orig_soname] = new_soname

        # Remove old links.
        for orig_path in orig_paths:
            if orig_path.is_symlink():
                print(f"Removing original link: {orig_path}")
                orig_path.unlink()

        # Establish new dev symlink.
        lib_path.symlink_to(new_lib_path.name)

    # Now go back and replace updated sonames.
    for soname_from, soname_to in soname_updates.items():
        for updated_lib in updated_libs:
            run_command(
                [
                    args.patchelf,
                    "--replace-needed",
                    soname_from,
                    soname_to,
                    updated_lib,
                ],
                cwd=Path.cwd(),
            )


def run(args: argparse.Namespace):
    if args.add_prefix:
        add_prefix(args)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--patchelf", default="patchelf", help="Patchelf command")
    p.add_argument(
        "so_files", type=Path, nargs="*", help="Shared library files to patch"
    )
    p.add_argument(
        "--add-prefix",
        help="Add a prefix to all shared libraries (and update all of their "
        "DT_NEEDED to match)",
    )
    args = p.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
