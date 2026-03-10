#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Post-install patching utility for ROCm sysdeps libraries.

Handles the common case of sysdeps libraries that are installed with a
rocm_sysdeps_ prefix and need to be normalized: real file renamed to match
the ELF SONAME, linker-visible symlink created, static/libtool artifacts
removed, and pkg-config files made relocatable.

Usage:
  python patch_sysdeps_install.py <install_prefix> \\
      --normalize librocm_sysdeps_foo.so:libfoo.so \\
      [--normalize librocm_sysdeps_bar.so:libbar.so ...]
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def relativize_pc_file(pc_file: Path) -> None:
    """Make a .pc file relocatable by using pcfiledir-relative paths.

    Replaces the absolute prefix= line with a pcfiledir-relative path,
    then replaces all other occurrences of the absolute prefix with ${prefix}.
    Assumes the .pc file is located at $PREFIX/lib/pkgconfig/.
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
    pc_file.write_text(content)


def update_library_links(
    libfile: Path, linker_name: str, patchelf: str = "patchelf"
) -> None:
    """Normalize a shared library so its real file matches its ELF SONAME.

    Given a library installed as a prefixed name (e.g. librocm_sysdeps_gmp.so),
    renames the underlying real file to the SONAME, creates a linker-visible
    symlink at linker_name, and removes the original prefixed file/symlink.
    """
    if not libfile.exists():
        raise FileNotFoundError(f"File '{libfile}' not found")

    dir_path = libfile.parent

    try:
        lib_soname = subprocess.check_output(
            [patchelf, "--print-soname", str(libfile)],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"patchelf --print-soname failed on '{libfile}' (exit code {e.returncode})"
        ) from e

    if not lib_soname:
        raise RuntimeError(f"No SONAME found in '{libfile}'")

    try:
        realname = libfile.resolve(strict=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"resolve() failed for '{libfile}'") from e

    target_real = dir_path / lib_soname
    if realname != target_real:
        # Move real file to $dir/$soname
        shutil.move(str(realname), str(target_real))

        # Create/update symlink
        symlink_path = dir_path / linker_name
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()
        symlink_path.symlink_to(lib_soname)

        # Remove the original symlink or file
        if libfile.is_symlink() or libfile.exists():
            libfile.unlink()
    else:
        # Rename symlink in the same directory
        new_path = dir_path / linker_name
        if new_path.exists():
            new_path.unlink()
        libfile.rename(new_path)


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("install_prefix", type=Path, help="Installation prefix path")
    p.add_argument(
        "--patchelf",
        default=os.environ.get("PATCHELF", "patchelf"),
        help="patchelf executable (default: $PATCHELF env var or 'patchelf')",
    )
    p.add_argument(
        "--normalize",
        action="append",
        default=[],
        metavar="PREFIXED:LINKER",
        help=(
            "Normalize a library: rename PREFIXED to its ELF SONAME and create a "
            "LINKER symlink. Format: librocm_sysdeps_foo.so:libfoo.so. "
            "May be specified multiple times."
        ),
    )
    args = p.parse_args(argv)

    normalize_pairs: list[tuple[str, str]] = []
    for spec in args.normalize:
        parts = spec.split(":", 1)
        if len(parts) != 2:
            p.error(f"--normalize requires PREFIXED:LINKER format, got: {spec!r}")
        normalize_pairs.append((parts[0], parts[1]))

    install_prefix = args.install_prefix
    patchelf = args.patchelf

    if platform.system() == "Linux":
        lib_dir = install_prefix / "lib"

        # Remove static libs (*.a) and descriptors (*.la).
        for file_path in lib_dir.iterdir():
            if file_path.suffix in (".a", ".la"):
                file_path.unlink(missing_ok=True)

        for prefixed_name, linker_name in normalize_pairs:
            update_library_links(lib_dir / prefixed_name, linker_name, patchelf)

        # Fix .pc files to use relocatable paths.
        pkgconfig_dir = lib_dir / "pkgconfig"
        if pkgconfig_dir.exists():
            for pc_file in pkgconfig_dir.glob("*.pc"):
                relativize_pc_file(pc_file)

    elif platform.system() == "Windows":
        sys.exit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
