#!/usr/bin/env python
"""Performs various surgical operations on linux shared libraries."""

import argparse
import glob
from pathlib import Path
import shlex
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
