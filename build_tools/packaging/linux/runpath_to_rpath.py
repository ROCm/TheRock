#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _get_rpath(filepath: Path):
    """Read DT_RPATH or DT_RUNPATH from an ELF binary via readelf.

    patchelf --print-rpath only reads DT_RPATH, not DT_RUNPATH, so using it
    directly on a binary that has only DT_RUNPATH returns an empty string and
    would cause --set-rpath to wipe the rpath. readelf reads both tags.
    """
    try:
        out = subprocess.check_output(
            ["readelf", "-d", str(filepath)],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return None
    m = re.search(r"\(R(?:UN)?PATH\)\s+Library r(?:un)?path: \[(.+)\]", out)
    return m.group(1) if m else None


def update_rpath(search_path: Path, excludes):
    """Change DT_RUNPATH to DT_RPATH in all ELF files under search_path.

    Uses patchelf --force-rpath --set-rpath, which is the same mechanism used
    by the Python wheel packaging path. readelf is used to read the existing
    rpath value because patchelf --print-rpath only reads DT_RPATH, not
    DT_RUNPATH, and would return empty for binaries that only have DT_RUNPATH.
    """
    for path, dirs, files in os.walk(search_path, topdown=True, followlinks=True):
        dirs[:] = [d for d in dirs if d not in excludes]
        for filename in files:
            filepath = Path(path) / filename
            if filepath.is_symlink():
                continue
            # Quick ELF magic check before invoking readelf/patchelf
            try:
                if filepath.read_bytes()[:4] != b"\x7fELF":
                    continue
            except OSError:
                continue
            rpath = _get_rpath(filepath)
            if not rpath:
                continue
            # Write the rpath back, forcing DT_RPATH tag instead of DT_RUNPATH
            try:
                subprocess.check_call(
                    [
                        "patchelf",
                        "--force-rpath",
                        "--set-rpath",
                        rpath,
                        str(filepath),
                    ]
                )
                print(f"DT_RUNPATH changed to DT_RPATH: {filepath}")
            except subprocess.CalledProcessError as ex:
                print(f"patchelf failed for {filepath}: {ex}")


def update_config_file(cfg_path: Path):
    """Update rocm llvm config file to default to DT_RPATH."""
    print("Updating cfg file in", cfg_path)
    if cfg_path.exists():
        print("cfg file exist in path, going ahead with update ")
        try:
            file_string = cfg_path.read_text(encoding="utf-8")
            file_string = re.sub("enable-new-dtags", "disable-new-dtags", file_string)
            cfg_path.write_text(file_string, encoding="utf-8")
        except Exception as ex:
            print("Couldnt update rocm.cfg file. ", ex)
    else:
        print("Config path doesnt exist", cfg_path)


def update_compiler_config(search_path: Path):
    """Search for rocm.cfg in search_path and update it to default to DT_RPATH."""
    cfg_file_name = "rocm.cfg"
    found_cfg = False
    print("Searching for ", cfg_file_name)
    for path, _, files in os.walk(search_path):
        if cfg_file_name in files:
            cfg_path = Path(path) / cfg_file_name
            print(" Found cfg file cfg_path")
            found_cfg = True
            update_config_file(cfg_path)
            # Continue with the search as there could be cfg files in llvm and llvm/alt
    if found_cfg:
        return
    # rocm.cfg config file not found in search path. Search in the ROCM_PATH.
    print(cfg_file_name, " not found in search_path. Trying to search in ROCM_PATH")
    try:
        rocm_path = Path(os.environ["ROCM_PATH"])
        print(" Found ROCM_PATH trying for rocm.cfg")
        # There are multiple possible paths for cfg file.
        # ROCM_PATH/llvm/bin and ROCM_PATH/lib/llvm/bin. Also alt location
        update_config_file(rocm_path / "llvm" / "bin" / cfg_file_name)
        update_config_file(rocm_path / "llvm" / "alt" / "bin" / cfg_file_name)
        update_config_file(rocm_path / "lib" / "llvm" / "bin" / cfg_file_name)
        update_config_file(rocm_path / "lib" / "llvm" / "alt" / "bin" / cfg_file_name)
    except Exception as ex:
        print("ROCM_PATH not found ", ex)


def convert_runpath_to_rpath(search_path):
    """Update ELF binaries and libraries in `search_path` by changing DT_RUNPATH to DT_RPATH.

    This function performs the following steps:
    1. Iterate through all files in `search_path`, skipping any directories listed in `excludes`.
    2. Check if each file is an ELF binary.
    3. Locate the DT_RUNPATH tag and its offset within the file.
    4. Modify the tag byte from DT_RUNPATH (0x1d) to DT_RPATH (0xf) and write the change back.

    Parameters:
    search_path : str
        The root directory to search for ELF files.
    excludes : list[str]
        A list of directory names to exclude from processing.

    Returns: None
    """
    print(f"convert_runpath_to_rpath {search_path}")
    excludes = []
    # Find the elf files in the search path and update DT_RUNPATH to DT_RPATH
    update_rpath(search_path, excludes)
    # Update rocm clang configs to default to DT_RPATH
    update_compiler_config(search_path)
    print("Done with rpath update")


def main():
    # The script expect a search folder as parameter. It finds all ELF files and updates RPATH
    argparser = argparse.ArgumentParser(
        usage="usage: %(prog)s  <folder-to-search>",
        description="Find the ELF files in the specified folder and convert the RUNPATH to RPATH. \n",
        add_help=False,
        prog="runpath_to_rpath.py",
    )

    argparser.add_argument(
        "searchdir",
        nargs="?",
        type=Path,
        default=None,
        help="Folder to search for ELF file. \nPlease note: Any folder with name llvm in that path will be discarded",
    )
    argparser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="help",
        help="Display this information",
    )

    args = argparser.parse_args()
    if args.help or not args.searchdir:
        argparser.print_help()
        sys.exit(0)

    convert_runpath_to_rpath(args.searchdir)


if __name__ == "__main__":
    main()
