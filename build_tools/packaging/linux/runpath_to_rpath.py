#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import os
import pathlib
import re
import subprocess
import sys

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.dynamic import DynamicSection
    from elftools.common.exceptions import ELFError
    from elftools.elf.dynamic import ENUM_D_TAG
except ImportError:
    sys.exit("Error : pyelftools failed to import. Make sure its installed\n")


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
    for path, dirs, files in os.walk(search_path, topdown=True, followlinks=True):
        dirs[:] = [d for d in dirs if d not in excludes]
        for filename in files:
            filename = os.path.join(path, filename)
            print("Opening file ", filename)
            # Open the file and check if its ELF file
            try:
                with open(filename, "rb+") as file:
                    elffile = ELFFile(file)
                    # Find the dynamic section and look for DT_RUNPATH tag
                    section = elffile.get_section_by_name(".dynamic")
                    if not section:
                        break
                    n = 0
                    for tag in section.iter_tags():
                        # DT_RUNPATH tag found. Toggle the byte to DT_RPATH
                        if tag.entry.d_tag == "DT_RUNPATH":
                            offset = section.header.sh_offset + n * section._tagsize
                            section.stream.seek(offset)
                            section.stream.write(bytes([ENUM_D_TAG["DT_RPATH"]]))
                            print("DT_RUNPATH changed to DT_RPATH ")
                            break
                        # DT_RUNPATH tag not found. Loop to the next tag
                        n = n + 1
            except ELFError:
                print("Discarding file as its not an ELF file", filename)
                continue
            except FileNotFoundError:
                print("Discarding file with bad links", filename)
                continue
            except OSError:
                print("Discarding file with OS error", filename)
                continue
            except Exception as ex:
                print("Discarding file ", filename, ex)
                continue


def update_config_file(cfg_path):
    """Update the ROCm LLVM configuration file to default to DT_RPATH.

    This function modifies the specified configuration file so that
    DT_RPATH is used as the default instead of DT_RUNPATH.

    Parameters:
    cfg_path : str
        Path to the ROCm LLVM configuration file.

    Returns: None
    """

    print("Updating cfg file in", cfg_path)
    config_file_exist = os.path.exists(cfg_path)
    if config_file_exist:
        print("cfg file exist in path, going ahead with update ")
        search_str = "enable-new-dtags"
        replace_str = "disable-new-dtags"
        try:
            # Read contents from file as a single string
            file_string = ""
            with open(cfg_path, "r", encoding="utf-8") as f:
                file_string = f.read()

            # Use RE package for string replacement
            file_string = re.sub(search_str, replace_str, file_string)

            # Write contents back to file. Using mode 'w' truncates the file.
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(file_string)
        except Exception as ex:
            print("Couldnt update rocm.cfg file. ", ex)
    else:
        print("Config path doesnt exist", cfg_path)


def update_compiler_config(search_path):
    """Search for the ROCm LLVM configuration file (rocm.cfg) and update it to default to DT_RPATH.

    This function performs the following steps:
    1. Look for the `rocm.cfg` file in the specified `search_path` directory.
    2. If not found, attempt to locate the file in the `ROCM_PATH` environment variable.
    3. Once the configuration file is found, modify its settings so that DT_RPATH is used as the default.

    Parameters:
    search_path : str
        Path to the directory where the function should search for the configuration file.

    Returns:None
    """
    cfg_file_name = "rocm.cfg"
    print("Searching for ", cfg_file_name)
    for path, dirs, files in os.walk(search_path):
        # Search for rocm.cfg in the search path and default to DT_RPATH
        if cfg_file_name in files:
            cfg_path = os.path.join(path, cfg_file_name)
            print(" Found cfg file cfg_path")
            update_config_file(cfg_path)
            return


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
        type=pathlib.Path,
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

    # pyelftools is a mandatory requirement for this script. Exit if requirement is not met
    if "ELFFile" not in globals():
        print(
            "Please install pyelftools using 'pip3 install pyelftools' "
            + "before using the script : runpath_to_rpath.py"
        )
        sys.exit(0)

    # Find the elf files in the search path and update DT_RUNPATH to DT_RPATH
    convert_runpath_to_rpath(args.searchdir)
    # Update rocm clang configs to default to DT_RPATH
    update_compiler_config(args.searchdir)
    print("Done with rpath update")


if __name__ == "__main__":
    main()
