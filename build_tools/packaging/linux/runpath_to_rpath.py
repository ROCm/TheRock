#!/usr/bin/env python3
################################################################################
##
## MIT License
##
## Copyright (c) 2017 - 2023 Advanced Micro Devices, Inc. All rights Reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

import os
import sys
import subprocess
import argparse
import pathlib
import re

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.dynamic import DynamicSection
    from elftools.common.exceptions import ELFError
    from elftools.elf.dynamic import ENUM_D_TAG
except ImportError:
    print("Error : pyelftools failed to import.\n"
          "Run \'pip3 install pyelftools\' to install the prerequisite\n")

def update_rpath(search_path, excludes) :
    ''' Function helps to change DT_RUNPATH in libraries and binaries in search_path to DT_RPATH.
        Its done with the following steps :
        1. Check all if the file is an ELF except in excludes folder
        2. Find the DT_RUNPATH tag and its offset from file.
        3. Toggle the DT_RUNPATH(0x1d) tag byte to DT_RPATH(0xf) and write back to file '''
    for path, dirs, files in os.walk(search_path, topdown=True, followlinks=True):
        dirs[:] = [d for d in dirs if d not in excludes]
        print( dirs )
        for filename in files:
            filename = os.path.join(path, filename)
            print("Opening file ",  filename)
            # Open the file and check if its ELF file
            try :
                with open(filename, 'rb+') as file:
                    elffile = ELFFile(file)
                    # Find the dynamic section and look for DT_RUNPATH tag
                    section = elffile.get_section_by_name('.dynamic')
                    if not section: break
                    n = 0
                    for tag in section.iter_tags():
                        # DT_RUNPATH tag found. Toggle the byte to DT_RPATH
                        if tag.entry.d_tag == 'DT_RUNPATH':
                            offset = section.header.sh_offset + n* section._tagsize
                            section.stream.seek(offset)
                            section.stream.write(bytes([ENUM_D_TAG['DT_RPATH']])) # DT_PATH
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
    ''' Function helps to update rocm llvm config file to default to DT_RPATH. '''
    print("Updating cfg file in", cfg_path)
    config_file_exist = os.path.exists(cfg_path)
    if config_file_exist:
        print("cfg file exist in path, going ahead with update ")
        search_str = "enable-new-dtags"
        replace_str = "disable-new-dtags"
        try:
            # Read contents from file as a single string
            file_string = ''
            with open(cfg_path, 'r') as f:
                file_string = f.read()

            # Use RE package for string replacement
            file_string = (re.sub(search_str, replace_str, file_string))

            # Write contents back to file. Using mode 'w' truncates the file.
            with open(cfg_path, 'w') as f:
                f.write(file_string)
        except Exception as ex:
            print("Couldnt update rocm.cfg file. ",  ex)
    else:
        print("Config path doesnt exist", cfg_path)

def update_compiler_config(search_path):
    ''' Function search for rocm llvm config(rocm.cfg) file in the search_path folder.
        If the config file is not foung search in ROCM_PATH. Once the config file is found,
        update llvm config to default to DT_RPATH '''
    cfg_file_name = "rocm.cfg"
    found_cfg = False
    print("Searching for ", cfg_file_name)
    for path, dirs, files in os.walk(search_path):
        # Search for rocm.cfg in the search path and default to DT_RPATH
        if cfg_file_name in files:
            cfg_path = os.path.join(path, cfg_file_name)
            print(" Found cfg file cfg_path")
            found_cfg = True
            update_config_file(cfg_path)
            # Continue with the search as there could be cfg files in llvm and llvm/alt
            continue;
    if found_cfg:
        return
    # rocm.cfg config file not found in search path. Search in the ROCM_PATH.
    print(cfg_file_name, " not found in search_path. Trying to search in ROCM_PATH")
    try :
        rocm_path = os.environ["ROCM_PATH"]
        print(" Found ROCM_PATH trying for rocm.cfg")
        # There are multiple possible paths for cfg file.
        # ROCM_PATH/llvm/bin and ROCM_PATH/lib/llvm/bin. Also alt location
        update_config_file(rocm_path + "/llvm/bin/" + cfg_file_name)
        update_config_file(rocm_path + "/llvm/alt/bin/" + cfg_file_name)
        update_config_file(rocm_path + "/lib/llvm/bin/" + cfg_file_name)
        update_config_file(rocm_path + "/lib/llvm/alt/bin/" + cfg_file_name)
            # Found config file. Change default DT_RUNPATH setting to DT_RPATH
    except Exception as ex:
        print("ROCM_PATH not found ", ex)

def main():
    # The script expect a search folder as parameter. It finds all ELF files and updates RPATH
    argparser = argparse.ArgumentParser(
            usage='usage: %(prog)s  <folder-to-search>',
            description='Find the ELF files in the specified folder and convert the RUNPATH to RPATH. \n',
            add_help=False,
            prog='runpath_to_rpath.py')

    argparser.add_argument('searchdir',
            nargs='?', type=pathlib.Path, default=None,
            help='Folder to search for ELF file. \nPlease note: Any folder with name llvm in that path will be discarded')
    argparser.add_argument('-h', '--help',
            action='store_true', dest='help',
            help='Display this information')

    args = argparser.parse_args()
    if args.help or not args.searchdir:
        argparser.print_help()
        sys.exit(0)

    # pyelftools is a mandatory requirement for this script. Exit if requirement is not met
    if 'ELFFile' not in globals():
        print('Please install pyelftools using \'pip3 install pyelftools\' ' +
              'before using the script : runpath_to_rpath.py')
        sys.exit(0)

    # Find the elf files in the serach path and update DT_RUNPATH to DT_RPATH
    # SWDEV-467155 : remove the exclusion of llvm folder
    excludes = []
    update_rpath(args.searchdir, excludes)
    # Update rocm clang configs to default to DT_RPATH
    update_compiler_config(args.searchdir)
    print("Done with rpath update")

if __name__ == "__main__":
    main()
