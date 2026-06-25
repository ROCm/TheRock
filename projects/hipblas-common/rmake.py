#!/usr/bin/python3
"""Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
   ies of the Software, and to permit persons to whom the Software is furnished
   to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
   PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
   FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
   CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import platform
import subprocess
import argparse
import pathlib

args = {}
OS_info = {}


# yapf: disable
def parse_args():
    """Parse command-line arguments"""
    global OS_info

    parser = argparse.ArgumentParser(description="""Checks build arguments""")
    general_opts = parser.add_argument_group('General Build Options')

    general_opts.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Specify path to configure & build process output directory.(optional, default: ./build)')

    general_opts.add_argument('-i', '--install', required=False, default=False, dest='install', action='store_true',
                        help='Generate and install library package after build. Windows only. Linux use install.sh (optional, default: False)')

    return parser.parse_args()
# yapf: enable

def os_detect():
    global OS_info
    if os.name == "nt":
        OS_info["ID"] = platform.system()
    else:
        inf_file = "/etc/os-release"
        if os.path.exists(inf_file):
            with open(inf_file) as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=")
                        OS_info[k] = v.replace('"', '')

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join(os.getcwd(), dir_path)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path):
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd("RMDIR", f"/S /Q {dir_path}")
    else:
        run_cmd("rm", f"-rf {dir_path}")


def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os.path.realpath(os_path)

def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []
    src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    if os.name == "nt":
        generator = f"-G Ninja"
        cmake_options.append(generator)

        # CMAKE_PREFIX_PATH set to rocm_path and HIP_PATH set BY SDK Installer
        raw_rocm_path = cmake_path(os.getenv('HIP_PATH', "C:/hip"))
        rocm_path = f'"{raw_rocm_path}"' # guard against spaces in path
        # CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append(f"-DCPACK_PACKAGING_INSTALL_PREFIX=")
        cmake_platform_opts.append(f'-DCMAKE_INSTALL_PREFIX="C:/hipSDK"')
    else:
        rocm_raw_path = os.getenv('ROCM_PATH', "/opt/rocm")
        rocm_path = rocm_raw_path
        cmake_platform_opts.append(f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}")

    print(f"Build source path: {src_path}")

    cmake_options.extend(cmake_platform_opts)

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}"
    cmake_options.append(cmake_base_options)

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF"
    cmake_options.append(cmake_pack_options)

    # build type
    cmake_config = ""
    build_path = os.path.realpath(args.build_dir)

    # clean
    delete_dir(build_path)

    create_dir(os.path.join(build_path))
    os.chdir(build_path)

    cmake_options.append(f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts


def make_cmd():
    global args
    global OS_info

    make_options = []

    if os.name == "nt":
        make_executable = f"ninja.exe"
        make_options.append("all")  # for cmake "--target all" )
        if args.install:
            make_options.append("package install")  # for cmake "--target package --target install" )
    else:
        make_executable = f"make package"
        if args.install:
            make_options.append("install")
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts


def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode


def main():
    global args
    os_detect()
    args = parse_args()

    print(OS_info)

    root_dir = os.curdir

    # configure
    exe, opts = config_cmd()
    if run_cmd(exe, opts):
        fatal("Configuration failed. Not continuing.")

    # make
    exe, opts = make_cmd()
    if run_cmd(exe, opts):
        fatal("Build failed. Not continuing.")


if __name__ == '__main__':
    main()
