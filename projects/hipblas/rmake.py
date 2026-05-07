#!/usr/bin/python3
"""Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.

Manage build and installation"""

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from shutil import which
from fnmatch import fnmatchcase

args = {}
param = {}
OS_info = {}

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""Checks build arguments""")

    parser.add_argument(       '--address-sanitizer', dest='address_sanitizer', required=False, default=False,
                        help='Build with address sanitizer enabled. (optional, default: False)')

    parser.add_argument(      '--build_dir', type=str, required=False, default = "build",
                        help='Configure & build process output directory.(optional, default: ./build)')

    parser.add_argument('-c', '--clients', required=False, default = False, dest='build_clients', action='store_true',
                        help='Build the library clients benchmark and gtest (optional, default: False. Generated binaries will be located at builddir/clients/staging)')

    parser.add_argument(      '--clients-only', dest='clients_only', required=False, default=False, action='store_true',
                        help='Skip building the library and only build the clients with a pre-built library.')

    parser.add_argument(     '--clients_no_fortran', required=False, default=False, action='store_true',
                        help='When building clients, build them without Fortran API testing or Fortran examples. (optional, default:False)')

    parser.add_argument(     '--clients_no_blis', required=False, default=False, action='store_true',
                        help='When building clients, build them without using the BLIS host library. (optional, default:False)')

    parser.add_argument(      '--cmake-arg', dest='cmake_args', type=str, required=False, default="",
                        help='Forward the given arguments to CMake when configuring the build.')

    parser.add_argument(      '--cmake-darg', required=False, dest='cmake_dargs', action='append', default=[],
                        help='List of additional cmake defines for builds (optional, e.g. CMAKE)')

    parser.add_argument(      '--cmake_install', required=False, default=False, action='store_true',
                        help='Linux only: Handled by install.sh')

    parser.add_argument(      '--codecoverage', required=False, default=False, action='store_true',
                        help='Code coverage build. Requires Debug (-g|--debug) or RelWithDebInfo mode (-k|--relwithdebinfo), (optional, default: False)')

    parser.add_argument(      '--compiler', type=str, required=False, default=None, dest='compiler',
                        help='Specify path to host compiler. Default is amdclang++.')

    parser.add_argument('--cuda', '--use-cuda', dest='use_cuda', required=False, default=False, action='store_true',
                        help='[DEPRECATED] Build library for CUDA backend. Deprecated, use HIP_PLATFORM environment variable to override default which is determined by `hipconfig --platform`')

    parser.add_argument(      '--cudapath', type=str, required=False, default='/usr/local/cuda', dest='cuda_path',
                        help='Specify path of CUDA install.')

    parser.add_argument(      '--custom-target', type=str, required=False, default='host', dest='custom_target',
                        help="Specify custom target to link the library against (e.g. host, device).",)

    parser.add_argument( '-d', '--dependencies', required=False, default=False, action='store_true',
                        help='Build and install external dependencies. (Handled by install.sh on linux and rdeps.py on Windows')

    parser.add_argument('-g', '--debug', required=False, default = False, action='store_true',
                        help='Build in Debug mode.(optional, default: False)')

    parser.add_argument('-i', '--install', required=False, default = False, dest='install', action='store_true',
                        help='Generate and install library package after build. (optional, default: False)')

    parser.add_argument(      '--installcuda', required=False, default=False, dest='install_cuda',
                        help='Install cuda package.')

    parser.add_argument(      '--installcudaversion', type=str, required=False, default='default', dest='cuda_version_install',
                        help='Used with --installcuda, optionally specify cuda version to install.')

    parser.add_argument(      '--install_invoked', required=False, default=False, action='store_true',
                        help='rmake invoked from install.sh so do not do dependency or package installation (default: False)')

    parser.add_argument('-k', '--relwithdebinfo', required=False, default = False, action='store_true',
                        help='Build in Release with Debug Info (optional, default: False)')

    parser.add_argument(      '--library-path', dest='library_dir_installed', type=str, required=False, default="",
                        help='Specify path to a pre-built hipBLAS library when building with --clients-only flag. (optional, default: /opt/rocm/hipblas)')

    parser.add_argument('-n', '--no-solver', dest='build_solver', required=False, default=True, action='store_false',
                        help='Build hipLBAS library without rocSOLVER dependency')

    parser.add_argument( '-r', '--relocatable', required=False, default=False, action='store_true',
                        help='Linux only: Add RUNPATH (based on ROCM_RPATH) and remove ldconf entry.')

    parser.add_argument('--rocblas-path', dest='rocblas_path', type=str, required=False, default=None,
                        help='Specify path to an existing rocBLAS install directory(optional, e.g. /src/rocBLAS/build/release/rocblas-install).')

    parser.add_argument('--rocsolver-path', dest='rocsolver_path', type=str, required=False, default=None,
                        help='Specify path to an existing rocSOLVER install directory (optional, e.g. /src/rocSOLVER/build/release/rocsolver-install).')

    parser.add_argument(      '--skip_ld_conf_entry', action='store_true', required=False, default = False,
                        help='Linux only: Skip ld.so.conf entry.')

    parser.add_argument('-s', '--static', required=False, default = False, dest='static_lib', action='store_true',
                        help='Build hipblas as a static library.(optional, default: False). hipblas must be built statically when the used companion rocblas is also static')

    parser.add_argument(      '--src_path', type=str, required=False, default="",
                        help='Source path. (optional, default: Current directory)')

    parser.add_argument(      '--hip-clang', dest='use_amdclang_compiler', required=False, default=True, action='store_true',
                        help='[DEPRECATED] Linux only: Build hipBLAS using hipcc compiler. Deprecated, use --compiler instead.')

    parser.add_argument(      '--no-hip-clang', dest='use_amdclang_compiler', required=False, default=False, action='store_false',
                        help='[DEPRECATED] Linux only: Build hipBLAS with g++ compiler instead of hipcc compiler. Deprecated, use --compiler instead.')

    parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
                        help='Verbose build (optional, default: False)')

    return parser.parse_args()

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
                        k,v = line.strip().split("=")
                        OS_info[k] = v.replace('"','')
    OS_info["NUM_PROC"] = os.cpu_count()
    print(OS_info)

def create_dir(dir_path):
    full_path = ""
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        run_cmd( "rm" , f"-rf {dir_path}")

def cmake_path(os_path):
    if os.name == "nt":
        return os_path.replace("\\", "/")
    else:
        return os.path.realpath(os_path)

def fatal(msg, code=1):
    print(msg)
    exit(code)

def get_rocm_path():
    if os.name == "nt":
        raw_rocm_path = cmake_path(os.getenv('HIP_PATH', 'C:/hip'))
        rocm_path = f'"{raw_rocm_path}"' # guard against spaces in path
    else:
        rocm_path = os.getenv('ROCM_PATH', '/opt/rocm')
    return rocm_path

def deps_cmd():
    if os.name == "nt":
        exe = f"python3 rdeps.py"
        all_args = ""
    else:
        exe = f"./install.sh --rmake_invoked -d"
        all_args = ' '.join(sys.argv[1:])
    return exe, all_args

def config_cmd():
    global args
    global OS_info
    cwd_path = os.getcwd()
    cmake_executable = "cmake"
    cmake_options = []
    if len(args.src_path):
        src_path = args.src_path
    else:
        src_path = cmake_path(cwd_path)
    cmake_platform_opts = []
    rocm_path = get_rocm_path()
    if os.name == "nt":
        generator = f"-G Ninja"
        cmake_options.append( generator )
        # CMAKE_PREFIX_PATH set to rocm_path and HIP_PATH set BY SDK Installer
        cmake_executable = "cmake"
        #set CPACK_PACKAGING_INSTALL_PREFIX= defined as blank as it is appended to end of path for archive creation
        cmake_platform_opts.append( f"-DCPACK_PACKAGING_INSTALL_PREFIX=" )
        cmake_platform_opts.append( f"-DCMAKE_INSTALL_PREFIX=\"C:/hipSDK\"" )
        toolchain = os.path.join( src_path, "toolchain-windows.cmake" )
    else:
        if (OS_info["ID"] in ['centos', 'rhel']):
          cmake_executable = "cmake" # was cmake3 but now we built cmake
        else:
          cmake_executable = "cmake"
        cmake_platform_opts.append( f"-DROCM_DIR:PATH={rocm_path} -DCPACK_PACKAGING_INSTALL_PREFIX={rocm_path}" )
        cmake_platform_opts.append( f'-DCMAKE_INSTALL_PREFIX="hipblas-install"' )
        toolchain = "toolchain-linux.cmake"

    print( f"Build source path: {src_path}")

    tools = f"-DCMAKE_TOOLCHAIN_FILE={toolchain}"
    cmake_options.append( tools )

    cmake_options.extend( cmake_platform_opts )

    if args.cmake_args:
        cmake_options.append( args.cmake_args )

    cmake_base_options = f"-DROCM_PATH={rocm_path} -DCMAKE_PREFIX_PATH:PATH={rocm_path}"
    cmake_options.append( cmake_base_options )

    # packaging options
    cmake_pack_options = f"-DCPACK_SET_DESTDIR=OFF"
    cmake_options.append( cmake_pack_options )

    if os.getenv('CMAKE_CXX_COMPILER_LAUNCHER'):
        cmake_options.append( f'-DCMAKE_CXX_COMPILER_LAUNCHER={os.getenv("CMAKE_CXX_COMPILER_LAUNCHER")}' )

    # build type
    cmake_config = ""
    build_dir = os.path.realpath(args.build_dir)
    if args.debug:
        build_path = os.path.join(build_dir, "debug")
        cmake_config="Debug"
    elif args.relwithdebinfo:
        build_path = os.path.join(build_dir, "release-debug")
        cmake_config ="RelWithDebInfo"
    else:
        build_path = os.path.join(build_dir, "release")
        cmake_config="Release"

    cmake_options.append( f"-DCMAKE_BUILD_TYPE={cmake_config}" )

    if args.codecoverage:
        if args.debug or args.relwithdebinfo:
            cmake_options.append( f"-DBUILD_CODE_COVERAGE=ON" )
        else:
            fatal( "*** Code coverage is not supported for Release build! Aborting. ***" )

    if args.address_sanitizer:
        cmake_options.append( f"-DBUILD_ADDRESS_SANITIZER=ON" )

    # clean
    delete_dir( build_path )

    create_dir( os.path.join(build_path, "clients") )
    os.chdir( build_path )

    # compiler
    if args.compiler is not None:
        cmake_options.append(f"-DCMAKE_CXX_COMPILER={args.compiler}")

    # amdclang++ default in linux toolchain, clang++ in windows toolchain
    if os.name != "nt" and args.compiler is None:
        if not args.use_amdclang_compiler:
            cmake_options.append(f"-DCMAKE_CXX_COMPILER=g++")

    if args.static_lib:
        cmake_options.append( f"-DBUILD_SHARED_LIBS=OFF" )

    if args.custom_target:
        cmake_options.append( f"-DCUSTOM_TARGET={args.custom_target}")

    if args.relocatable:
        rocm_rpath = os.getenv( 'ROCM_RPATH', "/opt/rocm/lib:/opt/rocm/lib64")
        cmake_options.append( f'-DCMAKE_SHARED_LINKER_FLAGS=" -Wl,--enable-new-dtags -Wl,--rpath,{rocm_rpath}"' )

    if args.skip_ld_conf_entry or args.relocatable:
        cmake_options.append( f"-DROCM_DISABLE_LDCONFIG=ON" )

    if args.clients_only:
        args.build_clients = True # Implied
        if args.library_dir_installed:
            library_dir = args.library_dir_installed
        else:
            library_dir = f"{rocm_path}"
        cmake_lib_dir = cmake_path(library_dir)
        cmake_options.append(f"-DSKIP_LIBRARY=ON -DHIPBLAS_LIBRARY_DIR={cmake_lib_dir}")

    if args.build_clients:
        cmake_build_dir = cmake_path(build_dir)
        cmake_options.append( f"-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON -DBUILD_CLIENTS_SAMPLES=ON -DBUILD_DIR={cmake_build_dir} " )
        if(args.clients_no_blis):
            cmake_options.append( f"-DLINK_BLIS=OFF")
        else:
            cmake_options.append( f"-DLINK_BLIS=ON")
        if args.clients_no_fortran:
            cmake_options.append(f"-DBUILD_FORTRAN_CLIENTS=OFF")
        if os.name == "nt":
            cmake_options.append( f"-DCREATE_TEST_APP_LOCAL_DEPLOY=ON")


    if args.build_solver:
        cmake_options.append (f"-DBUILD_WITH_SOLVER=ON")
    else:
        cmake_options.append(f"-DBUILD_WITH_SOLVER=OFF")

    if args.rocblas_path is not None:
        # "Custom" rocblas
        raw_rocblas_path = cmake_path(args.rocblas_path)
        rocblas_path_cmake =  f'"{raw_rocblas_path}"'
        cmake_options.append( f"-DCUSTOM_ROCBLAS={rocblas_path_cmake}")

    if args.rocsolver_path is not None:
        # "Custom" rocsolver
        raw_rocsolver_path = cmake_path(args.rocsolver_path)
        rocsolver_path_cmake =  f'"{raw_rocsolver_path}"'
        cmake_options.append( f"-DCUSTOM_ROCSOLVER={rocsolver_path_cmake}")

    if args.cuda_path:
        os.environ['CUDA_BIN_PATH'] = args.cuda_path

    if args.cmake_dargs:
        for i in args.cmake_dargs:
          cmake_options.append( f"-D{i}" )

    cmake_options.append( f"{src_path}")
    cmd_opts = " ".join(cmake_options)

    return cmake_executable, cmd_opts

def make_cmd():
    global args
    global OS_info

    make_options = []

    nproc = OS_info["NUM_PROC"]
    if os.name == "nt":
        make_executable = f"cmake.exe --build . " # ninja
        if args.verbose:
          make_options.append( "--verbose" )
        make_options.append( "--target all" )
        if args.install:
          make_options.append( "--target package --target install" )
    else:
        make_executable = f"make -j{nproc}"
        if args.verbose:
          make_options.append( "VERBOSE=1" )
        if not args.clients_only:
         make_options.append( "install" )
    cmd_opts = " ".join(make_options)

    return make_executable, cmd_opts

def run_cmd(exe, opts):
    program = f"{exe} {opts}"
    print(program)
    proc = subprocess.run(program, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode

def main():
    global args
    print("OS Info:")
    os_detect()
    args = parse_args()

    rocm_path = get_rocm_path()

    # support --cuda flag for now by overwriting HIP_PLATFORM envvar
    if args.use_cuda:
        os.environ['HIP_PLATFORM'] = 'nvidia'
        print("--cuda option is deprecated (use environment variable HIP_PLATFORM=nvidia)")

    # otherwise query hipconfig to get platform
    if which('hipconfig') is not None:
        os.environ['HIP_PLATFORM'] = subprocess.getoutput('hipconfig --platform')
    elif which(f'{rocm_path}/bin/hipconfig') is not None:
        os.environ['HIP_PLATFORM'] = subprocess.getoutput(f'{rocm_path}/bin/hipconfig --platform')
    else:
        os.environ['HIP_PLATFORM'] = 'amd' # if can't find hipconfig, default amd

    if os.environ['HIP_PLATFORM'] == 'nvidia' and args.static_lib:
        fatal("Static library not supported for CUDA backend. Not continuing.")

    if args.install_invoked:
        # ignore any install handled options
        args.dependencies = False
        args.install = False

    if args.dependencies:
        exe, opts = deps_cmd()
        if run_cmd(exe, opts):
            fatal("Dependency install failed. Not continuing.")

    # configure
    exe, opts = config_cmd()
    if run_cmd(exe, opts):
        fatal("Configuration failed. Not continuing.")

    # make
    exe, opts = make_cmd()
    if run_cmd(exe, opts):
        fatal("Build failed. Not continuing.")

    # Linux install and cleanup not supported from rmake yet

if __name__ == '__main__':
    main()
