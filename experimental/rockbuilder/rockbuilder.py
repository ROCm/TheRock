#!/usr/bin/env python

import argparse
import configparser
import sys
import os
import time
import platform
import subprocess
import lib_python.project_builder as project_builder
import lib_python.rockbuilder_config as RockBuilderConfig
from pathlib import Path, PurePosixPath

ROCK_BUILDER_VERSION = "2025-06-25_01"


def get_rocm_builder_root_dir():
    current_file_path = os.path.abspath(__file__)
    ret = Path(os.path.dirname(current_file_path)).resolve()
    return ret


def printout_rock_builder_info():
    print("RockBuilder " + ROCK_BUILDER_VERSION)
    print("")
    print("ROCK_BUILDER_HOME_DIR: " + os.environ["ROCK_BUILDER_HOME_DIR"])
    print("ROCK_BUILDER_SRC_DIR: " + os.environ["ROCK_BUILDER_SRC_DIR"])
    print("ROCK_BUILDER_BUILD_DIR: " + os.environ["ROCK_BUILDER_BUILD_DIR"])


def printout_build_env_info():
    printout_rock_builder_info()
    print("Build environment:")
    print("-----------------------------")
    print("ROCM_HOME: " + os.environ["ROCM_HOME"])
    print("ROCK_PYTHON_PATH: " + os.environ["ROCK_PYTHON_PATH"])
    print("PATH: " + os.environ["PATH"])
    print("-----------------------------")
    time.sleep(1)


# This argument parser is used to parse the project list.
# This is needed to do first before parsing other arguments
# because we need to add the --<project-name>-version arguments for the second parser that
# is used to parse all possible parameters.
def get_project_list_manager(rock_builder_home_dir: Path):
    parser = argparse.ArgumentParser(description="Project and Project List Parser")

    # Add arguments
    parser.add_argument(
        "--project_list",
        type=str,
        help="select project list for the actions.",
        default=None,
    )
    parser.add_argument(
        "--project",
        type=str,
        help="select target for the action. Must be one project from projects-directory.",
        default=None,
    )
    prj = None
    prj_list = None
    args, unknown = parser.parse_known_args()
    ret = project_builder.RockExternalProjectListManager(
        rock_builder_home_dir, args.project_list, args.project
    )
    return ret


# create user parameter parser
def create_build_argument_parser(
    rock_builder_home_dir, default_src_base_dir: Path, project_list
):
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="ROCK Project Builders")

    # Add arguments
    parser.add_argument(
        "--project_list",
        type=str,
        help="select project list for the actions.",
        default=None,
    )
    parser.add_argument(
        "--project",
        type=str,
        help="select target for the action. Must be one project from projects-directory.",
        default=None,
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="init build environment by installing dependencies",
        default=False,
    )
    parser.add_argument(
        "--clean", action="store_true", help="clean build files", default=False
    )
    parser.add_argument(
        "--checkout",
        action="store_true",
        help="checkout source code for the project",
        default=False,
    )
    parser.add_argument(
        "--hipify",
        action="store_true",
        help="hipify command for project",
        default=False,
    )
    parser.add_argument(
        "--pre_config",
        action="store_true",
        help="pre-config command for project",
        default=False,
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="config command for project",
        default=False,
    )
    parser.add_argument(
        "--post_config",
        action="store_true",
        help="post-config command for project",
        default=False,
    )
    parser.add_argument(
        "--build", action="store_true", help="build project", default=False
    )
    parser.add_argument(
        "--install", action="store_true", help="install build project", default=False
    )
    parser.add_argument(
        "--post_install",
        action="store_true",
        help="post-install command for project",
        default=False,
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        help="Directory where to checkout single project source code. Can only be used with the --project parameter.",
        default=None,
    )
    parser.add_argument(
        "--src-base-dir",
        type=Path,
        help="Base directory where each projects source code is checked out. Default is src_projects.",
        default=default_src_base_dir,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to copy built wheels to",
        default=rock_builder_home_dir / "packages" / "wheels",
    )
    for ii, prj_item in enumerate(project_list):
        arg_name = "--" + prj_item + "-version"
        print("arg_name: " + arg_name)
        parser.add_argument(
            "--" + prj_item + "-version",
            help=prj_item + " version used for the operations",
            default=None,
        )
    return parser


def parse_build_arguments(parser):
    # parse command line parameters
    args = parser.parse_args()
    if (
        ("--checkout" in sys.argv)
        or ("--clean" in sys.argv)
        or ("--init" in sys.argv)
        or ("--hipify" in sys.argv)
        or ("--pre_config" in sys.argv)
        or ("--config" in sys.argv)
        or ("--post_config" in sys.argv)
        or ("--build" in sys.argv)
        or ("--install" in sys.argv)
        or ("--post_install" in sys.argv)
    ):
        print(
            "checkout/init/clean/hipify/pre_config/config/post_config/build/install/post_install argument specified"
        )
    else:
        # print("Action not specified.(checkout/init/clean/hipify/pre_config/config/post_config/build/install/post_install)")
        # print("Using default values")
        # enable everything except clean
        args.checkout = True
        args.init = True
        args.clean = False
        args.hipify = True
        args.pre_config = True
        args.config = True
        args.post_config = True
        args.build = True
        args.install = True
        args.post_install = True
    # force the hipify step always as a part of the checkout
    if args.checkout:
        args.hipify = True

    # add output dir to environment variables
    if args.project and args.src_dir:
        # single project case with optional src_dir specified
        parent_dir = args.src_dir.parent
        if parent_dir == args.src_dir:
            print("Error, --src-dir parameter is not allowed to be a root-directory")
            sys.exit(1)
        os.environ["ROCK_BUILDER_SRC_DIR"] = parent_dir.as_posix()
    else:
        # directory where each projects source code is checked out
        os.environ["ROCK_BUILDER_SRC_DIR"] = args.src_base_dir.as_posix()
    os.environ["ROCK_BUILDER_PACKAGE_OUTPUT_DIR"] = args.output_dir.as_posix()

    return args


def printout_build_arguments(args):
    # printout arguments enabled
    print("Actions Enabled:")
    print("    checkout: ", args.checkout)
    print("    init:    ", args.init)
    print("    clean:    ", args.clean)
    print("    hipify:    ", args.hipify)
    print("    pre_config:", args.pre_config)
    print("    config:", args.config)
    print("    post_config:", args.post_config)
    print("    build:    ", args.build)
    print("    install:  ", args.install)
    print("    post_install:  ", args.post_install)
    print("Projects:", args.project)


def verify_build_env__python(self):
    # check the python used. It needs to be by default an virtual env but
    # this can be overriden to real python version by setting up ENV variable
    # ROCK_PYTHON_PATH
    python_home_dir = os.path.dirname(sys.executable)
    if "VIRTUAL_ENV" in os.environ:
        os.environ["ROCK_PYTHON_PATH"] = python_home_dir
    else:
        if "ROCK_PYTHON_PATH" in os.environ:
            if not os.path.abspath(python_home_dir) == os.path.abspath(
                os.environ["ROCK_PYTHON_PATH"]
            ):
                print("Error, virtual python environment is not active and")
                print(
                    "PYTHON location is different than specified by the ROCK_PYTHON_PATH"
                )
                print("    PYTHON location: " + python_home_dir)
                print("    ROCK_PYTHON_PATH: " + os.environ["ROCK_PYTHON_PATH"])
                print(
                    "If you want use this python location instead of using a virtual python env, define ROCK_PYTHON_PATH:"
                )
                if is_posix:
                    print("    export ROCK_PYTHON_PATH=" + python_home_dir)
                else:
                    print("    set ROCK_PYTHON_PATH=" + python_home_dir)
                print("Alternatively activate the virtual python environment")
                sys.exit(1)
            else:
                print("Using python from location: " + python_home_dir)
        else:
            print(
                "Error, virtual python environment is not active and ROCK_PYTHON_PATH is not defined"
            )
            print("    PYTHON location: " + python_home_dir)
            print(
                "If you want use this python location instead of using a virtual python env, define ROCK_PYTHON_PATH:"
            )
            if is_posix:
                print("    export ROCK_PYTHON_PATH=" + python_home_dir)
            else:
                print("    set ROCK_PYTHON_PATH=" + python_home_dir)
            print("Alternatively activate the virtual python environment")
            sys.exit(1)


def get_rocm_sdk_targets_on_linux(exec_dir: Path):
    ret = ""
    if exec_dir is not None:
        exec_cmd = "rocm_agent_enumerator"
        print("exec_dir: " + str(exec_dir))
        print("exec_cmd: " + exec_cmd)
        result = subprocess.run(
            exec_cmd, cwd=exec_dir, shell=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout)
                gpu_list = result.stdout.splitlines()
                # remove duplicates in case there are multiple instances of same gpu
                gpu_list = list(set(gpu_list))
                for ii, val in enumerate(gpu_list):
                    if ii == 0:
                        ret = val
                    else:
                        ret = ret + ";" + val
        else:
            ret = False
            print(result.stdout)
            print(
                f"Failed use to rocm_agent_enumerator to get gpu list: {result.stderr}"
            )
            sys.exit(1)
    return ret


def verify_build_env(args, rock_builder_home_dir: Path, rock_builder_build_dir: Path):
    # we may actually need to have the build environment
    # even during the checkout as we do the hipify during the checkout process
    """"
    if ((args.checkout == True) and\
        (args.init == False) and\
        (args.clean == False) and\
        (args.hipify == False) and\
        (args.pre_config == False) and\
        (args.config == False) and\
        (args.post_config == False) and\
        (args.build == False) and\
        (args.install == False) and\
        (args.post_install == False)):
        # allow code checkout even before TheRock build has been installed
        printout_rock_builder_info()
        return
    """

    # set the ENV_VARIABLE_NAME__LIB to be either LD_LIBRARY_PATH or LIBPATH depending
    # whether code is executed on Linux or Windows (it is later used to set env-variables)
    if is_posix:
        ENV_VARIABLE_NAME__LIB = "LD_LIBRARY_PATH"
    else:
        ENV_VARIABLE_NAME__LIB = "LIBPATH"

    # read and set up env based on to rockbuilder.ini file if it exist
    rcb_config = RockBuilderConfig.RockBuilderConfig(
        rock_builder_home_dir, rock_builder_build_dir
    )
    res = rcb_config.read_cfg()
    if res:
        rcb_config.setup_build_env()

    # check rocm
    if "ROCM_HOME" in os.environ:
        rocm_home_root_path = Path(os.environ["ROCM_HOME"])
        rocm_home_root_path = rocm_home_root_path.resolve()
    else:
        rocm_home_root_path = rock_builder_home_dir / "../../build/dist/rocm"
        rocm_home_root_path = rocm_home_root_path.resolve()
        print("using locally build rocm sdk from TheRock:")
        print("    " + rocm_home_root_path.as_posix())
    if rocm_home_root_path.exists():
        rocm_home_bin_path = rocm_home_root_path / "bin"
        rocm_home_lib_path = rocm_home_root_path / "lib"
        rocm_home_bin_path = rocm_home_bin_path.resolve()
        rocm_home_lib_path = rocm_home_lib_path.resolve()
        rocm_home_llvm_path = rocm_home_root_path / "lib" / "llvm" / "bin"
        rocm_home_llvm_path = rocm_home_llvm_path.resolve()
        if rocm_home_bin_path.exists() and rocm_home_lib_path.exists():
            # set ROCM_HOME if not yet set
            if not "ROCM_HOME" in os.environ:
                # print("ROCM_HOME: " + rocm_home_root_path.as_posix())
                os.environ["ROCM_HOME"] = rocm_home_root_path.as_posix()
            # set ROCM_PATH to always point to same location than ROCM_HOME
            # ROCM_PATH is used by some ROCM applications instead of ROCM_HOME
            os.environ["ROCM_PATH"] = rocm_home_root_path.as_posix()
            if not is_directory_in_env_variable_path(
                "PATH", rocm_home_bin_path.as_posix()
            ):
                # print("Adding " + rocm_home_bin_path.as_posix() + " to PATH")
                os.environ["PATH"] = (
                    rocm_home_bin_path.as_posix()
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
            if not is_directory_in_env_variable_path(
                "PATH", rocm_home_llvm_path.as_posix()
            ):
                # print("Adding " + rocm_home_llvm_path.as_posix() + " to PATH")
                os.environ["PATH"] = (
                    rocm_home_llvm_path.as_posix()
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )
            if not is_directory_in_env_variable_path(
                ENV_VARIABLE_NAME__LIB, rocm_home_lib_path.as_posix()
            ):
                # print("Adding " + rocm_home_lib_path.as_posix() + " to " + ENV_VARIABLE_NAME__LIB)
                os.environ[ENV_VARIABLE_NAME__LIB] = (
                    rocm_home_lib_path.as_posix()
                    + os.pathsep
                    + os.environ.get(ENV_VARIABLE_NAME__LIB, "")
                )
            # find bitcode and put it to path
            for folder_path in Path(rocm_home_root_path).glob("**/bitcode"):
                folder_path = folder_path.resolve()
                os.environ["ROCK_BUILDER_BITCODE_HOME"] = folder_path.as_posix()
                break
            # find hipcc
            if is_posix:
                hipcc_exec_name = "hipcc"
            else:
                hipcc_exec_name = "hipcc.bat"
            for folder_path in Path(rocm_home_root_path).glob("**/" + hipcc_exec_name):
                hipcc_home = folder_path.parent
                # make sure that we found bin/clang and not clang folder
                if hipcc_home.name.lower() == "bin":
                    hipcc_home = hipcc_home.parent
                    if hipcc_home.is_dir():
                        hipcc_home = hipcc_home.resolve()
                        os.environ["ROCK_BUILDER_HIPCC_HOME"] = hipcc_home.as_posix()
                        break
            # find clang
            if is_posix:
                clang_exec_name = "clang"
            else:
                clang_exec_name = "clang.exe"
            for folder_path in Path(rocm_home_root_path).glob("**/" + clang_exec_name):
                clang_home = folder_path.parent
                # make sure that we found bin/clang and not clang folder
                if clang_home.name.lower() == "bin":
                    clang_home = clang_home.parent
                    if clang_home.is_dir():
                        clang_home = clang_home.resolve()
                        os.environ["ROCK_BUILDER_CLANG_HOME"] = clang_home.as_posix()
                        break
            # check that THEROCK_AMDGPU_TARGETS environment variable is set.
            # If not:
            #   - Linux: check the gpus available and assign them to THEROCK_AMDGPU_TARGETS
            #   - Windows: exit on error, because it can not be queried automatically
            if not "THEROCK_AMDGPU_TARGETS" in os.environ:
                if is_posix:
                    gpu_targets = get_rocm_sdk_targets_on_linux(rocm_home_bin_path)
                    os.environ["THEROCK_AMDGPU_TARGETS"] = gpu_targets
                    print("gpu_targets: " + gpu_targets)
                else:
                    print(
                        "Error, THEROCK_AMDGPU_TARGETS must be set on Windows to select the target GPUs"
                    )
                    print(
                        "Target GPU must match with the GPU selected on TheRock core build"
                    )
                    print("Example for building for AMD Strix Halo and RX 9070:")
                    print("  set THEROCK_AMDGPU_TARGETS=gfx1151;gfx1201")
                    sys.exit(1)
        else:
            print(
                "Error, could not find directory ROCM_SDK/lib: "
                + rocm_home_lib_path.as_posix()
            )
            sys.exit(1)
    else:
        print("Error, ROCM_HOME is not defined and")
        print("ROCM build is not detected in the RockBuilder build environment:")
        print("    " + rocm_home_root_path.as_posix())
        print("")
        sys.exit(1)

    printout_build_env_info()


# check whether specified directory is included in the specified environment variable
def is_directory_in_env_variable_path(env_variable, directory):
    """
    Checks if a directory is in the env_variable specified as a parameter.
    (path, libpath, etc)

    Args:
      env_variable: Environment variable used to check the path
      directory: The path searched from the environment variable

    Returns:
      True if the directory is in PATH, False otherwise.
    """
    path_env = os.environ.get(env_variable, "")
    path_directories = path_env.split(os.pathsep)
    return directory in path_directories


# do all build steps for given process
def do_therock(prj_builder):
    ret = False
    if prj_builder is not None:
        if prj_builder.check_skip_on_os() == False:
            # setup first the project specific environment variables
            prj_builder.printout("start")
            prj_builder.do_env_setup()
            # print("do_env_setup done")

            # then do all possible commands requested for the project
            # multiple steps possible, so do not use else's here
            if args.clean:
                prj_builder.printout("clean")
                prj_builder.clean()
            if args.checkout:
                prj_builder.printout("checkout")
                prj_builder.checkout()
                # enable hipify always when doing the code checkout
                # even if it is not requested explicitly to be it's own command
                args.hipify = True
            if args.init:
                prj_builder.printout("init")
                prj_builder.init()
            if args.hipify:
                prj_builder.printout("hipify")
                prj_builder.hipify()
            if args.pre_config:
                prj_builder.printout("pre_config")
                prj_builder.pre_config()
            if args.config:
                prj_builder.printout("config")
                prj_builder.config()
            if args.post_config:
                prj_builder.printout("post_config")
                prj_builder.post_config()
            if args.build:
                prj_builder.printout("build")
                prj_builder.build()
            if args.install:
                prj_builder.printout("install")
                prj_builder.install()
            if args.post_install:
                prj_builder.printout("post_install")
                prj_builder.post_install()
            # in the end restore original environment variables
            # so that they do not cause problem for next possible project handled
            prj_builder.undo_env_setup()
            prj_builder.printout("done")
            ret = True
        else:
            print("skip_windows or skip_linux enabled for project")
            prj_builder.printout("skip")
            ret = True
    return ret


is_posix = not any(platform.win32_ver())

rock_builder_home_dir = get_rocm_builder_root_dir()
rock_builder_build_dir = rock_builder_home_dir / "builddir"
default_src_base_dir = rock_builder_home_dir / "src_projects"

os.environ["ROCK_BUILDER_HOME_DIR"] = rock_builder_home_dir.as_posix()
os.environ["ROCK_BUILDER_BUILD_DIR"] = rock_builder_build_dir.as_posix()

project_manager = get_project_list_manager(rock_builder_home_dir)
project_list = project_manager.get_external_project_list()
print(project_list)

arg_parser = create_build_argument_parser(
    rock_builder_home_dir, default_src_base_dir, project_list
)
args = parse_build_arguments(arg_parser)
# store the arguments to dictionary to make it easier to get "project_name"-version parameters
args_dict = args.__dict__
printout_build_arguments(args)
verify_build_env__python(args)
verify_build_env(args, rock_builder_home_dir, rock_builder_build_dir)

for ii, prj_item in enumerate(project_list):
    print(f"    Project [{ii}]: {prj_item}")

# let user to see env variables for a while before build start
time.sleep(1)

if not args.project:
    # process all projects specified in the core_project.pcfg
    if args.src_dir:
        print(
            '\nError, "--src-dir" parameter requires also to specify the project with the "--project"-parameter'
        )
        print('Alternatively you could use the "--src-base-dir" parameter.')
        print("")
        sys.exit(1)
    for ii, prj_item in enumerate(project_list):
        print(f"[{ii}]: {prj_item}")
        # argparser --> Keyword for parameter "--my-project-version=xyz" = "my_project_version"
        prj_version_keyword = project_list[ii] + "_version"
        prj_version_keyword = prj_version_keyword.replace("-", "_")
        version_override = args_dict[prj_version_keyword]
        # when issuing a command for all projects, we assume that the src_base_dir
        # is the base source directory under each project specific directory is checked out.
        prj_builder = project_manager.get_rock_project_builder(
            args.src_base_dir / project_list[ii],
            project_list[ii],
            args.output_dir,
            version_override,
        )
        if prj_builder is None:
            sys.exit(1)
        else:
            do_therock(prj_builder)
else:
    # process only a single project specified with the "--project" parameter
    # argparser --> Keyword for parameter "--my-project-version=xyz" = "my_project_version"
    prj_version_keyword = args.project + "_version"
    prj_version_keyword = prj_version_keyword.replace("-", "_")
    version_override = args_dict[prj_version_keyword]
    if args.src_dir:
        # source checkout dir = "--src-dir"
        prj_builder = project_manager.get_rock_project_builder(
            args.src_dir, args.project, args.output_dir, version_override
        )
    else:
        # source checkout dir = "--src-base-dir" / project_name
        prj_builder = project_manager.get_rock_project_builder(
            args.src_base_dir / args.project,
            args.project,
            args.output_dir,
            version_override,
        )
    if prj_builder is None:
        print("Error, failed to get the project builder")
        sys.exit(1)
    else:
        do_therock(prj_builder)
