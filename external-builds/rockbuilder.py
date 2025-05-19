#!/usr/bin/env python

import argparse
import configparser
import lib_python.project_builder as project_builder
import sys
import os
import time
import platform
from pathlib import Path, PurePosixPath

ROCK_BUILDER_VERSION = "2025-05-27_01"

def is_directory_in_path_env_variable(env_variable, directory):
  """
  Checks if a directory is in the env_variable environment variable.

  Args:
    env_variable: Environment variable used to check the path
    directory: The path searched from the environment variable

  Returns:
    True if the directory is in PATH, False otherwise.
  """
  path_env = os.environ.get(env_variable, '')
  path_directories = path_env.split(os.pathsep)
  return directory in path_directories

is_posix = not any(platform.win32_ver())

if is_posix:
    ENV_VARIABLE_NAME__LIB="LD_LIBRARY_PATH"
else:
    ENV_VARIABLE_NAME__LIB="LIBPATH"

#os.environ["ROCK_BUILDER_HOME_DIR"] = os.getcwd()
current_file_path = os.path.abspath(__file__)
rock_builder_home_dir = os.path.dirname(current_file_path)
os.environ["ROCK_BUILDER_HOME_DIR"] = rock_builder_home_dir
os.environ["ROCK_BUILDER_SRC_DIR"] = os.environ["ROCK_BUILDER_HOME_DIR"] + "/src_projects"
os.environ["ROCK_BUILDER_BUILD_DIR"] = os.environ["ROCK_BUILDER_HOME_DIR"] + "/builddir"

if "ROCM_HOME" not in os.environ:
    rocm_home_root_path = Path(rock_builder_home_dir) / "../build/dist/rocm"
    #print("ROCM_HOME: " + rocm_home_root_path)
    if rocm_home_root_path.exists():
        os.environ["ROCM_HOME"] = str(rocm_home_root_path)
        rocm_home_bin_path = rocm_home_root_path / "bin"
        rocm_home_lib_path = rocm_home_root_path / "lib"
        if rocm_home_bin_path.exists():
            if rocm_home_lib_path.exists():
                if not is_directory_in_path_env_variable("PATH", str(rocm_home_bin_path)):
                    #print("Adding " + str(rocm_home_bin_path) + " to PATH")
                    os.environ["PATH"] = str(rocm_home_bin_path) + os.pathsep + os.environ.get("PATH", "")
                if not is_directory_in_path_env_variable(ENV_VARIABLE_NAME__LIB, str(rocm_home_lib_path)):
                    #print("Adding " + str(rocm_home_lib_path) + " to " + ENV_VARIABLE_NAME__LIB)
                    os.environ[ENV_VARIABLE_NAME__LIB] = str(rocm_home_bin_path) + os.pathsep + os.environ.get(ENV_VARIABLE_NAME__LIB, "")
            else:
                print("Error, could not find directory " + str(rocm_home_lib_path))
        else:
            print("Error, could not find directory " + str(rocm_home_bin_path))
    else:
        print("Error, ROCM_HOME was not defined and ROCM build could not be found from the expected location: " + rocm_path)
        sys.exit(1)

python_home_dir = os.path.dirname(sys.executable)
if "VIRTUAL_ENV" in os.environ:
    os.environ["ROCK_PYTHON_PATH"] = python_home_dir
else:
    if "ROCK_PYTHON_PATH" in os.environ:
        if not os.path.abspath(python_home_dir) == os.path.abspath(os.environ["ROCK_PYTHON_PATH"]):
            print("Error, virtual python environment is not active and")
            print("PYTHON location is different than ROCK_PYTHON_PATH")
            print("    PYTHON location: " + python_home_dir)
            print("    ROCK_PYTHON_PATH: " + os.environ["ROCK_PYTHON_PATH"])
            print("If you want use this python location instead of using a virtual python env, define ROCK_PYTHON_PATH:")
            if is_posix:
                print("    export ROCK_PYTHON_PATH=" + python_home_dir)
            else:
                print("    set ROCK_PYTHON_PATH=" + python_home_dir)
            print("Alternatively activate the virtual python environment")
            sys.exit(1)
        else:
            print("Using python from location: " + python_home_dir)
    else:
        print("Error, virtual python environment is not active and ROCK_PYTHON_PATH is not defined")
        print("    PYTHON location: " + python_home_dir)
        print("If you want use this python location instead of using a virtual python env, define ROCK_PYTHON_PATH:")
        if is_posix:
            print("    export ROCK_PYTHON_PATH=" + python_home_dir)
        else:
            print("    set ROCK_PYTHON_PATH=" + python_home_dir)
        print("Alternatively activate the virtual python environment")
        sys.exit(1)

print("RockBuilder " + ROCK_BUILDER_VERSION)
print("")
print("Build environment:")
print("-----------------------------")
print("ROCM_HOME: " + os.environ["ROCM_HOME"])
print("ROCK_PYTHON_PATH: " + os.environ["ROCK_PYTHON_PATH"])
print("ROCK_BUILDER_HOME_DIR: " + os.environ["ROCK_BUILDER_HOME_DIR"])
print("ROCK_BUILDER_SRC_DIR: " + os.environ["ROCK_BUILDER_SRC_DIR"])
print("ROCK_BUILDER_BUILD_DIR: " + os.environ["ROCK_BUILDER_BUILD_DIR"])
print("PATH: " + os.environ["PATH"])
print(ENV_VARIABLE_NAME__LIB + ": " + os.environ[ENV_VARIABLE_NAME__LIB])
print("-----------------------------")
time.sleep(1)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='ROCK Project Builders')

# Add arguments
parser.add_argument('--project', type=str, help='select target for the action. Can be either one project or all projects in core_apps.pcfg.', default='all')
parser.add_argument('--checkout',  action='store_true', help='checkout source code for the project', default=False)
parser.add_argument('--clean',  action='store_true', help='clean build files', default=False)
parser.add_argument('--configure',  action='store_true', help='configure project for build', default=False)
parser.add_argument('--build',  action='store_true', help='build project', default=False)
parser.add_argument('--install',  action='store_true', help='install build project', default=False)

# Parse the arguments
args = parser.parse_args()

if ("--checkout" in sys.argv) or ("--clean" in sys.argv) or ("--configure" in sys.argv) or\
   ("--build" in sys.argv) or ("--install" in sys.argv):
    print("checkout/clean/configure/build or install argument specified")
else:
    #print("Action not specified.(checkout/clean/configure/build or install)")
    #print("Using default values")
    args.checkout=True
    args.configure=True
    args.build=True
    args.install=True

# Access the arguments
print("Actions Enabled:")
print('    Checkout: ', args.checkout)
print('    Clean:    ', args.clean)
print('    Configure:', args.configure)
print('    Build:    ', args.build)
print('    Install:  ', args.install)
print('Projects:', args.project)

project_manager = project_builder.RockExternalProjectListManager(rock_builder_home_dir)
# allow_no_value param says that no value keys are ok
sections = project_manager.sections()
#print(sections)
project_list = project_manager.get_external_project_list()
#print(project_list)

for ii, prj_item in enumerate(project_list):
    print(f"    Project [{ii}]: {prj_item}")

time.sleep(1)

# checkout all projects
if args.checkout:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"checkout[{ii}]: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is None):
                sys.exit(1)
            else:
                prj_builder.printout()
                prj_builder.checkout()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is None):
            sys.exit(1)
        else:
            prj_builder.printout()
            prj_builder.checkout()

if args.clean:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"clean[{ii}]: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is None):
                sys.exit(1)
            else:
                prj_builder.printout()
                prj_builder.clean()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is None):
            sys.exit(1)
        else:
            prj_builder.printout()
            prj_builder.clean()

if args.configure:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"configure[{ii}]: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is None):
                sys.exit(1)
            else:
                prj_builder.printout()
                prj_builder.configure()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is None):
            sys.exit(1)
        else:
            prj_builder.printout()
            prj_builder.configure()

if args.build:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"build[{ii}]: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is None):
                sys.exit(1)
            else:
                prj_builder.printout()
                prj_builder.build()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is None):
            sys.exit(1)
        else:
            prj_builder.printout()
            prj_builder.build()

if args.install:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"install[{ii}]: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is None):
                sys.exit(1)
            else:
                prj_builder.printout()
                prj_builder.install()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is None):
            sys.exit(1)
        else:
            prj_builder.printout()
            prj_builder.install()
