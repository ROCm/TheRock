#! python
import argparse
import configparser
import lib_python.project_builder as project_builder
import sys
import os

os.environ["ROCK_BUILDER_ROOT_DIR"] = os.getcwd()
os.environ["ROCK_BUILDER_SRC_ROOT_DIR"] = os.getcwd() + "/src_projects"
print("ROCK_BUILDER_ROOT_DIR: " + os.environ["ROCK_BUILDER_ROOT_DIR"])
print("ROCK_BUILDER_SRC_ROOT_DIR: " + os.environ["ROCK_BUILDER_SRC_ROOT_DIR"])

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='ROCK Project Builders')

# Add arguments
parser.add_argument('--project', type=str, help='select whether to target only one or projects', default='all')
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
    print("no checkout/clean/configure/build or install argument specified")
    print("assuming all of them are enabled")
    args.checkout=True
    args.configure=True
    args.build=True
    args.install=True

# Access the arguments
print('project:', args.project)
print('checkout:', args.checkout)
print('clean:', args.clean)
print('configure:', args.configure)
print('build:', args.build)
print('install:', args.install)

project_manager = project_builder.RockExternalProjectListManager()
# allow_no_value param says that no value keys are ok
sections = project_manager.sections()
print(sections)
project_list = project_manager.get_external_project_list()
print(project_list)

# checkout all projects
if args.checkout:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"index: {ii}, project: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is not None):
                prj_builder.printout()
                prj_builder.checkout()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is not None):
            prj_builder.printout()
            prj_builder.checkout()

if args.clean:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"index: {ii}, project: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is not None):
                prj_builder.printout()
                prj_builder.clean()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is not None):
            prj_builder.printout()
            prj_builder.clean()

if args.configure:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"index: {ii}, project: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is not None):
                prj_builder.printout()
                prj_builder.configure()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is not None):
            prj_builder.printout()
            prj_builder.configure()

if args.build:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"index: {ii}, project: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is not None):
                prj_builder.printout()
                prj_builder.build()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is not None):
            prj_builder.printout()
            prj_builder.build()

if args.install:
    if (args.project == "all"):
        for ii, prj_item in enumerate(project_list):
            print(f"index: {ii}, project: {prj_item}")
            prj_builder = project_manager.get_rock_project_builder(project_list[ii])
            if (prj_builder is not None):
                prj_builder.printout()
                prj_builder.install()
    else:
        prj_builder = project_manager.get_rock_project_builder(args.project)
        if (prj_builder is not None):
            prj_builder.printout()
            prj_builder.install()

