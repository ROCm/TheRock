#!/usr/bin/python3

"""Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

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

import re
import sys
import os
import platform
import subprocess
import argparse
import pathlib
from xml.dom import minidom
from fnmatch import fnmatchcase

SCRIPT_VERSION = 0.1

args = {}
param = {}
OS_info = {}
var_subs = {}

vcpkg_script = ['tdir %IDIR%',
                'git clone -b 2024.02.14 https://github.com/microsoft/vcpkg %IDIR%', 'cd %IDIR%', 'bootstrap-vcpkg.bat -disableMetrics' ]

xml_script = [ '%XML%' ]


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="""
    Checks build arguments
    """)
    # parser.add_argument('--install', required=False, default = True,  action='store_true',
    #                     help='Install dependencies (optional, default: True)')
    parser.add_argument('-i', '--install_dir', type=str, required=False, default = ("" if os.name == "nt" else "./build/deps"),
                        help='Install directory path (optional, windows default: C:\\github\\vcpkg, linux default: ./build/deps)')
    # parser.add_argument('-v', '--verbose', required=False, default = False, action='store_true',
    #                     help='Verbose install (optional, default: False)')
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
    if os.path.isabs(dir_path):
        full_path = dir_path
    else:
        full_path = os.path.join( os.getcwd(), dir_path )
    return pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)

def delete_dir(dir_path) :
    if (not os.path.exists(dir_path)):
        return
    if os.name == "nt":
        return run_cmd( "RMDIR" , f"/S /Q {dir_path}")
    else:
        linux_path = pathlib.Path(dir_path).absolute()
        return run_cmd( "rm" , f"-rf {linux_path}")

def run_cmd(cmd):
    global args
    if (cmd.startswith('cd ')):
        return os.chdir(cmd[3:])
    if (cmd.startswith('mkdir ')):
        return create_dir(cmd[6:])
    cmdline = f"{cmd}"
    print(cmdline)
    proc = subprocess.run(cmdline, check=True, stderr=subprocess.STDOUT, shell=True)
    return proc.returncode


def install_deps( os_node ):
    global var_subs

    cwd = pathlib.Path.absolute(pathlib.Path(os.curdir))

    if os.name == "nt":
        vc_node = os_node.getElementsByTagName('vcpkg')
        if vc_node:
            cmdline = "cd %IDIR%"
            cd_vcpkg = cmdline.replace('%IDIR%', args.install_dir)
            run_cmd(cd_vcpkg)
            for p in vc_node[0].getElementsByTagName('pkg'):
                name = p.getAttribute('name')
                package = p.firstChild.data
                if name:
                    print( f'***\n*** VCPKG Installing: {name}\n***' )
                raw_cmd = p.firstChild.data
                var_cmd = raw_cmd.format_map(var_subs)
                error = run_cmd( f'vcpkg.exe install {var_cmd}')
    else:
        create_dir( args.install_dir )
        # TODO

    os.chdir(cwd)

    pip_node = os_node.getElementsByTagName('pip')
    if pip_node:
        for p in pip_node[0].getElementsByTagName('pkg'):
            name = p.getAttribute('name')
            package = p.firstChild.data
            if name:
                print( f'***\n*** Pip Installing: {name}\n***' )
            raw_cmd = p.firstChild.data
            var_cmd = raw_cmd.format_map(var_subs)
            error = run_cmd( f'pip install {var_cmd}')
        for p in pip_node[0].getElementsByTagName('req'):
            name = p.getAttribute('name')
            package = p.firstChild.data
            if name:
                print( f'***\n*** Pip Requirements: {name}\n***' )
            raw_cmd = p.firstChild.data
            var_cmd = raw_cmd.format_map(var_subs)
            requirements_file = os.path.abspath( os.path.join( cwd, var_cmd ) )
            error = run_cmd( f'pip install -r {requirements_file}')

def run_install_script(script, xml):
    '''executes a simple batch style install script, the scripts are defined at top of file'''
    global OS_info
    global args
    global var_subs
    #
    cwd = pathlib.Path.absolute(pathlib.Path(os.getcwd()))

    fail = False
    last_cmd_index = 0
    for i in range(len(script)):
        last_cmd_index = i
        cmdline = script[i]
        cmd = cmdline.replace('%IDIR%', args.install_dir)
        if cmd.startswith('tdir '):
            if pathlib.Path(cmd[5:]).exists():
                break # all further cmds skipped
            else:
                continue
        error = False
        if cmd.startswith('%XML%'):
            fileversion = xml.getElementsByTagName('fileversion')
            if len(fileversion) == 0:
                print("WARNING: Could not find the version of this xml configuration file.")
            elif len(fileversion) > 1:
                print("WARNING: Multiple version tags found.")
            else:
                version = float(fileversion[0].firstChild.data)
                if version > SCRIPT_VERSION:
                    print(f"ERROR: This xml requires script version >= {version}, running script version {SCRIPT_VERSION}")
                    exit(1)

            for var in xml.getElementsByTagName('var'):
                name = var.getAttribute('name')
                if var.hasAttribute('value'):
                    val = var.getAttribute('value')
                elif var.firstChild is not None:
                    val = var.firstChild.data
                else:
                    val = ""
                var_subs[name] = val

            for os_node in xml.getElementsByTagName('os'):
                os_names = os_node.getAttribute('names')
                os_list = os_names.split(',')
                if (OS_info['ID'] in os_list) or ("all" in os_list):
                    error = install_deps( os_node )
        else:
            error = run_cmd(cmd)
        fail = fail or error
        if fail:
            break

    os.chdir( cwd )
    if (fail):
        if (script[last_cmd_index] == "%XML%"):
            print(f"FAILED xml dependency installation!")
        else:
            print(f"ERROR running: {script[last_cmd_index]}")
        return 1
    else:
        return 0

def installation():
    global vcpkg_script
    global xml_script
    global xmlDoc

    # install
    cwd = os.getcwd()

    xmlPath = os.path.join( cwd, 'rdeps.xml')
    xmlDoc = minidom.parse( xmlPath )

    scripts = []

    if os.name == "nt" and xmlDoc.getElementsByTagName('vcpkg'):
        scripts.append( vcpkg_script )
    scripts.append( xml_script )

    for i in scripts:
        if (run_install_script(i, xmlDoc)):
            #print("Failure in script. ABORTING")
            os.chdir( cwd )
            return 1
    os.chdir( cwd )
    return 0

def main():
    global args

    os_detect()
    args = parse_args()

    if os.name == "nt" and not args.install_dir:
        vcpkg_root = os.getenv( 'VCPKG_PATH', "C:\\github\\vcpkg")
        args.install_dir = vcpkg_root

    installation()

if __name__ == '__main__':
    main()
