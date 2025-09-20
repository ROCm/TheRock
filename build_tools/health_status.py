#!/usr/bin/env python

#
#   TheRock Project building system pre-build diagnosis script
#   License follows TheRock project
#
#   This script doesn't raise/throw back warnings/errors.
#   If running this script has errors, please report it as a new issue.
#


import sys, time

sys.dont_write_bytecode = True

from hack.env_check.utils import RepoInfo, cstring
from hack.env_check.device import SystemInfo
from hack.env_check.check_tools import *
from hack.env_check import check_therock
import subprocess


def printFilesystemUsage():
    print("       FILE SYSTEM:", end="")
    proc_ccache = subprocess.run(["df", "-h"], capture_output=True, text=True)
    proc_ccache.check_returncode()

    # prettify formatting to match rest of output
    for idx, line in enumerate(proc_ccache.stdout.split("\n")):
        if idx == 0:
            print(" " + line)
        else:
            print("                    " + line)


def printCaccheConfig():
    print("     CCACHE CONFIG:", end="")
    try:
        configpath = os.getenv("CCACHE_CONFIGPATH")
        if configpath == None:
            print(cstring(" Env $CCACHE_CONFIGPATH not defined!", "warn"))
            return
        proc_ccache = subprocess.run(
            ["cat", configpath], capture_output=True, text=True
        )
        proc_ccache.check_returncode()

        # prettify formatting to match rest of output
        for idx, line in enumerate(proc_ccache.stdout.split("\n")):
            if idx == 0:
                print(" " + line)
            else:
                print("                    " + line)

    except subprocess.CalledProcessError:
        print(proc_ccache.stderr)
        pass


def printCcacheStats():
    try:
        proc_ccache = subprocess.run(
            ["ccache", "-s", "-v"], capture_output=True, text=True
        )
        proc_ccache.check_returncode()

        # prettify formatting to match rest of output
        print("        CCACHE:", end="")
        for idx, line in enumerate(proc_ccache.stdout.split("\n")):
            if idx == 0:
                print("     " + line)
            else:
                print("                    " + line)

    except subprocess.CalledProcessError:
        print("    CCACHE:    not detected")


def printPythonList():
    try:
        proc_python = subprocess.run(
            ["pip", "list", "--format=freeze"], capture_output=True, text=True
        )
        proc_python.check_returncode()

        for idx, line in enumerate(proc_python.stdout.split("\n")):
            if idx == 0:
                print("\t\t\t\t\t\t  " + line)
            else:
                print("\t\t\t\t\t\t  " + line)

    except subprocess.CalledProcessError:
        print("ERROR when listing python packages!")


def main():
    therock_detect_start = time.perf_counter()
    device = SystemInfo()
    RepoInfo.__logo__(monospace=True)
    build_type = cstring(check_therock.build_project, "hint")

    device.summary

    printFilesystemUsage()

    printCcacheStats()
    printCaccheConfig()
    print(
        f"""
        ===================\t\tStart detect compoments on: {build_type}\t\t===================
    """
    )

    check_list = [
        check
        for check in [
            CheckOS(device_info=device),
            CheckCPU(device_info=device),
            CheckDisk(device_info=device),
            Check_Max_PATH_LIMIT(device_info=device) if device.is_windows else None,
            CheckGit(),
            CheckGitLFS(required=False)
            if device.is_windows
            else CheckGitLFS(required=True),
            CheckCMake(),
            CheckCCache(required=False),
            CheckNinja(),
            CheckGFortran(),
            CheckPython(isGlobalEnvOK=True),
            CheckUV(required=False),
        ]
        if check is not None
    ]

    win_only_list = [
        CheckVS20XX(),
        CheckMSVC(),
        CheckATL(),
        CheckML64(),
        CheckLIB(),
        CheckLINK(),
        CheckRC(),
    ]

    linux_only_list = [
        CheckGCC(),
        CheckGXX(),
        CheckGCC_AS(),
        CheckGCC_AR(),
        CheckLD(),
    ]

    if device.is_windows:
        check_list += win_only_list
    if device.is_linux:
        check_list += linux_only_list

    diag_check = check_therock.test_list(check_list).summary

    print(
        f"""
        ===================\t    {diag_check}\t===================
    """
    )

    print(
        f"""
        ===================\t\t\t\tPython List \t\t\t\t===================
    """
    )

    printPythonList()

    print(
        f"""
        ===================\t\t\t    End Python List   \t\t\t\t===================
    """
    )

    therock_detect_terminate = time.perf_counter()
    therock_detect_time = float(therock_detect_terminate - therock_detect_start)
    therock_detect_runtime = cstring(f"{therock_detect_time:.2f}", "hint")
    print(
        f"""
        ===================\tTheRock build pre-diagnosis script completed in {therock_detect_runtime} seconds\t===================
    """
    )


if __name__ == "__main__":
    main()
