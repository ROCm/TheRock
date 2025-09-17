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
from tabulate import tabulate


def printCaccheConfig():
    try:
        proc_ccache = subprocess.run(
            ["cat", "$CCACHE_CONFIGPATH"], capture_output=True, text=True
        )
        proc_ccache.check_returncode()

        # prettify formatting to match rest of output
        print("        CCACHE CONFIG:", end="")
        for idx, line in enumerate(proc_ccache.stdout.split("\n")):
            if idx == 0:
                print("     " + line)
            else:
                print("                    " + line)

    except subprocess.CalledProcessError:
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

        rows = []
        row = []

        for idx, line in enumerate(proc_python.stdout.split("\n")):
            row.append(line)

            if (idx + 1) % 3 == 0:
                rows.append(row)
                row = []

        if len(row) > 0:
            rows.append(row)

        for line in tabulate(rows).split("\n"):
            print("\t\t" + line)

    except subprocess.CalledProcessError:
        print("ERROR when listing python packages!")


def main():
    therock_detect_start = time.perf_counter()
    device = SystemInfo()
    RepoInfo.__logo__()
    build_type = cstring(check_therock.build_project, "hint")

    device.summary

    printCaccheConfig()
    printCcacheStats()
    print(
        f"""
        ===================\t\tStart detect compoments on: {build_type}\t\t===================
    """
    )

    diag_check = check_therock.test_list().summary

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
