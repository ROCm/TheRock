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


def main():
    therock_detect_start = time.perf_counter()
    device = SystemInfo()
    RepoInfo.__logo__(monospace=True)
    build_type = cstring(check_therock.build_project, "hint")

    device.summary

    print("")
    print(device.section_bar(f"Start detect compoments on: {build_type}"))
    print("")

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
            CheckPython(is_global_env_ok=True),
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

    print("")
    print(device.section_bar(diag_check))
    print("\n")

    device.python_list

    therock_detect_terminate = time.perf_counter()
    therock_detect_time = float(therock_detect_terminate - therock_detect_start)
    therock_detect_runtime = cstring(f"{therock_detect_time:.2f}", "hint")

    print("\n")
    print(
        device.section_bar(
            f"  TheRock build pre-diagnosis script completed in {therock_detect_runtime} seconds  "
        )
    )


if __name__ == "__main__":
    main()
