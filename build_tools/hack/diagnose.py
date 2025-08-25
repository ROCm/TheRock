#!/usr/bin/env python

#
#   Designed by TaiXeflar, reviewed by Scott Todd, contribute to TheRock team
#
#   TheRock Project building system pre-build diagnosis script
#   License follows TheRock project
#
#   !  Hint: This script doesn't raise/throw back warnings/errors.
#   This script is for detecting environments use, We do a global scan on all requirements at once.
#   We do not want users have to fix its environment one-by-one and get frustrated,so the diagnosis won't throw errors on it.
#   If running this script have throwback errors, Please report it as new issue or open in a new disscus <3
#


import sys, time

sys.dont_write_bytecode = True

from env_check.utils import RepoInfo, cstring
from env_check.device import SystemInfo
from env_check.check_tools import *
from env_check import check_therock


def main():
    therock_detect_start = time.perf_counter()
    device = SystemInfo()
    RepoInfo.__logo__()

    device.summary

    check_therock.test_list().summary

    therock_detect_terminate = time.perf_counter()
    therock_detect_time = float(therock_detect_terminate - therock_detect_start)
    therock_detect_runtime = cstring(f"{therock_detect_time:2f}", "hint")
    print(
        f"""
        ===========    TheRock build pre-diagnosis script completed in {therock_detect_runtime} seconds    ===========
    """
    )


if __name__ == "__main__":
    main()
