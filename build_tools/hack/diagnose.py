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

from env_check.utils import RepoInfo, cstring
from env_check.device import SystemInfo
from env_check.check_tools import *
from env_check import check_therock


def main():
    therock_detect_start = time.perf_counter()
    device = SystemInfo()
    RepoInfo.__logo__()
    build_type = cstring(check_therock.build_project, "hint")

    device.summary

    print(
        f"""
        ===========\t\tStart detect compoments on: {build_type}\t\t===========
    """
    )

    diag_check = check_therock.test_list().summary

    therock_detect_terminate = time.perf_counter()
    therock_detect_time = float(therock_detect_terminate - therock_detect_start)
    therock_detect_runtime = cstring(f"{therock_detect_time:.2f}", "hint")
    print(
        f"""
        ===========\t    {diag_check}\t===========

        ===========\tTheRock build pre-diagnosis script completed in {therock_detect_runtime} seconds\t===========
    """
    )


if __name__ == "__main__":
    main()
