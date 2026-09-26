import time
from functools import wraps
from typing import Union, Literal
import subprocess as subproc
from .status import *
from pathlib import Path
from .system_function import os_type
from utils import config

from .color_string import cstring


#   =======================================================================================================================================================================================
#       clear function for clear terminal.
#
def clear():
    import os, sys, subprocess

    if config.CLEAR_HOST:
        if sys.version_info < (3, 14):
            os.system("cls" if os_type() == "Windows" else "clear")
        else:
            sh = True if os_type() == "Windows" else False
            cmd = "cls" if os_type() == "Windows" else "clear"
            subprocess.run([cmd], shell=sh)
    else:
        pass


#   =======================================================================================================================================================================================
#       Tic/Toc functions wraps execution functions at main()
#


def tic_toc():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()

            result = func(*args, **kwargs)

            toc = time.perf_counter()
            print(
                f"\n        ===========\t\tTheRock diagnose system progress finished in {(toc - tic):.6f} seconds \t\t===========\n"
            )
            return result

        return wrapper

    return decorator


#   =======================================================================================================================================================================================
#       githead run Repo git head query.
#


def git_head():
    _head = subproc.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    return _head


githead = git_head()


def git_repo():
    finder = subproc.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    ).stdout.strip()
    return Path(finder).resolve()


gitrepo = git_repo()


def therock_ver():
    import json

    with open((gitrepo / "version.json"), "r", encoding="utf-8") as f:
        content: dict[Literal["rocm-version"], str] = json.loads(f.read())

    return content["rocm-version"]


def cmd_record() -> str:
    import sys
    import os

    args = sys.argv
    python_alias = os.path.basename(sys.executable)

    return f"{python_alias} {' '.join(args)}"


COMMAND_LINE = cmd_record().replace("\\", "/")


#   =======================================================================================================================================================================================
#       AMD logo.
#


def logo():
    """"""
    print(
        f"""


    {cstring("\t\t\t    # # # # # # # # # # #","ERROR")}
    {cstring("\t\t\t      # # # # # # # # # #","error")}
    {cstring("\t\t\t        # # # # # # # # #","error")}\t  {cstring("AMD ROCm/TheRock Project","error")}
    {cstring("\t\t\t                    # # #","error")}
    {cstring("\t\t\t        #           # # #","error")}\t  Build Environment Diagnose System {config.THEROCK_BUILD_DIAGNOSE_SYSTEM_VERSION}
    {cstring("\t\t\t      # #           # # #","error")}
    {cstring("\t\t\t    # # #           # # #","error")}\t  Version TheRock {therock_ver()} (current HEAD: {cstring(githead, "error")})
    {cstring("\t\t\t    # # # # # # #   # # #","error")}
    {cstring("\t\t\t    # # # # # #       # #","error")}
    {cstring("\t\t\t    # # # # #           #","error")}


    """
    )
