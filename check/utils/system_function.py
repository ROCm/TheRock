from typing import Literal

from .status import *

import platform, os


def os_type() -> Literal["Windows", "Linux", "BSD", "macOS"]:

    name: str = platform.system()

    if name == "Windows":
        sysname = name
    elif name == "Linux":
        sysname = name
    elif name == "Darwin":
        sysname = "macOS"
    elif "BSD" in name:
        sysname = "BSD"
    else:
        sysname = name

    return sysname


environ_dict = dict(os.environ)


def ENV(var: str = None):
    """
    Return env variable with defined value or NOTDEFINED.
    """
    from .status import NOTDEFINED

    if var is None:
        raise ValueError("Environment Variable query must have a variable name.")

    val = environ_dict.get(var)

    return val if val is not None else NOTDEFINED


def where(exe: str, /):
    from shutil import which

    return which(exe)
