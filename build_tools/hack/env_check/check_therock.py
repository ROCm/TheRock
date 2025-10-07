from .check_tools import *
from .device import SystemInfo

build_project = "ROCm/TheRock"

device = SystemInfo()

if device.is_windows:
    my_list = [
        CheckOS(device_info=device),
        CheckCPU(device_info=device),
        CheckDisk(device_info=device),
        Check_Max_PATH_LIMIT(device_info=device),
        CheckGit(),
        CheckGitLFS(required=False),
        CheckCMake(),
        CheckCCache(required=False),
        CheckNinja(),
        CheckGFortran(),
        CheckPython(),
        CheckUV(required=False),
        CheckVS20XX(),
        CheckMSVC(),
        CheckATL(),
        CheckML64(),
        CheckLIB(),
        CheckLINK(),
        CheckRC(),
    ]

elif device.is_linux:
    my_list = [
        CheckOS(device_info=device),
        CheckCPU(device_info=device),
        CheckDisk(device_info=device),
        CheckGit(),
        CheckGitLFS(required=True),
        CheckCMake(),
        CheckCCache(required=False),
        CheckNinja(),
        CheckPython(),
        CheckUV(required=False),
        CheckGCC(),
        CheckGXX(),
        CheckGFortran(),
        CheckGCC_AS(),
        CheckGCC_AR(),
        CheckLD(),
    ]


def test_list(entry_list=my_list) -> list:
    return Check_List(entry_list)
