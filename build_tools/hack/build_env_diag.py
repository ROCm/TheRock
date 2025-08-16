#!/usr/bin/env python

#
#   Designed by TaiXeflar, reviewed by Scott Todd, contribute to TheRock team
#
#   TheRock Project building system pre-build diagnosis script
#   License follows TheRock project
#
#   Variables: "FULL_UPPER_VARIABLE" are been called to CMake-Like style.
#      *CMake                  *Python
#               WIN32 -->               WINDOWS
#               Linux -->               LINUX
#       CMAKE_MAJOR_VERSION -->         PYTHON_MAJOR_VERSION
#
#   !  Hint: This script doesn't raise/throw back warnings/errors.
#   This script is for detecting environments use, We do a global scan on all requirements at once.
#   We do not want users have to fix its environment one-by-one and get frustrated,so the diagnosis won't throw errors on it.
#   If Users(Yes, It's you! maybe.) running this script have throwback errors, It must have bug in it.
#       PLZ! Please report it as new issue or open in a new disscus <3
#

from __future__ import annotations
from typing import Literal, Optional, Union, Tuple
import os, re, platform, sys, shutil
from pathlib import Path
import time

# Define Color string print.


def cstring(
    msg: Union[str],
    color: Union[
        Optional[Literal["err", "warn", "hint", "Discord"]],
        Tuple[int, int, int],
    ]
    | None = None,
) -> str:
    """
    ## Color String
    Returns with ANSI escape code formated string, with colors by (R, G, B).

    This feature passed on Linux Terminal, Windows Terminal, VSCode Terminal, and VSCode Jupyter Notebook.

    ### Usage
    `<STR_VAR> = cstring(string, color)`
    - msg: `str` type.
    - color: A user specified `tuple` with each value Ranged from `0` ~ `255` `(R, G, B)`.\t
    ```
    >>> your_text = cstring(msg="AMD RADEON RX 7800XT", color=(255, 0, 0))
    >>> your_text
    ```
    - If color's RGB not passed will be full white. Color also can be these keywords:
        - "err"
        - "warn"
        - "pass"
    """

    if isinstance(color, tuple):
        r, g, b = color
    else:
        match color:
            case "err":
                r, g, b = (255, 61, 61)
            case "warn":
                r, g, b = (255, 230, 66)
            case "hint":
                r, g, b = (150, 255, 255)
            case _:
                r, g, b = (255, 255, 255)

    return f"\033[38;2;{r};{g};{b}m{msg}\033[0m"


class Emoji:
    Pass = "✅"
    Warn = "⚠️"
    Err = "❌"


# Define AMD arrow logo and therock current head.
class TheRock:
    """
    ## TheRock class
    AMD ROCm/TheRock project.

    - `head()`: `str`. Returns Repo cloned main's head.
    - `repo()`: `str`. Returns Repo's abs path.

    - `__logo__()`: Advanced Micro Devices Logo. Displays AMD Arrow Logo and current git HEAD.

    ![image](https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png)
    """

    @staticmethod
    def head():
        try:
            with open(Path(f"{TheRock.repo()}/.git/refs/heads/main").resolve()) as f:
                local_sha = f.read().strip()
                return local_sha[:7]
        except FileNotFoundError as e:
            return "Unknown"

    @staticmethod
    def repo():
        import subprocess

        finder = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        ).stdout.strip()
        return finder

    @staticmethod
    def __logo__():

        """
        ![image](https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png)
        # Advanced Micro Devices Inc.
        """

        _REPO_HEAD = TheRock.head()

        print(
            f"""\n\n\n\n
    {cstring("   ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼","err")}
    {cstring("     ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼","err")}
    {cstring("       ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼ ◼","err")}\t  {cstring("AMD TheRock Project","err")}
    {cstring("                   ◼ ◼ ◼","err")}
    {cstring("       ◼           ◼ ◼ ◼","err")}\t  Build Environment diagnosis script
    {cstring("     ◼ ◼           ◼ ◼ ◼","err")}
    {cstring("   ◼ ◼ ◼           ◼ ◼ ◼","err")}\t  Version TheRock (current HEAD: {cstring(TheRock.head(), "err")})
    {cstring("   ◼ ◼ ◼ ◼ ◼ ◼ ◼   ◼ ◼ ◼","err")}
    {cstring("   ◼ ◼ ◼ ◼ ◼ ◼       ◼ ◼","err")}
    {cstring("   ◼ ◼ ◼ ◼ ◼           ◼","err")}
    """
        )


class Find_Program:
    """
    ## Find_Program
    This class finds the specified program, finds its location and analyze its version number.

    Find_Program object use attribute:
    - `exe`: The program's PATH. Found program will display the first priority only.
    - `version`: The program's complete version number.
    - `MAJOR_VERSION`: The program's major version number. Exclude MSVC toolchain and Windows SDK.
    - `MINOR_VERSION`: The program's minor version number. Exclude MSVC toolchain and Windows SDK.
    - `PATCH_VERSION`: The program's patch version number. Exclude MSVC toolchain and Windows SDK, GCC binutils.

    ```
     >>> python = Find_Program("python3")
     >>> python_path = python.exe
     >>> python_ver = python.version
     >>> X, Y, Z = python.MAJOR_VERSION, python.MINOR_VERSION, python.PATCH_VERSION
    ```

    ### Python
    If program name is python, it comes with other attribute:
    - `Free_Threaded`: Check if Python program have Global Interpreter Lock (GIL). If enabled (`True`), the Python is Free-Threaded version.
    - `env`: Python Environment Status. Could be `Global ENV`, `Virtual ENV`(Python VENV), `UV VENV`(Astral UV) or

            `Conda ENV`(Anaconda/Miniconda). pipx/poetry/pyenv etc. currently not supported.
    ```
     >>> python_env_type = python.env
     >>> python_have_GIL = python.have_GIL
     >>> python_free_threaded_status = not python.have_GIL
    ```
    """

    def __init__(self, executable: str):
        super().__init__()

        import shutil, subprocess, re

        self._name = executable.lower()
        self._find = shutil.which(self._name)

        if self.exe is None:
            self._version_num = None
        else:
            match executable:
                case "cl" | "link" | "lib" | "ml64":
                    self._version_num = (
                        os.getenv("VCToolsVersion")
                        if os.getenv("VCToolsVersion")
                        else None
                    )
                case "rc":
                    if (
                        os.getenv("WindowsSDKVersion") != "\\"
                        or os.getenv("WindowsSDKVersion") is not None
                    ):
                        self._version_num = os.getenv("WindowsSDKVersion").replace(
                            "\\", ""
                        )
                    else:
                        self._version_num = None
                case "ar" | "as" | "ld":
                    self._major_version, self._minor_version = map(
                        int,
                        re.search(
                            r"\b(\d+)\.(\d+)\b",
                            subprocess.run(
                                [self.exe, "--version"],
                                capture_output=True,
                                check=True,
                                text=True,
                            ).stdout.strip(),
                        ).groups(),
                    )
                    self._patch_version = None
                    self._version_num = f"{self._major_version}.{self._minor_version}"

                case "python" | "python3":
                    (
                        self._major_version,
                        self._minor_version,
                        self._patch_version,
                        self._release,
                        _,
                    ) = sys.version_info
                    self._version_num = f"{self._major_version}.{self._minor_version}.{self._patch_version}"

                    if os.getenv("CONDA_PREFIX") is not None:
                        self._is_env = True
                        self._env_type = "Conda ENV"
                    elif sys.prefix == sys.base_prefix:
                        self._is_env = False
                        self._env_type = "Global ENV"
                    elif os.getenv("VIRTUAL_ENV") is not None:
                        self._is_env = True
                        _cfg = Path(f"{sys.prefix}/pyvenv.cfg").resolve()
                        with open(_cfg, "r") as file:
                            _conf = file.read()
                            self._env_type = (
                                "uv VENV" if "uv" in _conf else "Python VENV"
                            )

                    if self._minor_version <= 12:
                        self.no_gil = False
                    elif self._minor_version >= 13 and sys._is_gil_enabled() == True:
                        self.no_gil = False
                    elif self._minor_version >= 13 and sys._is_gil_enabled() == False:
                        self.no_gil = True

                case _:
                    self._major_version, self._minor_version, self._patch_version = map(
                        int,
                        re.search(
                            r"\b(\d+)\.(\d+)(?:\.(\d+))?\b",
                            subprocess.run(
                                [self.exe, "--version"],
                                capture_output=True,
                                check=True,
                                text=True,
                            ).stdout.strip(),
                        ).groups(),
                    )
                    self._version_num = f"{self._major_version}.{self._minor_version}.{self._patch_version}"

    def __str__(self):
        return self.exe

    def __repr__(self):
        return self.__str__()

    @property
    def exe(self):
        return (
            self._find.replace("\\", "/").replace("EXE", "exe")
            if self._find is not None
            else None
        )

    @property
    def version(self):
        return self._version_num

    @property
    def MAJOR_VERSION(self):
        return self._major_version

    @property
    def MINOR_VERSION(self):
        return self._minor_version

    @property
    def PATCH_VERSION(self):
        return self._patch_version

    @property
    def name(self):
        return self._name

    @property
    def Free_Threaded(self):
        return self.no_gil if self.name in ("python", "python3") else None

    @property
    def env(self):
        return self._env_type if self.name in ("python", "python3") else None


class SystemInfo:
    """
    ## SystemInfo class
    A class for capturing system info.
    >>> device = SystemInfo()
    """

    def __init__(self):
        super().__init__()

        # Define OS version.
        self._device_os_stat = self.device_os_status()

        # Define CPU configuration.
        self._device_cpu_stat = self.device_cpu_status()

        # Define GPU configuration list.
        self._device_gpu_list = self.device_gpu_list()

        # Define Device Memory status.
        self._device_dram_stat = self.device_dram_status()

        # Define Device Storage status.
        self._device_disk_stat = self.device_disk_status()

    # Define is Operating system status.
    @property
    def OS(self):
        return platform.system().capitalize()

    # Define the device's system is Windows Operating System (Win32).
    @property
    def is_windows(self):
        return True if self.OS == "Windows" else False

    @property
    def is_linux(self):
        return True if self.OS == "Linux" else False

    @property
    def OS_NAME(self):
        if self.is_windows:
            return f"{self._device_os_stat[0]} {self._device_os_stat[1]} ({self._device_os_stat[2]})"
        elif self.is_linux:
            return f"{self._device_os_stat[0]} {self._device_os_stat[1]}"

    @property
    def OS_KERNEL(self):
        return self._device_os_stat[3]

    if is_windows:

        @property
        def is_cygwin(self):
            """
            Define the system environment is Cygwin.
            """
            return True if sys.platform == "cygwin" else False

        @property
        def is_msys2(self):
            """
            Define the system environment is MSYS2.
            """
            return True if sys.platform == "msys" else False

        @property
        def GPU_LIST(self):
            return self._device_gpu_list

        @property
        def VIRTUAL_MEMORY_AVAIL(self):
            return self._device_dram_stat[-1]

    if is_linux:

        @property
        def is_WSL2(self):
            with open("/proc/version", "r") as f:
                _f = f.read()
            return True if "microsoft-standard-WSL2" in _f else False

        @property
        def SWAP_MEMORY_AVAIL(self):
            return self._device_dram_stat[2]

    @property
    def CPU_NAME(self):
        return self._device_cpu_stat[0]

    @property
    def CPU_CORE(self):
        return self._device_cpu_stat[1]

    @property
    def CPU_ARCH(self):
        return self._device_cpu_stat[2]

    @property
    def PHYSICAL_MEMORY_TOTAL(self):
        return self._device_dram_stat[0]

    @property
    def PHYSICAL_MEMORY_AVAIL(self):
        return self._device_dram_stat[1]

    @property
    def DISK_REPO_PATH(self):
        return self._device_disk_stat[0]

    @property
    def DISK_REPO_MOUNT(self):
        return self._device_disk_stat[1]

    @property
    def DISK_TOTAL_SPACE(self):
        return self._device_disk_stat[2]

    @property
    def DISK_USED_SPACE(self):
        return self._device_disk_stat[3]

    @property
    def DISK_AVAIL_SPACE(self):
        return self._device_disk_stat[4]

    @property
    def DISK_USAGE_RATIO(self):
        return self._device_disk_stat[5]

    # Define Windows Registry Editor grep function in Windows platform.
    def get_regedit(
        self,
        root_key: Literal[
            "HKEY_LOCAL_MACHINE", "HKLM", "HKEY_CURRENT_USER", "HKCU"
        ] = "HKEY_LOCAL_MACHINE",
        path: str = any,
        key: str = any,
    ):
        """
        ## Get-Regedit
        Function to get Key-Value in Windows Registry Editor.
        `root_key`: Root Keys or Predefined Keys.You can type-in Regedit style or pwsh style as the choice below:
        - `HKEY_LOCAL_MACHINE` with pwsh alias `HKLM`
        - `HKEY_CURRENT_USER` with pwsh alias `HKCU`
        """

        from winreg import HKEY_LOCAL_MACHINE, HKEY_CURRENT_USER, QueryValueEx, OpenKey

        if root_key in ("HKEY_LOCAL_MACHINE", "HKLM"):
            _ROOT_KEY = HKEY_LOCAL_MACHINE
        elif root_key in ("HKEY_CURRENT_USER", "HKCU"):
            _ROOT_KEY = HKEY_CURRENT_USER
        else:
            raise TypeError("Unsupported Registry Root Key")

        try:
            regedit_val, _ = QueryValueEx(OpenKey(_ROOT_KEY, path), key)
        except FileNotFoundError as e:
            regedit_val = None
        return regedit_val

    # Define system status.
    def device_os_status(self):
        """
        Returns Device's operating system status.
        - Windows: -> `(Windows, 10/11, 2_H_, XXXXX)`
        - Linux:   -> `(LINUX_DISTRO_NAME, LINUX_DISTRO_VERSION, "GNU/Linux", LINUX_KERNEL_VERSION)`
        - Others: -> `None`.
        """

        import sys

        if self.OS == "Windows":
            _os_major = platform.release()
            _os_build = platform.version()
            _os_update = self.get_regedit(
                "HKLM",
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "DisplayVersion",
            )

            return (platform.system(), _os_major, _os_update, _os_build)
        elif self.OS == "Linux":
            with open("/proc/version", "r") as f:
                _f = f.read().splitlines()
                for _line in _f:
                    _kernel_match = re.match(
                        r"Linux version (\d+)\.(\d+)\.(\d+)\.(\d+)", _line
                    )
                    (
                        _LINUX_KERNEL_MAJOR_VERSION,
                        _LINUX_KERNEL_MINOR_VERSION,
                        _,
                        _,
                    ) = map(int, _kernel_match.groups())
                    _LINUX_KERNEL_VERSION = (
                        f"{_LINUX_KERNEL_MAJOR_VERSION}.{_LINUX_KERNEL_MINOR_VERSION}"
                    )
            with open("/etc/os-release") as f:
                _f = f.read().splitlines()
                for _line in _f:
                    _name_match = re.match(r'^NAME="?(.*?)"?$', _line)
                    _version_match = re.match(r'^VERSION_ID="?(.*?)"?$', _line)

                    if _name_match:
                        _LINUX_DISTRO_NAME = _name_match.group(1)
                    if _version_match:
                        _LINUX_DISTRO_VERSION = _version_match.group(1)

            return (
                _LINUX_DISTRO_NAME,
                _LINUX_DISTRO_VERSION,
                "GNU/Linux",
                _LINUX_KERNEL_VERSION,
            )
        else:
            pass

    def device_cpu_status(self):
        """
        **Warning:** This function may broken in Cluster systems.
        Return CPU status, include its name, architecture, total cpu count.
        -> `(CPU_NAME, CPU_ARCH, CPU_CORES)`
        """

        import os, platform, subprocess, re

        if self.is_windows:
            _cpu_name = self.get_regedit(
                "HKLM",
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "ProcessorNameString",
            )
            _cpu_arch = platform.machine()
            _cpu_core = os.cpu_count()

            return (_cpu_name, _cpu_core, _cpu_arch)

        elif self.is_linux:
            _cpu_name = (
                re.search(
                    r"^\s*Model name:\s*(.+)$",
                    subprocess.run(
                        ["lscpu"], capture_output=True, text=True, check=True
                    ).stdout,
                    re.MULTILINE,
                )
                .group(1)
                .strip()
            )
            _cpu_arch = platform.machine()
            _cpu_core = os.cpu_count()

            return (_cpu_name, _cpu_core, _cpu_arch)

        else:
            # <ADD BSD/Intel_MAC ???>
            ...
            pass

    def device_gpu_list(self):
        """
        Returns a list contains GPU info tuple on Windows platform.
        If on Linux or Windows python environment have no `pywin32` module, we skip test as return `None`.
        - Windows: `[(GPU_NUM, GPU_NAME, GPU_VRAM), (...), ...]` or `None`
        - Linux: `None`
        - Others: `None`
        """
        if self.is_windows:
            GPU_STATUS_LIST = []
            try:
                from win32com import client

                GPU_COUNT = len(
                    client.GetObject("winmgmts:").InstancesOf("Win32_VideoController")
                )

                for i in range(0, GPU_COUNT):
                    _GPU_REG_KEY = str(
                        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                        + f"\\000{i}\\"
                    )
                    GPU_CORE_NAME = self.get_regedit("HKLM", _GPU_REG_KEY, "DriverDesc")
                    if GPU_CORE_NAME != "Microsoft Basic Display Adapter":
                        GPU_VRAM = self.get_regedit(
                            "HKLM", _GPU_REG_KEY, "HardwareInformation.qwMemorySize"
                        )
                        GPU_STATUS_LIST.append(
                            (i, f"{GPU_CORE_NAME}", float(GPU_VRAM / (1024**3)))
                        )
                    else:
                        pass
                return GPU_STATUS_LIST
            except ModuleNotFoundError as e:
                return None
        else:
            return None

    def device_dram_status(self):
        """
        Analyze Device's DRAM Status. Both on Windows and Linux returns a tuple.
        - Windows: `(DRAM_PHYS_TOTAL, DRAM_PHYS_AVAIL, DRAM_VITURAL_AVAIL)`
        - Linux:   `(MEM_PHYS_TOTAL , MEM_PHYS_AVAIL , MEM_SWAP_AVAIL)`
        -  Others: `None`.
        """
        if self.is_windows:
            import ctypes

            class memSTAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem_status = memSTAT()
            mem_status.dwLength = ctypes.sizeof(memSTAT())

            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))

            MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_VITURAL_AVAIL = (
                float(mem_status.ullTotalPhys / (1024**3)),
                float(mem_status.ullAvailPhys / (1024**3)),
                float(mem_status.ullAvailPageFile / (1024**3)),
            )

            return (MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_VITURAL_AVAIL)
        elif self.is_linux:
            import re

            with open("/proc/meminfo", "r") as f:
                _f = f.read().splitlines()
                for line in _f:
                    mem_tol = re.search(r"^MemTotal:\s+(\d+)\s+kB$", line)
                    mem_avl = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", line)
                    mem_swp = re.search(r"^SwapTotal:\s+(\d+)\s+kB$", line)

                    if mem_tol:
                        MEM_PHYS_TOTAL = float(mem_tol.group(1)) / (1024**2)
                    elif mem_avl:
                        MEM_PHYS_AVAIL = float(mem_avl.group(1)) / (1024**2)
                    elif mem_swp:
                        MEM_SWAP_AVAIL = float(mem_swp.group(1)) / (1024**2)

            return (MEM_PHYS_TOTAL, MEM_PHYS_AVAIL, MEM_SWAP_AVAIL)
        else:
            return None
        ...

    def device_disk_status(self):
        """
        Return a tuple with Disk Total/Usage messages.
        `(DISK_DEVICE, DISK_MOUNT_POINT, DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE, DISK_USAGE_RATIO)`
        - `DISK_DEVICE`: Returns `str`. The device "contains this repo" name and its mounting point.
         - Windows: Returns a Drive Letter. eg `F:/` or `F:`
         - Linux: Returns disk's mounted device name and its mounting point. eg `/dev/sdd at: /`
        - `DISK_REPO_POINT`: Returns `str`. TheRock current repo abs path.
        - `DISK_TOTAL_SPACE`: Returns `float`. Current repo stored disk's total space.
        - `DISK_USAGE_SPACE`: Returns `float`. Current repo stored disk's used space.
        - `DISK_AVAIL_SPACE`: Returns `float`. Current repo stored disk's avail space.
        - `DISK_USAGE_RATIO`: Returns `float`. Current repo stored disk's current usage percentage.
        """

        import os, subprocess
        from shutil import disk_usage
        from pathlib import Path

        if self.is_windows:
            repo_path = TheRock.repo()
            repo_disk = os.path.splitdrive(repo_path)[0]

            DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE = disk_usage(repo_disk)

            DISK_USAGE_RATIO = float(DISK_USAGE_SPACE / DISK_TOTAL_SPACE) * 100.0
            DISK_TOTAL_SPACE = DISK_TOTAL_SPACE / (1024**3)
            DISK_USAGE_SPACE = DISK_USAGE_SPACE / (1024**3)
            DISK_AVAIL_SPACE = DISK_AVAIL_SPACE / (1024**3)

            return (
                repo_path,
                repo_disk,
                DISK_TOTAL_SPACE,
                DISK_USAGE_SPACE,
                DISK_AVAIL_SPACE,
                DISK_USAGE_RATIO,
            )

        elif self.is_linux:
            repo_path = TheRock.repo()
            DISK_STATUS_QUERY = (
                subprocess.run(
                    ["df", "-h", os.getcwd()],
                    capture_output=True,
                    check=True,
                    text=True,
                )
                .stdout.strip()
                .splitlines()[1]
                .split()
            )

            DISK_MOUNT_POINT, DISK_MOUNT_DEVICE = (
                DISK_STATUS_QUERY[-1],
                DISK_STATUS_QUERY[0],
            )
            DISK_TOTAL_SPACE, DISK_USAGE_SPACE, DISK_AVAIL_SPACE = disk_usage(
                DISK_MOUNT_POINT
            )
            DISK_USAGE_RATIO = DISK_USAGE_SPACE / DISK_TOTAL_SPACE * 100
            DISK_TOTAL_SPACE = DISK_TOTAL_SPACE / (1024**3)
            DISK_USAGE_SPACE = DISK_USAGE_SPACE / (1024**3)
            DISK_AVAIL_SPACE = DISK_AVAIL_SPACE / (1024**3)

            return (
                repo_path,
                f"{DISK_MOUNT_DEVICE} at: {DISK_MOUNT_POINT}",
                DISK_TOTAL_SPACE,
                DISK_USAGE_SPACE,
                DISK_AVAIL_SPACE,
                DISK_USAGE_RATIO,
            )

    # Define system's tools/utilities status.

    @property
    def git(self):
        return Find_Program("git")

    @property
    def git_lfs(self):
        return Find_Program("git-lfs")

    @property
    def python(self):
        return Find_Program("python") if self.is_windows else Find_Program("python3")

    @property
    def cmake(self):
        return Find_Program("cmake")

    @property
    def ccache(self):
        return Find_Program("ccache")

    @property
    def ninja(self):
        return Find_Program("ninja")

    ## Check if system's GCC toolchain exist.

    @property
    def gcc(self):
        return Find_Program("gcc")

    @property
    def gxx(self):
        return Find_Program("g++")

    @property
    def gfortran(self):
        return Find_Program("gfortran")

    @property
    def gcc_as(self):
        return Find_Program("as")

    @property
    def gcc_ar(self):
        return Find_Program("ar")

    @property
    def ld(self):
        return Find_Program("ld")

    ## Check if system's MSVC toolchain exist and VS20XX variables.

    if is_windows:

        @property
        def cl(self):
            return Find_Program("cl")

        @property
        def ml64(self):
            return Find_Program("ml64")

        @property
        def lib(self):
            return Find_Program("lib")

        @property
        def link(self):
            return Find_Program("link")

        @property
        def rc(self):
            return Find_Program("rc")

        @property
        def VSVER(self):
            """
            Define Visual Studio version.
            """
            _VSVER_NUM = os.getenv("VisualStudioVersion")
            return (
                None if (_VSVER_NUM is None or _VSVER_NUM == "") else float(_VSVER_NUM)
            )

        @property
        def VS20XX(self):
            """
            Define Visual Studio yearly version.
            """
            if self.VSVER is not None:
                match self.VSVER:
                    case 17.0:
                        return "VS2022"
                    case 16.0:
                        return "VS2019"
                    case 15.0:
                        return "VS2017"
                    case 14.0:
                        return "VS2015"
                    case _:
                        return "Legacy"
            else:
                return False

        @property
        def VC_VER(self):
            """Define MSVC build is v14X version."""
            _cl = self.cl.exe
            _vc_ver = os.getenv("VCToolsVersion")

            if _vc_ver == "14.43.34808":
                return "v143"
            elif _vc_ver == "14.29.30133":
                return "v142"
            elif _vc_ver == "14.16.27023":
                return "v141"
            elif (
                _cl
                == r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\cl.exe"
            ):
                return "v140"
            else:
                return None

        @property
        def VS20XX_INSTALL_DIR(self):
            """Find Environment Variable `VSINSTALLDIR` to show the current installed VS20XX location."""
            _dir = os.getenv("VSINSTALLDIR")
            return _dir if _dir is not None else None

        @property
        def VC_SDK(self):
            """Define Visual Studio current used Windows SDK version."""
            _sdk = os.getenv("WindowsSDKVersion")
            return _sdk.replace("\\", "") if _sdk is not None else None

        @property
        def VC_HOST(self):
            """Find VC++ compiler host environment."""
            _host = os.getenv("VSCMD_ARG_HOST_ARCH")
            return _host if _host is not None else None

        @property
        def VC_TARGET(self):
            """Find VC++ compiler target environment."""
            _target = os.getenv("VSCMD_ARG_TGT_ARCH")
            return _target if _target is not None else None

        @property
        def MAX_PATH_LENGTH(self):
            """Find if Windows machine enabled Long PATHs."""
            if self.is_windows:
                _long_path = self.get_regedit(
                    "HKLM",
                    r"SYSTEM\CurrentControlSet\Control\FileSystem",
                    key="LongPathsEnabled",
                )
                return True if _long_path == 1 else False
            else:
                return None

    # Find if device have ROCM_HOME/HIP_DIR.
    # TODO: Finds hipcc, AMD-LLVM toolchain if ROCM_HOME/HIP_DIR found.
    # Currently for detection yet, not yet add to tests.

    @property
    def ROCM_HOME(self):
        return os.getenv("ROCM_HOME")

    @property
    def HIP_PATH(self):
        return os.getenv("HIP_PATH")

    # Define OS configuration.
    @property
    def OS_STATUS(self):
        if self.is_windows:
            return self.OS_NAME
        elif self.is_linux:
            return (
                f"{self.OS_NAME}, GNU/Linux {self.OS_KERNEL} (WSL2)"
                if self.is_WSL2
                else f"{self.OS_NAME}, GNU/Linux {self.OS_KERNEL} (WSL2)"
            )
        else:
            pass

    # Define CPU status.
    @property
    def CPU_STATUS(self):
        return f"{self.CPU_NAME} {self.CPU_CORE} Cores ({self.CPU_ARCH})"

    # Define GPU list status.
    @property
    def GPU_STATUS(self):
        if self._device_gpu_list is not None:
            _gpulist = ""
            for _gpu_info in self._device_gpu_list:
                _gpu_num, _gpu_name, _gpu_vram = _gpu_info
                _gpulist += f"""GPU {_gpu_num}: \t{_gpu_name} ({_gpu_vram:.2f}GB VRAM)
    """
            return _gpulist

        elif self.is_linux:
            return cstring(f"{Emoji.Warn} Skip GPU detection on Linux.", "warn")

        elif self._device_gpu_list is None:
            return cstring(
                f"{Emoji.Warn} Python module 'pywin32' not found. Skip GPU detection.",
                "warn",
            )

    # Define Memory Device status.
    @property
    def MEM_STATUS(self):
        if self.is_windows:
            return f"""Total Physical Memory: {self.PHYSICAL_MEMORY_TOTAL:.2f} GB
                Avail Physical Memory: {self.PHYSICAL_MEMORY_AVAIL:.2f} GB
                Avail Virtual Memory: {self.VIRTUAL_MEMORY_AVAIL:.2f} GB
            """
        elif self.is_linux:
            return f"""Total Physical Memory: {self.PHYSICAL_MEMORY_TOTAL:.2f} GB
                Avail Physical Memory: {self.PHYSICAL_MEMORY_AVAIL:.2f} GB
                Avail Swap Memory: {self.SWAP_MEMORY_AVAIL:.2f} GB
            """
        else:
            pass

    # Define Disk Device status.
    @property
    def DISK_STATUS(self):
        return f"""Disk Total Space: {self.DISK_TOTAL_SPACE:.2f} GB
                Disk Avail Space: {self.DISK_AVAIL_SPACE:.2f} GB
                Disk Used: {self.DISK_USED_SPACE:.2f} GB
                Disk Usage: {self.DISK_USAGE_RATIO:.2f} %
                Current Repo path: {self.DISK_REPO_PATH}, Disk Device: {self.DISK_REPO_MOUNT}
        """

    @property
    def ENV_STATUS(self):
        if self.is_windows:
            return f"""Python ENV: {self.python.exe} ({self.python.env})
                Visual Studio: {self.VS20XX}
                Cygwin: {self.is_cygwin}
                MSYS2: {self.is_msys2}"""
        elif self.is_linux:
            return f"""Python3 VENV: {self.python.exe} ({self.python.env}) | WSL2: {self.is_WSL2}"""
        else:
            return f"""Python3 VENV: {self.python.exe} ({self.python.env}) """

    @property
    def SDK_STATUS(self):
        if self.is_windows:

            _vs20xx_stat = self.VS20XX if self.VS20XX else "Not Detected"
            _vs20xx_msvc = self.VC_VER if self.VC_VER else "Not Detected"
            _vs20xx_sdk = self.VC_SDK if self.VC_SDK else "Not Detected"

            _hipcc_stat = self.HIP_PATH if self.HIP_PATH else "Not Detected"
            _rocm_stat = self.ROCM_HOME if self.ROCM_HOME else "Not Detected"

            return f"""Visual Studio:  {_vs20xx_stat} | Host/Target: {self.VC_HOST} --> {self.VC_TARGET}
                VC++ Compiler:  {_vs20xx_msvc} ({self.cl.version})
                VC++ UCRT:      {_vs20xx_sdk}
                AMD HIP SDK:    {_hipcc_stat}
                AMD ROCm:       {_rocm_stat}
            """

    @property
    def summary(self):
        if self.is_windows:
            print(
                f"""
        ===========    Build Environment Summary    ===========

    OS:         {self.OS_STATUS}
    CPU:        {self.CPU_STATUS}
    {self.GPU_STATUS}
    RAM:        {self.MEM_STATUS}
    STORAGE:    {self.DISK_STATUS}

    ENV:        {self.ENV_STATUS}

    SDK:        {self.SDK_STATUS}

    MAX_PATH_ENABLED: {self.MAX_PATH_LENGTH}
    """
            )

        elif self.is_linux:
            print(
                f"""
        ===========    Build Environment Summary    ===========

    OS:         {self.OS_STATUS}
    CPU:        {self.CPU_STATUS}
    GPU:        {self.GPU_STATUS}
    RAM:        {self.MEM_STATUS}
    STORAGE:    {self.DISK_STATUS}
    """
            )


#####################################################
class DeviceChecker:
    def __init__(self, device: SystemInfo):
        self.device = device
        self.passed, self.warned, self.errs = 0, 0, 0
        self.check_record = []

    def msg_stat(
        self, status: Literal["pass", "warn", "err"], program: str, message: str
    ):
        if isinstance(program, Find_Program):
            match status:
                case "pass":
                    _emoji = Emoji.Pass
                case "warn":
                    _emoji = Emoji.Warn
                case "err":
                    _emoji = Emoji.Err

            return f"[{_emoji}][{program}] {message}"

        elif isinstance(program, str):
            match status:
                case "pass":
                    _emoji = Emoji.Pass
                case "warn":
                    _emoji = Emoji.Warn
                case "err":
                    _emoji = Emoji.Err

            return f"[{_emoji}][{program}] {message}"

    #
    #   check_PROGRAM() -> check_status, except_description, check_Countering_Measure
    #
    #   Defines the tools what we found.
    #   Generally, If tools we not found, we select what we need to print, pre-manually.
    #
    #
    #   Countering Measure on Found status:
    #     > True:   Found(pass)
    #     > False:  Failed(warn)
    #     > None:   Fatal(Err)
    #     > ... :   Not counting in and ignore components count.
    #               Additionally, Ellipsis can be a placeholder wait for future change if we need.
    #
    #     >>>       _result = True / False / None / ...
    #
    #   We handle the conditions like this:
    #
    #       if REQUIRED but NOTFOUND:   return FATAL
    #       elif NOTFOUND and OPTIONAL: return ... (Ellipsis)
    #       elif FOUND but UNEXCEPTED:  return FAILED/FATAL
    #       elif FOUND and FIT:         return True
    #
    # ===========      OS / CPU / Disk Testing      ===========

    def check_Device_OS(self):
        if self.device.is_windows and not (
            self.device.is_cygwin or self.device.is_msys2
        ):
            _stat = self.msg_stat(
                "pass",
                "Operating System",
                f"Detected OS is {self.device.OS_NAME}",
            )
            _except = ""
            _result = True
        elif self.device.is_cygwin or self.device.is_msys2:
            _stat = self.msg_stat(
                "err", "Operating System", f"Detected OS is Cygwin/MSYS2."
            )
            _except = cstring(
                f"""
    We found your platform is Cygwin/MSYS2.
    TheRock only supports pure Linux and pure Windows, currently have no plan to support and ETA on it.
    Please use Visual Studio Environment to build TheRock.

        traceback: Detected on invalid Windows platform Cygwin or MSYS2
    """,
                "err",
            )
            _result = None
        elif self.device.is_linux and (not self.device.is_WSL2):
            _stat = self.msg_stat(
                "pass",
                "Operating System",
                f"Detected OS is {self.device.OS_NAME} {self.device.OS_KERNEL}",
            )
            _except = ""
            _result = True
        elif self.device.is_linux and self.device.is_WSL2:
            _stat = self.msg_stat(
                "err",
                "Operating System",
                f"Detected OS is {self.device.OS_NAME} (GNU/Linux {self.device.OS_KERNEL})",
            )
            _except = cstring(
                f"""
    Found Linux distro {self.device.OS_NAME} is WSL2 environment.
    WSL2 is not yet supported. Please use native Linux or native Windows instead.
        traceback: Detected Linux is WSL2
    """,
                "err",
            )
            _result = None
        else:
            _os = platform.system()
            _stat = self.msg_stat("err", "Operating System", f"Detected OS is {_os}")
            _except = cstring(
                f"""
    We found your Operating System is {_os},  and it's not supported yet.
    Please select x86-64 based Linux or Windows platform for TheRock build.
    We're sorry for any inconvinence.
        traceback: Invalid Operating System {_os}
    """,
                "err",
            )
            _result = None

        return _stat, _except, _result

    def check_Device_ARCH(self):
        _cpu_arch = self.device.CPU_ARCH
        if _cpu_arch in (
            "x64",
            "AMD64",
            "amd64",
            "Intel 64",
            "intel 64",
            "x86-64",
            "x86_64",
        ):
            _stat = self.msg_stat(
                "pass", "CPU Arch", f"Detected Available CPU Arch {_cpu_arch}."
            )
            _except = ""
            _result = True
        else:
            _stat = self.msg_stat(
                "pass", "CPU Arch", f"Detected Invalid CPU Arch {_cpu_arch}."
            )
            _except = cstring(
                f"""
    We detected unsupported CPU Architecture {_cpu_arch}.
    TheRock project currently support x86-64 Architectures, Like AMD RYZEN/Althon/EPYC or Intel Core/Core Ultra/Xeon.
    We're sorry for any inconvinence.
        traceback: Unsupported CPU Architecture {_cpu_arch}
    """,
                "err",
            )
            _result = None

        return _stat, _except, _result

    def check_DISK_USAGE(self):
        _disk_avail = self.device.DISK_AVAIL_SPACE
        _disk_ratio = self.device.DISK_USAGE_RATIO
        _disk_drive = self.device.DISK_REPO_MOUNT

        if _disk_avail < 128 or _disk_ratio > 80:
            _stat = self.msg_stat("warn", "Disk Status", f"Disk space check attention.")
            _except = cstring(
                f"""
    We've checked the workspace disk {_disk_drive} available space could be too small to build TheRock (and PyTorch).
    TheRock builds may needs massive storage for the build, and we recommends availiable disk space with 128GB and usage not over 80%.
    """,
                "warn",
            )
            _result = False

        else:
            _stat = self.msg_stat("pass", "Disk Status", f"Disk space check pass.")
            _except = ""
            _result = True

        return _stat, _except, _result

    # ===========   General Tools/Utilities  ===========
    def check_python(self):
        python = self.device.python
        python.description = "Python 3"
        _env_ = python.env

        if python.MAJOR_VERSION == 2:
            _stat = self.msg_stat("err", "Python 3", f"Found Python 2: {python.exe}")
            _except = cstring(
                f"""
    Found Python is Python 2.
    TheRock not support Python 2. Please Switch to Python 3.9+ versions.
        traceback: Python major version too old
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {python.version}
    """,
                "err",
            )
            _result = None

        elif python.MINOR_VERSION <= 8:
            _stat = self.msg_stat(
                "err",
                "Python 3",
                f"Found Python {python.version} at {python.exe} {_env_}.",
            )
            _except = cstring(
                f"""
    Found Python version: {python.version}
    TheRock not support Python 3.8 and older versions. Please Switch to Python 3.9+ versions.
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {python.version}
    """,
                "err",
            )
            _result = None

        elif python.Free_Threaded == True:
            _stat = self.msg_stat(
                "err", "Python 3", f"Found Python {python.exe} is No-GIL version."
            )
            _except = cstring(
                f"""
    TheRock have not support Free-Threaded Python yet.

        traceback: Found Python3 using Free-Threaded (No-GIL) version
    """,
                "err",
            )
            _result = None

        elif python.MINOR_VERSION >= 14:
            _stat = self.msg_stat(
                "warn",
                "Python 3",
                f"Found Python {python.version} at {python.exe} {_env_}.",
            )
            _except = cstring(
                f"""
    Found Python version: {python.version}
    Warning: These newer Python versions are not yet tested.

        traceback: Python3 version may unstable due to Python3 version too new.
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {python.version}
    """,
                "warn",
            )
            _result = False

        elif python.env == "Global ENV":
            _stat = self.msg_stat(
                "warn",
                "Python 3",
                f"Found Python {python.version} at {python.exe} {_env_}.",
            )
            _except = cstring(
                f"""
    Found Python is Global ENV ({python.exe}).
    We recommends you using venv create a clear Python environment to build TheRock.
    Parts of TheRock installed Python dependices may pollute your Global ENV.

        traceback: Detected Global ENV Python3 environment
            > expected Python3 is Virtual ENV, found Global ENV: {python.exe}
    """,
                "warn",
            )
            _result = False

        elif python.env == "Conda ENV":
            _stat = self.msg_stat(
                "pass",
                "Python 3",
                f"Found Python {python.version} at {python.exe} ({_env_}).",
            )
            _except = cstring(
                f"""
    Note: Found Conda ENV.
    """,
                "hint",
            )
            _result = True

        else:
            _stat = self.msg_stat(
                "pass",
                "Python 3",
                f"Found Python {python.version} at {python.exe} ({_env_}).",
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_git(self):
        git = self.device.git
        git.description = "Git version control system"

        if git.exe is None:
            _stat = self.msg_stat("warn", "Git", f"Cannot found git.")
            _except = cstring(
                f"""
    We cannot find git ({git.description}).
    TheRock needs git program to fetch patches and sub-projects.
    For Windows users, please install it from Git for Windows installer, or winget/choco.
    For Linux users please install it from your Linux distro's package manager.
        PS > winget install --id Git.Git -e --source winget
        PS > choco install git -y
        sh $ apt/dnf install git
        traceback: Required program git not found
        """,
                "err",
            )
            _result = False

        else:
            _stat = self.msg_stat(
                "pass", "Git", f"Found git {git.version} at {git.exe}"
            )
            _except = ""
            _result = True
        return _stat, _except, _result

    def check_gitlfs(self):
        gitlfs = self.device.git_lfs
        gitlfs.description = "Git Large File System"

        if gitlfs.exe is not None:
            _stat = self.msg_stat(
                "pass", "Git-LFS", f"Found git-lfs {gitlfs.version} at {gitlfs.exe}"
            )
            _except = ""
            _result = True

        else:
            if self.device.is_windows:
                _stat = self.msg_stat(
                    "warn", "Git-LFS", f"Cannot found git-lfs {gitlfs.version}."
                )
                _except = cstring(
                    f"""
    We cannot find git-lfs ({gitlfs.description}). We recommends git-lfs for additional tools.
    For Windows users, you can install it from Git-LFS for Windows installer, or winget/choco.
        PS > winget install --id GitHub.GitLFS -e
        PS > choco install git-lfs -y
        traceback: Optional program git-lfs not found
        """,
                    "warn",
                )
                _result = False

            elif self.device.is_linux:
                _stat = self.msg_stat(
                    "err", "Git-LFS", f"Cannot found git-lfs {gitlfs.version}."
                )
                _except = cstring(
                    f"""
    We cannot find git-lfs program as TheRock required.
    For Linux users please install it from your Linux distro's package manager.
        sh $ apt/dnf install git-lfs

        traceback: Required program git-lfs not found
        """,
                    "err",
                )
                _result = None

        return _stat, _except, _result

    def check_cmake(self):
        cmake = self.device.cmake
        cmake.description = "Cross-Platform Make (CMake)"

        if cmake.exe is None:
            _stat = self.msg_stat("err", "CMake", f"Cannot find CMake program.")
            _except = cstring(
                f"""
    We cannot find any possiable CMake program. Please check CMake program is installed and in PATH.
    TheRock is a CMake super project requires CMake program.
    For Windows users, please install CMake for Windows support via Visual Studio Installer.
    For Linux users please install it via package manager.
        sh $ apt/dnf install cmake
        sh $ pacman -S cmake

        traceback: Required CMake program not installed or in PATH
    """,
                "err",
            )
            _result = None

        elif cmake.MAJOR_VERSION >= 4:
            _stat = self.msg_stat(
                "warn", "CMake", f"Found CMake {cmake.MAJOR_VERSION} at {cmake.exe}"
            )
            _except = cstring(
                f"""
    We found you're using CMake program is CMake 4 (cmake {cmake.version}).
    The support of CMake 4 is still not confirmed, and the different CMake behavior may effect TheRock build.
    Please downgrade it and re-try again.

        traceback: CMake program too new may cause unstable
            expected: 3.25.X ≤ cmake ≤ 3.31.X, found {cmake.version}
    """,
                "warn",
            )
            _result = False

        elif cmake.MAJOR_VERSION == 3 and cmake.MINOR_VERSION < 25:
            _stat = self.msg_stat(
                "warn", "CMake", f"Found CMake {cmake.version} at {cmake.exe}"
            )
            _except = cstring(
                f"""
    We found you're CMake program is ({cmake.version}).
    Your CMake version is too old to TheRock project that requires version 3.25.
    Please upgrade your CMake program version.

        traceback: CMake program too old excluded by TheRock Top-Level CMakeLists.txt rules `cmake_minimum_required()`
            expected: 3.25.X ≤ cmake ≤ 3.31.X, found {cmake.version}
    """,
                "warn",
            )
            _result = False

        else:
            _stat = self.msg_stat(
                "pass", "CMake", f"Found CMake {cmake.version} at {cmake.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_ninja(self):
        ninja = self.device.ninja
        ninja.description = "A small buildsystem focus on speed"

        if ninja.exe is None:
            _stat = self.msg_stat("err", "Ninja-build", f"Ninja Generator not found.")
            _except = cstring(
                f"""
    We can't find required generator "Ninja".
    Ninja is TheRock project current supported generator.
    For Windows users, please use ninja from VS20XX, or build from source. Install from command line, please avoid version 1.11.
    For Linux users, please install it via package manager or build from source.
        PS > pip/uv pip/choco/winget install ninja
        sh $ apt/dnf install ninja-build
        sh $ pacman -S ninja-build
        traceback: Missing Required generator 'Ninja'
    """,
                "err",
            )
            _result = None

        elif ninja.MINOR_VERSION == 11:
            _stat = self.msg_stat(
                "warn", "Ninja-build", f"Found Ninja {ninja.version} at {ninja.exe}"
            )
            _except = cstring(
                f"""
    We found your ninja generator is {ninja.version}.
    This version of ninja program could unstable and hit some unknown CMake re-run deadloop.
    Please consider downgrade it <= 1.10, or upgrade it to >= 1.12, or self build a ninja generator from source.
    """,
                "warn",
            )
            _result = False

        else:
            _stat = self.msg_stat(
                "pass", "Ninja-build", f"Found Ninja {ninja.version} at {ninja.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_ccache(self):
        ccache = self.device.ccache
        ccache.description = "Compiler Cache"

        if ccache.exe is None:
            _stat = self.msg_stat("warn", "ccache", f"ccache not found.")
            _except = ""
            _result = ...

        else:
            _stat = self.msg_stat(
                "pass", "ccache", f"Found ccache {ccache.version} at {ccache.exe}"
            )
            _except = cstring(
                f"""
    Note: ccache is a optional compoment. You can Ignore it if you not using ccache.
    Note: ccache is still investigating the exact proper options on Windows platform.
    If you want to avoid this issue, please ignore ccache setup and to avoid use
       CMake cache `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER`.
    """,
                "warn",
            )
            _result = True

        return _stat, _except, _result

    #  ===========  GNU GCC Compiler Toolchain  ===========
    def check_gcc(self):
        gcc = self.device.gcc

        if self.device.is_linux:
            if gcc.exe is None:
                _stat = self.msg_stat("err", "GCC", f"Cannot find {gcc.name} compiler.")
                _except = cstring(
                    f"""
    We can't find required C/C++ compilers.
    On Linux platform we need GCC compilers to compile.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GCC program {gcc.name} not installed or not in PATH
           > Hint: Missing GCC Compiler C Language Frontend
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    "GCC",
                    f"Found GCC compiler program gcc {gcc.version} at {gcc.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_gxx(self):
        gxx = self.device.gxx

        if self.device.is_linux:
            if gxx.exe is None:
                _stat = self.msg_stat("err", "GCC", f"Cannot find g++ .")
                _except = cstring(
                    f"""
    We can't find required C/C++ compilers.
    On Linux platform we need GCC compilers to compile.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GCC program {gxx.name} not installed or not in PATH
           > Hint: Missing GCC Compiler C++ Language Frontend
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    "GCC",
                    f"Found GCC compiler program g++ {gxx.version} at {gxx.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_gfortran(self):
        gfortran = self.device.gfortran

        if gfortran.exe is None:
            _stat = self.msg_stat(
                "err", "gfortran", f"Cannot found GCC Fortran Compiler."
            )
            _except = cstring(
                f"""
    We cannot find any available Fortran Compiler.
    On Windows, please install gfortran in your device, via Strawberry/MinGW-builds etc.
    On Linux, please install via package managers.
        sh $ apt/dnf install gfortran
        sh $ pacman -S gfortran

        traceback: No available fortran compiler
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass",
                "gfortran",
                f"Found Fortran compiler gfortran {gfortran.version} at {gfortran.exe}",
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_gcc_ar(self):
        gcc_ar = self.device.gcc_ar

        if self.device.is_linux:
            if gcc_ar.exe is None:
                _stat = self.msg_stat("err", "Binutils", f"Can't found GNU Archiver.")
                _except = cstring(
                    f"""
    We can't found GNU toolchain required Archiver/Linker.
    Please configure your binutils is installed correctly.
    Please install it via package managers.
        sh $ apt/dnf install binutils

        traceback: GNU binutils {gcc_ar} not installed or not in PATH
           > Hint: Missing GNU Archiver
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    "Binutils",
                    f"Found GNU Archiver {gcc_ar.version} at {gcc_ar.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_gcc_as(self):
        gcc_as = self.device.gcc_as

        if self.device.is_linux:
            if gcc_as.exe is None:
                _stat = self.msg_stat("err", "Binutils", f"Can't found GNU Assembler.")
                _except = cstring(
                    f"""
    We can't found GNU {gcc_as}.
    Please configure your binutils is installed correctly.
    Please install it via package managers.
        sh $ apt/dnf install binutils

        traceback: GNU binutils {gcc_as} not installed or not in PATH
           > Hint: Missing GNU Assembler
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    "Binutils",
                    f"Found GNU Assembler {gcc_as.version} at {gcc_as.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_ld(self):
        ld = self.device.ld
        ld.description = "GNU Linker"

        if self.device.is_linux:
            if ld.exe is None:
                _stat = self.msg_stat("err", "Binutils", f"Can't found GNU Linker.")
                _except = cstring(
                    f"""
    We can't found GNU toolchain required Archiver/Linker.
    Please configure your binutils is installed correctly.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GNU binutils {ld} not installed or not in PATH
           > Hint: Missing GNU Linker
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass", "Binutils", f"Found GNU Linker {ld.version} at {ld.exe}"
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    #  ===========  MSVC Compiler Toolchain  ===========
    def check_msvc(self):
        cl = self.device.cl

        if cl.exe is None:
            _stat = self.msg_stat(
                "err",
                "MSVC",
                f"Cannot found Microsoft Optimized C/C++ compiler Driver cl.exe.",
            )
            _except = cstring(
                f"""
    We can't found any available MSVC compiler on your Windows device.
    MSVC (Microsoft Visual C/C++), The C/C++ compliler for native Windows development.
    Please re-configure your Visual Studio installed C/C++ correctly.
        Visual Studio Installer > C/C++ Development for Desktop:
        - MSVC v14X
        - MSVC MFC
        - MSVC ALT
        - Windows SDK 10.0.XXXXX
        - CMake for Windows
        - C++ Address Sanitizer
        traceback: Required VC++ compiler not found
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass",
                "MSVC",
                f"Found MSVC {self.device.VC_VER} ({cl.version}) at {cl.exe}",
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_ml64(self):
        ml64 = self.device.ml64

        if ml64.exe is None:
            _stat = self.msg_stat(
                "err",
                "MSVC",
                f"Cannot found Microsoft Macro Assembler.",
            )
            _except = cstring(
                f"""
    We can't found any available Microsoft Macro Assembler on your Windows device.
    ml64.exe is the Assembler for native Windows development.
    Please re-configure your Visual Studio installed C/C++ correctly.
        Visual Studio Installer > C/C++ Development for Desktop:
        - MSVC v14X
        - MSVC MFC
        - MSVC ALT
        - Windows SDK 10.0.XXXXX
        - CMake for Windows
        - C++ Address Sanitizer
        traceback: Required Macro Assembler not found
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass",
                "MSVC",
                f"Found Microsoft Macro Assembler {ml64.version} at {ml64.exe}",
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_lib(self):
        lib = self.device.lib

        if lib.exe is None:
            _stat = self.msg_stat("err", "MSVC", f"Cannot found MSVC program lib.exe.")
            _except = cstring(
                f"""
    We cannot find MSVC toolchain's archiver.
    Please check your Microsoft VC++ installation is correct, or re-check if the install is broken.

        traceback: MSVC lib.exe ({lib.exe}) not found
    """,
                "err",
            )
            _result = None
        else:
            _stat = self.msg_stat(
                "pass", "MSVC", f"Found MSVC program lib.exe at {lib.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_link(self):
        link = self.device.link

        if link.exe is None:
            _stat = self.msg_stat(
                "err", "MSVC", f"Cannot found Microsoft Incremental Linker."
            )
            _except = cstring(
                f"""
    We cannot find MSVC linker link.exe (Microsoft Incremental Linker).
    Please re-check your MSVC installation if it's broken.
        traceback: Missing MSVC required linker compoments link.exe
    """,
                "err",
            )
            _result = None
        else:
            _stat = self.msg_stat(
                "pass", "MSVC", f"Found Microsoft Incremental Linker at {link.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_rc(self):
        rc = self.device.rc

        if rc.exe is None:
            _stat = self.msg_stat(
                "err", "Windows SDK", f"Cannot found Microsoft Resource Compiler."
            )
            _except = cstring(
                f"""
    We cannot find rc.exe (Microsoft Resource Compiler) in your Windows SDK, or you have no Windows SDK installed.
    Please re-configure your MSVC and Windows SDK installation via Visual Studio.
        Visual Studio Installer > C/C++ Development for Desktop:
            - MSVC v14X
            - MSVC MFC
            - MSVC ALT
            - Windows SDK 10.0.XXXXX
            - CMake for Windows
            - C++ Address Sanitizer
        traceback: Microsoft Resource Compiler not found in Windows SDK or Windows SDK not installed
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass", "Windows SDK", f"Found Microsoft Resource Compiler at {rc.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    #  ==============    Find Environment    =============
    #
    #   TaiXeflar: Only Windows needs to do this: VS20XX and MAX_PATH.
    #   Windows: Visual Studio have different profiles to select compile host machine and targeted machine.
    #   Windows: Detects Long PATHs are enabled.
    #
    #   All platforms: Detects Disk usage.
    #       TheRock is a CMake super-project with lots of builds.
    #       It could easy take over 100GB usage.

    def check_VS20XX(self):
        _env = self.device.VS20XX
        if _env is None:
            _stat = self.msg_stat(
                "err", "Visual Studio", f"Cannot found Visual Studio Environment."
            )
            _except = cstring(
                f"""
                We can't found a available Visual Studio install version.
    This error might be you don't have Visual Studio installed, or running out of Visual Studio environment profile.
    Please open a Visual Studio environment Terminal and re-run this diagnosis script.
    By open A Windows Terminal `wt.exe` and open VS20XX profile from tab.
    - Developer Command prompt for Visual Studio 20XX
    - Developer PowerShell for Visual Studio 20XX
        traceback:  TheRock on Windows build requires Visual Studio 2022/2019/2017/2015 environment
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass", "Visual Studio", f"Found Visual Studio {_env}."
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_VC_HOST(self):
        _host = self.device.VC_HOST
        _cl = self.device.cl
        if _host == "x64":
            _stat = self.msg_stat("pass", "VC Host", f"MSVC Host compiler is {_host}.")
            _except = ""
            _result = True

        else:
            _stat = self.msg_stat("err", "VC Host", f"MSVC Host compiler is {_host}.")
            _except = cstring(
                f"""
    We detected your CPU architecture is {self.device.CPU_ARCH}, but your VC++ Host is {_host}.
    This might hit compile runtime error. Please re-open and select correct Visual Studio environment profile.
        traceback: Expected CPU Architecture {self.device.CPU_ARCH}, but VC++ compiler host is {_host}
    """,
                "err",
            )
            _result = None

        return _stat, _except, _result

    def check_VC_TARGET(self):
        _target = self.device.VC_TARGET
        _cl = self.device.cl
        if _target == "x64":
            _stat = self.msg_stat(
                "pass", "VC Target", f"MSVC compile target is {_target}."
            )
            _except = ""
            _result = True
        else:
            _stat = self.msg_stat(
                "err", "VC Target", f"MSVC Host compiler is {_target}."
            )
            _except = cstring(
                f"""
    We found you are compiling to target {_target}.
    TheRock project not supporting compile to {_target} target.
    If you insisting compile may cause the build unstable.
        traceback: Unexpected compile target {_target}
    """,
                "err",
            )
            _result = None

        return _stat, _except, _result

    def check_VC_SDK(self):
        _sdk = self.device.VC_SDK
        if _sdk is None:
            _stat = self.msg_stat("err", "WindowsSDK", f"Cannot found Windows SDK.")
            _except = cstring(
                f"""
    We cannot find available Windows SDK.
    Windows SDK provides Universal CRT(UCRT) library and Resource Compiler.
    Please re-configure Visual Studio installed compoments.

        traceback: No available Windows SDK detected
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass", "Windows SDK", f"Found Windows SDK version {_sdk}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_MAX_PATH(self):
        _status = self.device.MAX_PATH_LENGTH
        if _status:
            _stat = self.msg_stat("pass", "Long Path", f"Windows Long PATHs Enabled.")
            _except = ""
            _result = True
        else:
            _stat = self.msg_stat("warn", "Long Path", f"Windows Long PATHs Disabled.")
            _except = cstring(
                f"""
    We found you have not enable Windows Long PATH support yet.
    This could hits unexpected error while we compile/generates long name files.
    Please enable this feature via one of these solution:
        > Using Registry Editor(regedit) or using Group Policy
        > Restart your device.

        traceback: Windows Enable Long PATH support feature is Disabled
            Registry Key Hint: HKLM:/SYSTEM/CurrentControlSet/Control/FileSystem LongPathsEnabled = 0 (DWORD)
    """,
                "warn",
            )
            _result = False

        return _stat, _except, _result

    #  ==============   Summarize   ================
    #               Summarize count.


def check_summary(result: DeviceChecker):
    pass_num = cstring(result.check_record.count(True), color=(55, 255, 125))
    warn_num = cstring(result.check_record.count(False), color="warn")
    err_num = cstring(result.check_record.count(None), color="err")

    print(
        f"""
                            Compoments check {pass_num} Passed, {warn_num} Warning, {err_num} Fatal Error"""
    )


def run_test():

    device = SystemInfo()
    tester = DeviceChecker(device=device)

    if device.is_windows:
        test = [
            tester.check_Device_OS,
            tester.check_Device_ARCH,
            tester.check_DISK_USAGE,
            tester.check_MAX_PATH,
            tester.check_python,
            tester.check_git,
            tester.check_gitlfs,
            tester.check_cmake,
            tester.check_ccache,
            tester.check_ninja,
            tester.check_gfortran,
            tester.check_VS20XX,
            tester.check_msvc,
            tester.check_VC_HOST,
            tester.check_VC_TARGET,
            tester.check_ml64,
            tester.check_lib,
            tester.check_link,
            tester.check_VC_SDK,
            tester.check_rc,
        ]

    elif device.is_linux:
        test = [
            tester.check_Device_OS,
            tester.check_Device_ARCH,
            tester.check_DISK_USAGE,
            tester.check_python,
            tester.check_git,
            tester.check_gitlfs,
            tester.check_cmake,
            tester.check_ccache,
            tester.check_ninja,
            tester.check_gcc,
            tester.check_gxx,
            tester.check_gfortran,
            tester.check_gcc_as,
            tester.check_gcc_ar,
            tester.check_ld,
        ]

    for items in test:
        _stat, _except, _result = items.__call__()
        print(f"{_stat} {_except}")
        tester.check_record.append(_result)

    check_summary(tester)


def main():
    therock_detect_start = time.perf_counter()
    TheRock.__logo__()
    device = SystemInfo()
    device.summary

    print(
        r"""
        ===========    Start detecting components for building ROCm\TheRock    ===========
    """
    )

    run_test()

    therock_detect_terminate = time.perf_counter()
    therock_detect_time = float(therock_detect_terminate - therock_detect_start)
    therock_detect_runtime = cstring(f"{therock_detect_time:2f}", "hint")
    print(
        f"""
        ===========    TheRock build pre-diagnosis script completed in {therock_detect_runtime} seconds    ===========
    """
    )

    time.sleep(0.5)


# Launcher.
if __name__ == "__main__":
    main()
