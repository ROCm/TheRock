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
import os, re, platform, sys
from pathlib import Path
import time

# Define Color string print.
#   > cprint is for print()
#   > cstring is for colored string.
class cstring:
    """
    ## Color String \n\n
    Returns with ANSI escape code formated string, with colors by (R, G, B).\n
    This display feature is supported on macOS/Linux Terminal, Windows Terminal, VSCode Terminal, and VSCode Jupyter Notebook. (etc?)

    ### Usage
    `<STR_VAR> = cstring(string, color)`
    - msg: `str` or `cstring` type. `C5H8NO4Na` is Invalid type.
    - color: A user specified `tuple` with each value Ranged from `0` ~ `255` `(R, G, B)`.\n\t
    ```
    >>> your_text = cstring(msg="AMD RADEON", color=(255, 0, 0))
    >>> your_text
    ```
    - If color's RGB not passed will be full white. Color also can be these keywords:
        - "err"
        - "warn"
        - "pass"
        - "Windows"
        - "Cygwin"
        - "msys2"
        - "ubuntu"
        - "fedora"
        - "Discord"
        - `Any`: Any type will ignore it as default white (255 ,255, 255).
    """

    def __init__(
        self,
        msg: Union[str, cstring],
        color: Union[
            Optional[
                Literal[
                    "err",
                    "warn",
                    "hint",
                    "Windows",
                    "Cygwin",
                    "msys2",
                    "ubuntu",
                    "fedora",
                    "Discord",
                ]
            ],
            Tuple[int, int, int],
        ]
        | None = None,
    ) -> str:
        # super().__

        if isinstance(color, tuple):
            self.r, self.g, self.b = color
        else:
            match color:
                case "err":
                    r, g, b = (255, 61, 61)
                case "warn":
                    r, g, b = (255, 230, 66)
                case "hint":
                    r, g, b = (150, 255, 255)
                case "Windows":
                    r, g, b = (0, 79, 225)
                case "Cygwin":
                    r, g, b = (0, 255, 0)
                case "msys2":
                    r, g, b = (126, 64, 158)
                case "ubuntu":
                    r, g, b = (221, 72, 20)
                case "fedora":
                    r, g, b = ...
                case "Discord":
                    r, g, b = (88, 101, 242)
                case _:
                    r, g, b = (255, 255, 255)
            self.r, self.g, self.b = r, g, b

        if type(msg) is cstring:
            self.info = msg.info  # != C5H8NO4Na
        else:
            self.info = msg  # != C5H8NO4Na

    def __str__(self):
        return f"\033[38;2;{self.r};{self.g};{self.b}m{self.info}\033[0m"

    def __repr__(self):
        return self.__str__()


class Emoji:
    Pass = "✅"
    Warn = "⚠️"
    Err = "❌"


# Define AMD arrow logo and therock current head.
class TheRock:
    """
    ## TheRock class
    AMD ROCm/TheRock project.
    ### Methods
    `head()`: `str`. Returns Repo cloned main's head.\n
    `repo()`: `str`. Returns Repo's abs path.\n
    `license()`: `None`. Displays TheRock repo's Public License.\n\n
    ### Fake magic methods
    `__logo__()`: Advanced Micro Devices Logo. Displays AMD Arrow Logo and current git HEAD.\n
    ![image](https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png)\n
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
    def license():
        with open(Path(f"{TheRock.repo()}/LICENSE").resolve()) as f:
            lic = f.read()
        TheRock.__logo__()
        print(cstring(lic, "hint"))

    @staticmethod
    def __logo__():

        """
        ![image](https://upload.wikimedia.org/wikipedia/commons/6/6a/AMD_Logo.png)\n
        # Advanced Micro Devices Inc.
        """

        _REPO_HEAD = TheRock.head()

        print(
            f"""\n\n\n\n\n
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

    @staticmethod
    def help():
        TheRock.__logo__()
        _repo_ = cstring("https://github.com/ROCm/TheRock", "err")
        _discord_ = cstring("https://discord.com/invite/amd-dev", "Discord")
        _warn_, _FAILED_ = cstring("Warning", "warn"), cstring("Failed", "warn")
        _err_, _FATAL_ = cstring("Error", "err"), cstring("Fatal", "err")
        print(
            f"""
        - Diagnosis: No arguments pass.
            Direct run will detect your system's hardware/software info.
            There will several messages, by using colored {_warn_}/{_err_} message, shows tests are {_FAILED_}/{_FATAL_}.
            This script shouldn't throw a python running error -- report it if you hits error.
                sh $ python3 ./diagnose.py
                PS > python  ./diagnose.py
            This diagnose script have POSIX/MS-DOS arg-parsing style compatability.
                sh $ python3 ./diagnosis.py --ARGS -ARGS /ARGS
                PS > python  ./diagnosis.py --ARGS -ARGS /ARGS

        - Help: Display this command line usage.
                sh $ python3 ./diagnosis.py --help -help -? /help /?
                PS > python  ./diagnosis.py --help -help -? /help /?

        - Issue: This diagnosis script could be somewhere buggy or broken.
                Please report issue to GitHub repository here. {_repo_}
                Or join AMD Developer Community Discord server here. {_discord_}
    """
        )


class where:
    """
    ## Where
    Find program that similar to Windows `where.exe`.
    ```
     PowerShell: PS > where.exe python
     CMD:           > where.exe python
     Python REPL: >>> where("python")
    ```
    Let's make something elation.
    ### init(executable:`str`)
    Analyze a program with its lower name and find its PATH in system.
    ```
     python = where("Python")
    ```
    ### exe:
    Returns where object founded program's PATH.
    ```
     >>> python = where("python")
     >>> py_path = python.exe
    ```
    ### version:
    Returns where object founded program's version number.
    ```
     >>> python = where("python")
     >>> X, Y, Z = python.version
     >>> XYZ = python.version
     >>> X, Y, Z = python.MAJOR_VERSION, python.MINOR_VERSION, python.PATCH_VERSION
    ```
    """

    # name map:
    # gcc -> GCC
    # cl.exe -> MSVC

    __name_map__ = {
        "git": "Git",
        "git-lfs": "Git-LFS",
        "python": "Python 3",
        "python3": "Python 3",
        "uv": "Astral UV",
        "cmake": "CMake",
        "ccache": "ccache",
        "ninja": "Ninja",
        "cl": "MSVC",
        "ml64": "MSVC",
        "lib": "MSVC",
        "link": "MSVC",
        "rc": "Windows SDK",
        "gcc": "gcc",
        "g++": "g++",
        "gfortran": "gfortran",
        "as": "as",
        "ar": "ar",
        "ld": "ld",
    }

    def __init__(self, executable: str):
        super().__init__()
        """
        Set where object's specified progranm name.
        executable: `str`
        """
        import shutil, subprocess, re

        self._name = executable.lower()
        _find = shutil.which(self._name)
        self.exe = (
            _find.replace("\\", "/").replace("EXE", "exe")
            if _find is not None
            else None
        )

        if self.exe is None:
            self.version_num = None
        else:
            match executable:
                case "cl" | "link" | "lib" | "ml64":
                    self.version_num = (
                        os.getenv("VCToolsVersion")
                        if os.getenv("VCToolsVersion")
                        else None
                    )
                case "rc":
                    if (
                        os.getenv("WindowsSDKVersion") != "\\"
                        or os.getenv("WindowsSDKVersion") is not None
                    ):
                        self.version_num = os.getenv("WindowsSDKVersion").replace(
                            "\\", ""
                        )
                    else:
                        self.version_num = None
                case "ar" | "as" | "ld":
                    self.MAJOR_VERSION, self.MINOR_VERSION = map(
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
                    self.version_num = f"{self.MAJOR_VERSION}.{self.MINOR_VERSION}"
                case "nvcc":
                    self.MAJOR_VERSION, self.MINOR_VERSION, self.PATCH_VERSION = map(
                        int,
                        re.search(
                            r"V(\d+)\.(\d+)\.(\d+)",
                            subprocess.run(
                                [self.exe, "--version"],
                                capture_output=True,
                                check=True,
                                text=True,
                            ).stdout.strip(),
                        ).groups(),
                    )
                    self.version_num = f"{self.MAJOR_VERSION}.{self.MINOR_VERSION}.{self.PATCH_VERSION}"

                case "python" | "python3":
                    (
                        self.MAJOR_VERSION,
                        self.MINOR_VERSION,
                        self.PATCH_VERSION,
                        self.release,
                        _,
                    ) = sys.version_info
                    self.version_num = f"{self.MAJOR_VERSION}.{self.MINOR_VERSION}.{self.PATCH_VERSION}"

                    if os.getenv("CONDA_PREFIX") is not None:
                        self._env = True
                        self.env = "Conda ENV"
                        self.env_name = os.getenv("CONDA_DEFAULT_ENV")
                        self.env_dir = os.getenv("CONDA_PREFIX")
                    elif sys.prefix == sys.base_prefix:
                        self._env = False
                        self.env = "Global ENV"
                        self.env_name = ""
                        self.env_dir = sys.prefix
                    elif os.getenv("VIRTUAL_ENV") is not None:
                        self._env = True
                        _cfg = Path(f"{sys.prefix}/pyvenv.cfg").resolve()
                        with open(_cfg, "r") as file:
                            _conf = file.read()
                        self.env = "uv VENV" if "uv" in _conf else "Python VENV"
                        self.env_dir = sys.exec_prefix
                        self.env_name = os.getenv("VIRTUAL_ENV_PROMPT")

                case _:
                    self.MAJOR_VERSION, self.MINOR_VERSION, self.PATCH_VERSION = map(
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
                    self.version_num = f"{self.MAJOR_VERSION}.{self.MINOR_VERSION}.{self.PATCH_VERSION}"

        self._desc_: Optional[str] = None

    def __str__(self):
        return self.exe

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        return self.__name_map__.get(self._name, self._name)

    @property
    def description(self) -> str:
        return f"{self._desc_}" if self._desc_ is not None else ""

    @description.setter
    def description(self, info: Optional[str] = None):
        self._desc_ = info

    @property
    def version(self):
        if self.exe is None:
            return None
        else:
            return self.version_num


class Device:
    """
    ## class Device \n\n
    A Device class for capturing system info.
    """

    def __init__(self):
        super().__init__()

        import platform

        # Define OS version.
        _device_status_set = self.device_os_status()
        if platform.system() == "Windows":
            self.OS = "Windows"
            self.OS_NAME = f"{_device_status_set[0]} {_device_status_set[1]}"
            self.OS_PATCH = _device_status_set[2]
            self.OS_BUILD = _device_status_set[3]

        elif platform.system() == "Linux":
            self.OS = "Linux"
            self.OS_TYPE = f"{_device_status_set[0]} {_device_status_set[1]}"
            self.OS_KERNEL = _device_status_set[2]
        else:
            ...

        # Define CPU configuration.
        self.CPU_NAME, self.CPU_ARCH, self.CPU_CORE = self.device_cpu_status()

        # Define GPU configuration list.
        self.GPU_LIST = self.device_gpu_list()

        # Define Device Memory status.
        if self.WINDOWS:
            (
                self.MEM_PHYS_TOTAL,
                self.MEM_PHYS_AVAIL,
                self.MEM_VIRTUAL_AVAIL,
            ) = self.device_dram_status()
        elif self.LINUX:
            (
                self.MEM_PHYS_TOTAL,
                self.MEM_PHYS_AVAIL,
                self.MEM_SWAP_AVAIL,
            ) = self.device_dram_status()
        else:
            pass

        # Define Device Storage status.
        (
            self.DISK_REPO_PATH,
            self.DISK_REPO_MOUNT,
            self.DISK_TOTAL_SPACE,
            self.DISK_USED_SPACE,
            self.DISK_AVAIL_SPACE,
            self.DISK_USAGE_RATIO,
        ) = self.device_disk_status()

        # Define is Windows status.

    @property
    def WINDOWS(self):
        return True if self.OS == "Windows" else False

    if WINDOWS:
        # Define if Windows environment is Cygwin/MSYS2, or We expected VS20XX.

        @property
        def CYGWIN(self):
            return True if sys.platform == "cygwin" else False

        @property
        def MSYS2(self):
            return True if sys.platform == "msys" else False

        @property
        def VSVER(self):
            if os.getenv("VisualStudioVersion") is not None:
                return float(os.getenv("VisualStudioVersion"))
            else:
                None

        @property
        def VS20XX(self):
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
                False

    # Define is Linux status.
    @property
    def LINUX(self):
        return True if self.OS == "Linux" else False

    if LINUX:
        # Define if Linux is WSL2.
        @property
        def WSL2(self):
            with open("/proc/version", "r") as _:
                _ = _.read().splitlines()
            return True if "microsoft-standard-WSL2" in _ else False

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
        `root_key`: Root Keys or Predefined Keys.\nYou can type-in Regedit style or pwsh style as the choice below:\n
        - `HKEY_LOCAL_MACHINE` with pwsh alias `HKLM` \n
        - `HKEY_CURRENT_USER` with pwsh alias `HKCU` \n
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

        if sys.platform == "win32":
            _os_major = platform.release()
            _os_build = platform.version()
            _os_update = self.get_regedit(
                "HKLM",
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
                "DisplayVersion",
            )

            return (platform.system(), _os_major, _os_update, _os_build)
        elif sys.platform.capitalize() == "Linux":
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
        **Warning:** This function may broken in Cluster systems.\n
        Return CPU status, include its name, architecture, total cpu count.
        -> `(CPU_NAME, CPU_ARCH, CPU_CORES)`
        """

        import os, platform, subprocess, re

        if self.WINDOWS:
            _cpu_name = self.get_regedit(
                "HKLM",
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                "ProcessorNameString",
            )
            _cpu_arch = platform.machine()
            _cpu_core = os.cpu_count()

            return (_cpu_name, _cpu_arch, _cpu_core)

        elif self.LINUX:
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

            return (_cpu_name, _cpu_arch, _cpu_core)

        else:
            # <ADD BSD/Intel_MAC ???>
            ...
            pass

    def device_gpu_list(self):
        """
        Returns a list contains GPI info tuple on Windows platform.\n
        If on Linux or Windows python environment have no `pywin32` module, we skip test as return `None`. \n
        - Windows: `[(GPU_NUM, GPU_NAME, GPU_VRAM), (...), ...]` or `None`\n
        - Linux: `None`
        - Others: `None`
        """
        if self.WINDOWS:
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
        Analyze Device's DRAM Status. Both on Windows and Linux returns a tuple.\n
        - Windows: `(DRAM_PHYS_TOTAL, DRAM_PHYS_AVAIL, DRAM_VITURAL_AVAIL)`\n
        - Linux:   `(MEM_PHYS_TOTAL , MEM_PHYS_AVAIL , MEM_SWAP_AVAIL)`\n
        -  Others: `None`.
        """
        if self.WINDOWS:
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
        elif self.LINUX:
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
        - `DISK_DEVICE`: Returns `str`. The device "contains this repo" name and its mounting point.\n
         - Windows: Returns a Drive Letter. eg `F:/` or `F:`\n
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

        if self.WINDOWS:
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

        elif self.LINUX:
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
    if True:

        @property
        def git(self):
            return where("git")

        @property
        def git_lfs(self):
            return where("git-lfs")

        @property
        def python(self):
            return where("python") if self.WINDOWS else where("python3")

        @property
        def uv(self):
            return where("uv")

        @property
        def cmake(self):
            return where("cmake")

        @property
        def ccache(self):
            return where("ccache")

        @property
        def ninja(self):
            return where("ninja")

    ## Check if system's GCC toolchain exist.
    if True:

        @property
        def gcc(self):
            return where("gcc")

        @property
        def gxx(self):
            return where("g++")

        @property
        def gfortran(self):
            return where("gfortran")

        @property
        def gcc_as(self):
            return where("as")

        @property
        def gcc_ar(self):
            return where("ar")

        @property
        def ld(self):
            return where("ld")

    ## Check if system's MSVC toolchain exist. If not WINDOWS just return Not found and None.

    if WINDOWS:

        @property
        def msvc(self):
            return where("cl")

        @property
        def ml64(self):
            return where("ml64")

        @property
        def lib(self):
            return where("lib")

        @property
        def link(self):
            return where("link")

        @property
        def rc(self):
            return where("rc")

    if WINDOWS:

        @property
        def hipcc(self):
            return where("hipcc")

    # Check if Windows have HIP SDK, ROCM_HOME.
    if WINDOWS:

        @property
        def ROCM_HOME(self):
            _rocm_home = os.getenv("ROCM_HOME")
            return _rocm_home if _rocm_home is not None else None

        @property
        def HIP_PATH(self):
            hipcc = self.hipcc
            if hipcc.exe is not None:
                _hip_path = Path(hipcc.exe).parent.parent.resolve()
            else:
                _hip_path = os.getenv("HIP_PATH")
            return _hip_path if _hip_path is not None else None

        @property
        def VC_VER(self):
            _cl = self.msvc.exe
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
            _dir = os.getenv("VSINSTALLDIR")
            return _dir if _dir is not None else None

        @property
        def VS_SDK(self):
            _sdk = os.getenv("WindowsSDKVersion")
            return _sdk.replace("\\", "") if _sdk is not None else None

        @property
        def VS_HOST(self):
            _host = os.getenv("VSCMD_ARG_HOST_ARCH")
            return _host if _host is not None else None

        @property
        def VS_TARGET(self):
            _target = os.getenv("VSCMD_ARG_TGT_ARCH")
            return _target if _target is not None else None

        @property
        def MAX_PATH_LENGTH(self):
            if self.WINDOWS:
                _long_path = self.get_regedit(
                    "HKLM",
                    r"SYSTEM\CurrentControlSet\Control\FileSystem",
                    key="LongPathsEnabled",
                )
                return True if _long_path == 1 else False
            else:
                return None

    # Define OS configuration.
    @property
    def OS_STATUS(self):
        if self.WINDOWS:
            return f"{self.OS_NAME} {self.OS_PATCH}, build {self.OS_BUILD}"
        elif self.LINUX:
            return (
                f"{self.OS_TYPE}, GNU/Linux {self.OS_KERNEL} (WSL2)"
                if self.WSL2
                else f"{self.OS_TYPE}, GNU/Linux {self.OS_KERNEL} (WSL2)"
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
        if self.device_gpu_list() is not None:
            _gpulist = ""
            for _gpu_info in self.device_gpu_list():
                _gpu_num, _gpu_name, _gpu_vram = _gpu_info
                _gpulist += (
                    f"GPU {_gpu_num}: \t{_gpu_name} ({_gpu_vram:.2f}GB VRAM)\n    "
                )
            return _gpulist
        elif self.device_gpu_list() is None:
            cstring(
                f"{Emoji.Warn} Python module 'pywin32' not found. Skip GPU detection.",
                "warn",
            )
        else:
            cstring(f"{Emoji.Warn} Skip GPU detection on Linux.", "warn")

    # Define Memory Device status.
    @property
    def MEM_STATUS(self):
        if self.WINDOWS:
            return f"Total Physical Memory: {self.MEM_PHYS_TOTAL:.2f} GB, Avail Physical Memory: {self.MEM_PHYS_AVAIL:.2f} GB, Avail Virtual Memory: {self.MEM_VIRTUAL_AVAIL:.2f} GB"
        elif self.LINUX:
            return f"Total Physical Memory: {self.MEM_PHYS_TOTAL:.2f} GB, Avail Physical Memory: {self.MEM_PHYS_AVAIL:.2f} GB, Avail Swap Memory: {self.MEM_SWAP_AVAIL:.2f} GB"
        else:
            pass

    # Define Disk Device status. DRIVE_STATUS <--> DISK_STATUS.
    @property
    def DRIVE_STATUS(self):
        return f"""Disk Total Space: {self.DISK_TOTAL_SPACE:.2f} GB | Disk Avail Space: {self.DISK_AVAIL_SPACE:.2f} GB | Disk Used: {self.DISK_USED_SPACE:.2f} GB |  Disk Usage: {self.DISK_USAGE_RATIO:.2f} %
                Current Repo path: {self.DISK_REPO_PATH}, Disk Device: {self.DISK_REPO_MOUNT}
                """

    @property
    def DISK_STATUS(self):
        return f"""Disk Total Space: {self.DISK_TOTAL_SPACE:.2f} GB | Disk Avail Space: {self.DISK_AVAIL_SPACE:.2f} GB | Disk Used: {self.DISK_USED_SPACE:.2f} GB |  Disk Usage: {self.DISK_USAGE_RATIO:.2f} %
                Current Repo path: {self.DISK_REPO_PATH}, Disk Device: {self.DISK_REPO_MOUNT}
                """

    @property
    def ENV_STATUS(self):
        if self.WINDOWS:
            return f"""Python ENV: {self.python.exe} ({self.python.env})
                VS20XX: {self.VS20XX}
                Cygwin: {self.CYGWIN}
                MSYS2: {self.MSYS2}"""
        elif self.LINUX:
            return f"""Python3 VENV: {self.python.exe} ({self.python.env}) | WSL2: {self.WSL2}"""
        else:
            return f"""Python3 VENV: {self.python.exe} ({self.python.env}) """

    @property
    def SDK_STATUS(self):
        if self.WINDOWS:

            _vs20xx_stat = self.VS20XX if self.VS20XX else "Not Detected"
            _vs20xx_msvc = self.VC_VER if self.VC_VER else "Not Detected"
            _vs20xx_sdk = self.VS_SDK if self.VS_SDK else "Not Detected"

            _hipcc_stat = self.HIP_PATH if self.HIP_PATH else "Not Detected"
            _rocm_stat = self.ROCM_HOME if self.ROCM_HOME else "Not Detected"

            return f"""Visual Studio:  {_vs20xx_stat} | Host/Target: {self.VS_HOST} --> {self.VS_TARGET}
                VC++ Compiler:  {_vs20xx_msvc} ({self.msvc.version})
                VC++ UCRT:      {_vs20xx_sdk}
                AMD HIP SDK:    {_hipcc_stat}
                AMD ROCm:       {_rocm_stat}
            """

    @property
    def summary(self):
        if self.WINDOWS:
            print(
                f"""
        ===========    Build Environment Summary    ===========
    OS:         {self.OS_STATUS}
    CPU:        {self.CPU_STATUS}
    {self.GPU_STATUS}
    RAM:        {self.MEM_STATUS}
    STORAGE:    {self.DISK_STATUS}
    MAX_PATH_ENABLED: {self.MAX_PATH_LENGTH}
    ENV:        {self.ENV_STATUS}
    SDK:        {self.SDK_STATUS}
    """
            )

        elif self.LINUX:
            print(
                f"""
    ===========    Build Environment Summary    ===========
    OS:         {self.OS_STATUS}
    CPU:        {self.CPU_STATUS}
    RAM:        {self.MEM_STATUS}
    STORAGE:    {self.DISK_STATUS}
    """
            )


#####################################################
class DeviceChecker:
    def __init__(self, device: Device):
        self.device = device
        self.passed, self.warned, self.errs = 0, 0, 0
        self.check_record = []

    def msg_stat(
        self, status: Literal["pass", "warn", "err"], program: where | str, message: str
    ):
        if isinstance(program, where):
            match status:
                case "pass":
                    _emoji = Emoji.Pass
                case "warn":
                    _emoji = Emoji.Warn
                case "err":
                    _emoji = Emoji.Err

            return f"[{_emoji}][{program.name}] {message}"

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
    #   check_UTILITIES(self, exception)
    #
    #   Defines the tools what we found.
    #   Generally, If tools we not found, we select what we need to print, pre-manually.
    #   >>>
    #       if NOTFOUND and REQUIRED:
    #           return FATAL
    #       elif NOTFOUND but OPTIONAL:
    #           return FAILED
    #       elif FOUND but UNEXCEPTED:
    #           return FAILED/FATAL
    #
    #   Countering Mesure on Found status:
    #     > True:   Found
    #     > False:  Failed
    #     > None:   Fatal

    #
    #    check_PROGRAM() -> check_status, except_description, check_Countering_Mesure
    #

    # ===========      OS / CPU / Disk Testing      ===========

    def check_Device_OS(self):
        if self.device.WINDOWS and not (self.device.CYGWIN or self.device.MSYS2):
            _stat = self.msg_stat(
                "pass",
                "Operating System",
                f"Detected OS is {self.device.OS_NAME} {self.device.OS_PATCH}",
            )
            _except = ""
            _result = True
        elif self.device.CYGWIN or self.device.MSYS2:
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
        elif self.device.LINUX and self.device.WSL2:
            _stat = self.msg_stat(
                "pass",
                "Operating System",
                f"Detected OS is {self.device.OS_TYPE} {self.device.OS_KERNEL}",
            )
            _except = ""
            _result = True
        elif self.device.LINUX and (not self.device.WSL2):
            _stat = self.msg_stat(
                "warn",
                "Operating System",
                f"Detected OS is {self.device.OS_TYPE} {self.device.OS_KERNEL}",
            )
            _except = cstring(
                f"""
    We detect your Linux distro {self.device.OS_TYPE} is WSL2 environment.
    TheRock team still not examined on WSL2 environment. We cannot guarantee the build on WSL2.
    In current early state developement, TheRock have no ETA on WSL2 environment.
    For developers want to try on WSL2, We're welcome for 3rd-party/anyone deploy it.
    For nightly-stable-builds, please build it on Original Linux or Windows.
        traceback: Detected Linux is WSL2
        """,
                "warn",
            )
            _result = False
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
    def check_py(self):
        py = self.device.python
        py.description = "Python"
        _env_ = py.env

        if py.MAJOR_VERSION == 2:
            _stat = self.msg_stat("err", py, f"Found Python is Python 2 at {py.exe}")
            _except = cstring(
                f"""
    Found {py.name} is Python 2.
    TheRock not support build on Python 2 environment. Please Switch to Python 3 environment.
    We recommends you use Python 3.9 and newer versions.
        traceback: Python major version too old
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {py.version}
    """,
                "err",
            )
            _result = None

        elif py.MINOR_VERSION <= 8:
            _stat = self.msg_stat(
                "warn", py, f"Found Python {py.version} at {py.exe} {_env_}."
            )
            _except = cstring(
                f"""
    Found {py.name} version: {py.version}
    TheRock team still not examine on these older Python versions yet, and seems will be deprecated in future release.
    The build maybe success, but we do not promise the stability on older versions.
        traceback: Python3 version may unstable due to Python3 version too old.
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {py.version}
    """,
                "warn",
            )
            _result = False

        elif py.MINOR_VERSION >= 14:
            _stat = self.msg_stat(
                "warn", py, f"Found Python {py.version} at {py.exe} {_env_}."
            )
            _except = cstring(
                f"""
    Found {py.name} version: {py.version}
    TheRock team still not test on these newer Python versions yet.
    The build maybe success, but we do not promise the stability on new versions.

        traceback: Python3 version may unstable due to Python3 version too new.
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {py.version}
    """,
                "warn",
            )
            _result = False

        elif not py._env:
            _stat = self.msg_stat(
                "warn", py, f"Found Python {py.version} at {py.exe} {_env_}."
            )
            _except = cstring(
                f"""
    Found {py.name} is Global ENV ({py.exe}).
    We recommends you using venv like uv to create a clear Python environment to build TheRock.
    Parts of TheRock installed Python dependices may pollute your Global ENV.
        traceback: Detected Global ENV Python3 environment
            > expected Python3 is Virtual ENV, found Global ENV: {py.exe}
    """,
                "warn",
            )
            _result = False

        elif py.env == "Conda ENV":
            _stat = self.msg_stat(
                "pass", py, f"Found Python {py.version} at {py.exe} ({_env_})."
            )
            _except = cstring(
                f"""
    Note: Found Python ENV is Conda ENV.
    """,
                "hint",
            )
            _result = True

        else:
            _stat = self.msg_stat(
                "pass", py, f"Found Python {py.version} at {py.exe} ({_env_})."
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_git(self):
        git = self.device.git
        git.description = "Git version control system"

        if git.exe is None:
            _stat = self.msg_stat("warn", git, f"Cannot found git.")
            _except = cstring(
                f"""
    We cannot found git ({git.description}).
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
            _stat = self.msg_stat("pass", git, f"Found git {git.version} at {git.exe}")
            _except = ""
            _result = True
        return _stat, _except, _result

    def check_gitlfs(self):
        gitlfs = self.device.git_lfs
        gitlfs.description = "Git Large File System"

        if gitlfs.exe is not None:
            _stat = self.msg_stat(
                "pass", gitlfs, f"Found git-lfs {gitlfs.version} at {gitlfs.exe}"
            )
            _except = ""
            _result = True

        else:
            if self.device.WINDOWS:
                _stat = self.msg_stat(
                    "warn", gitlfs, f"Cannot found git-lfs {gitlfs.version}."
                )
                _except = cstring(
                    f"""
    We cannot found git-lfs ({gitlfs.description}). We recommends git-lfs for additional tools.
    For Windows users, you can install it from Git-LFS for Windows installer, or winget/choco.
        PS > winget install --id GitHub.GitLFS -e
        PS > choco install git-lfs -y
        traceback: Optional program git-lfs not found
        """,
                    "warn",
                )
                _result = False

            elif self.device.LINUX:
                _stat = self.msg_stat(
                    "err", gitlfs, f"Cannot found git-lfs {gitlfs.version}."
                )
                _except = cstring(
                    f"""
    We cannot found git-lfs program as TheRock required.
    For Linux users please install it from your Linux distro's package manager.
        sh $ apt/dnf install git-lfs

        traceback: Required program git-lfs not found
        """,
                    "err",
                )
                _result = None

        return _stat, _except, _result

    def check_uv(self):
        uv = self.device.uv
        uv.description = "Astral uv Python package and project manager"

        if uv.exe is None:
            _stat = self.msg_stat("warn", uv, f"Cannot find Astral uv.")
            _except = cstring(
                f"""
    We recommends using uv ({uv.description}) to fastly build and manage Python VENV.
    For Windows users can install via Global ENV Python PyPI or use Astral official powershell script.
    For Linux   users can install via wget/curl command.
        PS > powershell/pwsh -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        PS > pip install uv
        sh $ wget -qO- https://astral.sh/uv/install.sh | sh
        sh $ curl -LsSf https://astral.sh/uv/install.sh | sh
    Note: uv is a optional compoment. You can Ignore it if you prefer using venv or conda.
        traceback: Optional program uv not found""",
                "warn",
            )
            _result = False

        else:
            _stat = self.msg_stat(
                "pass", uv, f"Found Astral uv {uv.version} at {uv.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_cmake(self):
        cmake = self.device.cmake
        cmake.description = "Cross-Platform Make (CMake)"

        if cmake.exe is None:
            _stat = self.msg_stat("err", cmake, f"Cannot find CMake program.")
            _except = cstring(
                f"""
    We cannot find any possiable cmake program. Please check cmake program is installed and in PATH.
    TheRock is a CMake super project requires cmake program.
    For Windows users can install VS20XX, Strawberry Perl with its combined cmake or via command.
    For Linux users please install it via package manager.
        PS > pip/uv pip/winget/choco install cmake
        sh $ apt/dnf install cmake
        sh $ pacman -S cmake

        traceback: Required CMake program not installed or in PATH
    """,
                "err",
            )
            _result = None

        elif cmake.MAJOR_VERSION >= 4:
            _stat = self.msg_stat(
                "warn", cmake, f"Found CMake {cmake.MAJOR_VERSION} at {cmake.exe}"
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
                "warn", cmake, f"Found CMake {cmake.version} at {cmake.exe}"
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
                "pass", cmake, f"Found CMake {cmake.version} at {cmake.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_ninja(self):
        ninja = self.device.ninja
        ninja.description = "A small buildsystem focus on speed"

        if ninja.exe is None:
            _stat = self.msg_stat("err", ninja, f"Ninja Generator not found.")
            _except = cstring(
                f"""
    We can't find required generator "Ninja".
    Ninja is TheRock project current supported generator.
    For Windows users, please use ninja from VS20XX or Strawberry Perl, or build from source. Install from command line, please avoid version 1.11.
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
                "warn", ninja, f"Found Ninja {ninja.version} at {ninja.exe}"
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
                "pass", ninja, f"Found Ninja {ninja.version} at {ninja.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_ccache(self):
        ccache = self.device.ccache
        ccache.description = "Compiler Cache"

        if ccache.exe is None:
            _stat = self.msg_stat("warn", ccache, f"ccache not found.")
            _except = cstring(
                f"""
    We cannot find ccache ({ccache.description}).
    Ccache is a program stores compiler cache, ready for accelerates re-build speed.
    You can install it with Strawberry Perl, via package manager, or build it from source.
    Note: ccache is a optional compoment. You can Ignore it if you not using ccache.
    Note: ccache is still investigating the exact proper options on Windows platform.
    If you want to avoid this issue, please ignore ccache setup and to avoid use
       CMake cache variable `CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER`.
        traceback: Optional program ccache not installed or in PATH
    """,
                "warn",
            )
            _result = False

        else:
            _stat = self.msg_stat(
                "pass", ccache, f"Found ccache {ccache.version} at {ccache.exe}"
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
        gcc.description = "GCC Compiler C Language Frontend"

        if self.device.LINUX:
            if gcc.exe is None:
                _stat = self.msg_stat("err", gcc, f"Cannot find {gcc.name} compiler.")
                _except = cstring(
                    f"""
    We can't find required C/C++ compilers.
    On Linux platform we need GCC compilers to compile.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GCC program {gcc.name} not installed or not in PATH
           > Hint: Missing GNU {gcc.description}
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    gcc,
                    f"Found GCC compiler program {gcc.name} {gcc.version} at {gcc.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_gxx(self):
        gxx = self.device.gxx
        gxx.description = "GCC Compiler C++ Language Frontend"

        if self.device.LINUX:
            if gxx.exe is None:
                _stat = self.msg_stat("err", gxx, f"Cannot find {gxx.name} compiler.")
                _except = cstring(
                    f"""
    We can't find required C/C++ compilers.
    On Linux platform we need GCC compilers to compile.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GCC program {gxx.name} not installed or not in PATH
           > Hint: Missing GNU {gxx.description}
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    gxx,
                    f"Found GCC compiler program {gxx.name} {gxx.version} at {gxx.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_gfortran(self):
        gfortran = self.device.gfortran
        gfortran.description = "GNU Fortran Compiler"

        if gfortran.exe is None:
            _stat = self.msg_stat(
                "err", gfortran, f"Cannot found {gfortran.description}."
            )
            _except = cstring(
                f"""
    We cannot found any available {gfortran.description}.
    On Windows, please install gfortran in your device, via Strawberry/MinGW-builds etc.
    On Linux, please install via package managers.
    Note: This requirement will be deprecated when TheRock team enables Flang/flang-new (LLVM based Fortran Compiler)
        build on TheRock sub-project amd-llvm (ROCM/llvm-project).
        traceback: No available Fortran compiler 'gfortran'
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat(
                "pass",
                gfortran,
                f"Found Fortran compiler {gfortran.name} {gfortran.version} at {gfortran.exe}",
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_gcc_ar(self):
        gcc_ar = self.device.gcc_ar
        gcc_ar.description = "GNU Binutils Archiver"

        if self.device.LINUX:
            if gcc_ar.exe is None:
                _stat = self.msg_stat(
                    "err", gcc_ar, f"Can't found {gcc_ar.description}."
                )
                _except = cstring(
                    f"""
    We can't found GNU toolchain required Archiver/Linker.
    Please configure your binutils is installed correctly.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GNU binutils {gcc_ar} not installed or not in PATH
           > Hint: Missing GNU {gcc_ar.description}
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    gcc_ar,
                    f"Found {gcc_ar.description} {gcc_ar.version} at {gcc_ar.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_gcc_as(self):
        gcc_as = self.device.gcc_as
        gcc_as.description = "GNU Binutils Assembler"

        if self.device.LINUX:
            if gcc_as.exe is None:
                _stat = self.msg_stat(
                    "err", gcc_as, f"Can't found {gcc_as.description}."
                )
                _except = cstring(
                    f"""
    We can't found GNU {gcc_as}.
    Please configure your binutils is installed correctly.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GNU binutils {gcc_as} not installed or not in PATH
           > Hint: Missing GNU {gcc_as.description}
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass",
                    gcc_as,
                    f"Found {gcc_as.description} {gcc_as.version} at {gcc_as.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_ld(self):
        ld = self.device.ld
        ld.description = "GNU Binutils Linker"

        if self.device.LINUX:
            if ld.exe is None:
                _stat = self.msg_stat("err", ld, f"Can't found {ld.description}.")
                _except = cstring(
                    f"""
    We can't found GNU toolchain required Archiver/Linker.
    Please configure your binutils is installed correctly.
    Please install it via package managers.
        sh $ apt/dnf install gcc g++ binutils

        traceback: GNU binutils {ld} not installed or not in PATH
           > Hint: Missing GNU {ld.description}
    """,
                    "err",
                )
                _result = None

            else:
                _stat = self.msg_stat(
                    "pass", ld, f"Found {ld.description} {ld.version} at {ld.exe}"
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    #  ===========  MSVC Compiler Toolchain  ===========
    def check_msvc(self):
        cl = self.device.msvc
        cl.description = "Microsoft C/C++ compiler Driver"

        if self.device.WINDOWS:
            if cl.exe is None:
                _stat = self.msg_stat(
                    "err", cl, f"Cannot found MSVC program cl.exe {cl.description}"
                )
                _except = cstring(
                    f"""
    We can't found any available MSVC compiler on your Windows device.
    MSVC (Microsoft Optimized Visual C/C++ compiler Driver), The C/C++ compliler for native Windows development.
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
                    cl,
                    f"Found MSVC {self.device.VC_VER} ({cl.version}) at {cl.exe}",
                )
                _except = ""
                _result = True

            return _stat, _except, _result

    def check_ml64(self):
        ml64 = self.device.ml64
        ml64.description = "Microsoft Macro Assembler"

        if self.device.WINDOWS:
            if ml64.exe is None:
                _stat = self.msg_stat(
                    "err",
                    ml64,
                    f"Cannot found MSVC program ml64.exe ({ml64.description}).",
                )
                _except = cstring(
                    f"""
    We can't found any available MSVC compiler on your Windows device.
    MSVC{ml64.description}, The C/C++ compliler for native Windows environment.
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
                    ml64,
                    f"Found MSVC Macro Assembler {ml64.version} at {ml64.exe}",
                )
                _except = ""
                _result = True

        return _stat, _except, _result

    def check_lib(self):
        lib = self.device.lib
        lib.description = "Microsoft Linker Stub"

        if lib.exe is None:
            _stat = self.msg_stat(
                "err", lib, f"Cannot found MSVC program lib.exe ({lib.description})."
            )
            _except = cstring(
                f"""
    We cannot found MSVC toolchain's archiver.
    Please check your Microsoft VC++ installation is correct, or re-check if the install is broken.

        traceback: MSVC Archiver lib.exe ({lib.exe}) not found
    """,
                "err",
            )
            _result = None
        else:
            _stat = self.msg_stat(
                "pass", lib, f"Found MSVC program lib.exe at {lib.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_link(self):
        link = self.device.link
        link.description = "Microsoft Incremental Linker"

        if link.exe is None:
            _stat = self.msg_stat("err", link, f"Cannot found {link.description}.")
            _except = cstring(
                f"""
    We cannot found MSVC toolchain's linker link.exe ({link.description}).
    Please re-check your MSVC installation if it's broken.
        traceback: Missing MSVC required linker compoments link.exe
    """,
                "err",
            )
            _result = None
        else:
            _stat = self.msg_stat(
                "pass", link, f"Found {link.description} at {link.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result

    def check_rc(self):
        rc = self.device.rc
        rc.description = "Microsoft Resource Compiler"

        if rc.exe is None:
            _stat = self.msg_stat("err", rc, f"Cannot found {rc.description}.")
            _except = cstring(
                f"""
    We cannot found rc.exe ({rc.description}) in your Windows SDK, or yuo have not Windows SDK installed.
    Please re-configure your MSVC and Windows SDK installation via Visual Studio.
        Visual Studio Installer > C/C++ Development for Desktop:
            - MSVC v14X
            - MSVC MFC
            - MSVC ALT
            - Windows SDK 10.0.XXXXX
            - CMake for Windows
            - C++ Address Sanitizer
        traceback: {rc.description} not found in Windows SDK or Windows SDK not installed
    """,
                "err",
            )
            _result = None

        else:
            _stat = self.msg_stat("pass", rc, f"Found {rc.description} at {rc.exe}")
            _except = ""
            _result = True

        return _stat, _except, _result

    #  ==============    Find Environment    =============
    #
    #   TaiXeflar: Only Windows needs to do VS20XX and MAX_PATH.
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
                f"""We can't found a available Visual Studio install version.
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
        _host = self.device.VS_HOST
        _cl = self.device.msvc
        if _host == "x64":
            _stat = self.msg_stat("pass", _cl, f"MSVC Host compiler is {_host}.")
            _except = ""
            _result = True

        else:
            _stat = self.msg_stat("err", _cl, f"MSVC Host compiler is {_host}.")
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
        _target = self.device.VS_TARGET
        _cl = self.device.msvc
        if _target == "x64":
            _stat = self.msg_stat("pass", _cl, f"MSVC compile target is {_target}.")
            _except = ""
            _result = True
        else:
            _stat = self.msg_stat("err", _cl, f"MSVC Host compiler is {_target}.")
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
        _sdk = self.device.VS_SDK
        if _sdk is None:
            _stat = self.msg_stat("err", "WindowsSDK", f"Cannot found Windows SDK.")
            _except = cstring(
                f"""
    We cannot found available Windows SDK.
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
    \t Registry Key Hint: HKLM:/SYSTEM/CurrentControlSet/Control/FileSystem LongPathsEnabled = 0 (DWORD)
    """,
                "warn",
            )
            _result = False

        return _stat, _except, _result

    #  ==============   Summarize   ================


def check_summary(result: DeviceChecker):
    pass_num = cstring(result.check_record.count(True), color=(55, 255, 125))
    warn_num = cstring(result.check_record.count(False), color="warn")
    err_num = cstring(result.check_record.count(None), color="err")

    print(
        f"""
                            Compoments check {pass_num} Passed, {warn_num} Warning, {err_num} Fatal Error"""
    )


def run_test():

    device = Device()
    tester = DeviceChecker(device=device)

    if device.WINDOWS:
        test = [
            tester.check_Device_OS,
            tester.check_Device_ARCH,
            tester.check_DISK_USAGE,
            tester.check_MAX_PATH,
            tester.check_py,
            tester.check_git,
            tester.check_gitlfs,
            tester.check_uv,
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

    elif device.LINUX:
        test = [
            tester.check_Device_OS,
            tester.check_Device_ARCH,
            tester.check_DISK_USAGE,
            tester.check_py,
            tester.check_git,
            tester.check_gitlfs,
            tester.check_uv,
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


# Define Main() as main diagnosis, Help() as `--help`, Track() as `--track`.
def main():

    therock_detect_start = time.perf_counter()

    # os.system("cls" if os.name == "nt" else "clear")  Disabled clear prompt.

    TheRock.__logo__()

    device = Device()
    device.summary

    print(
        r"""
        ===========    Start detecting conpoments for building ROCm\TheRock    ===========
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


# Possiable Args.
help_args = ("--help", "-help", "-?", "/help", "/?", "--i-need-help")


# Launcher.
if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        args = [arg.lower() for arg in sys.argv[1:]]
        if any(arg in help_args for arg in args):
            TheRock.help()
        else:
            TheRock.help()
