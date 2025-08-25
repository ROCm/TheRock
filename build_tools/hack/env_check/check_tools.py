from __future__ import annotations
from typing import Literal
import os
from pathlib import Path
from abc import ABC
from .utils import cstring, Emoji
from .find_tools import *
from .device import SystemInfo


def msg_stat(status: Literal["pass", "warn", "err"], program: str, message: str):
    if isinstance(program, FindProgram):
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

        return f"[{_emoji}][{cstring(program, status)}] {message}"


class Check_List:
    def __init__(self, checks: list):
        self.todo = checks
        self.passed, self.warned, self.errs = 0, 0, 0
        self.check_record = []

    def run(self):
        for each_test in self.todo:
            _stat, _except, _result = each_test.__call__()
            print(f"{_stat} {_except}")
            self.check_record.append(_result)

    @property
    def summary(self):
        self.run()
        pass_num = cstring(self.check_record.count(True), color=(55, 255, 125))
        warn_num = cstring(self.check_record.count(False), color="warn")
        err_num = cstring(self.check_record.count(None), color="err")

        print(
            f"""
                                Compoments check {pass_num} Passed, {warn_num} Warning, {err_num} Fatal Error"""
        )


# Program Checkers.


class CheckProgram(ABC):
    def __init__(self, required: bool = True):
        super().__init__()
        self.program = FindProgram()
        self.condition = required
        self.name = "Name"

    def __call__(self):
        return self.check()

    def check(self):
        program = self.program
        name = self.name
        required = self.condition

        if program.exe is None and required:
            _stat = msg_stat("err", name, f"Cannot Find {name}.")
            _except = cstring(
                f"""
    Program {self.name} not found. Please Install it.""",
                "err",
            )
            _result = None
        elif program.exe is None and not required:
            _stat = msg_stat("warn", name, f"Cannot Find {name}.")
            _except = cstring(
                f"""
    Optional {name} not found.""",
                "warn",
            )
            _result = ...
        elif program.exe:
            _stat = msg_stat("pass", name, f"Found {name} at {program.exe}")
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckPython(CheckProgram):
    def __init__(self):
        super().__init__()
        self.python = FindPython()
        self.name = "Python 3"

    def check(self):
        python = self.python
        name = self.name
        if python.MAJOR_VERSION == 2:
            # This only could happens on Windows.
            _stat = msg_stat("err", name, f"Found Python 2: {python.exe}")
            _except = cstring(
                f"""
    Found Python is Python 2.
    TheRock not support Python 2. Please Switch to Python 3.9+ versions.

        traceback: Unsupported Python versions
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {python.version}
    """,
                "err",
            )
            _result = None

        elif python.MINOR_VERSION <= 8:
            _stat = msg_stat(
                "err",
                name,
                f"Found Python {python.version} at {python.exe} {python.ENV_TYPE}.",
            )
            _except = cstring(
                f"""
    Found Python version: {python.version}
    TheRock not support Python 3.8 and older versions. Please Switch to Python 3.9+ versions.

        traceback: Unsupported Python versions
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {python.version}
    """,
                "err",
            )
            _result = None

        elif python.Free_Threaded == True:
            _stat = msg_stat(
                "err", name, f"Found Python {python.exe} is No-GIL version."
            )
            _except = cstring(
                f"""
    TheRock have not support Free-Threaded Python yet.

        traceback: Found Python 3 using Free-Threaded (No-GIL) version
    """,
                "err",
            )
            _result = None

        elif python.interpreter != "cpython":
            _stat = msg_stat(
                "err",
                name,
                f"Found Python {python.exe} interpreter is {python.interpreter}.",
            )
            _except = cstring(
                f"""
    TheRock only supports CPython interpreter.

        traceback: Found Python 3 unexpected interpreter type {python.interpreter}
    """,
                "err",
            )
            _result = None

        elif python.MINOR_VERSION >= 14:
            _stat = msg_stat(
                "warn",
                name,
                f"Found Python {python.version} at {python.exe} {python.ENV_TYPE}.",
            )
            _except = cstring(
                f"""
    Found Python version: {python.version}
    Warning: These newer Python versions are not yet tested.

        traceback: Python 3 version too new may unstable
            > expected version: 3.9.X ≤ python ≤ 3.13.X, found {python.version}
    """,
                "warn",
            )
            _result = False

        elif python.ENV_TYPE == "Global ENV":
            _stat = msg_stat(
                "warn",
                name,
                f"Found Python {python.version} at {python.exe} {python.ENV_TYPE}.",
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

        elif python.ENV_TYPE == "Conda ENV":
            _stat = msg_stat(
                "pass",
                name,
                f"Found Python {python.version} at {python.exe} ({python.ENV_TYPE}).",
            )
            _except = cstring(
                f"""
    Note: Found Conda ENV.
    """,
                "hint",
            )
            _result = True

        else:
            _stat = msg_stat(
                "pass",
                name,
                f"Found Python {python.version} at {python.exe} ({python.ENV_TYPE}).",
            )
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckGit(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGit()
        self.name = "Git"


class CheckGitLFS(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGitLFS()
        self.name = "Git-LFS"


class CheckUV(CheckProgram):
    def __init__(self, required=False):
        super().__init__(required)
        self.program = FindUV()
        self.name = "Astral UV"


class CheckCMake(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindCMake()
        self.name = "CMake"

    def check(self):
        if self.program.exe is None:
            _stat = msg_stat("err", "CMake", f"Cannot find CMake.")
            _except = cstring(
                f"""
    No CMake program found. Please check CMake program is installed and in PATH.
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
        elif (self.program.MAJOR_VERSION, self.program.MINOR_VERSION) < (3, 25):
            _stat = msg_stat(
                "err",
                "CMake",
                f"Found CMake version {self.program.version} at {self.program.exe}",
            )
            _except = cstring(
                f"""
    TheRock requires CMake version >= 3.25, but found {self.program.version}.

        traceback: CMake program too old excluded by TheRock Top-Level CMakeLists.txt rules `cmake_minimum_required()`
            expected: 3.25.X ≤ cmake ≤ 3.31.X, found {self.program.version}
    """,
                "warn",
            )
            _result = None
        elif self.program.MAJOR_VERSION == 4:
            _stat = msg_stat(
                "warn",
                "CMake",
                f"Found CMake version {self.program.version} at {self.program.exe}",
            )
            _except = cstring(
                f"""
    The support of CMake 4 is still under development, and the different CMake version behavior may effect TheRock build.
    If your build requires stable build, please downgrade it and re-try again.

        traceback: CMake program too new may cause unstable
            expected: 3.25.X ≤ cmake ≤ 3.31.X, found {self.program.version}
    """,
                "warn",
            )
            _result = False

        else:
            _stat = msg_stat(
                "pass",
                "CMake",
                f"Found CMake {self.program.version} at {self.program.exe}",
            )
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckCCache(CheckProgram):
    def __init__(self, required=False):
        super().__init__(required)
        self.program = FindCCache()
        self.name = "CCache"


class CheckNinja(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindNinja()
        self.name = "Ninja-Build"


# GCC compiler toolchain
"""
GCC:: gcc
GCC:: g++
GCC:: gfortran
GCC:: Binutils/ar
GCC:: Binutils/as
GCC:: Binutils/ld
"""


class CheckGCC(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGCC()
        self.name = "gcc"


class CheckGXX(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGXX()
        self.name = "g++"


class CheckGFortran(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGFortran()
        self.name = "gfortran"


class CheckGCC_AR(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGCC_AR()
        self.name = "Binutils"


class CheckGCC_AS(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindGCC_AS()
        self.name = "Binutils"


class CheckLD(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindLD()
        self.name = "Binutils"


# MSVC Compiler toolchain
"""
MSVC:: cl.exe
MSVC:: ml64.exe
MSVC:: lib.exe
MSVC:: link.exe
MSVC:: ATL/MFC
MSVC:: Windows SDK
"""


class CheckMSVC(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindMSVC()
        self.name = "MSVC/VC++"

    def check(self):
        cl = self.program

        if cl.exe is None:
            _stat = msg_stat(
                "err",
                self.name,
                f"Cannot found Microsoft Optimized C/C++ compiler Driver cl.exe.",
            )
            _except = cstring(
                f"""
    We can't found any available MSVC compiler on your Windows device.
    MSVC (Microsoft Visual C/C++), The C/C++ compliler for native Windows development.
    Please re-configure your Visual Studio installed C/C++ correctly.
        Visual Studio Installer > C/C++ Development for Desktop:
        - MSVC v14X
        - MSVC ALT
        - Windows SDK 10.0.XXXXX
        - CMake for Windows
        - C++ Address Sanitizer

        traceback: Required MSVC compiler not found
    """,
                "err",
            )
            _result = None

        elif (cl.exe is not None) and (cl.host != "x64"):
            _stat = msg_stat("err", self.name, f"Found VC++ compiler unsupported host.")
            _except = cstring(
                f"""
    TheRock project not support this VC++ host compiler.

        traceback: Unexpected VC++ compiler host {cl.host}
    """,
                "err",
            )
            _result = None

        elif (cl.exe is not None) and (cl.target != "x64"):
            _stat = msg_stat("err", self.name, f"Found VC++ compiler unsupported host.")
            _except = cstring(
                f"""
    TheRock project not support compile to target {cl.target}.

        traceback: Unexpected VC++ compile target {cl.target}
    """,
                "err",
            )
            _result = None

        elif (cl.exe is not None) and (cl.host == "x64") and (cl.target == "x64"):
            _stat = msg_stat("pass", self.name, f"Found MSVC {cl.version} at {cl.exe}")
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckML64(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindML64()
        self.name = "MSVC/ML64"

    def check(self):
        ml64 = self.program

        if ml64.exe is None:
            _stat = msg_stat(
                "err", self.name, f"Cannot found Microsoft Macro Assembler."
            )
            _except = cstring(
                f"""
    We can't found any available Microsoft Macro Assembler on your Windows device.
    ml64.exe is the Assembler program for native Windows development.
    Please re-configure your Visual Studio installed C/C++ correctly.
        Visual Studio Installer > C/C++ Development for Desktop:
        - MSVC v14X
        - MSVC MFC
        - MSVC ALT
        - Windows SDK 10.0.XXXXX
        - CMake for Windows
        - C++ Address Sanitizer

        traceback: Required MSVC Assembler not found
    """,
                "err",
            )
            _result = None

        else:
            _stat = msg_stat(
                "pass",
                self.name,
                f"Found Microsoft Macro Assembler {ml64.version} at {ml64.exe}",
            )
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckLIB(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindLIB()
        self.name = "MSVC/Microsoft Linker stub"

    def check(self):
        lib = self.program
        if lib.exe is None:
            _stat = msg_stat("err", self.name, f"Cannot found MSVC program lib.exe.")
            _except = cstring(
                f"""
    We cannot find MSVC toolchain's archiver.
    Please check your Microsoft VC++ installation is correct, or re-check if the install is broken.

        traceback: Missing MSVC required linker compoments
    """,
                "err",
            )
            _result = None
        else:
            _stat = msg_stat(
                "pass", self.name, f"Found MSVC program lib.exe at {lib.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckLINK(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindLINK()
        self.name = "MSVC/Microsoft Linker stub"

    def check(self):
        link = self.program

        if link.exe is None:
            _stat = msg_stat(
                "err", self.name, f"Cannot found Microsoft Incremental Linker."
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
            _stat = msg_stat(
                "pass", self.name, f"Found Microsoft Incremental Linker at {link.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckRC(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindRC()
        self.name = "Windows SDK"

    def check(self):
        rc = self.program
        if rc.exe is None:
            _stat = msg_stat(
                "err", self.name, f"Cannot found Microsoft Resource Compiler."
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
            _stat = msg_stat(
                "pass", self.name, f"Found Microsoft Resource Compiler at {rc.exe}"
            )
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckATL(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.name = "MSVC/ATL"

        if os.getenv("VCToolsInstallDir") is None:
            self.program = None
        else:
            _vc_inst_dir = os.getenv("VCToolsInstallDir")
            _atl_inst = Path(f"{_vc_inst_dir}/atlmfc").exists()
            self.program = _atl_inst

    def check(self):
        atl = self.program
        name = self.name
        if (atl is None) or (atl is False):
            _stat = msg_stat("err", name, f"Cannot find MSVC ATL libraries.")
            _except = cstring(
                f"""
    TheRock requires MSVC installed ATL (Active Template Library).
    Please reconfig your Visual Studio install.

        traceback: MSVC ATL not found
    """,
                "err",
            )
            _result = None
        else:
            _stat = msg_stat("pass", name, f"Found MSVC ATL.")
            _except = ""
            _result = True

        return _stat, _except, _result


class CheckVS20XX(CheckProgram):
    def __init__(self, required=True):
        super().__init__(required)
        self.program = FindVS20XX()
        self.name = "Visual Studio"

    def check(self):
        vs20xx = self.program.get_version()
        name = self.name
        if vs20xx is None:
            _stat = msg_stat(
                "err",
                name,
                f"Cannot Find Visual Studio Environment or VS version too old.",
            )
            _except = cstring(
                f"""
    TheRock needs Visual Studio 2015/2017/2019/2022. Please install it or update Visual studio Environment.

        traceback:  TheRock on Windows build requires Visual Studio 2022/2019/2017/2015 environment
    """,
                "err",
            )
            _result = None

        else:
            _stat = msg_stat("pass", name, f"Found Visual Studio {vs20xx}.")
            _except = ""
            _result = True

        return _stat, _except, _result


# System Info


class check_device(ABC):
    def __init__(self, device_info: SystemInfo):
        super().__init__()
        self.device = device_info
        self.name = "Name"

    def __call__(self):
        return self.check()

    def check(self):
        device = self.device

        _stat = _except = _result = None
        return _stat, _except, _result


class CheckOS(check_device):
    def __init__(self, device_info):
        super().__init__(device_info)
        self.device = device_info
        self.name = "Operating System"

    def check(self):

        if self.device.is_windows and not (
            self.device.is_cygwin or self.device.is_msys2
        ):
            _stat = msg_stat(
                "pass",
                self.name,
                f"Detected OS is {self.device.OS_NAME}",
            )
            _except = ""
            _result = True

        elif self.device.is_cygwin or self.device.is_msys2:
            _stat = msg_stat("err", self.name, f"Detected OS is Cygwin/MSYS2.")
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
            _stat = msg_stat(
                "pass",
                self.name,
                f"Detected OS is {self.device.OS_NAME} {self.device.OS_KERNEL}",
            )
            _except = ""
            _result = True

        elif self.device.is_linux and self.device.is_WSL2:
            _stat = msg_stat(
                "err",
                self.name,
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
            _os = self.device.OS
            _stat = msg_stat("err", self.name, f"Detected OS is {_os}")
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


class CheckCPU(check_device):
    def __init__(
        self,
        device_info,
    ):
        super().__init__(device_info)
        self.device = device_info
        self.name = "CPU Arch"

    def check(self):
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
            _stat = msg_stat(
                "pass", self.name, f"Detected Available CPU Arch {_cpu_arch}."
            )
            _except = ""
            _result = True
        else:
            _stat = msg_stat(
                "pass", self.name, f"Detected Invalid CPU Arch {_cpu_arch}."
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


class CheckDisk(check_device):
    def __init__(self, device_info):
        super().__init__(device_info)
        self.device = device_info
        self.name = "Disk Usage"

    def check(self):
        name = self.name
        _disk_avail = self.device._device_disk_stat[4]
        _disk_ratio = self.device._device_disk_stat[5]
        _disk_drive = self.device._device_disk_stat[2]
        if _disk_avail < 128 or _disk_ratio > 80:
            _stat = msg_stat("warn", self.name, f"Disk space check attention.")
            _except = cstring(
                f"""
    We've checked the workspace disk {_disk_drive} available space could be too small to build TheRock (and PyTorch).
    TheRock builds may needs massive storage for the build, and we recommends availiable disk space with 128GB and usage not over 80%.
    """,
                "warn",
            )
            _result = False

        else:
            _stat = msg_stat("pass", self.name, f"Disk space check pass.")
            _except = ""
            _result = True

        return _stat, _except, _result


class Check_Max_PATH_LIMIT(check_device):
    def __init__(self, device_info):
        super().__init__(device_info)
        self.device = device_info
        self.name = "Max PATH Limit"

    def check(self):
        _status = self.device.MAX_PATH_LENGTH
        if _status:
            _stat = msg_stat("pass", self.name, f"Windows Long PATHs Enabled.")
            _except = ""
            _result = True
        else:
            _stat = msg_stat("warn", self.name, f"Windows Long PATHs Disabled.")
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
