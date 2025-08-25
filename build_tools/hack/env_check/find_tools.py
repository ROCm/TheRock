from __future__ import annotations
import subprocess
import os, re, sys, shutil
from pathlib import Path
from abc import ABC


class FindProgram(ABC):
    """
    # class FindProgram

    Generally, this class stores programs's version, location.

    ## methods
    - `run_query`: Using a subprocess executes the program's version output (possiably `self.exe --version`).
    - `get_version`: Analyze the version info inputs from `run_query()` or overriden by rules(like Python and MSVC).

    ## Property
    - `exe`: Programs PATH.
    - `version`: Programs version number `<Major>.<Minor>.<Patch>`.
    - `MAJOR_VERSION`: Programs major version number `<Major>`.
    - `MINOR_VERSION`: Programs minor version number `<Minor>`.
    - `PATCH_VERSION`: Programs patch version number `<Patch>`.

    ## General programs
    Nothing special. Programs like `cmake` or `git` with it's version number fits `<Major>.<Minor>.<Patch>` are same.

    ## GCC Binutils
    Binutils (`ar`, `as`, `ld`) comes with version number from `<Major>.<Minor>` to `<Major>.<Minor>.0`.

    ## Python 3
    Python use its self libraries like sys, platform etc. Additional Properties:
    - `is_VENV`: `True` if Shell environment in Python VENV, Astral uv VENV, or Conda ENV. Otherwise (Global Env, pipx, poetry, pyenv etc.) False.
    - `ENV_TYPE`: Print VENV type. Supports `Global Env`, `Python VENV`, `Astral UV`, `Conda ENV`.
    - `Free_Threaded`: If this Python 3 program is Free Threaded version.
    - `no_gil`: If this Python 3 program have no Global Interpreter Lock(GIL). Value same as `Free_Threaded`.
    - `interpreter`: Check Python interpreter is CPython, PyPy, Jython or anything else.

    ## MSVC
    MSVC program's version number is bonded with different version naming conventions and meanings.

    So, `MAJOR_VERSION`, `MINOR_VERSION`, `PATCH_VERSION` are None, `version` will saying to MSVC build version numbers, with a analyze of `v14X`.

    Resource Compiler comes with Windows SDK, it's version num will be the version of Windows SDK.

    Other Microsoft Visual Studio compoments as same to special cases.
    """

    def __init__(self):
        self._major_version = None
        self._minor_version = None
        self._patch_version = None
        self._version = None
        self.name = None

    @property
    def exe(self):
        _exe = shutil.which(self.name)
        if _exe is not None:
            return _exe.replace("\\", "/").replace("EXE", "exe")
        else:
            return None

    def get_version(self):
        if self.exe is None:
            self._version = None
        else:
            query = re.search(
                r"(\d+)\.(\d+)(?:\.(\d+))?", self.run_query([self.exe, "--version"])
            ).groups()
            if query:
                major, minor, patch = query
                self._major_version = int(major)
                self._minor_version = int(minor)
                self._patch_version = int(patch) if patch else 0
                self._version = (
                    f"{self._major_version}.{self._minor_version}.{self._patch_version}"
                )

    def run_query(self, cmd) -> str:
        try:
            return subprocess.run(
                cmd, text=True, capture_output=True, check=True
            ).stdout.strip()
        except Exception as e:
            return None

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
    def version(self):
        return self._version


# Find Programs.


class FindPython(FindProgram):
    def __init__(self):
        super().__init__()
        self.get_version()
        self._py_env()

    @property
    def exe(self):
        return sys.executable

    def get_version(self):
        self._major_version = sys.version_info.major
        self._minor_version = sys.version_info.minor
        self._patch_version = sys.version_info.micro
        self._relse_version = sys.version_info.releaselevel
        self._version = (
            f"{self._major_version}.{self._minor_version}.{self._patch_version}"
        )

    def _py_env(self):
        if os.getenv("CONDA_PREFIX") is not None:
            return True, "Conda ENV"
        elif sys.prefix == sys.base_prefix:
            return False, "Global ENV"
        elif os.getenv("VIRTUAL_ENV") is not None:
            with open(Path(f"{sys.prefix}/pyvenv.cfg").resolve(), "r") as file:
                _conf = file.read()
            _env_type = "uv VENV" if "uv" in _conf else "Python VENV"
            return True, _env_type
        else:
            return False, "Unknown ENV"

    @property
    def is_VENV(self):
        return self._py_env()[0]

    @property
    def ENV_TYPE(self):
        return self._py_env()[1]

    @property
    def Free_Threaded(self):
        if self.MINOR_VERSION <= 12:
            return False
        elif self.MINOR_VERSION >= 13 and sys._is_gil_enabled():
            return False
        elif self.MINOR_VERSION >= 13 and not sys._is_gil_enabled():
            return True

    @property
    def no_gil(self):
        return self.Free_Threaded

    @property
    def interpreter(self):
        return sys.implementation.name


class FindGit(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "git"
        self.get_version()


class FindGitLFS(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "git-lfs"
        self.get_version()


class FindUV(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "uv"
        self.get_version()


class FindCMake(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "cmake"
        self.get_version()


class FindCCache(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "ccache"
        self.get_version()


class FindNinja(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "ninja"
        self.get_version()


# MSVC


class FindMSVC(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "cl"
        self.get_version()

    def get_version(self):
        _vc_ver = os.getenv("VCToolsVersion")
        if _vc_ver is None:
            self._version = None
        else:
            _vc_ver_split = tuple(int(x) for x in _vc_ver.split("."))
            if _vc_ver_split >= (14, 30, 00000):
                self._version = f"v143 ({_vc_ver})"
            elif _vc_ver_split <= (14, 29, 30133):
                self._version = f"v142 ({_vc_ver})"
            elif _vc_ver_split <= (14, 16, 27023):
                self._version = f"v141 ({_vc_ver})"
            elif (
                self.exe
                == r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\cl.exe"
            ):
                self._version = "v140"

    @property
    def host(self):
        return os.getenv("VSCMD_ARG_HOST_ARCH")

    @property
    def target(self):
        return os.getenv("VSCMD_ARG_TGT_ARCH")


class FindML64(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "ml64"
        self.get_version()

    def get_version(self):
        _vc_ver = os.getenv("VCToolsVersion")
        if _vc_ver is None:
            self._version = None
        else:
            _vc_ver_split = tuple(int(x) for x in _vc_ver.split("."))
            if _vc_ver_split >= (14, 30, 00000):
                self._version = f"v143 ({_vc_ver})"
            elif _vc_ver_split <= (14, 29, 30133):
                self._version = f"v142 ({_vc_ver})"
            elif _vc_ver_split <= (14, 16, 27023):
                self._version = f"v141 ({_vc_ver})"
            elif (
                self.exe
                == r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\ml64.exe"
            ):
                self._version = "v140"


class FindLIB(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "lib"
        self.get_version()

    def get_version(self):
        _vc_ver = os.getenv("VCToolsVersion")
        if _vc_ver is None:
            self._version = None
        else:
            _vc_ver_split = tuple(int(x) for x in _vc_ver.split("."))
            if _vc_ver_split >= (14, 30, 00000):
                self._version = f"v143 ({_vc_ver})"
            elif _vc_ver_split <= (14, 29, 30133):
                self._version = f"v142 ({_vc_ver})"
            elif _vc_ver_split <= (14, 16, 27023):
                self._version = f"v141 ({_vc_ver})"
            elif (
                self.exe
                == r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\lib.exe"
            ):
                self._version = "v140"


class FindLINK(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "link"
        self.get_version()

    def get_version(self):
        _vc_ver = os.getenv("VCToolsVersion")
        if _vc_ver is None:
            self._version = None
        else:
            _vc_ver_split = tuple(int(x) for x in _vc_ver.split("."))
            if _vc_ver_split >= (14, 30, 00000):
                self._version = f"v143 ({_vc_ver})"
            elif _vc_ver_split <= (14, 29, 30133):
                self._version = f"v142 ({_vc_ver})"
            elif _vc_ver_split <= (14, 16, 27023):
                self._version = f"v141 ({_vc_ver})"
            elif (
                self.exe
                == r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\link.exe"
            ):
                self._version = "v140"


class FindRC(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "rc"
        self.get_version()

    def get_version(self):
        if os.getenv("WindowsSDKVersion"):
            return os.getenv("WindowsSDKVersion").replace("\\", "")
        else:
            return None


class FindGCC(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "gcc"
        self.get_version()


class FindGXX(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "g++"
        self.get_version()


class FindGFortran(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "gfortran"
        self.get_version()


class FindLD(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "ld"
        self.get_version()


class FindGCC_AR(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "ar"
        self.get_version()


class FindGCC_AS(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "as"
        self.get_version()


# Find SDKs.
class FindVS20XX(FindProgram):
    def __init__(self):
        super().__init__()
        self.name = "Visual Studio"
        self.get_version()

    def get_version(self) -> str:
        _vs_ver = os.getenv("VisualStudioVersion")
        if _vs_ver is None:
            return None
        else:
            _vs_ver = float(os.getenv("VisualStudioVersion"))
        match _vs_ver:
            case 17.0:
                return "VS2022"
            case 16.0:
                return "VS2019"
            case 15.0:
                return "VS2017"
            case 14.0:
                return "VS2015"
            case _:
                return None


# TODO: Complete AMD ROCM/HIP SDK and amd-llvm detection.
# AMD ROCm/HIP SDKs.
# Find_ROCM/HIP ...
#
# AMD HIPCC and LLVM compiler.
# Find_HIPCC/Clang/Clang++/Flang/LLD/LLVM-AR/LLVM-AS/LLVM-MN/ ...
