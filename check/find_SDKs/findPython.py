import os
import sys
from pathlib import Path
from typing import Literal, Union, Any, overload
from textwrap import dedent

from utils import *
from utils.status import _SuccessType, _WarningType, _FatalType, _HintType
from utils.compare_functions import *
from utils.system_function import *
from . import FindSDK


THEROCK_REQUIRED_PYTHON_VERSION = "3.10.0"
THEROCK_DEPRECATED_PYTHON_VERSION_MAX = "3.9.15"

interps = ["PyPy", "IronPython", "GraalPy", "Jython", "MicroPython", "RustPython"]


class FindPython(FindSDK):

    def __init__(self):
        super().__init__()

        self._dispatch_map = {
            "VERSION": self._post_python_version,
            "INTERPRETER": self._post_python_interpreter,
            "VENV": self._post_python_venv,
            "GIL": self._post_python_GIL,
            "JIT": self._post_python_JIT,
            "module": self._post_python_module,
        }

        self._allowed_rel: Literal["candidate", "final"] = ["candidate", "final"]
        self._interpreters = {}

    #   ===================================

    @property
    def Release(self):
        return self.info["Release"]

    @property
    def Interpreter(self) -> Literal["cpython"]:
        return self.info["Interpreter"]

    @property
    def VENV(self) -> Literal["venv", "uv", "Conda", "Global"]:
        return self.info["VENV"]

    @property
    def GIL(self) -> bool:
        return self.info["GIL"]

    @property
    def JIT(self):
        return self.info["JIT"]

    #   ===================================

    def __WINDOWS__(self):
        return self._find_python()

    def __LINUX__(self):
        return self._find_python()

    def _find_python(self):
        exe = Path(sys.executable)
        ver = self._python_version()
        interp = self._python_interpreter()
        venv = self._python_venv()
        gil = self._python_GIL()
        jit = self._python_JIT()
        rel = sys.version_info.releaselevel

        return {
            "exe": exe,
            "Version": ver,
            "Interpreter": interp,
            "VENV": venv,
            "GIL": gil,
            "JIT": jit,
            "Release": rel,
        }

    #   ====================================

    def _python_version(self):
        _1 = sys.version_info.major
        _2 = sys.version_info.minor
        _3 = sys.version_info.micro

        return VersionNum(f"{_1}.{_2}.{_3}")

    def _python_interpreter(self):
        if STREQUAL(sys.implementation.name, "cpython"):
            _1 = "CPython"
        elif STREQUAL(sys.implementation.name, "pypy"):
            _1 = "PyPy"
        elif STREQUAL(sys.implementation.name, "ironpython"):
            _1 = "IronPython"
        elif STREQUAL(sys.implementation.name, "rustpython"):
            _1 = "RustPython"
        elif STREQUAL(sys.implementation.name, "micropython"):
            _1 = "MicroPython"
        elif STREQUAL(sys.implementation.name, "graalpy"):
            _1 = "GraalPy"
        elif STREQUAL(os.name, "Java"):
            _1 = "Jython"
        else:
            raise Exception("Unexpected Python interpreter type")
        return _1

    def _python_venv(self):

        if ENV("VIRTUAL_ENV"):
            venv_path = ENV("VIRTUAL_ENV").replace("\\", "/")
            venv_conf = f"{venv_path}/pyvenv.cfg"

            with open(venv_conf) as f:
                cfg = f.read()

            VENV = "venv" if "uv" not in cfg else "uv"
        elif ENV("CONDA_PREFIX"):
            VENV = "Conda"
        else:
            VENV = "Global"

        return VENV

    def _python_GIL(self):
        ver = self._python_version()
        return sys._is_gil_enabled() if VERSION(ver, ">=", "3.13.0") else True

    def _python_JIT(self) -> bool:
        ver = self._python_version()
        return sys._jit.is_active() if VERSION(ver, ">=", "3.14.0") else False

    def _experiment(self):

        stat = sys.version_info.releaselevel

        if stat == "alpha":
            suffix = stat
        if stat == "beta":
            suffix = stat
        if stat == "candidate":
            suffix = "rc"
        if stat == "final":
            suffix = "release"

        return suffix

    #   ====================================

    @overload
    def post(
        self,
        kwd: Literal["VERSION"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        mode: Literal["IN_RANGE", "COMPARE"] = "IN_RANGE",
        version: Union[str, VersionNum, Literal["3.10.0"]] = "3.10.0",
        op: Literal["<", "<=", ">", ">="] = None,
        op_ceil: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        *,
        experimental: bool = False,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["INTERPRETER"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        whitelist: Literal[
            "CPython",
            "PyPy",
            "IronPython",
            "GraalPy",
            "Jython",
            "MicroPython",
            "RustPython",
        ] = [],
        *,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["VENV"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        envs: Literal["Global", "Conda", "uv"] = None,
        *,
        conda_base: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "WARNING",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["GIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        ENABLE_FREE_THREADING: bool = False,
        *,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["JIT"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        ENABLE_JIT: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["module"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        build_type: Literal[
            "TheRock", "PyTorch", "JAX", "TensorFlow", "XGBoost", "MXNet"
        ],
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    #   ====================================

    def _post_python_version(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        mode: Literal["IN_RANGE", "COMPARE"] = "IN_RANGE",
        version: Union[str, VersionNum] = config.THRROCK_REQUIRED_PYTHON_VERSION_MIN,
        op: Literal["<", "<=", ">", ">="] = None,
        op_ceil: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = "3.13.15",
        experimental: bool = False,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):

        message("STATUS", "Check for Python 3 versions")

        # First, We need handle compoments test level and fail level.
        fail_result = self._compoment_fail(condition, fail_level)

        # Then we open 2 compare mode with "IN_RANGE" and "COMPARE."
        #   IN_RANGE mode: set exam range is a </<= x </<= b in open/closed (a, b).
        if mode == "IN_RANGE":

            if op is None or op_ceil is None or version_max is None:
                err_msg = dedent(
                    f"""\
                                 Found Invalid argument settings. We found you have incorrect argument set in FindPython class.
                                 The "IN_RANGE" examine containes with None.
                                  >>> FindPython().post(...,
                                    version: {version},
                                    op: {op},
                                    op_ceil: {op_ceil},
                                    version_max: {version_max},
                                    ..."""
                )
                message("FATAL_ERROR", err_msg)

            check = VERSION_IN_RANGE(
                version,
                op,
                self.Version,
                op_ceil,
                version_max,
                blacklist=blacklist,
                fuzzy=fuzzy,
            )
            result = SUCCESS if check else fail_result

            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                Found incompatiable Python version.
                                TheRock supported Python 3 versions from {version} to {version_max}, but you have version {self.Version}.
                                Please upgrade/downgrade to a compatiable version and activate a venv for TheRock.

                                 >>> traceback: Incompatable Python version {self.Version} out of support range"""
                )

        #   COMPARE mode: we take only one operator and exam if version check success.
        elif mode == "COMPARE":
            check = VERSION(self.Version, op, version, blacklist=blacklist, fuzzy=fuzzy)
            result = SUCCESS if check else fail_result
            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                Found incompatiable Python version.
                                TheRock supported Python 3 versions {op} {version}, but you have version {self.Version}.
                                Please upgrade/downgrade to a compatiable version and activate a venv for TheRock.

                                 >>> traceback: Incompatable Python version {self.Version} out of support range"""
                )

        else:
            message("FATAL_ERROR", f'Unsupported Python version exam mode "{mode}".')

        if VERSION(self.Version, "<", config.THRROCK_REQUIRED_PYTHON_VERSION_MIN):
            result = FATAL
            reason = dedent(
                f"""\
                            Found Python is EOL (End-Of-Life) version.
                            TheRock requires PSF supported Python which in 5-years life support. Python 3.9, 3.8
                            and earlier versions are deprecated/unsupported.
                            Please update to {config.THRROCK_REQUIRED_PYTHON_VERSION_MIN} version and higher versions.

                             >>> traceback: Found EOL Python versions {self.Version}"""
            )

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            message(result, reason)

        if not STRMATCH(self.Release, self._allowed_rel):
            experiment_version_reason = dedent(
                f"""\
                                               Found incompatiable Python version.
                                                TheRock found Python 3 version is an experimental build. Please use a stable Python 3
                                                build and activate a venv for TheRock.

                                                 >>> traceback: Unexpected experimental Python version {sys.version_info} out of support range"""
            )
            result = (
                self._compoment_fail(condition, fail_level)
                if not experimental
                else WARNING
            )

            message(result, experiment_version_reason)

        return result

    def _post_python_interpreter(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        whitelist: Iterable[
            Literal[
                "CPython",
                "PyPy",
                "IronPython",
                "GraalPy",
                "Jython",
                "MicroPython",
                "RustPython",
            ]
        ] = [],
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", f"Check for Python 3 Interpreter type")

        if whitelist is None:
            whitelist = ["CPython"]

        if STREQUAL(self.Interpreter, "CPython"):
            result = SUCCESS
            message("CHECK", self.Interpreter, check_result=result)

        elif self.Interpreter in whitelist:
            result = SUCCESS
            message("CHECK", f"{self.Interpreter} (In WhiteList)", check_result=result)
        else:
            result = self._compoment_fail(condition, fail_level)
            message("CHECK", self.Interpreter, check_result=result)
            reason = dedent(
                f"""\
                            Found Invalid Python 3 Interpreter type.
                            TheRock not support {self.Interpreter} type Python. Please use CPython type Python Interpreter based
                            venv to build TheRock.
                            If you're a Developer working on this, make sure contact to TheRock dev team with compatability
                                configuration, and add {self.Interpreter} to whitelist.

                             >>> traceback: Found Invalid Python interpreter implementation type {self.Interpreter}"""
            )
            message(result, reason)

        return result

    def _post_python_venv(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        envs: Literal["Global", "Conda", "uv"] = None,
        conda_base: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "WARNING",
    ):

        message("STATUS", "Check for Python 3 venv")

        env = self.VENV

        glob_reason = dedent(
            f"""\
                             Found Python is Global ENV.
                             TheRock recommends using a clear venv to install Python requirements, by not pollute your Global ENV
                             or hit Global ENV conflict TheRock required conpoments.

                              >>> traceback:  Found Python is Global ENV."""
        )

        conda_reason = dedent(
            f"""\
                              Found Python is Conda ENV.
                              TheRock recommends using a clear venv to install Python requirements, but found your environment
                              is Conda ENV. Please configure Conda package support status via TheRock information.

                               >>> traceback:  Found Python is Conda ENV."""
        )

        uv_reason = dedent(
            f"""\
                           Found Python is uv VENV.
                            TheRock recommends using `python -m venv` to create and install Python requirements, but found
                            your environment is uv VENV. Please configure uv environment created python wheel package
                            support status via TheRock information.

                             >>> traceback:  Found Python is uv VENV."""
        )

        conda_base_reason = dedent(
            f"""\
                                   Found Conda is using base environment.
                                   If you want individual conda env, please avoid use base environment, and create a new one with conda command:
                                    > conda create -n <CONDA_ENV_NAME> python=...
                                   If tests allowed conda env using base env, please set 'conda_base=True' in post("VENV") options.

                                    >>> traceback: Found Conda env using base env"""
        )

        if env == "venv":
            result = SUCCESS
            message("CHECK", "Python VENV", check_result=result)
        elif env == "Global" and env in envs:
            result = SUCCESS
            message("CHECK", "Global ENV (Enabled)", check_result=result)
        elif env == "Conda" and env in envs:
            # Allowcation with Conda base,

            conda_env_name = ENV("CONDA_DEFAULT_ENV")
            conda_env_dir = ENV("CONDA_PREFIX")
            conda_env_py = Path(conda_env_dir, "python.exe").as_posix()

            if not conda_base and conda_env_name == "base":
                result = self._compoment_fail(condition, fail_level)
                reason = conda_base_reason

            elif self.exe != conda_env_py:
                result = self._compoment_fail(condition, fail_level)
                reason = dedent(
                    f"""Found Conda ENV but with incorrect/no self contained Python.
                                    It seems your created conda env have not installed its python. If you insist use conda envs python, please create
                                    it by specify python installation:
                                        > conda create -n <CONDA_ENV_NAME> python=...

                                    Found Conda ENV name: {conda_env_name}
                                        expected python executable path: {conda_env_py}
                                        current  python executable path: {self.exe}

                                    >>> traceback:  Found Python is Conda ENV."""
                )
            else:
                result = SUCCESS
            message("CHECK", "Conda ENV (Enabled)", check_result=result)

            if result is not SUCCESS:
                message(result, reason)
        elif env == "uv" and env in envs:
            result = SUCCESS
            message("CHECK", "UV VENV (Enabled)", check_result=result)

        else:
            result = self._compoment_fail(condition, fail_level)
            message("CHECK", env, check_result=result)
            if env == "Global":
                reason = glob_reason
            if env == "Conda":
                reason = conda_reason
            if env == "uv":
                reason = uv_reason
            message(result, reason)

        return result

    def _post_python_GIL(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        ENABLE_FREE_THREADING: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for Python 3 is GIL enabled")
        gil = self.GIL

        if gil:
            result = SUCCESS
            message("CHECK", "Yes", check_result=result)
        elif not gil and ENABLE_FREE_THREADING:
            result = SUCCESS
            message("CHECK", "No", check_result=result)
        else:
            result = self._compoment_fail(condition, fail_level)
            message("CHECK", "No", check_result=result)
            reason = dedent(
                f"""\
                            Found Python 3 is Free-Threaded (No GIL) version.
                            TheRock has not confirm the build process with free-threaded Python executable.
                            Please ensure all Python library and its related requirements are production
                                ready with Free-Threaded Python support.

                            For TheRock dev teams, please set Free_Threaded and force_FATAL to True and False
                                respectively while Free-Threaded Python is allowed to generate stable venv
                                and enable to build TheRock.

                            >>> traceback: Found Python is Free-Threaded (No GIL) version"""
            )
            message(result, reason)
        return result

    def _post_python_JIT(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        ENABLE_JIT: bool = False,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for Python 3 is JIT enabled")
        jit = self.JIT

        if not jit:
            result = SUCCESS
            message("CHECK", "No", check_result=result)
        elif jit and ENABLE_JIT:
            result = SUCCESS
            message("CHECK", "Yes", check_result=result)
        else:
            result = self._compoment_fail(condition, fail_level)
            message("CHECK", "Yes", check_result=result)
            reason = dedent(
                f"""\
                            Found Python 3 enabled Just-In-Time feature.
                            TheRock has not confirm the build stability with Python JIT runtime enabled.
                            PSF has explictly mentioned JIT is a experimental feature under development,
                                not a stable feature for usual and production ready cases.
                            Please avoid to enable JIT feature until PSF docuemntation/PEP released that
                                JIT feature is confirmed ready for official release.

                            For TheRock dev teams, please set Enable_JIT and force_FATAL to True and False
                                respectively while Just-In-Time feature is in production ready status during
                                Python runtime to build TheRock.

                             >>> traceback: Found Python enabled JIT experimental feature"""
            )
            message(result, reason)

        return result

    def _post_python_module(
        self,
        *,
        build_type: Literal[
            "TheRock", "PyTorch", "JAX", "TensorFlow", "XGBoost", "MXNet", "UCCL"
        ],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        import re
        from importlib.metadata import version, PackageNotFoundError

        require_case_record = {}

        match build_type:
            case "TheRock":
                requirements_txt = gitrepo / r"requirements.txt"
            case "PyTorch":
                requirements_txt = (
                    gitrepo / r"external-builds/pytorch/pytorch/requirements.txt"
                )
            case "JAX":
                requirements_txt = gitrepo / r"external-builds/jax/requirements-jax.txt"

            case _:
                if build_type in ("TensorFlow", "XGBoost", "MXNet", "UCCL"):
                    message("FATAL_ERROR", "Unsupported build type")
                else:
                    message("FATAL_ERROR", "Unsupported")

        if not requirements_txt.resolve().exists():
            message(
                "FATAL_ERROR",
                dedent(
                    f"""\
                                          Fatal Error on analyzing requirements.txt.
                                          Please re-configure repository have contain it.
                                           > Expected file: {requirements_txt.resolve().as_posix()}
                                           > File existed:  {requirements_txt.exists()}

                                           >>> traceback: Cannot find requirements.txt with incomplete or broken repo"""
                ),
            )

        with open(requirements_txt, "r", encoding="utf-8") as req:
            for line in req:
                # 1. 清理註解與空白
                clean_line = line.split("#")[0].strip()
                if not clean_line:
                    continue

                # 2. 略過遞迴引入 (-r 或 --requirement)
                # 注意：如果你需要解析這個，必須像上一篇那樣寫成遞迴函式
                if clean_line.startswith("-r") or clean_line.startswith(
                    "--requirement"
                ):
                    message(
                        "STATUS", f"Skipping external requirement file: {clean_line}"
                    )
                    continue

                # 3. 處理環境標記 (Marker)
                if ";" in clean_line:
                    pkg_part, marker = clean_line.split(";", 1)
                    # 保留你原本的 OS 判斷，可視需求擴充
                    if (
                        'platform_system != "Windows"' in marker
                        and self.os == "Windows"
                    ):
                        continue
                    clean_line = pkg_part.strip()

                # 4. 精準提取套件名稱 (乾淨剝離 build[uv] 的 [uv] 與 >=1.0 版本號)
                match = re.match(r"^([A-Za-z0-9_\-\.]+)", clean_line)
                if not match:
                    continue
                pkg_name = match.group(1)

                message("STATUS", f"Check for Python package {pkg_name}")

                try:
                    # 5. 關鍵改動：檢查「套件」，免去 require_case_fix 轉換
                    installed_ver = version(pkg_name)
                    result = SUCCESS
                    message("CHECK", "Found", check_result=result)
                    # 如果你想在 LOG 印出版本，也可以改成 f"Found (v{installed_ver})"

                except PackageNotFoundError:
                    result = self._compoment_fail(condition, fail_level)
                    message("CHECK", "Not Installed", check_result=result)

                    # 使用 pkg_name 即可，不需要 import_target
                    _msg = dedent(
                        f"""\
                        Failed to find package {pkg_name}.
                        TheRock requires Python package {pkg_name} to be installed.
                        Please check your environment have installed via Package managers (pip, conda, uv etc.)"""
                    )
                    message(result, _msg)

                except Exception as e:
                    message(
                        "FATAL_ERROR", f"Check package {pkg_name} has {e} occurred."
                    )
                    result = "FATAL_ERROR"  # 或你原本對應的處理

                finally:
                    require_case_record[pkg_name] = result

        return require_case_record
