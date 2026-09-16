from typing import Literal, Any, Union, overload
from pathlib import Path
import subprocess
from textwrap import dedent

from find_SDKs import FindSDK
from utils import *
from . import FindSDK
from .refs._findROCm import (
    rocX_config_version_cmake_phonebook,
    ROC_X_LIBRARIES_LIST,
    ROC_X_LIBS_TYPEHINT,
)
from .findAMD_LLVM import FindAMDLLVM


class FindROCm(FindSDK):
    def __init__(self):
        super().__init__()

        self.info: dict[ROC_X_LIBS_TYPEHINT, Any]

        self._dispatch_map = {
            "ROCM_TYPE": self._post_rocm_dist,
            "roc-X": self._post_rocX,
        }

    @property
    def __ROCM__(self):
        return self.info["__ROCM__"]

    @property
    def ROCM_HOME(self):
        return self.info["ROCM_PATH"]

    @property
    def ROCM_PATH(self):
        return self.info["ROCM_PATH"]

    @property
    def ROCM_VERSION(self):
        return self.info["rocm-core"]

    @property
    def ROCM_TYPE(self):
        return self.info["ROCM_DIST"]

    @property
    def stats(self):
        return self.__stat__()

    def __repr__(self):
        return f"AMD ROCm {self.ROCM_VERSION} at ROCM_HOME: {self.ROCM_HOME}"

    def __stat__(self):

        stat = f"ROCm Software Stack Configuration\n"

        stat += f"        {self.__ROCM__}\n"

        for k, v in self.info.items():
            if k == ("ROCM_DIST", "ROCM_PATH", "__ROCM__"):
                pass
            else:
                stat += f"        {k:<25} {v}\n"
        return stat

    def __WINDOWS__(self):
        return self._find_ROCm()

    def __LINUX__(self):
        return self._find_ROCm()

    def _find_ROCm(self):

        rocX: ROC_X_LIBS_TYPEHINT
        llvm = FindAMDLLVM()
        llvm_dir = llvm.info["LLVM Configuration"]["LLVM DIR"]

        amd_base_type = llvm.info["LLVM Configuration"]["LLVM Dist"]

        if amd_base_type == "AMD-LLVM":
            rocm_dir = Path(llvm_dir.parent.parent).resolve()
        elif amd_base_type == "AMD HIP SDK":
            rocm_dir = llvm_dir
        else:
            # Linux ROCm wait for examine
            rocm_dir = NOTDEFINED

        rocX_status: dict[ROC_X_LIBS_TYPEHINT, VersionNum] = {}

        rocX_status["__LLVM__"] = llvm
        rocX_status["ROCM_PATH"] = rocm_dir

        for rocX in ROC_X_LIBRARIES_LIST:
            subdir = rocX_config_version_cmake_phonebook[rocX.lower()]

            if rocm_dir and subdir is not None:
                rocX_config_version_cmake = Path(rocm_dir, subdir).resolve()
                if rocX == "TheRock":
                    if amd_base_type == "AMD-LLVM":
                        with open(
                            rocX_config_version_cmake, "r", encoding="utf-8"
                        ) as f:

                            rocX_version = VersionNum(f.read())
                    else:
                        rocX_version = NOTDEFINED
                elif rocX == "rocGDB":
                    rocgdb = (rocm_dir / "bin/rocgdb").resolve()

                    if rocgdb.exists():
                        rocgdb_version_query = (
                            subprocess.run(
                                [rocgdb.as_posix(), "--version"],
                                capture_output=True,
                                text=True,
                                check=True,
                            )
                            .stdout.splitlines()[0]
                            .split("GNU gdb (ROCm) ")[-1]
                        )
                    else:
                        rocgdb_version_query = NOTDEFINED

                    rocX_version = rocgdb_version_query

                elif rocX_config_version_cmake.exists():
                    rocX_version_query = cmake_variable_finder(
                        file=rocX_config_version_cmake,
                        hint=["PACKAGE_VERSION"],
                        output="all",
                    )
                    rocX_version = rocX_version_query["PACKAGE_VERSION"]
                elif subdir == "Same as AMD-LLVM":
                    rocX_version = llvm.Version

                else:
                    rocX_version = NOTDEFINED
            else:
                rocX_version = NOTDEFINED

            rocX_status[rocX] = rocX_version

        if amd_base_type == "AMD-LLVM" and rocX_status["TheRock"]:
            rocX_status["ROCM_DIST"] = "TheRock"
            rocX_status["__ROCM__"] = f"TheRock {rocX_status["TheRock"]} at {rocm_dir}"
        elif amd_base_type == "AMD HIP SDK":
            rocX_status["ROCM_DIST"] = "AMD HIP SDK"
            rocX_status["__ROCM__"] = (
                f"AMD HIP SDK {llvm.info["LLVM Configuration"]["AMD HIP SDK version"]} at {rocm_dir}"
            )
        else:
            rocX_status["__ROCM__"] = NOTDEFINED
            rocX_status["ROCM_DIST"] = NOTDEFINED

        return rocX_status

    @overload
    def post(
        self,
        kwd: Literal["ROCM_TYPE"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        ROCM_TYPE: Literal["ROCm", "AMD HIP SDK", "TheRock"],
        fail_level: Literal["ERROR", "WARNING", "SUCCESS"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["roc-X"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        roc_X: list[ROC_X_LIBS_TYPEHINT],
        optional: list[ROC_X_LIBS_TYPEHINT] = [],
        fail_level: Literal["ERROR", "WARNING", "SUCCESS"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_rocm_dist(
        self,
        *,
        ROCM_TYPE: Literal["ROCm", "AMD HIP SDK", "TheRock", "ALL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "SUCCESS"] = "ERROR",
    ):
        rocm_type = self.ROCM_TYPE

        if ROCM_TYPE == "ALL":
            result = (
                SUCCESS
                if rocm_type in ("ROCm", "AMD HIP SDK", "TheRock")
                else self._compoment_fail(condition, fail_level)
            )
        else:
            result = (
                SUCCESS
                if rocm_type == ROCM_TYPE
                else self._compoment_fail(condition, fail_level)
            )

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found invalid ROCm Build dist/type.
                            TheRock requires {ROCM_TYPE} build/dist, but you have {rocm_type}.

                             >>> traceback: Found invalid ROCm dist"""
            )
            message(result, reason)

        return result

    def _post_rocX(
        self,
        *,
        roc_X: list[ROC_X_LIBS_TYPEHINT],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        optional: list[ROC_X_LIBS_TYPEHINT] = [],
        fail_level: Literal["ERROR", "WARNING", "SUCCESS"] = "ERROR",
    ):

        roc_X = sorted(set(roc_X))
        optional = sorted(set(optional))

        if any(rocX not in ROC_X_LIBRARIES_LIST for rocX in roc_X):
            invalid = [rocx for rocx in roc_X if rocx not in ROC_X_LIBRARIES_LIST]
            message(
                "FATAL_ERROR",
                f"Invalid or unsupported rocm library {invalid} defined in roc_X list.",
            )
        if any(rocX not in ROC_X_LIBRARIES_LIST for rocX in optional):
            invalid = [rocx for rocx in optional if rocx not in ROC_X_LIBRARIES_LIST]
            message(
                "FATAL_ERROR",
                f"Invalid or unsupported rocm library {invalid} defined in optional list.",
            )
        if not roc_X or not isinstance(roc_X, list):
            message("FATAL_ERROR", f"ROCm library is empty{invalid}.")

        rocm_check_result = {}

        for rocX in roc_X:

            message("STATUS", f"Check for ROCm Library {rocX}")

            rocX_version = self.info[rocX]

            if rocX_version is NOTDEFINED:
                result = (
                    self._compoment_fail(condition, fail_level)
                    if rocX not in optional
                    else WARNING
                )
            else:
                result = SUCCESS

            message("CHECK", rocX_version, check_result=result)
            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                Cannot find required ROCm libraries: {rocX}.
                                You may need configure your ROCm build.

                                 > ROCm directory: {self.ROCM_HOME}

                                 >>> traceback: Cannot configure ROCm library {rocX}"""
                )
                message(result, reason)

            rocm_check_result[rocX] = result

        return rocm_check_result
