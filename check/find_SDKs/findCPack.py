from find_SDKs import FindSDK
from typing import Literal, overload
from textwrap import dedent

from utils.status import *
from utils.compare_functions import *
from utils import *


class FindCPack(FindSDK):

    cpack = "cpack.exe"

    def __init__(self):
        super().__init__()

        self.info: dict[Literal["exe", "Version", "msvc"], Any]

        self._dispatch_map = {
            "AVAIL": self._post_cpack_avail,
        }

    @property
    def exe(self):
        return self.info["exe"]

    @property
    def Version(self):
        return self.info["Version"]

    @property
    def msvc(self):
        return self.info["msvc"]

    def __WINDOWS__(self):
        return self._cpack_info()

    def __LINUX__(self):
        return self._cpack_info()

    def _cpack_info(self):

        exe = self._find_program(self.cpack)

        if "/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/" in exe:
            ver = self._find_version(
                exe, pattern=r"\d+\.\d+(?:\.\d+)*[-\w]*", args="--version"
            )
            msvc = True
        else:
            ver = self._find_version(
                exe, pattern=r"\d+\.\d+(?:\.\d+)*", args="--version"
            )
            msvc = False

        return {"exe": exe, "Version": ver, "msvc": msvc}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    def _post_cpack_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for CPack program")

        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)
        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot find avail CMake packing program (CPack).
                            This diag system required CMake packing with cpack program, but you don't have it in callable cmake.
                            CPack is one of the subprogram of CMake. Please check your CMake installation is broken or not.

                             >>> traceback: Missing CPack program from CMake"""
            )
            message(result, reason)

        return result


class FindInnoSetup(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {
            "AVAIL": self._post_innosetup_avail,
        }

    def __WINDOWS__(self):
        return {}

    def __LINUX__(self):
        message("FATAL_ERROR", "InnoSetup is not supported on non Windows OS.")

    def _find_innosetup(self):
        version_ispp_content = """
        [Setup]
        AppName=VerCheck
        AppVersion=1.0
        DefaultDirName={tmp}
        #pragma message "VERSION_MARKER:" + DecodeVer(Ver)
        """

        exe = self._find_program("iscc.exe")
        ver = self._find_version(
            exe, args=["-"], input=version_ispp_content, vertemp="X.Y.Z"
        )

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_innosetup_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found required CPack generator 'InnoSetup'.
                            This Diagnose system has enabled CPack generator check with InnoSetup installation, but can't
                            find one onb this device.
                            Please install it via InnoSetup installer.

                             >>> traceback: Required CPack Generator 'InnoSetup' not installed"""
            )
            message(result, reason)

        return result
