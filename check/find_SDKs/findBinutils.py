from textwrap import dedent
from pathlib import Path
from typing import Literal, Any, overload

from find_SDKs import FindSDK
from utils import *


class FindAR(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_binutils_ar_avail}

    def __WINDOWS__(self):
        return self._find_binutils_ar()

    def __LINUX__(self):
        return self._find_binutils_ar()

    def _find_binutils_ar(self):
        exe = self._find_program("ar.exe")
        ver = self._find_version(exe, args="--version", vertemp="X.Y")

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        /,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_binutils_ar_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", f"Check for binutils archiver")

        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found required GNU/binutils archiver.
                            TheRock required complete GCC compiler and GNU/Binutils toolchain.
                            Please check your toolchain installation.

                            On Linux platform, please use package managers to install binutils.

                             >>> traceback: Binutils archiver 'ar' missing."""
            )
            message(result, reason)
        return result


class FindAS(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_binutils_as_avail}

    def __WINDOWS__(self):
        return self._find_binutils_as()

    def __LINUX__(self):
        return self._find_binutils_as()

    def _find_binutils_as(self):
        exe = self._find_program("as.exe")
        ver = self._find_version(exe, args="--version", vertemp="X.Y")

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        /,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_binutils_as_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", f"Check for binutils assembler")

        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found required GNU/binutils assembler.
                            TheRock required complete GCC compiler and GNU/Binutils toolchain.
                            Please check your toolchain installation.

                            On Linux platform, please use package managers to install binutils.

                             >>> traceback: Binutils assembler 'as' missing."""
            )
            message(result, reason)
        return result


class FindLD(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_binutils_ld_avail}

    def __WINDOWS__(self):
        return self._find_binutils_ld()

    def __LINUX__(self):
        return self._find_binutils_ld()

    def _find_binutils_ld(self):
        exe = self._find_program("ld.exe")
        ver = self._find_version(exe, args="--version", vertemp="X.Y")

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        /,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_binutils_ld_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", f"Check for binutils linker")

        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found required GNU/binutils linker.
                            TheRock required complete GCC compiler and GNU/Binutils toolchain.
                            Please check your toolchain installation.

                            On Linux platform, please use package managers to install binutils.

                            >>> traceback: Binutils linker 'ld' missing."""
            )
            message(result, reason)
        return result
