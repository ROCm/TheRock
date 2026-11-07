from typing import overload, Any, Literal
from textwrap import dedent

from . import FindSDK
from utils import *
from pathlib import Path


class FindPkgConfig(FindSDK):
    pkgconf_win = "pkg-config.bat"
    pkgconf_lin = "pkgconf"

    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_pkgconf_avail}

    @property
    def pkgconf(self) -> Literal["pkgconf", "pkg-config.bat"]:
        return self.info["pkgconf"]

    def __WINDOWS__(self):
        bat = self._find_program(self.pkgconf_win)
        ver = self._find_version(bat, args="--version", vertemp="X.Y")

        return {"exe": bat, "Version": ver, "pkgconf": self.pkgconf_win}

    def __LINUX__(self):
        exe = self._find_program(self.pkgconf_lin)
        ver = self._find_version(exe, args="--version", vertemp="X.Y")

        return {"exe": exe, "Version": ver, "pkgconf": self.pkgconf_lin}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    def _post_pkgconf_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", f"Check for pkg-config program `{self.pkgconf}`")

        if self.exe and self.os == "Windows":
            result = WARNING
            dep = True
        elif self.exe:
            result = SUCCESS
            dep = False
        else:
            result = self._compoment_fail(condition, fail_level)
            dep = False

        message("CHECK", self.Version, check_result=result)

        if result is WARNING and dep:
            reason = dedent(
                f"""\
                            Found diagnose system requires {self.pkgconf_win}.
                            TheRock have already changed PkgConfig dependices with pkg-config-lite program on Windows.
                            Please use FindPkgConfigLite class instead of FindPkgConfig to finding pkg-config.

                             >>> traceback: PkgConfig program {self.pkgconf_win} is deprecated."""
            )
            message("WARNING", reason)

        elif result is not SUCCESS and not dep:
            reason = dedent(
                f"""\
                            Cannot find available pkg-config program.
                            TheRock requires Pkg-Config program for dependices configure.
                            - On Windows platform, please install via Strawberry Perl.
                            - On Linux platform, please install it via package managers.

                             >>> traceback: Missing required pkg-config program {self.pkgconf}"""
            )
            message(result, reason)

        else:
            pass

        return result


class FindPkgConfigLite(FindSDK):

    pkg_config_lite = "pkg-config.exe"

    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_pkgconflite_avail}

    def __WINDOWS__(self):
        exe = self._find_program(self.pkg_config_lite)
        ver = self._find_version(exe)

        return {"exe": exe, "Version": ver}

    def __LINUX__(self):
        message("FATAL_ERROR", "Pkg-Config-Lite is not avail on Linux platform.")

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"],
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"],
    ): ...
    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    def _post_pkgconflite_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for PkgConfig lite program available")

        if self.exe:
            result = SUCCESS
        else:
            result = self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            TheRock can't find available Pkg-Config lite program.
                            TheRock requires Pkg-Config Lite program `{self.pkg_config_lite}` as a dependicy.
                            Please install it via winget/choco/scoop install source from bloodrock.

                             >>> traceback: No avail Pkg-Config Lite program {self.pkg_config_lite} or not in PATH"""
            )

            message(result, reason)

        return result
