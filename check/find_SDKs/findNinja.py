from typing import Literal, overload
from textwrap import dedent

from utils.status import *
from utils.compare_functions import *
from utils import *
from find_SDKs import FindSDK


class FindNinja(FindSDK):

    THEROCK_REQUIRED_NINJA = VersionNum("1.13.1")

    def __init__(self):
        super().__init__()

        self._dispatch_map = {
            "AVAIL": self._post_ninja_avail,
            "MAX_JOBS": self._post_ninja_maxjob,
        }

    def __WINDOWS__(self):
        return self._find_ninja()

    def __LINUX__(self):
        return self._find_ninja()

    def _find_ninja(self):
        exe = self._find_program("ninja")
        ver = self._find_version(exe, args="--version", vertemp="X.Y.Z")
        return {
            "exe": exe,
            "Version": ver,
        }

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        min: Union[str, VersionNum] = "1.13.1",
        max: Union[str, VersionNum] = None,
        blacklist: Iterable[Union[str, VersionNum]],
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["MAX_JOBS"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        limit: int = 0,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_ninja_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        min: Union[str, VersionNum, Literal["1.13.1"]] = "1.13.1",
        max: Union[str, VersionNum] = None,
        blacklist: Iterable[Union[str, VersionNum]] = None,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for Ninja generator available")

        MIN = (
            VersionNum(min)
            if isinstance(min, (str, VersionNum))
            else self.THEROCK_REQUIRED_NINJA
        )
        MAX = VersionNum(max) if isinstance(max, (str, VersionNum)) else NOTDEFINED

        if self.exe is NOTFOUND:
            result = FATAL

        elif MAX is NOTDEFINED:
            result = (
                SUCCESS
                if VERSION(self.Version, ">=", MIN, blacklist=blacklist, fuzzy=False)
                else self._compoment_fail(condition, fail_level)
            )

        elif MIN and MAX:
            result = (
                SUCCESS
                if VERSION_IN_RANGE(MIN, "<=", self.Version, "<=", MAX)
                else self._compoment_fail(condition, fail_level)
            )

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            No available Ninja Generator Found. TheRock requires Ninja-Build as CMake generator. Both
                            Visual Studio and Strawberry Perl will provide a redistributable `ninja.exe` program from their installation.
                            > For Windows users, You can download/install from:
                                - Visual Studio Installer: Check CMake for Windows support from Desktop C++ development.
                                - GitHub: You can visit https://github.com/ninja-build/ninja and get latest release.
                            > For Linux users, You can download/install via package managers or build from release.
                                - command: sudo apt/dnf ninja-build
                                - command: sudo pacman -S ninja-build

                            TheRock also requires Ninja Generator with version greater equal then {min}.

                             >>> traceback: Cannot find a available Ninja Generator: Version {self.Version}"""
            )
            message(result, reason)
        return result

    def _post_ninja_maxjob(
        self,
        *,
        limit: int = None,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):

        message("STATUS", "Check for Ninja generator core limit")

        mx = ENV("MAX_JOBS")

        mx = 0 if mx is NOTDEFINED else mx

        limit = 0 if limit is None else limit

        if limit > 0:
            if mx is NOTDEFINED:
                result = self._compoment_fail(condition, fail_level)
                reason = dedent(
                    f"""\
                                Found Environment Variable 'MAX_JOBS' is not set.
                                TheRock requires set a maximum cpu core alloc with {limit}, but your MAX_JOBS is None.
                                Please export MAX_JOBS with an integer with value below {limit}.

                                 >>> traceback: TheRock required MAX_JOBS is {limit} but system not defined (allow all cores)"""
                )
            elif mx >= limit:
                result = self._compoment_fail(condition, fail_level)
                reason = dedent(
                    f"""\
                                Found Environment Variable 'MAX_JOBS' is set.
                                TheRock requires set a maximum cpu core alloc with {limit}, but your MAX_JOBS is {mx}.
                                Please lower your MAX_JOBS below {limit}.

                                 >>> traceback: TheRock required MAX_JOBS is {limit} but found {mx}"""
                )
            else:
                result = SUCCESS
        elif limit == 0:
            if mx:
                result = WARNING
                reason = dedent(
                    f"""\
                                Found MAX_JOBS has set.
                                TheRock testing on Ninja-Build cores allocation has not set max limit, but your device has set to {mx}.
                                If you requires set a max job on it, please ignore this warning.

                                 >>> traceback: Found defined CPU cores alloc max at {mx} but TheRock recommends all cores"""
                )
            else:
                result = SUCCESS
        else:
            message(
                "FATAL_ERROR",
                f"Posting on MAX cores must higher than 0 or allow all cores by set to 0.",
            )

        message("CHECK", f"Limit: {limit}, MAX_JOBS: {mx}", check_result=result)

        if result is not SUCCESS:
            message(result, reason)

        return result
