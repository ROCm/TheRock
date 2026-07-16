from typing import Literal, overload, Any
from textwrap import dedent

from . import FindSDK
from utils import *


_no_git_reason = dedent(
    f"""\
                        No available Git program.
                        TheRock requires Git program to fetch submodule and 3rd-party library to build
                        TheRock core libraries and its dependicies. Please install Git program into
                        your computer, and make sure git program is in PATH.
                         > Windows users please download Git for Windows installer via git-scm website, or install
                            it via winget/chocolatey/scoop CLI package manager.
                         > Linux users please install via your Linux distro package manager.

                         >>> traceback: No available Git program.
                        """
)


class FindGit(FindSDK):
    def __init__(self):
        super().__init__()

        self.info: dict[Literal["exe", "Version"], Any]

        self._dispatch_map = {"AVAIL": self._post_git_avail}

    def __WINDOWS__(self):
        return self._find_git()

    def __LINUX__(self):
        return self._find_git()

    def _find_git(self):
        git = self._find_program("git")
        ver = self._find_version(git, args="--version", vertemp="X.Y.Z")

        return {
            "exe": git,
            "Version": ver,
        }

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

    def _post_git_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for Git program")

        result = (
            self._compoment_fail(condition, fail_level)
            if self.exe is NOTFOUND
            else SUCCESS
        )
        message("CHECK", self.Version, check_result=result)
        if result is not SUCCESS:
            message(result, _no_git_reason)
        return result
