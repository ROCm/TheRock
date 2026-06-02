from typing import overload, Literal, Any
from textwrap import dedent

from . import FindSDK
from utils import *


_no_dvc_reason = dedent(
    f"""\
                        TheRock cannot find available dvc program.
                        DVC is used for version control of pre-compiled MIOpen kernels for reducing compile time.
                        On Windows platform, please install it via package manager winget/chocolatey/scoop
                         > `winget install --id Iterative.DVC --silent --accept-source-agreements`
                         > `choco install dvc`
                         > `scoop install dvc`
                        On Linux platform, please install it via snap command
                         > `snap install --classic dvc`

                         >>> traceback: No avail dvc program."""
)


class FindDVC(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_dvc_avail}

    def __WINDOWS__(self):
        return self._find_dvc()

    def __LINUX__(self):
        return self._find_dvc()

    def _find_dvc(self):

        dvc = self._find_program("dvc")
        ver = self._find_version(
            dvc, args="--version", vertemp="X.Y.Z", pattern=r"\d+\.\d+(?:\.\d+)*"
        )
        return {"exe": dvc, "Version": ver}

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

    def _post_dvc_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for DVC program available")
        result = (
            self._compoment_fail(condition, fail_level)
            if self.exe is NOTFOUND
            else SUCCESS
        )
        message("CHECK", self.Version, check_result=result)
        if self.exe is NOTFOUND:
            message(result, _no_dvc_reason)
        return result
