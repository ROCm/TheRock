from typing import Literal, Union, Any, overload

from . import FindSDK
from utils import *
from utils.compare_functions import *
from utils.system_function import *


class FindConda(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_conda_avail}

    def __WINDOWS__(self):
        return self._find_conda()

    def __LINUX__(self):
        return self._find_conda()

    def _find_conda(self):
        exe = self._find_program("conda")
        ver = self._find_version(exe, args="--version", vertemp="X.Y.Z")

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "FATAL"] = "HINT",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_conda_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "HINT",
    ):
        message("STATUS", "Check for Conda")

        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = "Optional Conda not found."
            message(result, reason)

        return result


class FindUV(FindSDK):

    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_uv_avail}

    def __WINDOWS__(self):
        return self._find_uv()

    def __LINUX__(self):
        return self._find_uv()

    def _find_uv(self):
        exe = self._find_program("uv")
        ver = self._find_version(exe, args="--version", vertemp="X.Y.Z")

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "FATAL"] = "HINT",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_uv_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "HINT",
    ):
        message("STATUS", "Check for Astral UV")

        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)
        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = "Optional Astral UV not found."
            message(result, reason)

        return result
