import sys
from textwrap import dedent
from utils import *
from utils.system_function import *
from utils.compare_functions import *
from find_SDKs import FindSDK


_wsl2_err = dedent(
    f"""\
                   Found Linux is WSL2 environment.
                   TheRock is a project targeting ROCm builds with native Linux/Windows. If you want your Machine is
                   Windows NT based, please build on Windows native with MSVC instead.
                   You can have build guides from here: > https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md

                    >>> traceback: Found Linux is not native with WSL2 kernel"""
)


class FindLinux(FindSDK):

    def __init__(self):
        super().__init__()
        self.info: dict[
            Literal["OS_TYPE", "OS_NAME", "OS_VERSION", "OS_KERNEL", "WSL2", "ENDIAN"],
            Any,
        ]

        self._dispatch_map = {
            "WSL2": self._post_wsl2,
            "BYTE_ORDER": self._post_endian,
            "NETWORK": self._post_network,
        }

    def __repr__(self):
        return f"{self.OS_NAME} {self.OS_VERSION} (Linux Kernel {self.OS_KERNEL}), WSL2 = {self.WSL2}"

    @property
    def OS_TYPE(self):
        return "Linux"

    @property
    def OS_NAME(self):
        return self.info["OS_NAME"]

    @property
    def OS_VERSION(self):
        return self.info["OS_VERSION"]

    @property
    def OS_KERNEL(self):
        return self.info["OS_KERNEL"]

    @property
    def WSL2(self):
        return self.info["WSL2"]

    @property
    def ENDIAN(self):
        return self.info["ENDIAN"]

    def __LINUX__(self):
        return self._linux_info()

    def _linux_info(self):
        import subprocess

        kernel = subprocess.check_output(["uname", "-r"], text=True).strip()
        kernel_version = kernel.split("-")[0]

        with open("/etc/os-release") as f:
            _f = f.read().splitlines()
            for _line in _f:
                _name_match = re.match(r'^NAME="?(.*?)"?$', _line)
                _version_match = re.match(r'^VERSION_ID="?(.*?)"?$', _line)

                if _name_match:
                    _LINUX_DISTRO_NAME = _name_match.group(1)
                if _version_match:
                    _LINUX_DISTRO_VERSION = _version_match.group(1)

        return {
            "OS_TYPE": "Linux",
            "OS_NAME": _LINUX_DISTRO_NAME,
            "OS_VERSION": _LINUX_DISTRO_VERSION,
            "OS_KERNEL": kernel_version,
            "WSL2": "Yes" if "microsoft-standard-WSL2" in kernel else "No",
            "ENDIAN": self._endian_type(),
        }

    def _endian_type(self) -> Literal["Little Endian", "Big Endian"]:
        return "Little Endian" if sys.byteorder == "little" else "Big Endian"

    @overload
    def post(
        self,
        kwd: Literal[r"WSL2"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["BYTE_ORDER"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        endian: Literal["Little Endian", "Big Endian"] = "Little Endian",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["NETWORK"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_wsl2(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check Linux Kernel is Windows Subsystem (WSL2)")
        result = self._compoment_fail(condition, fail_level) if self.WSL2 else SUCCESS
        message("CHECK", self.WSL2, check_result=result)
        if result is not SUCCESS:
            reason = _wsl2_err
            message(result, reason)

        return result

    def _post_endian(
        self,
        *,
        endian: Literal["Little Endian", "Big Endian"] = "Little Endian",
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Linux Byte Order type")
        byteOrder = self.ENDIAN
        if byteOrder == endian:
            result = SUCCESS
        elif byteOrder != endian or byteOrder not in ["Little Endian", "Big Endian"]:
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            Byte Order examination failed.
                            TheRock required System Byte Order is {endian}, but you have {byteOrder}.

                             >>> traceback: Invalid System Byte Order {byteOrder}"""
            )

        message("CHECK", byteOrder, check_result=result)
        if result is not SUCCESS:
            message(result, reason)

        return result

    def _post_network(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        import urllib.request, urllib.error

        message("STATUS", "Check for Network connection")

        url = r"https://github.com/ROCm/TheRock"
        timeout = 3
        try:
            urllib.request.urlopen(url, timeout=timeout)
            res = "Yes"
            result = SUCCESS
        except (TimeoutError, urllib.error.URLError):
            res = "No"
            result = self._compoment_fail(condition, fail_level)

        message("CHECK", res, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found your device have no available internet connection.
                            TheRock building subprojects will need network connection for downloading required source tarballs.
                            Please re-connect your device back online.

                             >>> traceback: No internet connection available"""
            )
            message(result, reason)

        return result

    def __WINDOWS__(self):
        message(
            "FATAL_ERROR",
            "Windows Device doesn't support Linux build mode!",
            RuntimeError,
        )

    def __BSD__(self):
        message(
            "FATAL_ERROR",
            "Windows Device doesn't support Linux build mode!",
            RuntimeError,
        )
