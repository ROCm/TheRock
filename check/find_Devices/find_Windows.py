import platform, sys

from utils import *
from utils.system_function import *
from utils.compare_functions import *
from find_SDKs import FindSDK
from utils.Registry import regedit
import subprocess
from textwrap import dedent


_win10_warn = dedent(
    f"""\
                     Found build target OS is Windows 10.
                     TheRock supports build for Windows 11 (build 10.0.22000.0) with Visual Studio 2022.
                     The official support status for Windows 10 has no ETA or no current plan on Windows 10.
                     Please build on Windows 11 platform.

                      >>> traceback: Found Windows is Windows 10."""
)

_win7_err = dedent(
    f"""\
                   Found build target OS is Legacy Windows version.
                   Please build TheRock on Windows 11.

                    >>> traceback: Found Unsupported/Legacy Windows version."""
)

_posix_err = dedent(
    f"""\
                    Found your Windows build environment using Cygwin/MSYS2.
                    TheRock project implementation on Windows platform using MSVC toolchain.
                    For Windows OS full support, so please build in Visual Studio Developer Command Prompt/
                    Developer PowerShell instead.
                    TheRock has no official support for Cygwin, MSYS2, MinGW64 platforms.
                    Also, if you continue to build in Cygwin/MSYS2 shell, you may hit errors like
                        `ld: error: export ordinal too large: XXXXX` link error or missing cygwin/msys2
                        runtime DLLs when you packing for redistributables.

                     >>> traceback: Found unsupported Cygwin/MSYS2 based env"""
)

_wine_err = dedent(
    f"""\
                   Found your Windows configuration is Wine.
                    TheRock not support Windows on Wine emulate layer. Please use Windows
                    native, Complete Vitural Machine, or CI/CD container to build TheRock.

                     >>> traceback: Found Windows is Wine emulate layer"""
)

_long_path_warn = dedent(
    f"""\
                         Found your Windows have not enable long PATH yet.
                         TheRock needs Windows enable Long Path to generate long name files during
                         build. If filename reach MAX_PATH limit will cause long name objects
                         cannot save correctly or other unexpected errors.

                          >>> traceback: Windows Long PATHs is disabled"""
)


class FindWindows(FindSDK):

    East_Asia_Encode = {
        "950": "Traditional Chinese (Big5)",
        "936": "Simplfied Chinese (GBK)",
        "932": "MS Japanese (Shift-JIS)",
        "949": "Korean UHC Encoding (UHC)",
    }

    def __init__(self):
        self.info: dict[
            Literal[
                "OS_TYPE",
                "OS_NAME",
                "OS_KERNEL",
                "OS_BUILD",
                "ENDIAN",
                "POSIX",
                "WINE",
                "CHCP",
                "MAX_PATH",
            ],
            Any,
        ] = self._windows_info()
        self.env = self._sysdm_cpl_env()

        self._dispatch_map = {
            r"VERSION": self._post_os_name,
            r"CHCP": self._post_chcp,
            r"POSIX": self._post_posix,
            r"WINE": self._post_wine,
            r"MAX_PATH": self._post_maxpath,
            r"BYTE_ORDER": self._post_endian,
            r"NETWORK": self._post_network,
            r"HIP_PATH": self._post_hip_path,
        }

    def __str__(self):
        return (
            f"{self.OS_NAME} {self.OS_VERSION}, Feature Experience pack {self.OS_BUILD}"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def OS_TYPE(self):
        return self.info["OS_TYPE"]

    @property  # Windows 11
    def OS_NAME(self):
        return self.info["OS_NAME"]

    @property  # 25H2
    def OS_VERSION(self):
        return self.info["OS_VERSION"]

    @property  # 26100
    def OS_KERNEL(self):
        return self.info["OS_KERNEL"]

    @property
    def OS_BUILD(self):
        return self.info["OS_BUILD"]

    @property
    def ENDIAN(self):
        return self.info["ENDIAN"]

    @property
    def POSIX(self) -> bool:
        return self.info["POSIX"]

    @property
    def WINE(self) -> bool:
        return self.info["WINE"]

    @property
    def MAX_PATH(self) -> bool:
        return self.info["MAX_PATH"]

    @property
    def CHCP(self):
        return self.info["CHCP"]

    def __WINDOWS__(self):
        return self._windows_info()

    def _windows_info(self):
        _os_major = platform.release()
        _os_build = platform.version()
        _os_update = regedit(
            "HKLM",
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
            key_name="DisplayVersion",
        )

        return {
            "OS_TYPE": "WINDOWS",
            "OS_NAME": f"Windows {_os_major}",
            "OS_VERSION": _os_update,
            "OS_KERNEL": _os_build,
            "OS_BUILD": f"{_os_build}",
            "ENDIAN": self._endian_type(),
            "POSIX": self._test_posix(),
            "WINE": self._test_wine(),
            "MAX_PATH": self._test_path(),
            "CHCP": self._test_chcp(),
        }

    def _endian_type(self) -> Literal["Little Endian", "Big Endian"]:
        return "Little Endian" if sys.byteorder == "little" else "Big Endian"

    def _sysdm_cpl_env(self):
        return os.environ

    def _test_chcp(self):
        _, query = (
            subprocess.run(["chcp.COM"], text=True, capture_output=True)
            .stdout.strip()
            .replace(" ", "")
            .split(":")
        )

        return query

    def _test_path(self) -> bool:
        _long_path = regedit(
            "HKLM",
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            key_name=r"LongPathsEnabled",
        )

        return True if _long_path == "1" else False

    def _test_posix(self) -> Literal["Cygwin", "MSYS2", False]:

        if ENV("CYGWIN"):
            return "Cygwin"
        if ENV("MSYSTEM"):
            e = ENV("MSYSTEM")
            return f"MSYS2: {e}"

        return False

    def _test_wine(self) -> bool:

        return True if ENV("WINEPREFIX") else False

    @overload
    def post(
        self,
        kwd: Literal[r"VERSION"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal[r"POSIX"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal[r"WINE"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal[r"MAX_PATH"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal[r"CHCP"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
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
    @overload
    def post(
        self,
        kwd: Literal["HIP_PATH"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_os_name(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        ## Return
        - Windows 11:   `SUCCESS`
        - Windows 10:   `WARNING`
        - Windows 8/7:  `FATAL`
        """
        message("STATUS", "Check for Windows version")

        if STREQUAL(self.OS_NAME, "Windows 11"):
            result = SUCCESS
        elif STREQUAL(self.OS_NAME, "Windows 10"):
            result = self._compoment_fail(condition, fail_level)
            reason = _win10_warn
        else:
            result = self._compoment_fail(condition, fail_level)
            reason = _win7_err

        message("CHECK", f"{self.OS_NAME} ({self.OS_KERNEL})", check_result=result)

        if result is not SUCCESS:
            message(fail_level, reason)

        return result

    def _post_posix(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Windows is Cygwin/MSYS2 ENV")

        result = self._compoment_fail(condition, fail_level) if self.POSIX else SUCCESS
        message("CHECK", f"{self.POSIX}", check_result=result)

        if result is not SUCCESS:
            message(fail_level, _posix_err)

        return result

    def _post_wine(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Windows is WINE enumulate layer")

        result = self._compoment_fail(condition, fail_level) if self.WINE else SUCCESS
        message("CHECK", f"{self.WINE}", check_result=result)

        if result is not SUCCESS:
            message(fail_level, _wine_err)

        return result

    def _post_maxpath(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "WARNING",
    ):
        message("STATUS", "Check for Windows Long PATHs enabled")

        result = (
            SUCCESS
            if self.MAX_PATH == 1
            else self._compoment_fail(condition, fail_level)
        )

        message("CHECK", f"{self.MAX_PATH}", check_result=result)
        if result is not SUCCESS:
            message(result, _long_path_warn)

        return result

    def _post_chcp(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "WARNING",
    ):
        message("STATUS", "Check for Windows Terminal Code Page")

        encode = self.CHCP

        if STREQUAL(encode, "65001"):
            result = SUCCESS
            message("CHECK", encode, check_result=result)
        else:
            result = self._compoment_fail(condition, fail_level)
            lang = (
                self.East_Asia_Encode[encode]
                if encode in self.East_Asia_Encode.keys()
                else ""
            )
            message("CHECK", encode, check_result=result)

            reason = dedent(
                f"""
                            Found Current Terminal Session is not UTF-8 encoding.
                            In general compile process, if tools not support native locale languange will print lots of
                                massive unreadable code/texts/status. It is not FATAL to TheRock build, but will confuses
                                Users/Devs with unrecognizable char in outputs and logs (probably).
                            To avoid this problem, in current session type 'chcp 65001' to switch Code Pages to UTF-8.
                             >>> traceback: Current Terminal session code page is cp{encode} {lang}"""
            )
            message(result, reason)
        return result

    def _post_endian(
        self,
        *,
        endian: Literal["Little Endian", "Big Endian"] = "Little Endian",
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Windows Byte Order type")

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

    def _post_hip_path(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Windows have HIP SDK releated paths")

        existing_hip_paths = {
            key: Path(value).as_posix()
            for key, value in os.environ.items()
            if "HIP_PATH" in key.upper() and Path(value).exists()
        }

        if existing_hip_paths or Path("C:/Program Files/AMD/ROCm").exists():
            result = self._compoment_fail(condition, fail_level)
        else:
            result = SUCCESS

        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            program_file_amd_rocm = Path("C:/Program Files/AMD/ROCm").exists()
            reason = dedent(
                f"""\
                            TheRock Disgnose system found HIP_PATH and its related default path folder.
                            TheRock build system recommends current build environment without existed HIP SDK.
                            Please remove the entire C:/Program Files/AMD/ROCm folder and installed HIP SDK inside.
                            If your HIP SDK installed at other locations, please remove 'C:/Program Files/AMD/ROCm'
                                folder (Generated by installer) and remove any HIP_PATH environment variables.

                             > Possiable existed HIP_PATH* variable:
                             {existing_hip_paths}

                             > Path("C:/Program Files/AMD/ROCm"): {program_file_amd_rocm}

                             >>> traceback: Found Existed HIP_PATH variable and its installation"""
            )
            message(result, reason)

        return result

    def __LINUX__(self):
        message("FATAL_ERROR", "Linux Device doesn't support Windows Build Mode!")

    def __BSD__(self):
        message("FATAL_ERROR", "Linux Device doesn't support Windows Build Mode!")

    def __MACOS__(self):
        message("FATAL_ERROR", "Linux Device doesn't support Windows Build Mode!")
