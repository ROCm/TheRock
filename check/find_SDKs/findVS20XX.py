from typing import Literal, Iterable, Tuple, overload, override
from pathlib import Path
from textwrap import dedent

from . import FindSDK
from .refs._find_VS20XX import VISUAL_STUDIO_20XX_INFO_PROPERTY
from utils import *
from utils.utility_functions import *
from utils.compare_functions import *
from utils.system_function import *

VS20XX_VERSION_LIST = ["VS2015", "VS2017", "VS2019", "VS2022", "VS2026"]
VS20XX_VERSION_OPTIONS = Literal[*VS20XX_VERSION_LIST]

MSVC_VERSION_LIST = ["v140", "v141", "v142", "v143", "v145"]
MSVC_VERSION_OPTIONS = Literal[*MSVC_VERSION_LIST]


class FindVS20XX(FindSDK):

    VS20XX_VERSION_DICT = {
        "18.0": "VS2026",
        "17.0": "VS2022",
        "16.0": "VS2019",
        "15.0": "VS2017",
        "14.0": "VS2015",
    }

    def __init__(self):
        super().__init__()

        self.info: dict[VISUAL_STUDIO_20XX_INFO_PROPERTY, Any]

        self._dispatch_map = {
            "VS20XX": self._post_VS20XX,
            "VC_Host": self._post_msvc_host,
            "VC_Target": self._post_msvc_arch,
            "VC_Version": self._post_msvc_ver,
            "UCRT": self._post_ucrt,
            "VC_ATL": self._post_msvc_atl,
            "MSVC": self._post_msvc_cl,
            "ML64": self._post_msvc_ml64,
            "LIB": self._post_msvc_lib,
            "LINK": self._post_msvc_link,
            "MT": self._post_ucrt_mt,
            "RC": self._post_ucrt_rc,
            "dumpbin": self._post_dumpbin,
        }

    @property
    def VS20XX(self):
        return self.info["VS20XX"]

    @property
    def VS_DIR(self):
        return self.info["VS_DIR"]

    @property
    def VC_HOST(self):
        return self.info["VC_HOST"]

    @property
    def VC_ARCH(self):
        return self.info["VC_ARCH"]

    @property
    def VC_VERSION(self):
        return self.info["VC_VER"]

    @property
    def VC_ATLMFC(self):
        return self.info["VC_ATLMFC"]

    @property
    def UCRT_VERSION(self):
        return self.info["VC_UCRT"]

    @property
    def MSBuild(self) -> Path:
        return self.info["msbuild.exe"]

    @property
    def cmake(self) -> Path:
        return self.info["cmake.exe"]

    @property
    def cl(self) -> Path:
        return self.info["cl.exe"]

    @property
    def ml64(self) -> Path:
        return self.info["ml64.exe"]

    @property
    def lib(self) -> Path:
        return self.info["lib.exe"]

    @property
    def link(self) -> Path:
        return self.info["link.exe"]

    @property
    def rc(self) -> Path:
        return self.info["rc.exe"]

    @property
    def mt(self) -> Path:
        return self.info["mt.exe"]

    def __WINDOWS__(self) -> dict[VISUAL_STUDIO_20XX_INFO_PROPERTY, Any]:
        VS20XX = self._VS20XX_VERISON()

        VS_DIR = (
            Path(ENV("VSINSTALLDIR")).resolve().as_posix()
            if ENV("VSINSTALLDIR")
            else "N/A"
        )

        VC_DETAILS = self._VS20XX_VC_DETAILS()
        UCRT_DETAILS = self._VS20XX_UCRT_DETAILS()
        MSBUILD = self._VS20XX_MSBuild()
        CMAKE = self._VS20XX_CMake(VS_DIR)

        CL = self._find_program("cl.exe")
        ML64 = self._find_program(CL.parent / "ml64")
        LIB = self._find_program(CL.parent / "lib.exe")
        LINK = self._find_program(CL.parent / "link.exe")
        RC = self._find_program("rc.exe")
        dumpbin = self._find_program("dumpbin.exe")
        nmake = self._find_program("nmake.exe")
        mt = self._find_program("mt.exe")

        vsdict = {
            "VS20XX": VS20XX,
            "VS_DIR": VS_DIR,
            "VC_VER": VC_DETAILS[2],
            "VC_HOST": VC_DETAILS[0],
            "VC_ARCH": VC_DETAILS[1],
            "VC_ATLMFC": VC_DETAILS[3],
            "VC_UCRT": UCRT_DETAILS,
            "msbuild.exe": MSBUILD,
            "cmake.exe": CMAKE,
            "nmake.exe": nmake,
            "cl.exe": CL,
            "ml64.exe": ML64,
            "lib.exe": LIB,
            "link.exe": LINK,
            "rc.exe": RC,
            "mt.exe": mt,
            "dumpbin.exe": dumpbin,
        }

        return vsdict

    def _VS20XX_VERISON(self):
        with EXISTS(env="VisualStudioVersion") as VAR:
            return self.VS20XX_VERSION_DICT[VAR] if VAR else NOTDEFINED

    def _VS20XX_VC_DETAILS(self):

        VAR: str

        with EXISTS(env="VSCMD_ARG_HOST_ARCH") as VAR:
            VC_HOST = VAR if VAR else NOTDEFINED

        with EXISTS(env="VSCMD_ARG_HOST_ARCH") as VAR:
            VC_ARCH = VAR if VAR else NOTDEFINED

        with EXISTS(env="VCToolsVersion") as VAR:
            VC_VERSION = self._VC_v14X(VAR) if VAR else NOTDEFINED

        with EXISTS(env="VCToolsInstallDir") as VAR:
            if VAR is NOTDEFINED:
                VC_DIR, VC_ATLMFC = NOTDEFINED, NOTDEFINED
            else:
                VC_DIR = VAR.replace("\\", "/")
                VC_ATLMFC = (
                    Path(f"{VC_DIR}/atlmfc/include/atlbase.h").resolve().exists()
                )

        return VC_HOST, VC_ARCH, VC_VERSION, VC_ATLMFC

    def _VS20XX_UCRT_DETAILS(self):

        VAR: str

        with EXISTS(env="WindowsSDKVersion") as VAR:
            UCRT_VER = VAR.replace("\\", "") if VAR else NOTDEFINED

        return UCRT_VER

    def _VC_v14X(self, VC_VERSION: str):
        # https://learn.microsoft.com/zh-tw/cpp/overview/compiler-versions?view=msvc-170
        # https://learn.microsoft.com/en-us/visualstudio/releases/2026/release-notes

        vc_ver = VersionNum(VC_VERSION)

        if VERSION(vc_ver, ">=", "14.50"):
            return "v145"

        if VERSION_IN_RANGE("14.30", "<=", vc_ver, "<", "14.50"):
            return "v143"

        if VERSION_IN_RANGE("14.20", "<=", vc_ver, "<", "14.30"):
            return "v142"

        if VERSION_IN_RANGE("14.10", "<=", vc_ver, "<", "14.20"):
            return "v141"

        return NOTDEFINED

    def _VS20XX_MSBuild(self):

        with EXISTS(env="VSINSTALLDIR") as VAR:
            if VAR is NOTDEFINED:
                return NOTFOUND
            msbuild = Path(VAR, "MSBuild/Current/Bin/amd64/msbuild.exe")
            if msbuild.exists():
                return msbuild

    def _VS20XX_CMake(self, VS_DIR: Union[Path, str]):
        if VS_DIR:
            vscmake = Path(
                VS_DIR,
                "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe",
            )
            if vscmake.exists():
                return vscmake
            else:
                return NOTFOUND
        else:
            return NOTFOUND

    #   ===============================================================

    @overload
    def post(
        self,
        kwd: Literal["VS20XX"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        whitelist: Literal["VS2015", "VS2017", "VS2019", "VS2022", "VS2026"] = "VS2022",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "WARNING",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["VC_Host"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["VC_Target"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["VC_Version"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        msvc: Literal["v140", "v141", "v142", "v143", "v145"] = "v143",
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["UCRT"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["VC_ATL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["MSVC"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["ML64"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["LIB"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["LINK"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["RC"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["MT"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...
    @overload
    def post(
        self,
        kwd: Literal["dumpbin"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_VS20XX(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        whitelist: Literal["VS2015", "VS2017", "VS2019", "VS2022", "VS2026"] = [
            "VS2022"
        ],
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "WARNING",
    ):

        message("STATUS", "Check for Visual Studio version")

        if isinstance(whitelist, str):
            if whitelist not in VS20XX_VERSION_LIST:
                raise TypeError
            result = (
                SUCCESS
                if self.VS20XX == whitelist
                else self._compoment_fail(condition, fail_level)
            )

        elif isinstance(whitelist, (list, Tuple)):
            result = (
                SUCCESS
                if self.VS20XX in whitelist
                else self._compoment_fail(condition, fail_level)
            )
        else:
            raise TypeError

        message("CHECK", self.VS20XX, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found Visual Studio is not supported (suggested) version.
                            TheRock will need Microsoft supported Visual Studio Versions to gain required MSVC compilers.
                            Your Visual Studio version {self.VS20XX} is out of whitelist.
                                > TheRock enabled Visual Studio whitelist: {whitelist}
                                > current: {self.VS20XX}
                             >>> traceback:  Found unsupported Visual Studio versions"""
            )
            message(result, reason)

        return result

    def _post_msvc_ver(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        msvc: Iterable[Literal["v140", "v141", "v142", "v143", "v145"]] = "v143",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for MSVC compiler version")
        if isinstance(msvc, str):
            if msvc not in MSVC_VERSION_LIST:
                raise ValueError(f"MSVC version {msvc} is not valid version.")
            result = (
                SUCCESS
                if VERSION_WHITELIST(self.VC_VERSION, ["v141", "v142", "v143"])
                else self._compoment_fail(condition, fail_level)
            )
        elif isinstance(msvc, Iterable):
            result = (
                SUCCESS
                if VERSION_WHITELIST(self.VC_VERSION, msvc)
                else self._compoment_fail(condition, fail_level)
            )

        message("CHECK", self.VC_VERSION, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found avail MSVC versions.
                            TheRock supports MSVC version range is {msvc}, but found {self.VC_VERSION}.
                            Please use supported MSVC compiler versions.

                             >>> traceback: Invalid MSVC version {self.VC_VERSION} out of supported whitelist"""
            )
            message(result, reason)
        return result

    def _post_msvc_host(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for MSVC compile host")
        result = (
            SUCCESS
            if STREQUAL(self.VC_HOST, "x64")
            else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", self.VC_HOST, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found MSVC compiler Host is invalid.
                            If you are x64 CPU users, Please check your VC++ compiler host. Incorrect VC++ compiler host will
                                make unstable during compile time.
                            - In 'Developer Command Prompt for VS20XX' batch terminal profile, Type '%VSCMD_ARG_HOST_ARCH%'
                                to check Value is 'x64' or not. If not, please open a correct host one.
                            - In 'Developer PowerShell for VS20XX' ps1 terminal profile, Type '$env:VSCMD_ARG_HOST_ARCH' to
                                check Value is 'x64' or not. If not, the profile might be broken. Please switch to CMD profiles
                                for direct 'Developer Command Prompt for VS20XX' batch file then re-enter powershell.

                            If you are x86 (32-bit) CPU users, Please switch to a 64-bit CPU environment computer to build
                                TheRock. TheRock only supports build on x64 native machines.

                             >>> traceback: Invalid VC++ compiler host: {self.VC_HOST}"""
            )
            message(result, reason)

        return result

    def _post_msvc_arch(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for MSVC compile target")
        result = (
            SUCCESS
            if STREQUAL(self.VC_ARCH, "x64")
            else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", self.VC_ARCH, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found Visual C++ compile Target is invalid.
                            TheRock only supports compile to x64 CPUs (AMD64/EM64T/Intel64/x86-64). Please check your VC++ Target.
                            - In 'Developer Command Prompt for VS20XX' batch terminal profile, Type '%VSCMD_ARG_TGT_ARCH%'
                                to check Value is 'x64' or not. If not, please open a correct target one.
                            - In 'Developer PowerShell for VS20XX' ps1 terminal profile, Type '$env:VSCMD_ARG_TGT_ARCH' to
                                check Value is 'x64' or not. If not, the profile might be broken. Please switch to CMD profiles
                                for direct 'Developer Command Prompt for VS20XX' batch file then re-enter powershell.

                            If you are x86 (32-bit) CPU users, Please switch to a 64-bit CPU environment computer to build
                                TheRock. TheRock only supports build on x64 native to target x64 machines.

                             >>> traceback: Invalid VC++ compile target: {self.VC_ARCH}"""
            )
            message(result, reason)

        return result

    def _post_msvc_atl(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for MSVC have ATL/MFC Library")

        result = (
            SUCCESS if self.VC_ATLMFC else self._compoment_fail(condition, fail_level)
        )

        res_disp = "Found" if self.VC_ATLMFC else NOTFOUND

        message("CHECK", res_disp, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Failes posting MSVC ATL library.
                            TheRock requires your MSVC toolchain have Active Template Library (ATL) installed.
                            Please use Visual Studio Installer to reconfigure ATL options is checked.

                             >>> traceback: Missing Required MSVC ATL"""
            )
            message(result, reason)
        return result

    def _post_msvc_cl(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for MSVC compiler driver program cl.exe")

        cl = self.cl

        c1 = Path(cl.parent / "c1.dll").resolve()
        c1xx = Path(cl.parent / "c1xx.dll").resolve()
        c2 = Path(cl.parent / "c2.dll").resolve()
        c2dd = Path(cl.parent / "c2dd.dll").resolve()

        if any(not path.exists() for path in [cl, c1, c1xx, c2, c2dd]):
            result = self._compoment_fail(condition, fail_level)
        else:
            result = SUCCESS

        message("CHECK", "", check_result=result)
        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found MSVC compiler toolchain is incomplete.
                            The MSVC compiler toolchain contains with: {{
                                MSVC compiler driver (cl.exe):                  {cl.as_posix()}
                                MSVC Compiler C language frontend (c1.dll):     {c1.as_posix()}
                                MSVC Compiler C++ language frontend (c1xx.dll): {c1xx.as_posix()}
                                MSVC Compiler Backend (c2.dll):                 {c2.as_posix()}
                                MSVC Compiler Backend (c2dd.dll):               {c2dd.as_posix()}
                            }}
                             >>> traceback: Found Incomplete/Broken Install MSVC"""
            )
            message(result, reason)

        return result

    def _post_msvc_ml64(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Microsoft Macro Assembler program ml64.exe")

        ml64 = (self.cl.parent / "ml64.exe").resolve()
        ml = (self.cl.parent / "ml64.exe").resolve()

        result = (
            SUCCESS if ml64.exists() else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            if ml.exists() and not ml64.exists():
                reason = dedent(
                    f"""\
                                Found invalid Microsoft Macro assembler.
                                TheRock is finding 'ml64.exe' (64-bit version), but found 'ml.exe' (32-bit version).
                                Please configure your MSVC installation is broken.
                                 >>> traceback: Found unexpected Microsoft Macro assembler"""
                )
            elif ml64.exists():
                reason = dedent(
                    f"""\
                                Cannot find Microsoft Macro assembler.
                                TheRock is finding for MSVC's assembler 'ml64.exe'.
                                Please configure your MSVC installation is broken.
                                 >>> traceback: No available Microsoft Macro assembler"""
                )

            message(result, reason)
        return result

    def _post_msvc_lib(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for MSVC linker stub lib.exe")

        lib = self.lib
        result = (
            SUCCESS if lib.exists() else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found MSVC static linker (archiver) 'lib.exe'.
                            If this program is not with cl.exe, It probably be broken MSVC install.
                            Please re-configure MSVC installation.
                             >>> traceback: Cannot find MSVC linker tool 'lib.exe'"""
            )
            message(result, reason)
        return result

    def _post_msvc_link(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for MSVC linker stub link.exe")

        link = self.link
        result = (
            SUCCESS if link.exists() else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found MSVC Incremental linker 'link.exe'.
                            If this program is not with cl.exe, It probably be broken MSVC install.
                            Please re-configure MSVC installation.
                             >>> traceback: Cannot find MSVC linker tool 'link.exe'"""
            )
            message(result, reason)
        return result

    def _post_ucrt(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):
        message("STATUS", "Check for MSVC have Windows SDK/Universal CRT")
        result = (
            SUCCESS
            if self.UCRT_VERSION
            else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", self.UCRT_VERSION, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found available Windows SDK and its Universal C Runtime Library (CRT).
                            MSVC compiler toolset requires UCRT for UCRT runtime libraries, also Resource compiler,
                            Manifest tool are included in UCRT. Please check your UCRT installation.

                             >>> traceback: No available UCRT library version found"""
            )
            message(result, reason)
        return result

    def _post_ucrt_mt(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Windows SDK/UCRT Manifest tool mt.exe")

        mt = self.rc.parent / "mt.exe"
        result = SUCCESS if mt.exists() else self._compoment_fail(condition, fail_level)
        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found Windows SDK/UCRT Manifest tool 'mt.exe'.
                            If this program is not in Windows SDK/UCRT, the compilation with MSVC toolchain in cmake
                                build process will be broken.
                            Please re-configure Windows SDK/UCRT installation via VS20XX Community/Build Tools.
                             >>> traceback: Cannot found Windows SDK/UCRT Manifest tool mt.exe"""
            )
            message(result, reason)
        return result

    def _post_ucrt_rc(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["WARNING", "ERROR"] = "ERROR",
    ):

        message("STATUS", "Check for Windows SDK/UCRT Resource compiler rc.exe")

        rc = self.rc

        result = SUCCESS if rc.exists() else self._compoment_fail(condition, fail_level)
        message("CHECK", "", check_result=result)
        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot found Windows SDK/UCRT Resource Compiler 'rc.exe'.
                            If this program is not in Windows SDK/UCRT, the compilation with MSVC toolchain in cmake
                                build process will be broken.
                            Please re-configure Windows SDK/UCRT installation via VS20XX Community/Build Tools.
                             >>> traceback: Cannot found Windows SDK/UCRT Resource Compiler rc.exe"""
            )
            message(result, reason)
        return result

    def _post_dumpbin(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["WARNING", "ERROR"] = "WARNING",
    ):
        message("STATUS", "Check for MSVC toolchain dumpbin.exe")
        dumpbin: Path = self.info["dumpbin.exe"]

        result = (
            SUCCESS if dumpbin.exists() else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", "", check_result=result)
        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Cannot find MSVC toolchain 'dumpbin.exe'.
                            If you are following TheRock dev's guidance tracing back with broken binaries,
                                you need dumpbin program to extract exe/DLLs information. Since this tool
                                is included with MSVC, it means your MSVC toolchain is broken.
                            Please re-configure your MSVC installation.
                             >>> traceback: Required Diagnose tool dumpbin.exe is missing"""
            )
            message(result, reason)
        return result

    #   ===============================================================
    @override
    def __LINUX__(self):
        message("FATAL_ERROR", f"MSVC is not supported on native Linux and WINE.")

    @override
    def __BSD__(self):
        message("FATAL_ERROR", f"MSVC is not supported on BSD systems.")
