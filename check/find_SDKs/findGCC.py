import subprocess as subproc
from typing import Literal, Any, Union, overload
from textwrap import dedent

from . import FindSDK
from utils import *
from utils.compare_functions import *


GCC_TARGET_TRIPLE_LIST = [
    "x86_64-linux-gnu",
    "x86_64-redhat-linux",
    "x86_64-w64-mingw32",
    "i686-w64-mingw32",
    "arm-linux-gnueabi",
    "aarch64-linux-gnu",
    "riscv64-linux-gnu",
    "riscv32-linux-gnu",
    "mips64-linux-gnuabi64",
    "mips-linux-gnu",
    "powerpc64-linux-gnu",
    "powerpc-linux-gnu",
    "sparc64-linux-gnu",
    "ia64-linux-gnu",
]

GCC_TARGET_TRIPLE_OPTION = Literal[*GCC_TARGET_TRIPLE_LIST]


def _define_gcc_triple(gcc: str | Path):
    if isinstance(gcc, Path):
        gcc = gcc.as_posix()
    if gcc is NOTFOUND or gcc is NOTDEFINED:
        return NOTDEFINED
    query = subproc.run(
        [gcc, "-dumpmachine"], capture_output=True, text=True
    ).stdout.strip()

    return query if query in GCC_TARGET_TRIPLE_LIST else NOTDEFINED


class FindGCC(FindSDK):
    name = "gcc"

    def __init__(self):
        super().__init__()

        self._dispatch_map = {
            "AVAIL": self._post_gcc_avail,
            "TARGET": self._post_gcc_arch,
        }

    @property
    def target(self):
        return self.info["target"]

    def __WINDOWS__(self):
        return self._find_gcc()

    def __LINUX__(self):
        return self._find_gcc()

    def __BSD__(self): ...

    def _find_gcc(self):
        exe = self._find_program("gcc")
        ver = self._find_version(exe)
        arch = _define_gcc_triple(exe)

        return {
            "exe": exe,
            "Version": ver,
            "target": arch,
        }

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        version_require: Literal["IN_RANGE", "COMPARE", "ONLY", None] = None,
        version: Union[str, VersionNum] = None,
        op: Literal["<", "<="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        >>> gcc = FindGCC()
        >>> gcc.post("AVAIL", "REQUIRED", fail_level="ERROR")
        ```
        """

    @overload
    def post(
        self,
        kwd: Literal["TARGET"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        >>> gcc = FindGCC()
        >>> gcc.post("TARGET", "REQUIRED", target=...)
        ```
        """

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    #  ==================================================
    def _post_gcc_avail(
        self,
        *,
        version_require: Literal["IN_RANGE", "COMPARE", "ONLY"],
        version: Iterable[str | VersionNum] = None,
        op: Literal["<", "<=", ">", ">="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for GCC compiler gcc available")

        if self.exe:
            if version_require == "IN_RANGE":
                if op not in ("<", "<=") or op2 is None:
                    err_use = dedent(
                        f"""Illegal comnpare operator usage.
                                        Expected In Range compare with: {version} </<= {self.Version} </<= {version_max}    --> VERSION_IN_RANGE(...)
                                        Your Entry:                     {version} {op} {self.Version} {op2} {version_max}   """
                    )
                    message("FATAL_ERROR", err_use)

                check = VERSION_IN_RANGE(
                    version,
                    op,
                    self.Version,
                    op2,
                    version_max,
                    blacklist=blacklist,
                    fuzzy=fuzzy,
                )
            elif version_require == "COMPARE":
                check = VERSION(
                    self.Version, op, version, blacklist=blacklist, fuzzy=fuzzy
                )
            else:
                message(
                    "FATAL_ERROR", f"Invalid gcc version exam mode {version_require}."
                )

            result = SUCCESS if check else self._compoment_fail(condition, fail_level)
            message("CHECK", self.Version, check_result=result)
            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                TheRock found `gcc` program with invalid version: {self.Version}.
                                TheRock build pre-diagnose system have set the version rules '{version_require}' with parameters:
                                 > version conpare:    {version} {op} {self.Version} {op2} {version_max}
                                 > blacklist     {blacklist}, fuzzy = {fuzzy}

                                Please contact to ROCm/TheRock Devs for further information with your version.

                                 >>> traceback: Found GCC compiler program 'gcc' with invalid version {self.Version}"""
                )

        else:
            result = self._compoment_fail(condition, fail_level)
            message("CHECK", NOTFOUND, check_result=result)
            reason = dedent(
                f"""\
                            TheRock cannot find a GCC compiler C language frontend `gcc`.
                            TheRock requires gcc to compile ROCm AMD-LLVM. Please install it via package managers.

                             >>> traceback: GCC compiler missing C language frontend `gcc` program"""
            )

        if result is not SUCCESS:
            message(result, reason)

        return result

    def _post_gcc_arch(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for GCC compiler gcc target triple")

        result = (
            SUCCESS
            if self.target == "x86_64-linux-gnu"
            else self._compoment_fail(condition, fail_level)
        )

        message("CHECK", self.target, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found Invalid GCC target triple `{self.target}`.
                            TheRock only avail target on x64 target: "AMD64", "x64", "EMT64/Intel 64", "x86_64".
                            Please contact to TheRock dev team for '{self.target}' target development.

                             >>> traceback: Found unsupported gcc target triple `{self.target}`"""
            )
            message(result, reason)
        return result


class FindGXX(FindSDK):
    name = "g++"

    def __init__(self):
        super().__init__()

        self.info: dict[Literal["exe", "Version", "target"]]

        self._dispatch_map = {
            "AVAIL": self._post_gxx_avail,
            "TARGET": self._post_gxx_arch,
        }

    @property
    def target(self):
        return self.info["target"]

    def __WINDOWS__(self):
        return self._find_gxx()

    def __LINUX__(self):
        return self._find_gxx()

    def __BSD__(self): ...

    def _find_gxx(self):
        gxx = self._find_program("g++")
        ver = self._find_version(gxx)
        arch = _define_gcc_triple(gxx)

        return {"exe": gxx, "Version": ver, "target": arch}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        mode: Literal["IN_RANGE", "COMPARE", "ONLY", None] = None,
        version: Iterable[str | VersionNum] = None,
        op: Literal["<", "<="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        >>> gxx = FindGXX()
        >>> gxx.post("AVAIL", "REQUIRED", fail_level="ERROR")
        ```
        """

    @overload
    def post(
        self,
        kwd: Literal["TARGET"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        >>> gxx = FindGXX()
        >>> gxx.post("TARGET", "REQUIRED", target=...)
        ```
        """

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    #  ==================================================
    def _post_gxx_avail(
        self,
        *,
        version_require: Literal["IN_RANGE", "COMPARE", "ONLY"],
        version: Union[str, VersionNum] = None,
        op: Literal["<", "<=", ">", ">="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for GCC compiler g++ available")

        if self.exe:
            if version_require == "IN_RANGE":
                if op not in ("<", "<=") or op2 is None:
                    err_msg = dedent(
                        f"""\
                                     Illegal comnpare operator usage.
                                     Expected In Range compare with: {version} </<= {self.Version} </<= {version_max}    --> VERSION_IN_RANGE(...)
                                     Your Entry:                     {version} {op} {self.Version} {op2} {version_max}   """
                    )
                    message("FATAL_ERROR", err_msg)

                check = VERSION_IN_RANGE(
                    version,
                    op,
                    self.Version,
                    op2,
                    version_max,
                    blacklist=blacklist,
                    fuzzy=fuzzy,
                )
            elif version_require == "COMPARE":
                check = VERSION(
                    self.Version, op, version, blacklist=blacklist, fuzzy=fuzzy
                )
            else:
                message(
                    "FATAL_ERROR", f"Invalid g++ version exam mode {version_require}."
                )

            result = SUCCESS if check else self._compoment_fail(condition, fail_level)

            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                TheRock found `g++` program with invalid version: {self.Version}.
                                TheRock build pre-diagnose system have set the version rules '{version_require}' with parameters:
                                 > version conpare:    {version} {op} {self.Version} {op2} {version_max}
                                 > blacklist     {blacklist}, fuzzy = {fuzzy}

                                Please contact to ROCm/TheRock Devs for further information with your version.

                                 >>> traceback: Found GCC compiler program 'gcc' with invalid version {self.Version}"""
                )

        else:
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            TheRock cannot find a GCC compiler C++ language frontend `g++`.
                            TheRock requires g++ to compile ROCm AMD-LLVM. Please install it via package managers.

                             >>> traceback: GCC compiler missing C++ language frontend `g++` program"""
            )

        message("CHECK", self.Version, check_result=result)
        if result is not SUCCESS:
            message(result, reason)

        return result

    def _post_gxx_arch(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for GCC compiler g++ target triple")

        result = (
            SUCCESS
            if self.target == "x86_64-linux-gnu"
            else self._compoment_fail(condition, fail_level)
        )
        message("CHECK", self.target, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found Invalid g++ target triple `{self.target}`.
                            TheRock only avail target on x64 target: "AMD64", "x64", "EMT64/Intel 64", "x86_64".
                            Please contact to TheRock dev team for '{self.target}' target development.

                             >>> traceback: Found unsupported g++ target triple `{self.target}`"""
            )
            message(result, reason)
        return result


class FindGFortran(FindSDK):
    name: Literal["gfortran"] = "gfortran"

    def __init__(self):
        super().__init__()

        self._dispatch_map = {
            "AVAIL": self._post_gfortran_avail,
            "TARGET": self._post_gfortran_arch,
        }

    @property
    def target(self):
        return self.info["target"]

    def __WINDOWS__(self):
        return self._find_gfortran()

    def __LINUX__(self):
        return self._find_gfortran()

    def _find_gfortran(self):
        exe = self._find_program("gfortran")
        ver = self._find_version(exe, args="--version", vertemp="X.Y.Z")
        arch = _define_gcc_triple(exe)

        return {"exe": exe, "Version": ver, "target": arch}

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        mode: Literal["IN_RANGE", "COMPARE", "ONLY", None] = None,
        version: Union[str, VersionNum] = None,
        op: Literal["<", "<="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        >>> gfortran = FindGFortran()
        >>> gfortran.post("AVAIL", "REQUIRED", fail_level="ERROR")
        ```
        """

    @overload
    def post(
        self,
        kwd: Literal["TARGET"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):
        """
        >>> gfortran = FindGFortran()
        >>> gfortran.post("TARGET", "REQUIRED", target=...)
        ```
        """

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    #  ==================================================
    def _post_gfortran_avail(
        self,
        *,
        version_require: Literal["IN_RANGE", "COMPARE", "ONLY", None] = None,
        version: Union[str, VersionNum] = None,
        op: Literal["<", "<=", ">", ">="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        message("STATUS", "Check for GCC compiler gfortran available")

        if self.exe:
            if version_require is None:
                check = True
            elif version_require == "IN_RANGE":
                if op not in ("<", "<=") or op2 is None:
                    err_msg = dedent(
                        f"""\
                                     Illegal comnpare operator usage.
                                     Expected In Range compare with: {version} </<= {self.Version} </<= {version_max}    --> VERSION_IN_RANGE(...)
                                     Your Entry:                     {version} {op} {self.Version} {op2} {version_max}   """
                    )
                    message("FATAL_ERROR", err_msg)

                check = VERSION_IN_RANGE(
                    version,
                    op,
                    self.Version,
                    op2,
                    version_max,
                    blacklist=blacklist,
                    fuzzy=fuzzy,
                )
            elif version_require == "COMPARE":
                check = VERSION(
                    self.Version, op, version, blacklist=blacklist, fuzzy=fuzzy
                )
            else:
                message(
                    "FATAL_ERROR",
                    f"Invalid gfortran version exam mode {version_require}.",
                )

            result = SUCCESS if check else self._compoment_fail(condition, fail_level)

            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                TheRock found `gfortran` program with invalid version: {self.Version}.
                                TheRock build pre-diagnose system have set the version rules '{version_require}' with parameters:
                                 > version conpare:    {version} {op} {self.Version} {op2} {version_max}
                                 > blacklist     {blacklist}, fuzzy = {fuzzy}

                                Please contact to ROCm/TheRock Devs for further information with your version.

                                 >>> traceback: Found GCC compiler program 'gcc' with invalid version {self.Version}"""
                )
        else:
            result = FATAL
            reason = dedent(
                f"""\
                            TheRock cannot find a GCC compiler Fortran language frontend `gfortran`.
                            TheRock requires gfortran to compile ROCm libraries.
                             - Windows:    Please install via Strawberry/MinGW-w64.
                             - Linux:      Please install it via package managers.

                             >>> traceback: GCC compiler missing Fortran language frontend `gfortran` program
                            """
            )

        message("CHECK", self.Version, check_result=result)
        if result is not SUCCESS:
            message(result, reason)

        return result

    def _post_gfortran_arch(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):
        print(" -- Check for gfortran compiler triple:      ", end="", flush=True)

        if self.os == "Linux" and self.target == "x86_64-linux-gnu":
            result = SUCCESS
        elif self.os == "Windows" and self.target == "x86_64-w64-mingw32":
            result = SUCCESS
        else:
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            Found Invalid gfortran target triple `{self.target}`.
                            TheRock only avail target on x64 target: "AMD64", "x64", "EMT64/Intel 64", "x86_64".
                            Please contact to TheRock dev team for '{self.target}' target development.

                             >>> traceback: Found unsupported gfortran target triple `{self.target}`"""
            )

        message("CHECK", self.target, check_result=result)
        if result is not SUCCESS:
            print(cstring(reason, fail_level))

        return result


class FindGCC_AR(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_gccar_avail}

    def __WINDOWS__(self):
        return self._find_gcc_ar()

    def __LINUX__(self):
        return self._find_gcc_ar()

    def _find_gcc_ar(self):
        exe = self._find_program("gcc-ar")
        ver = self._find_version(exe, args="--version", vertemp="X.Y")

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
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_gccar_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):

        message("STATUS", "Check for GCC compiler gcc-ar")
        result = SUCCESS if self.exe else self._compoment_fail(condition, fail_level)

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Failed to find GCC compiler program gcc-ar.
                            If TheRock enables Linking-Time-Optimization flag `-flto`, we needs gcc-ar program.
                            But we seems gcc-ar is not found. Please check your GCC toolchain installation.

                             >>> traceback: GCC compiler program gcc-ar is missing"""
            )
            message(result, reason)

        return result


class FindGLIBC(FindSDK):
    def __init__(self):
        super().__init__()

        self._dispatch_map = {"AVAIL": self._post_glibc_avail}

    def __WINDOWS__(self):
        message("FATAL_ERROR", "GLIBC is not supported on Windows.")

    def __LINUX__(self):
        exe = self._find_program("ldd")
        ver = self._find_version(exe)

        return {"exe": exe, "Version": ver}

    @overload
    def post(
        self,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
        /,
        *,
        version_require: Literal["IN_RANGE", "COMPARE", "ONLY"],
        version: Union[str, VersionNum] = None,
        op: Literal["<", "<=", ">", ">="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_glibc_avail(
        self,
        *,
        version_require: Literal["IN_RANGE", "COMPARE", "ONLY"] = None,
        version: Union[str, VersionNum] = None,
        op: Literal["<", "<=", ">", ">="] = None,
        op2: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):

        message("STATUS", "Check for glibc version with `ldd` program")

        if self.exe:
            if version_require == "IN_RANGE":
                if op not in ("<", "<=") or op2 is None:
                    err_msg = dedent(
                        f"""\
                                     Illegal comnpare operator usage.
                                     Expected In Range compare with: {version} </<= {self.Version} </<= {version_max}    --> VERSION_IN_RANGE(...)
                                     Your Entry:                     {version} {op} {self.Version} {op2} {version_max}   """,
                    )
                    message("FATAL_ERROR", err_msg)

                check = VERSION_IN_RANGE(
                    version,
                    op,
                    self.Version,
                    op2,
                    version_max,
                    blacklist=blacklist,
                    fuzzy=fuzzy,
                )
            elif version_require == "COMPARE":
                check = VERSION(
                    self.Version, op, version, blacklist=blacklist, fuzzy=fuzzy
                )
            else:
                message(
                    "FATAL_ERROR", f"Invalid glibc version exam mode {version_require}."
                )

            result = SUCCESS if check else self._compoment_fail(condition, fail_level)
            if result is not SUCCESS:
                reason = dedent(
                    f"""\
                                TheRock found glibc with invalid version: {self.Version}.
                                TheRock build pre-diagnose system have set the version rules '{version_require}' with parameters:
                                 > version conpare:    {version} {op} {self.Version} {op2} {version_max}
                                 > blacklist     {blacklist}, fuzzy = {fuzzy}

                                Please contact to ROCm/TheRock Devs for further information with your version.

                                 >>> traceback: Found GCC compiler program 'gcc' with invalid version {self.Version}"""
                )
        else:
            result = self._compoment_fail(condition, fail_level)
        reason = dedent(
            f"""\
                        TheRock cannot find required glibc.
                        TheRock requires glibc to compile AMD-LLVM and other ROCm libraries.
                        Please install it via package managers.

                         >>> traceback: Missing glibc installation"""
        )

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            message(result, reason)

        return result
