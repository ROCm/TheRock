from typing import Literal, Any, overload
from pathlib import Path
from textwrap import dedent


from utils import *
from utils.compare_functions import *
from utils.status import *
from utils.status import _NotDefinedType, _NotFoundType
from . import FindSDK


class FindCMake(FindSDK):

    cmake = "cmake"

    def __init__(self):
        super().__init__()

        self.info: dict[Literal["exe", "Version", "msvc"]]

        self._dispatch_map = {
            "AVAIL": self._post_cmake_avail,
            "VS_CMAKE": self._post_cmake_vsdeps,
            "COMPILER_VAR": self._post_compiler_envvar,
        }

    @property
    def exe(self) -> Path | _NotFoundType:
        return self.info["exe"]

    @property
    def Version(self) -> VersionNum | _NotDefinedType:
        return self.info["Version"]

    @property
    def msvc(self):
        return self.info["msvc"]

    def __WINDOWS__(self):
        return self._cmake_info()

    def __LINUX__(self):
        return self._cmake_info()

    def _cmake_info(self):

        exe = self._find_program(self.cmake)

        if exe is NOTFOUND:
            ver = NOTDEFINED
            msvc = False

        elif (
            "/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/" in exe.as_posix()
        ):
            ver = self._find_version(
                exe, pattern=r"\d+\.\d+(?:\.\d+)*[-\w]*", args="--version"
            )
            msvc = True
        else:
            ver = self._find_version(
                exe, pattern=r"\d+\.\d+(?:\.\d+)*", args="--version"
            )
            msvc = False

        return {
            "exe": exe,
            "Version": ver,
            "msvc": msvc,
            "CC": ENV("CC"),
            "CXX": ENV("CXX"),
            "FC": ENV("FC"),
            "CMAKE_C_COMPILER": ENV("CMAKE_C_COMPILER"),
            "CMAKE_CXX_COMPILER": ENV("CMAKE_CXX_COMPILER"),
            "CMAKE_Fortran_COMPILER": ENV("CMAKE_Fortran_COMPILER"),
        }

    def _cmake_minimum_required_from_CMakeLists(self) -> str:

        from pathlib import Path
        import __main__

        if hasattr(__main__, "__file__"):
            Top_Level_CMakeLists_txt = (gitrepo / "CMakeLists.txt").resolve()
        else:
            message("FATAL_ERROR", "Please do not run diagnose in REPL or Jupyter NB!")

        if Top_Level_CMakeLists_txt.exists():
            pattern = re.compile(
                r"cmake_minimum_required\s*\(\s*VERSION\s+([\d\.]+)", re.IGNORECASE
            )
            with open(Top_Level_CMakeLists_txt, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        return match.group(1)
        else:
            message(
                "FATAL_ERROR", "TheRock Top-Level CMakeLists.txt is Missing !!!!!!!!!"
            )

    @overload
    def post(
        self,
        kwd: Literal["AVAIL"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        cmake_minimum_required: str = None,
        cmake_policy_max: str = None,
        cmake4: bool = False,
        *,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["VS_CMAKE"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        /,
        path: Union[Path, str] = None,
        *,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "HINT",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["COMPILER_VAR"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        lang: Literal["C", "C++", "Fortran"],
        mode: Literal["REQUIRES", "NOTDEFINED"] = "NOTDEFINED",
        expected_compiler: Union[Path, str] = None,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd.upper(), condition=condition, **kwargs)

    def _post_cmake_avail(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        cmake_minimum_required: str = None,
        cmake_policy_max: str = None,
        cmake4: bool = False,
        blacklist: Iterable[str | VersionNum] = None,
        fuzzy: bool = False,
        fail_level: Literal["FATAL", "WARNING", "HINT"] = "FATAL",
    ):

        message("STATUS", "Check for CMake program available")

        min_v = (
            VersionNum(self._cmake_minimum_required_from_CMakeLists())
            if cmake_minimum_required is None
            else VersionNum(cmake_minimum_required)
        )

        if not cmake4 and cmake_policy_max is None:
            cmake_policy_max = "3.31.11"

        if not self.exe:
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            TheRock cannot find CMake program.
                            TheRock is a CMake super-project, using CMake to build roc/hip-X Libraries, AMD-LLVM and HIPCC compiler etc.
                            AMD ROCm library tools/libraries/compoments.

                            Please install a available CMake program, at least with minimum required version {min}. The latest CMake
                                support please follow current issues to allow your build with CMake <= 3.31.X or CMake 4.

                             >>> traceback: No available CMake program found"""
            )
        elif VERSION(self.Version, "<=", min, blacklist=blacklist, fuzzy=fuzzy):
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            TheRock found invalid cmake versions.
                            TheRock requires CMake program version range from {cmake_minimum_required} to {cmake_policy_max}, but
                                found you're CMake is version {self.Version}.
                            Please select your CMake program by adjusting CMake programs' PATH priority.

                             >>> traceback: Found invalid CMake program {self.Version} not in supported versions: ({min_v}, {cmake_policy_max})"""
            )
        elif not cmake4 and VERSION(self.Version, ">=", "4.0.0"):
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            TheRock found CMake program is CMake 4. (Current version: {self.Version})
                            TheRock enables build with CMake 4, but will hit CMake configure error with 3rd-party libraries. Those
                                libraries may not upgrade their CMake versions cause CMake behavior/policy have conflict/ or been
                                deprecated.
                            Please downgrade your CMake versions that less than 4.0.0.

                             >>> traceback: CMake required version below 3.31.X, but found CMake 4 with {self.Version}"""
            )
        elif cmake_policy_max is not None and VERSION(
            self.Version, ">=", cmake_policy_max
        ):
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            TheRock found invalid cmake versions.
                            TheRock requires CMake program version range from {cmake_minimum_required} to {cmake_policy_max}, but
                                found you're CMake is version {self.Version}.
                            Please select your CMake program by adjusting CMake programs' PATH priority.

                             >>> traceback: Found invalid CMake program {self.Version} not in supported versions: ({cmake_minimum_required}, {cmake_policy_max})"""
            )
        elif blacklist is not None and not VERSION_BLACKLIST(
            self.Version, blacklist, fuzzy=fuzzy
        ):
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            TheRock found invalid cmake versions.
                            TheRock requires CMake program version range from {cmake_minimum_required} to {cmake_policy_max}, but
                                found you're CMake is version {self.Version}.
                            Please select your CMake program by adjusting CMake programs' PATH priority.

                             >>> traceback: Found invalid CMake program {self.Version} not in supported versions: ({cmake_minimum_required}, {cmake_policy_max})"""
            )
        else:
            result = SUCCESS

        message("CHECK", self.Version, check_result=result)

        if result is not SUCCESS:
            message(result, reason)
        return result

    def _post_cmake_vsdeps(
        self,
        *,
        entry: Union[Path, str],
        condition: Literal["REQUIRED", "OPTIONAL"] = "OPTIONAL",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "HINT",
    ):
        message("STATUS", "Checking CMake program is from Visual Studio")

        if isinstance(entry, (Path, str)):
            pth = Path(entry).resolve().as_posix()
        else:
            pth = NOTFOUND

        if pth is NOTFOUND:
            reason = dedent(
                f"""\
                            Found CMake program is not from Visual Studio.
                            This CMake test is examine to find cmake from Visual Studio, but your Visual Studio didn't install it.

                             >>> traceback: Found Visual Studio have not CMake installed"""
            )
            result = self._compoment_fail(condition, fail_level)

        if self.exe.resolve().as_posix() == pth:
            result = SUCCESS
        else:
            result = self._compoment_fail(condition, fail_level)
            reason = dedent(
                f"""\
                            Found CMake program is not Visual Studio version.
                            This Diagnose system have enabled testing cmake program from Visual Studio, but your callable cmake program
                                is prior than the Visual Studio integrated one. If you want use Visual Studio one, Please consider adjust
                            PATH variable or remove this cmake from PATH.
                                > Visual Studio installed CMake:    {pth}
                                > Your cmake (top one):             {self.exe}

                             >>> traceback: Callable cmake is not from Visual Studio installed version"""
            )
            message(result, reason)

        message("CHECK", "", check_result=result)

        if result is not SUCCESS:
            message(result, reason)

        return result

    def _post_compiler_envvar(
        self,
        *,
        lang: Literal["C", "C++", "Fortran"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        mode: Literal["REQUIRES", "NOTDEFINED"] = "NOTDEFINED",
        expected_compiler: Union[Path, str] = None,
        fail_level: Literal["HINT", "WARNING", "ERROR"] = "ERROR",
    ):

        if lang == "C":
            _lang_set = ("CC", "CMAKE_C_COMPILER")
            _compiler = self.info["CC"]
            _cmake_lang_compiler = self.info["CMAKE_C_COMPILER"]
        elif lang == "C++":
            _lang_set = ("CXX", "CMAKE_CXX_COMPILER")
            _compiler = self.info["CXX"]
            _cmake_lang_compiler = self.info["CMAKE_CXX_COMPILER"]
        elif lang == "Fortran":
            _lang = "Fortran"
            _lang_set = ("FC", "CMAKE_Fortran_COMPILER")
            _compiler = self.info["FC"]
            _cmake_lang_compiler = self.info["CMAKE_Fortran_COMPILER"]
        else:
            message(
                "FATAL_ERROR",
                f"Unsupport detecting {lang} / CMAKE_{lang}_COMPILER detection",
            )

        _1, _2 = _lang_set
        message("STATUS", f"Check for defined {_1} / {_2}")

        # CMAKE_<LANG>_COMPILER priority is higher than CC/CXX/FC
        if _compiler and _cmake_lang_compiler:
            _compiler_set = True
            _compiler_override = True if _compiler != _cmake_lang_compiler else False
        elif _compiler and not _cmake_lang_compiler:
            _compiler_set = True
            _cmake_lang_compiler = _compiler
            _compiler_override = False
        elif _cmake_lang_compiler and not _compiler:
            _compiler_set = True
            _compiler = _cmake_lang_compiler
            _compiler_override = False
        else:
            _compiler_set = False
            _compiler_override = False

        if mode == "NOTDEFINED":
            if _compiler_set:
                result = self._compoment_fail(condition, fail_level)
                reason = dedent(
                    f"""\
                                Found {_1} / {_2} is set.
                                TheRock requires CMake {_lang} language compiler default calling behavior is None, but you have already
                                    set to {_cmake_lang_compiler} with {_lang_set}. This will let CMake have incorrect {_lang} compiler
                                    during CMake configuration.
                                Please remove {_1} / {_2} Variable or set them to:
                                    {expected_compiler}

                                 >>> traceback: Found unexpected environment variable {_1} / {_2}"""
                )
            else:
                result = SUCCESS
        else:  # == "REQUIRES"
            if (
                _cmake_lang_compiler is None
                or _cmake_lang_compiler is NOTDEFINED
                or _cmake_lang_compiler is NOTFOUND
            ):
                result = WARNING
                reason = dedent(
                    f"""\
                                Cannot found specified {lang} compiler to cmake.
                                TheRock requires Environment Variable {_2} with expected compiler's path.
                                Please set it either with {_2} or {_1}, or define {_2} with cmake configure command.
                                 > expected {_2} value: {expected_compiler}
                                 > current: {_cmake_lang_compiler}

                                 >>> traceback: Expected {_2}/{_1} is not set"""
                )
            elif (
                Path(_cmake_lang_compiler).exists()
                and _cmake_lang_compiler == expected_compiler
            ):
                result = SUCCESS
            else:
                result = self._compoment_fail(condition, fail_level)
                reason = dedent(
                    f"""\
                                Found {_1} / {_2} is invalid.
                                TheRock have expected CMake {_lang} language compiler is {expected_compiler},
                                    but yours is {_cmake_lang_compiler} with {_1} and/or {_2} exported.
                                Please make sure {_1} / {_2} value is set to {expected_compiler}.

                                 >>> traceback: Found unexpected {_1} / {_2} value {_compiler}"""
                )

        message("CHECK", "Yes" if _compiler_set else "No", check_result=result)
        if _compiler_override:
            reason = dedent(
                f"""\
                            Found multiple defined compiler related environment variable.
                            CMake Compiler variable {_1} and {_2} has been defined with different values respectively.
                            Environment variable {_2} will override {_1} with value {_cmake_lang_compiler}.
                             >>> traceback: Found multiple defined {_lang} compiler specified environment variable"""
            )
            message("WARNING", reason)

        if result is not SUCCESS:
            message(result, reason)

        return result
