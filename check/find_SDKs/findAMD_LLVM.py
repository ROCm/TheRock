import subprocess
from typing import Literal, Any, overload, Union, TypedDict, Any
from pathlib import Path
from textwrap import dedent

from utils import *
from . import (
    FindSDK,
    AMDLLVM_TYPEHINT,
    AMD_LLVM_TOOLS,
    AMD_LLVM_COMPOMENTS_TYPEHINT,
    AMD_LLVM_CONFIGURE_TYPEHINT,
)


class FindAMDLLVM(FindSDK):
    def __init__(self):
        super().__init__()

        self.info: AMDLLVM_TYPEHINT

        self._dispatch_map = {"LLVM_DISTRIBUTION": self._post_llvm_distribution}

    @property
    def exe(self) -> Path:
        return self.LLVM_CONFIG["LLVM DIR"]

    @property
    def Version(self):
        return self.LLVM_CONFIG["LLVM Version"]

    @property
    def LLVM_CONFIG(self) -> dict[AMD_LLVM_CONFIGURE_TYPEHINT, Union[str, Path]]:
        return self.info["LLVM Configuration"]

    @property
    def LLVM_COMPILER(self) -> dict[AMD_LLVM_COMPOMENTS_TYPEHINT, Path]:
        return self.info["LLVM Toolchain"]

    def __repr__(self):
        return f"LLVM Version {self.Version} at {self.exe}"

    def __stat__(self):

        llvm_config = self.LLVM_CONFIG
        llvm_compiler = self.LLVM_COMPILER
        llvm_dir = self.exe  # (LLVM_DIR)
        llvm_version = self.Version

        hipcc_version = self.LLVM_CONFIG["HIP Version"]

        if not config.DISPLAY_LLVM:
            return ""
        else:

            if isinstance(llvm_config["LLVM Targets"], list):
                # No LLVM Targets list avail;
                llvm_targets_fmt = "     ".join(i for i in llvm_config["LLVM Targets"])
            else:
                llvm_targets_fmt = llvm_config["LLVM Targets"].strip()

            content = f"""\

    LLVM Configuration
        LLVM Configured:      {llvm_config['LLVM Configured']}
        LLVM_DIR:             {llvm_dir.as_posix()}
        LLVM dist type:       {llvm_config['LLVM Dist']}
        LLVM_VERSION:         {llvm_version}
        HIP Version:          {hipcc_version}
        AMD HIP SDK version:  {llvm_config['AMD HIP SDK version']}
        llvm-host-triple:     {llvm_config["LLVM Host Triple"].strip()}
        llvm-targets:         {llvm_targets_fmt}
        llvm-enable-RTTI:     {llvm_config['LLVM Enabled RTTI']}
        llvm-bin-dir:         {llvm_config["LLVM Binary Dir"].as_posix()}
        llvm-inc-dir:         {llvm_config["LLVM Include Dir"].as_posix()}
        llvm-lib-dir:         {llvm_config["LLVM Library Dir"].as_posix()}
        llvm-cmake-dir:       {llvm_config["LLVM CMake dir"].as_posix()}
        llvm-ld-library-dir:  {llvm_config["LLVM LD Library Dir"].as_posix()}
"""

        for flags in ("LLVM cflags", "LLVM cxxflags", "LLVM cppflags", "LLVM ldflags"):
            llvm_flags = llvm_config.get(flags, "")

            flags_list = [f for f in str(llvm_flags).split(" ") if f.strip()]
            label_with_colon = f"{flags}:"

            if flags_list:

                content += f"        {label_with_colon:<20}{flags_list[0]}\n"

                for flag in flags_list[1:]:
                    content += f"                            {flag}\n"
            else:
                content += f"        {label_with_colon:<22}None\n"

        for toolname, exe in llvm_compiler.items():
            exe: Path
            content_add_template = f"        {toolname:<22}{exe.as_posix()}\n"
            content += content_add_template

        return content

    def __WINDOWS__(self):
        return self._find_llvm()

    def __LINUX__(self):
        return self._find_llvm()

    def __BSD__(self):
        return ...

    def _find_llvm(self):
        # Findout LLVM with llvm-config and llvm-ar as LLVM_BIN_DIR anchor.
        # if not llvm-config.exe then use llvm-ar
        # phony_llvm_bin_dir:
        #   >   if      config.ROCM_HOME is set, use this one
        #   >   else                             use callable llvm compoemnts

        if config.ROCM_HOME == "build_dist":
            _phony_rocm_home = Path(gitrepo, "build/dist/rocm").resolve()
        elif config.ROCM_HOME:
            _phony_rocm_home = Path(config.ROCM_HOME).resolve()
        else:
            _phony_rocm_home = self._find_program("llvm-ar").parent.parent

        if _phony_rocm_home is NOTDEFINED:
            _llvm_bin_dir = NOTDEFINED
            _hip_bin_dir = NOTDEFINED
        elif _phony_rocm_home.exists():
            _llvm_bin_dir = (_phony_rocm_home / "lib/llvm/bin").resolve()
            _llvm_bin_dir = (
                _llvm_bin_dir
                if _llvm_bin_dir.exists()
                else (_phony_rocm_home / "bin").resolve()
            )
            _hip_bin_dir = (_phony_rocm_home / "bin").resolve()
        else:
            _llvm_bin_dir = NOTDEFINED
            _hip_bin_dir = NOTDEFINED

        amd_llvm_toolchain: dict[AMD_LLVM_COMPOMENTS_TYPEHINT, Path] = {}

        for llvm_tool in AMD_LLVM_TOOLS:
            if llvm_tool.startswith("hip"):
                amd_llvm_toolchain[llvm_tool] = self._find_program(
                    _hip_bin_dir, llvm_tool
                )
            else:
                amd_llvm_toolchain[llvm_tool] = self._find_program(
                    _llvm_bin_dir, llvm_tool
                )

        llvm_config = amd_llvm_toolchain["llvm-config"]
        llvm_ar = amd_llvm_toolchain["llvm-ar"]
        clang = amd_llvm_toolchain["clang"]
        amdclang = amd_llvm_toolchain["amdclang"]

        if llvm_config:
            is_llvm = True
            cmd = [
                llvm_config.as_posix(),
                "--version",
                "--targets-built",
                "--host-target",
                "--has-rtti",
                "--prefix",
                "--bindir",
                "--includedir",
                "--libdir",
                "--cmakedir",
                "--cflags",
                "--cxxflags",
                "--cppflags",
                "--ldflags",
            ]
            llvm_query = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            ).stdout.splitlines()
            try:
                llvm_query_aocc = subprocess.run(
                    [llvm_config.as_posix(), "--src-root"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            except subprocess.CalledProcessError:
                llvm_query_aocc = ""

            llvm_version = VersionNum(llvm_query[0])
            llvm_targets = llvm_query[1].split()
            llvm_host_triple = llvm_query[2]
            llvm_target_triple = subprocess.run(
                [clang.as_posix(), "--print-target-triple"],
                text=True,
                check=True,
                capture_output=True,
            ).stdout
            llvm_enable_rtti = llvm_query[3]
            llvm_dir = Path(llvm_query[4])
            llvm_dir_bin = Path(llvm_query[5])
            llvm_dir_inc = Path(llvm_query[6])
            llvm_dir_lib = Path(llvm_query[7])
            llvm_dir_cmake = Path(llvm_query[8])
            llvm_cflags = llvm_query[9].replace("\\", "/")
            llvm_cxxflags = llvm_query[10].replace("\\", "/")
            llvm_cppflags = llvm_query[11].replace("\\", "/")
            llvm_ldflags = llvm_query[12].replace("\\", "/")

            llvm_ld_library_path = llvm_dir_bin if self.os else llvm_dir_lib

        elif llvm_ar:
            is_llvm = True
            llvm_query_aocc = ""
            llvm_version = self._find_version(llvm_ar)
            llvm_targets = NOTDEFINED
            llvm_host_triple = subprocess.run(
                [clang.as_posix(), "-dumpmachine"],
                text=True,
                check=True,
                capture_output=True,
            ).stdout
            llvm_target_triple = subprocess.run(
                [clang.as_posix(), "--print-target-triple"],
                text=True,
                check=True,
                capture_output=True,
            ).stdout

            llvm_enable_rtti = NOTDEFINED
            llvm_dir = Path(_llvm_bin_dir.parent)
            llvm_dir_bin = _llvm_bin_dir
            llvm_dir_inc = Path(llvm_dir / "include")
            llvm_dir_lib = Path(llvm_dir / "lib")
            llvm_dir_cmake = Path(llvm_dir / "lib/cmake/llvm/")

            llvm_cflags = NOTDEFINED
            llvm_cxxflags = NOTDEFINED
            llvm_cppflags = NOTDEFINED
            llvm_ldflags = NOTDEFINED

            llvm_ld_library_path = llvm_dir_bin if self.os else llvm_dir_lib

        else:
            is_llvm = False
            llvm_version = NOTDEFINED
            llvm_query_aocc = ""
            llvm_dir_bin = NOTDEFINED
            llvm_targets = NOTDEFINED
            llvm_host_triple = NOTDEFINED
            llvm_target_triple = NOTDEFINED
            llvm_enable_rtti = NOTDEFINED
            llvm_dir_bin = NOTFOUND
            llvm_dir = NOTFOUND
            llvm_dir_inc = NOTFOUND
            llvm_dir_lib = NOTFOUND
            llvm_dir_cmake = NOTFOUND
            llvm_cflags = NOTDEFINED
            llvm_cxxflags = NOTDEFINED
            llvm_cppflags = NOTDEFINED
            llvm_ldflags = NOTDEFINED
            llvm_ld_library_path = NOTDEFINED

        llvm_version = self._find_version(llvm_ar)

        if llvm_host_triple is NOTDEFINED:
            llvm_host_triple = (
                subprocess.run(
                    [clang.as_posix(), "-dumpmachine"],
                    text=True,
                    check=True,
                    capture_output=True,
                ).stdout
                if clang
                else llvm_host_triple
            )
        if llvm_targets is NOTDEFINED:
            llvm_targets = llvm_host_triple

        hipcc = amd_llvm_toolchain["hipcc"]

        if (
            (amdclang and llvm_config and clang)
            and (amdclang.parent == clang.parent)
            and hipcc
        ):
            llvm_type = "AMD-LLVM"
            hipcc_version = self._find_version(hipcc, args="--version", vertemp="X.Y.Z")
        elif (not amdclang and not llvm_config and clang and hipcc) and (
            clang.parent == hipcc.parent
        ):
            llvm_type = "AMD HIP SDK"
            hipcc_version = self._find_version(hipcc, args="--version", vertemp="X.Y.Z")
        elif "/VC/Tools/Llvm/" in clang.as_posix():
            hipcc = NOTFOUND
            llvm_type = "Visual Studio"
            hipcc_version = NOTDEFINED
        elif "/aocc/llvm-project/llvm" in llvm_query_aocc:
            llvm_type = "AOCC"
            hipcc_version = NOTDEFINED
        else:
            llvm_type = "3rd-Party" if llvm_ar else NOTDEFINED
            hipcc_version = NOTDEFINED

        if (self.os == "Linux" and llvm_type == "AMD HIP SDK") or (
            self.os == "Windows" and llvm_type == "AOCC"
        ):
            message(
                "FATAL_ERROR",
                f"""Detected Invalid status with {self.os} have {llvm_type}.""",
            )

        amd_hip_version = NOTDEFINED if llvm_type != "AMD HIP SDK" else hipcc_version

        return {
            "LLVM Configuration": {
                "LLVM Configured": is_llvm,
                "LLVM DIR": llvm_dir,
                "LLVM Version": llvm_version,
                "LLVM Dist": llvm_type,
                "HIP Version": hipcc_version,
                "AMD HIP SDK version": amd_hip_version,
                "LLVM Host Triple": llvm_host_triple,
                "LLVM Target Triple": llvm_target_triple,
                "LLVM Targets": llvm_targets,
                "LLVM Enabled RTTI": llvm_enable_rtti,
                "LLVM Binary Dir": llvm_dir_bin,
                "LLVM Include Dir": llvm_dir_inc,
                "LLVM Library Dir": llvm_dir_lib,
                "LLVM LD Library Dir": llvm_ld_library_path,
                "LLVM CMake dir": llvm_dir_cmake,
                "LLVM cflags": llvm_cflags,
                "LLVM cxxflags": llvm_cxxflags,
                "LLVM cppflags": llvm_cppflags,
                "LLVM ldflags": llvm_ldflags,
            },
            "LLVM Toolchain": amd_llvm_toolchain,
        }

    @overload
    def post(
        self,
        kwd: Literal["LLVM_DISTRIBUTION"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        configure: Literal[
            "DISABLE", "AMD-LLVM", "HIP_SDK", "VS20XX", "AOCC", "3rd-Party"
        ] = "AMD-LLVM",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    @overload
    def post(
        self,
        kwd: Literal["LLVM_VERSION"],
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        /,
        *,
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ): ...

    def post(self, kwd: str, condition: str, **kwargs: Any):
        return super().post(kwd, condition=condition, **kwargs)

    def _post_llvm_distribution(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        configure: Literal[
            "DISABLE", "AMD-LLVM", "HIP_SDK", "AOCC", "VS20XX", "3rd-Party"
        ] = "AMD-LLVM",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
    ):

        message("STATUS", "Check for LLVM compiler distribution")

        is_llvm = self.LLVM_CONFIG["LLVM Configured"]
        llvm_dir = self.exe
        llvm_type = self.LLVM_CONFIG["LLVM Dist"]
        clang = self.LLVM_COMPILER["clang"]
        clang_cl = self.LLVM_COMPILER["clang-cl"]

        if configure == "DISABLE" and is_llvm:
            # Any LLVM is found
            if ENV("CC") == clang:
                result = self._compoment_fail(condition, fail_level)
            else:
                result = WARNING
        elif configure == "HIP_SDK" and llvm_type == "AMD HIP SDK":
            # Allow AMD HIP SDK
            result = SUCCESS
        elif configure == "AOCC" and llvm_type == "AOCC":
            # Allow AMD Optimizing C/C++ and Fortran Compilers
            result = SUCCESS
        elif configure == "AMD-LLVM" and llvm_type == "AMD-LLVM":
            # Use build AMD-LLVM from ROCm/TheRock
            result = SUCCESS
        elif configure == "VS20XX" and llvm_type == "Visual Studio":
            # Allow VS20XX LLVM/Clang
            result = SUCCESS
        elif configure == "3rd-Party" and llvm_type == "3rd-party":
            # Allow 3rd-party LLVM/Clang
            result = SUCCESS
        else:
            # Other configure failed
            result = FATAL

        message("CHECK", f"Requires {configure}", check_result=result)
        if result is not SUCCESS:
            # TheRock build, requires disable, but found existed AMD HIP SDK in PATH
            if (
                config.DEFAULT_TASK == "TheRock"
                and configure == "DISABLE"
                and llvm_type == "AMD HIP SDK"
            ):
                reason = dedent(
                    f"""\
                                Found existed AMD HIP SDK at {llvm_dir.as_posix()}.
                                TheRock requires MSVC to compile its LLVM compiler and need to exclude AMD HIP SDK environment.
                                There's several issues will make roc-X subprojects hit compile error if HIP SDK related toolchains
                                existed. If you have existed AMD HIP SDK installation, please exclude these LLVM directory from
                                LLVM_DIR, CMAKE_PREFIX_PATH and binary PATH.

                                 >>> traceback: Found existed AMD HIP SDK"""
                )
            # TheRock build, requires disable, but found existed 3rd-party LLVM in PATH
            elif (
                config.DEFAULT_TASK == "TheRock" and configure == "DISABLE" and is_llvm
            ):
                reason = dedent(
                    f"""\
                                Found existed LLVM compiler at {llvm_dir.as_posix()}.
                                TheRock requires MSVC to compile its LLVM compiler. If you have existed LLVM installation, includes
                                    Visual Studio LLVM/Clang, AMD HIP SDK, Intel oneAPI SDK, NVIDIA HPC SDK, and other 3rd-party
                                    LLVM distributions, please exclude these LLVM directory from LLVM_DIR, CMAKE_PREFIX_PATH and
                                    binary PATH.

                                 >>> traceback: Found invalid LLVM build/distribution"""
                )
            # PyTorch build, requires AMD-LLVM, but found other LLVM (include AMD HIP SDK)
            elif (
                config.DEFAULT_TASK == "PyTorch"
                and configure == "AMD-LLVM"
                and llvm_type != "AMD-LLVM"
            ):
                reason = dedent(
                    f"""\
                                Found existed LLVM compiler at {llvm_dir.as_posix()}.
                                TheRock building PyTorch requires AMD-LLVM, but found 3rd-party/other type/build/dist LLVM toolchain.
                                Please re-configure your LLVM type and specify to TheRock required one.
                                    Note: this LLVM toolchain info records with llvm type = {llvm_type}.

                                 >>> traceback: Found TheRock build PyTorch required LLVM toolchain is not AMD-LLVM"""
                )

            elif configure == "DISABLE" and config.ROCM_HOME and not is_llvm:
                reason = dedent(
                    f"""\
                                Cannot configure LLVM distribution with specified ROCM_HOME.
                                We cannot configure your ROCm software stack with specified ROCM_HOME.
                                Your specified ROCM_HOME dir:
                                    > {Path(config.ROCM_HOME).as_posix()}
                                From FindAMD_LLVM class reports:
                                    > LLVM Configured:  {is_llvm}
                                    > LLVM_DIR:         {llvm_dir.as_posix()}

                                Please check your passed ROCM_HOME path is valid. If this is an unexpected error, please
                                report to TheRock dev team, and provide your compoments, command line, other details to
                                help TheRock dev team fix this issue.

                                 >>> traceback: LLVM configuration is invalid"""
                )
            # Cannot configure LLVM distribution.
            else:
                reason = dedent(
                    f"""\
                                Cannot configure LLVM distribution.

                                 >>> traceback: LLVM configuration is invalid"""
                )

            message(result, reason)
        return result

    def _post_llvm_version(
        self,
        *,
        condition: Literal["REQUIRED", "OPTIONAL"] = "REQUIRED",
        fail_level: Literal["ERROR", "WARNING", "HINT"] = "ERROR",
        version: Union[str, VersionNum],
        op: Literal["=", "≠", "!=", "<", "<=", ">", ">="] = None,
        op_ceil: Literal["<", "<="] = None,
        version_max: Union[str, VersionNum] = None,
    ):

        message("STATUS", "Check for LLVM version")

        if op_ceil is None and version_max is None:
            result = (
                SUCCESS
                if VERSION(self.Version, op, version)
                else self._compoment_fail(condition, fail_level)
            )
            op_ceil, version_max = "", ""
        else:
            result = (
                SUCCESS
                if VERSION_IN_RANGE(version, op, self.Version, op_ceil, version_max)
                else self._compoment_fail(condition, fail_level)
            )

        message("CHECK", self.Version, check_result=result)
        if result is not SUCCESS:
            reason = dedent(
                f"""\
                            Found LLVM version is imcompatable.
                            TheRock required version range is:
                                {version} {op} {self.Version} {op_ceil} {version_max}

                             >>> traceback: Found imcompatable LLVM version"""
            )
            message(result, reason)

        return result
