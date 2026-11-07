from devices import WindowsNT, ManyLinux
from ._task_manager import BuildTasks
from utils import *


class TheRock(BuildTasks):

    build_type = "TheRock"

    def __init__(self):
        super().__init__()

    def __WINDOWS__(self):
        super().__WINDOWS__()
        device: WindowsNT = self.device

        test_dict = {
            "Windows Version": device.OS.post(
                "VERSION", "REQUIRED", fail_level="ERROR"
            ),
            "Windows is POSIX": device.OS.post("POSIX", "REQUIRED", fail_level="ERROR"),
            "Windows is WINE": device.OS.post("POSIX", "REQUIRED", fail_level="ERROR"),
            "Windows MAX_PATH": device.OS.post(
                "MAX_PATH", "REQUIRED", fail_level="ERROR"
            ),
            "Windows Code Page": device.OS.post(
                "CHCP", "OPTIONAL", fail_level="WARNING"
            ),
            "Windows Byteorder": device.OS.post(
                "BYTE_ORDER", "REQUIRED", endian="Little Endian", fail_level="ERROR"
            ),
            "Windows have network": device.OS.post(
                "NETWORK", "REQUIRED", fail_level="ERROR"
            ),
            "Windows have HIP SDK": device.OS.post(
                "HIP_PATH", "REQUIRED", fail_level="ERROR"
            ),
            "CPU Architecture": device.CPU.post("ARCH", "REQUIRED", fail_level="ERROR"),
            "CPU cores": device.CPU.post("CORES", "REQUIRED", fail_level="WARNING"),
            "GPU Diagnose": device.GPU.post("ROCm"),
            "DRAM Volume": device.RAM.post("VOLUME", "REQUIRED", fail_level="ERROR"),
            "DRAM Type": device.RAM.post(
                "TYPE", "OPTIONAL", memory="UDIMM", fail_level="WARNING"
            ),
            "Disk Volume": device.ROM.post("AVAIL", "REQUIRED", fail_level="WARNING"),
            "Python version": device.Python.post(
                "VERSION",
                "REQUIRED",
                mode="IN_RANGE",
                version="3.10.0",
                op="<=",
                op_ceil="<=",
                version_max="3.13.15",
                experimental=False,
                fail_level="ERROR",
            ),
            "Python Interpreter": device.Python.post(
                "INTERPRETER", "REQUIRED", whitelist=[], fail_level="ERROR"
            ),
            "Python venv": device.Python.post(
                "VENV",
                "REQUIRED",
                envs=["uv", "Conda"],
                conda_base=True,
                fail_level="ERROR",
            ),
            "Python GIL": device.Python.post(
                "GIL", "REQUIRED", ENABLE_FREE_THREADING=False, fail_level="ERROR"
            ),
            "Python JIT": device.Python.post(
                "JIT", "REQUIRED", ENABLE_JIT=False, fail_level="ERROR"
            ),
            "Python module": device.Python.post(
                "module", "REQUIRED", build_type=self.build_type, fail_level="ERROR"
            ),
            "DVC Avail": device.DVC.post("AVAIL", "REQUIRED", fail_level="ERROR"),
            "Git Avail": device.Git.post("AVAIL", "REQUIRED", fail_level="ERROR"),
            "Conda Avail": device.Conda.post("AVAIL", "OPTIONAL", fail_level="HINT"),
            "Astral uv Avail": device.UV.post("AVAIL", "OPTIONAL", fail_level="HINT"),
            "CMake Avail": device.CMake.post(
                "AVAIL",
                "REQUIRED",
                cmake_minimum_required=None,
                cmake4=False,
                cmake_policy_max="3.31.11",
                fail_level="ERROR",
            ),
            "CMake have CC": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="C",
                mode="NOTDEFINED",
                fail_level="ERROR",
            ),
            "CMake have CXX": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="C++",
                mode="NOTDEFINED",
                fail_level="ERROR",
            ),
            "CMake have FC": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="Fortran",
                mode="NOTDEFINED",
                fail_level="ERROR",
            ),
            "Ninja Avail": device.Ninja.post(
                "AVAIL", "REQUIRED", min="1.13.1", fail_level="ERROR"
            ),
            "Ninja MAX_JOBS": device.Ninja.post(
                "MAX_JOBS", "REQUIRED", limit=None, fail_level="ERROR"
            ),
            "CCache Avail": device.CCache.post("AVAIL", "OPTIONAL", fail_level="HINT"),
            "SCCache Avail": device.SCCache.post(
                "AVAIL", "OPTIONAL", fail_level="HINT"
            ),
            "GCC gortran Avail": device.gfortran.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "PkgConfig Lite Avail": device.PkgConfig.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "VS20XX Avail": device.VS20XX.post(
                "VS20XX",
                "OPTIONAL",
                whitelist=["VS2022", "VS2019"],
                fail_level="WARNING",
            ),
            "MSVC Version": device.VS20XX.post(
                "VC_Version", "REQUIRED", msvc=["v143", "v142"], fail_level="ERROR"
            ),
            "MSVC compiler": device.VS20XX.post("MSVC", "REQUIRED", fail_level="ERROR"),
            "MSVC assembler": device.VS20XX.post(
                "ML64", "REQUIRED", fail_level="ERROR"
            ),
            "MSVC archiver": device.VS20XX.post("LIB", "REQUIRED", fail_level="ERROR"),
            "MSVC linker": device.VS20XX.post("LINK", "REQUIRED", fail_level="ERROR"),
            "MSVC compiler host": device.VS20XX.post(
                "VC_Host", "REQUIRED", fail_level="ERROR"
            ),
            "MSVC compiler target": device.VS20XX.post(
                "VC_Target", "REQUIRED", fail_level="ERROR"
            ),
            "MSVC ATL/MFC": device.VS20XX.post(
                "VC_ATL", "REQUIRED", fail_level="ERROR"
            ),
            "UCRT Version": device.VS20XX.post("UCRT", "REQUIRED", fail_level="ERROR"),
            "UCRT RC": device.VS20XX.post("RC", "REQUIRED", fail_level="ERROR"),
            "UCRT MT": device.VS20XX.post("MT", "REQUIRED", fail_level="ERROR"),
            "MSVC dumpbin": device.VS20XX.post(
                "dumpbin", "OPTIONAL", fail_level="WARNING"
            ),
            "LLVM Avail": device.llvm.post(
                "LLVM_DISTRIBUTION", "REQUIRED", configure="DISABLE", fail_level="ERROR"
            ),
        }

        return test_dict

    def __LINUX__(self):
        super().__LINUX__()
        device: ManyLinux = self.device

        test_dict = {
            "Linux Version": device.OS.post("WSL2", "REQUIRED", fail_level="ERROR"),
            "Linux Byteorder": device.OS.post(
                "BYTE_ORDER", "REQUIRED", fail_level="ERROR"
            ),
            "Linux have Network": device.OS.post(
                "NETWORK", "REQUIRED", fail_level="ERROR"
            ),
            "CPU Architecture": device.CPU.post("ARCH", "REQUIRED", fail_level="ERROR"),
            "CPU cores": device.CPU.post("CORES", "REQUIRED", fail_level="WARNING"),
            "GPU Diagnose": device.GPU.post("ROCm"),
            "DRAM Volume": device.RAM.post("VOLUME", "REQUIRED", fail_level="ERROR"),
            "DRAM Type": device.RAM.post("TYPE", "OPTIONAL", fail_level="HINT"),
            "Disk Volume": device.ROM.post("AVAIL", "REQUIRED", fail_level="WARNING"),
            "Python version": device.Python.post(
                "VERSION",
                "REQUIRED",
                mode="IN_RANGE",
                version="3.10.0",
                op="<=",
                op_ceil="<=",
                version_max="3.13.15",
                experimental=False,
                fail_level="ERROR",
            ),
            "Python Interpreter": device.Python.post(
                "INTERPRETER", "REQUIRED", whitelist=[], fail_level="ERROR"
            ),
            "Python venv": device.Python.post(
                "VENV", "REQUIRED", envs=["uv", "Conda"], fail_level="ERROR"
            ),
            "Python GIL": device.Python.post(
                "GIL", "REQUIRED", ENABLE_FREE_THREADING=False, fail_level="ERROR"
            ),
            "Python JIT": device.Python.post(
                "JIT", "REQUIRED", ENABLE_JIT=False, fail_level="ERROR"
            ),
            "Python module": device.Python.post(
                "module", "REQUIRED", build_type=self.build_type, fail_level="ERROR"
            ),
            "DVC Avail": device.DVC.post("AVAIL", "REQUIRED", fail_level="ERROR"),
            "Git Avail": device.Git.post("AVAIL", "REQUIRED", fail_level="ERROR"),
            "Conda Avail": device.Conda.post("AVAIL", "OPTIONAL", fail_level="HINT"),
            "Astral uv Avail": device.UV.post("AVAIL", "OPTIONAL", fail_level="HINT"),
            "CMake Avail": device.CMake.post(
                "AVAIL",
                "REQUIRED",
                cmake_minimum_required=None,
                cmake4=False,
                cmake_policy_max="3.31.11",
                fail_level="ERROR",
            ),
            "CMake have CC": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="C",
                mode="NOTDEFINED",
                fail_level="ERROR",
            ),
            "CMake have CXX": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="C++",
                mode="NOTDEFINED",
                fail_level="ERROR",
            ),
            "CMake have FC": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="Fortran",
                mode="NOTDEFINED",
                fail_level="ERROR",
            ),
            "Ninja Avail": device.Ninja.post(
                "AVAIL", "REQUIRED", min="1.13.1", fail_level="ERROR"
            ),
            "CCache Avail": device.CCache.post("AVAIL", "OPTIONAL", fail_level="HINT"),
            "SCCache Avail": device.SCCache.post(
                "AVAIL", "OPTIONAL", fail_level="HINT"
            ),
            "PkgConfig Avail": device.PkgConfig.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "GCC gcc Avail": device.gcc.post(
                "AVAIL",
                "REQUIRED",
                version_require="COMPARE",
                op=">=",
                version="11.2.0",
                fail_level="ERROR",
            ),
            "GCC g++ Avail": device.gxx.post(
                "AVAIL",
                "REQUIRED",
                version_require="COMPARE",
                op=">=",
                version="11.2.0",
                fail_level="ERROR",
            ),
            "GCC gfortran Avail": device.gfortran.post(
                "AVAIL",
                "REQUIRED",
                version_require="COMPARE",
                op=">=",
                version="11.2.0",
                fail_level="ERROR",
            ),
            "GCC gcc-ar Avail": device.gcc_ar.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "Binutils ar Avail": device.bin_ar.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "Binutils as Avail": device.bin_as.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "Binutils ld Avail": device.ld.post(
                "AVAIL", "REQUIRED", fail_level="ERROR"
            ),
            "GLIBC Avail": device.glibc.post(
                "AVAIL", "REQUIRED", version_require="COMPARE", op=">=", version="2.38"
            ),
        }

        results_list = [
            subitem
            for item in test_dict.values()
            for subitem in (item.values() if isinstance(item, dict) else [item])
        ]

        test_dict["SUCCESS_Total"] = results_list.count(SUCCESS)
        test_dict["HINT_Total"] = results_list.count(HINT)
        test_dict["WARNING_Total"] = results_list.count(WARNING)
        test_dict["ERROR_Total"] = results_list.count(FATAL)

        return test_dict
