from ._task_manager import BuildTasks
from devices import WindowsNT, ManyLinux


class PyTorch(BuildTasks):

    build_type = "PyTorch"

    def __init__(self):
        super().__init__()

    def __WINDOWS__(self):

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
                mode="REQUIRES",
                expected_compiler=device.llvm.info["LLVM Toolchain"]["clang"],
                fail_level="ERROR",
            ),
            "CMake have CXX": device.CMake.post(
                "COMPILER_VAR",
                "REQUIRED",
                lang="C++",
                mode="REQUIRES",
                expected_compiler=device.llvm.info["LLVM Toolchain"]["clang++"],
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
            "LLVM Dist": device.llvm.post(
                "LLVM_DISTRIBUTION",
                "REQUIRED",
                configure="AMD-LLVM",
                fail_level="ERROR",
            ),
            "ROCm Dist": device.ROCm.post(
                "ROCM_TYPE", "REQUIRED", ROCM_TYPE="TheRock", fail_level="ERROR"
            ),
            "ROCm Library": device.ROCm.post(
                "roc-X",
                "REQUIRED",
                fail_level="ERROR",
                roc_X=[
                    "hip",
                    "rocm-core",
                    "amd_comgr",
                    "hipRAND",
                    "rocBLAS",
                    "hipBLAS",
                    "MIOpen",
                    "hipFFT",
                    "hipSPARSE",
                    "rocPRIM",
                    "hipCUB",
                    "rocThrust",
                    "hipSOLVER",
                    "rocSOLVER",
                    "hiprtc",
                    "hipBLASLt",
                    "hipSPARSELt",
                ],
                optional=["hipSPARSELt"],
            ),
        }

        return test_dict

    def __LINUX__(self):

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
            "LLVM Dist": device.llvm.post(
                "LLVM_DISTRIBUTION",
                "REQUIRED",
                configure="AMD-LLVM",
                fail_level="ERROR",
            ),
            "ROCm Dist": device.ROCm.post(
                "ROCM_TYPE", "REQUIRED", ROCM_TYPE="TheRock", fail_level="ERROR"
            ),
            "ROCm Library": device.ROCm.post(
                "roc-X",
                "REQUIRED",
                fail_level="ERROR",
                roc_X=[
                    "hip",
                    "rocm-core",
                    "amd_comgr",
                    "hipRAND",
                    "rocBLAS",
                    "hipBLAS",
                    "MIOpen",
                    "hipFFT",
                    "hipSPARSE",
                    "rocPRIM",
                    "hipCUB",
                    "rocThrust",
                    "hipSOLVER",
                    "rocSOLVER",
                    "hiprtc",
                    "hipBLASLt",
                    "hipSPARSELt",
                ],
                optional=["hipSPARSELt"],
            ),
        }

        return test_dict
