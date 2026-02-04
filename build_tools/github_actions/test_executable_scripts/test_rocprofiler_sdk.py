import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent
THEROCK_LIB_PATH = str(THEROCK_PATH / "lib")
ROCPROFILER_SDK_DIRECTORY = f"{THEROCK_PATH}/share/rocprofiler-sdk"
ROCPROFILER_SDK_TESTS_DIRECTORY = str(ROCPROFILER_SDK_DIRECTORY / "tests")

logging.basicConfig(level=logging.INFO)

# Set up HIP_PATH / ROCM_PATH
environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = str(THEROCK_PATH)
environ_vars["HIP_PATH"] = str(THEROCK_PATH)

# Set up ROCPROFILER_SDK_TEST_METRICS_PATH
environ_vars["ROCPROFILER_SDK_TEST_METRICS_PATH"] = str(ROCPROFILER_SDK_DIRECTORY)

# Env setup
environ_vars["HIP_PLATFORM"] = "amd"

# Set up LD_LIBRARY_PATH / ROCPROFILER_SDK_TEST_LD_LIBRARY_PATH
old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
sysdeps_path = f"{THEROCK_LIB_PATH}/rocm_sysdeps/lib"
if old_ld_lib_path:
    environ_vars["LD_LIBRARY_PATH"] = (
        f"{THEROCK_LIB_PATH}:{sysdeps_path}:{old_ld_lib_path}"
    )
    environ_vars["ROCPROFILER_SDK_TEST_LD_LIBRARY_PATH"] = (
        f"{THEROCK_LIB_PATH}:{sysdeps_path}:{old_ld_lib_path}"
    )
else:
    environ_vars["LD_LIBRARY_PATH"] = f"{THEROCK_LIB_PATH}:{sysdeps_path}"
    environ_vars["ROCPROFILER_SDK_TEST_LD_LIBRARY_PATH"] = (
        f"{THEROCK_LIB_PATH}:{sysdeps_path}"
    )

# CMake Configuration
cmake_config_cmd = [
    "cmake",
    "-B",
    "build",
    "-G",
    "Ninja",
    f"-DCMAKE_PREFIX_PATH={THEROCK_PATH}",
    f"-DCMAKE_HIP_COMPILER={THEROCK_PATH}/llvm/bin/amdclang++",
    f"-DHIP_HIPCC_EXECUTABLE={THEROCK_PATH}/bin/hipcc",
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmake_config_cmd)}")
subprocess.run(
    cmake_config_cmd,
    cwd=THEROCK_DIR,
    check=True,
    env=environ_vars,
)

# CMake Build
cmake_build_cmd = [
    "cmake",
    "--build",
    "build",
    "-j",
]

logging.info(
    f"++ Exec [{ROCPROFILER_SDK_TESTS_DIRECTORY}]$ {shlex.join(cmake_build_cmd)}"
)
subprocess.run(
    cmake_build_cmd,
    cwd=ROCPROFILER_SDK_TESTS_DIRECTORY,
    check=True,
    env=environ_vars,
)

# CTest
ctest_cmd = [
    "ctest",
    "--test-dir",
    "build",
    "--output-on-failure",
]

logging.info(f"++ Exec [{ROCPROFILER_SDK_TESTS_DIRECTORY}]$ {shlex.join(ctest_cmd)}")
subprocess.run(
    ctest_cmd,
    cwd=ROCPROFILER_SDK_TESTS_DIRECTORY,
    check=True,
    env=environ_vars,
)
