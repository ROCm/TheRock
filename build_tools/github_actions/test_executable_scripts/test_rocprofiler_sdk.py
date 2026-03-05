import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys

# Base Paths
THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent

# LIB Paths
THEROCK_LIB_PATH = THEROCK_PATH / "lib"
THEROCK_SYSDEPS_PATH = THEROCK_LIB_PATH / "rocm_sysdeps"
THEROCK_SYSDEPS_LIB_PATH = THEROCK_SYSDEPS_PATH / "lib"

# LLVM Paths
THEROCK_LLVM_BIN_PATH = THEROCK_PATH / "llvm" / "bin"
THEROCK_CLANG_PATH = THEROCK_LLVM_BIN_PATH / "amdclang"
THEROCK_CLANG_PLUS_PATH = THEROCK_LLVM_BIN_PATH / "amdclang++"

# SDK Paths
ROCPROFILER_SDK_PATH = THEROCK_PATH / "share" / "rocprofiler-sdk"
ROCPROFILER_SDK_TESTS_PATH = ROCPROFILER_SDK_PATH / "tests"

logging.basicConfig(level=logging.INFO)
environ_vars = os.environ.copy()


def setup_env():
    environ_vars["ROCM_PATH"] = str(THEROCK_PATH)
    environ_vars["HIP_PATH"] = str(THEROCK_PATH)
    environ_vars["ROCPROFILER_METRICS_PATH"] = str(ROCPROFILER_SDK_PATH)
    environ_vars["HIP_PLATFORM"] = "amd"

    old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
    if old_ld_lib_path:
        environ_vars["LD_LIBRARY_PATH"] = (
            f"{THEROCK_LIB_PATH}:{THEROCK_SYSDEPS_LIB_PATH}:{old_ld_lib_path}"
        )
    else:
        environ_vars["LD_LIBRARY_PATH"] = (
            f"{THEROCK_LIB_PATH}:{THEROCK_SYSDEPS_LIB_PATH}"
        )


def cmake_config():
    cmake_config_cmd = [
        "cmake",
        "-B",
        "build",
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={THEROCK_PATH};{THEROCK_SYSDEPS_PATH}",
        f"-DCMAKE_HIP_COMPILER={THEROCK_CLANG_PLUS_PATH}",
        f"-DCMAKE_C_COMPILER={THEROCK_CLANG_PATH}",
        f"-DCMAKE_CXX_COMPILER={THEROCK_CLANG_PLUS_PATH}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]

    logging.info(
        f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(cmake_config_cmd)}"
    )
    subprocess.run(
        cmake_config_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


# SDK requires test binaries to be built on the gfx architecture being tested on
# This requires building on the test stage as we cannot build these on a gpu-less environment
# Ensuring that these tests build properly is also part of the overall test coverage for SDK
def cmake_build():
    cmake_build_cmd = [
        "cmake",
        "--build",
        "build",
        "-j",
    ]

    logging.info(
        f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(cmake_build_cmd)}"
    )
    subprocess.run(
        cmake_build_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


def execute_tests():
    ctest_cmd = [
        "ctest",
        "--test-dir",
        "build",
        "--parallel",
        "8",
        "--output-on-failure",
    ]

    logging.info(f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(ctest_cmd)}")
    subprocess.run(
        ctest_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


if __name__ == "__main__":
    setup_env()
    cmake_config()
    cmake_build()
    execute_tests()
