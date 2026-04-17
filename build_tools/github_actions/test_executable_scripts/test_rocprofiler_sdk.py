# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent.parent
_DEFAULT_ROCM_BIN = _REPO_ROOT / "build" / "dist" / "rocm" / "bin"


def _resolve_therock_bin_dir() -> Path:
    env = os.getenv("THEROCK_BIN_DIR")
    if env:
        return Path(env).resolve()
    if _DEFAULT_ROCM_BIN.is_dir():
        return _DEFAULT_ROCM_BIN.resolve()
    raise SystemExit(
        "THEROCK_BIN_DIR is not set and "
        f"{_DEFAULT_ROCM_BIN} does not exist. "
        "Set THEROCK_BIN_DIR to your TheRock install bin directory "
        "(e.g. build/dist/rocm/bin)."
    )


# Base Paths
THEROCK_BIN_PATH = _resolve_therock_bin_dir()
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

    old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "").split(":")
    environ_vars["LD_LIBRARY_PATH"] = ":".join(
        [f"{THEROCK_LIB_PATH}", f"{THEROCK_SYSDEPS_LIB_PATH}"] + old_ld_lib_path
    )

def get_cmake_config_cmd() -> list[str]:
    """Configure command used for rocprofiler-sdk tests (matches local cmake_config).

    Uses absolute -S/-B paths because CTest runs CTEST_CONFIGURE_COMMAND with the
    working directory set to the *binary* directory (CMake 3.14+); a relative
    ``cmake -B build`` would then treat the build tree as the source tree and fail.
    ``--fresh`` avoids stale cache / generator mismatches when switching generators.
    """
    tests_dir = ROCPROFILER_SDK_TESTS_PATH
    build_dir = tests_dir / "build"
    return [
        "cmake",
        "-S",
        str(tests_dir),
        "-B",
        str(build_dir),
        "--fresh",
        "-G",
        "Ninja",
        f"-DCMAKE_PREFIX_PATH={THEROCK_PATH};{THEROCK_SYSDEPS_PATH}",
        f"-DCMAKE_HIP_COMPILER={THEROCK_CLANG_PLUS_PATH}",
        f"-DCMAKE_C_COMPILER={THEROCK_CLANG_PATH}",
        f"-DCMAKE_CXX_COMPILER={THEROCK_CLANG_PLUS_PATH}",
        f"-DPython3_EXECUTABLE={sys.executable}",
    ]


def get_cmake_build_cmd() -> list[str]:
    """Build command used for rocprofiler-sdk tests (matches local cmake_build)."""
    build_dir = ROCPROFILER_SDK_TESTS_PATH / "build"
    return [
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        "8",
    ]


def get_ctest_cmd() -> list[str]:
    """CTest invocation used for rocprofiler-sdk tests (matches local execute_tests)."""
    return [
        "ctest",
        "--test-dir",
        "build",
        "--parallel",
        "8",
        "--output-on-failure",
    ]


def _running_in_ci() -> bool:
    """True when running on a typical CI runner (GitHub Actions sets CI=true)."""
    ci = os.environ.get("CI", "").strip().lower()
    return ci in ("1", "true", "yes")


def run_therock_ci(
    cmake_config_cmd: list[str],
    cmake_build_cmd: list[str],
    ctest_cmd: list[str],
) -> None:
    """Run run-therock-ci.py with the same configure, build, and ctest arguments as this script.

    Passes shell-joined command lines so run-therock-ci can set CTEST_CONFIGURE_COMMAND,
    CTEST_BUILD_COMMAND, and CMAKE_CTEST_ARGUMENTS consistently with local runs.
    """
    ctest_args = (
        ctest_cmd[1:] if ctest_cmd and ctest_cmd[0] == "ctest" else list(ctest_cmd)
    )
    argv = [
        sys.executable,
        str(_REPO_ROOT / "rocm-systems/.github/scripts/run-therock-ci.py"),
        "--configure-cmd",
        shlex.join(cmake_config_cmd),
        "--build-cmd",
        shlex.join(cmake_build_cmd),
        "--ctest-args",
        shlex.join(ctest_args),
        "--rocprofiler-sdk-path",
        str(ROCPROFILER_SDK_PATH),
        "--rocprofiler-sdk-tests-path",
        str(ROCPROFILER_SDK_TESTS_PATH),
        "--therock-bin-path",
        str(THEROCK_BIN_PATH),
        "--therock-clang-path",
        str(THEROCK_CLANG_PATH),
        "--therock-clang-plus-path",
        str(THEROCK_CLANG_PLUS_PATH),
        "--therock-lib-path",
        str(THEROCK_LIB_PATH),
        "--therock-sysdeps-lib-path",
        str(THEROCK_SYSDEPS_LIB_PATH),
        "--therock-sysdeps-path",
        str(THEROCK_SYSDEPS_PATH),
        "--therock-path",
        str(THEROCK_PATH),
    ]
    logging.info(f"++ Exec [{_REPO_ROOT}]$ {shlex.join(argv)}")
    subprocess.run(
        argv,
        cwd=_REPO_ROOT,
        check=True,
        env=environ_vars,
    )


def cmake_config():
    cmake_config_cmd = get_cmake_config_cmd()

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
# Certain tests are enabled/disabled based on the GPU architecture.
# Ensuring that these tests build properly against an install is also part of the overall test coverage for SDK (emulates tool developers building tools with rocprofiler-sdk)
def cmake_build():
    cmake_build_cmd = get_cmake_build_cmd()

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
    ctest_cmd = get_ctest_cmd()

    logging.info(f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(ctest_cmd)}")
    subprocess.run(
        ctest_cmd,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


if __name__ == "__main__":
    setup_env()
    if _running_in_ci():
        run_therock_ci(
            get_cmake_config_cmd(),
            get_cmake_build_cmd(),
            get_ctest_cmd(),
        )
    else:
        cmake_config()
        cmake_build()
        execute_tests()
