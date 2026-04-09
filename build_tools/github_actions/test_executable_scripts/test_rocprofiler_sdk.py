# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shlex
import subprocess
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(_REPO_ROOT / "build_tools" / "github_actions"))
from amdgpu_family_matrix import is_asan

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

# Tests skipped under ASan (known failing/unstable in the ASan configuration).
ASAN_EXCLUDED_TESTS = [
    "rocprofiler_sdk.unit.spm_core.check_packet_generation",
    "rocprofiler_sdk.unit.spm_core.check_callbacks",
    "rocprofiler_sdk.unit.rocprofiler_lib.callback_external_correlation",
    "rocprofiler_sdk.unit.rocprofiler_lib.buffered_external_correlation",
    "rocprofiler_sdk.unit.rocprofiler_lib.callback_registration_lambda_with_result",
    "rocprofiler_sdk.unit.rocprofiler_lib.buffer_registration_lambda_with_result",
    "async-copy-tracing",
    "memory-allocation-tracing",
    "test-scratch-memory-tracing",
    "rocjpeg-tracing",
    "rocprofv3-test-hsa-multiqueue",
    "rocprofv3-test-att-hsa-multiqueue-cmd",
    "rocprofv3-test-att-hsa-multiqueue-cmd-env-att-lib-path",
    "rocprofv3-test-att-hsa-multiqueue-json",
    "rocprofv3-test-att-env-var",
    "rocpd-api-python-interface-test",
]

logging.basicConfig(level=logging.INFO)
environ_vars = os.environ.copy()


def get_asan_runtime_library():
    """Return the clang AddressSanitizer runtime path."""
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        arch = "x86_64"
    elif machine == "aarch64":
        arch = "aarch64"
    else:
        raise RuntimeError(f"Unsupported ASan runtime architecture: {machine}")

    asan_lib = f"libclang_rt.asan-{arch}.so"
    result = subprocess.run(
        [str(THEROCK_CLANG_PLUS_PATH), f"-print-file-name={asan_lib}"],
        check=True,
        capture_output=True,
        text=True,
        env=environ_vars,
    )
    resolved = result.stdout.strip()
    if not resolved or resolved == asan_lib or not Path(resolved).is_file():
        raise FileNotFoundError(
            f"Could not locate ASan runtime '{asan_lib}' via {THEROCK_CLANG_PLUS_PATH} "
            f"(got: '{resolved}')"
        )
    return str(Path(resolved).resolve())


def setup_env():
    environ_vars["ROCM_PATH"] = str(THEROCK_PATH)
    environ_vars["HIP_PATH"] = str(THEROCK_PATH)
    environ_vars["ROCPROFILER_METRICS_PATH"] = str(ROCPROFILER_SDK_PATH)
    environ_vars["HIP_PLATFORM"] = "amd"

    ld_lib_paths = [f"{THEROCK_LIB_PATH}", f"{THEROCK_SYSDEPS_LIB_PATH}"]

    if is_asan():
        # Installed test binaries are built with -shared-libsan, so the clang
        # resource dir holding libclang_rt.asan-<arch>.so must be on the loader
        # search path. Match rocprofiler-sdk sanitizer defaults for launchers.
        ld_lib_paths.append(str(Path(get_asan_runtime_library()).parent))

        existing_asan_options = os.getenv("ASAN_OPTIONS", "")
        asan_options = "detect_leaks=0:use_sigaltstack=0"
        if existing_asan_options:
            asan_options = f"{asan_options}:{existing_asan_options}"
        environ_vars["ASAN_OPTIONS"] = asan_options

    old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "").split(":")
    environ_vars["LD_LIBRARY_PATH"] = ":".join(ld_lib_paths + old_ld_lib_path)
    environ_vars["THEROCK_BIN_DIR"] = str(THEROCK_BIN_PATH)

    # Avoid conflicting agent visibility; HIP_VISIBLE_DEVICES supersedes.
    if environ_vars.get("HIP_VISIBLE_DEVICES"):
        environ_vars.pop("GPU_DEVICE_ORDINAL", None)

    # Paths for test_therock.py (single source of truth; subprocess inherits these).
    environ_vars["THEROCK_PATH"] = str(THEROCK_PATH)
    environ_vars["THEROCK_LIB_PATH"] = str(THEROCK_LIB_PATH)
    environ_vars["THEROCK_SYSDEPS_PATH"] = str(THEROCK_SYSDEPS_PATH)
    environ_vars["THEROCK_SYSDEPS_LIB_PATH"] = str(THEROCK_SYSDEPS_LIB_PATH)
    environ_vars["THEROCK_LLVM_BIN_PATH"] = str(THEROCK_LLVM_BIN_PATH)
    environ_vars["THEROCK_CLANG_PATH"] = str(THEROCK_CLANG_PATH)
    environ_vars["THEROCK_CLANG_PLUS_PATH"] = str(THEROCK_CLANG_PLUS_PATH)
    environ_vars["ROCPROFILER_SDK_PATH"] = str(ROCPROFILER_SDK_PATH)
    environ_vars["ROCPROFILER_SDK_TESTS_PATH"] = str(ROCPROFILER_SDK_TESTS_PATH)


def get_cmake_config_cmd() -> list[str]:
    """Configure command used for rocprofiler-sdk tests (matches local cmake_config)."""
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
    if is_asan():
        # Preload ASan for standalone tests loading instrumented ROCm libraries.
        asan_runtime_library = get_asan_runtime_library()
        cmake_config_cmd += [
            "-DROCPROFILER_MEMCHECK=AddressSanitizer",
            f"-DROCPROFILER_MEMCHECK_PRELOAD_ENV=LD_PRELOAD={asan_runtime_library}",
            f"-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE={asan_runtime_library}",
        ]
    return cmake_config_cmd


def get_cmake_build_cmd() -> list[str]:
    """Build command used for rocprofiler-sdk tests (matches local cmake_build)."""
    return [
        "cmake",
        "--build",
        "build",
        "--parallel",
        "8",
    ]


def get_ctest_cmd() -> list[str]:
    """CTest invocation used for rocprofiler-sdk tests (matches local execute_tests)."""
    ctest_cmd = [
        "ctest",
        "--test-dir",
        "build",
        "--parallel",
        "8",
        "--output-on-failure",
    ]
    if is_asan():
        # Exclude tests known to fail/hang in the ASan configuration.
        exclude_regex = "|".join(ASAN_EXCLUDED_TESTS)
        ctest_cmd += ["--exclude-regex", exclude_regex]
    return ctest_cmd


def run_test_therock_cdash(
    cmake_config_cmd: list[str],
    cmake_build_cmd: list[str],
    ctest_cmd: list[str],
) -> None:
    """Run test_therock.py with the same configure, build, and ctest arguments as this script.

    Passes shell-joined command lines so test_therock can set CTEST_CONFIGURE_COMMAND,
    CTEST_BUILD_COMMAND, and CMAKE_CTEST_ARGUMENTS consistently with local runs.
    """
    ctest_args = ctest_cmd[1:] if ctest_cmd and ctest_cmd[0] == "ctest" else list(ctest_cmd)
    argv = [
        sys.executable,
        str(SCRIPT_DIR / "test_therock.py"),
        "--configure-cmd",
        shlex.join(cmake_config_cmd),
        "--build-cmd",
        shlex.join(cmake_build_cmd),
        "--ctest-args",
        shlex.join(ctest_args),
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
    run_test_therock_cdash(
        get_cmake_config_cmd(),
        get_cmake_build_cmd(),
        get_ctest_cmd(),
    )
