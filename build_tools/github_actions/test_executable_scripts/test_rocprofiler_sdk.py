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
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from amdgpu_family_matrix import is_asan

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

# Determine host triple
if THEROCK_CLANG_PATH.exists():
    try:
        host_triple = subprocess.run(
            [str(THEROCK_CLANG_PATH), "--print-target-triple"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(
            f"'{THEROCK_CLANG_PATH} --print-target-triple' failed; "
            "this suggests a broken toolchain."
        ) from exc

if host_triple:
    THEROCK_LLVM_LIB_HOST_TRIPLE_PATH = THEROCK_LIB_PATH / "llvm" / "lib" / host_triple

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
    if host_triple:
        ld_lib_paths += [f"{THEROCK_LLVM_LIB_HOST_TRIPLE_PATH}"]

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

    # Avoid conflicting agent visibility; HIP_VISIBLE_DEVICES supersedes.
    if environ_vars.get("HIP_VISIBLE_DEVICES"):
        environ_vars.pop("GPU_DEVICE_ORDINAL", None)


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
    if is_asan():
        # Preload ASan for standalone tests loading instrumented ROCm libraries.
        asan_runtime_library = get_asan_runtime_library()
        cmake_config_cmd += [
            "-DROCPROFILER_MEMCHECK=AddressSanitizer",
            f"-DROCPROFILER_MEMCHECK_PRELOAD_ENV=LD_PRELOAD={asan_runtime_library}",
            f"-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE={asan_runtime_library}",
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
# Certain tests are enabled/disabled based on the GPU architecture.
# Ensuring that these tests build properly against an install is also part of the overall test coverage for SDK (emulates tool developers building tools with rocprofiler-sdk)
def cmake_build():
    cmake_build_cmd = [
        "cmake",
        "--build",
        "build",
        "--parallel",
        "8",
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
    if is_asan():
        # Exclude tests known to fail/hang in the ASan configuration.
        exclude_regex = "|".join(ASAN_EXCLUDED_TESTS)
        ctest_cmd += ["--exclude-regex", exclude_regex]

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
