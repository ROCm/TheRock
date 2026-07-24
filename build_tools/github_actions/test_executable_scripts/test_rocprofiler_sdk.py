#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Test rocprofiler-sdk from TheRock's installed test artifact.

This is distinct from rocprofiler-sdk's source-tree ``source/scripts/run-ci.py``.
The direct path matches the current TheRock runner; ``--enable-cdash`` uses the
same commands through CTest's dashboard API so configure, build, and test
results can be submitted without changing the artifact under test.
"""

import argparse
import logging
import os
import platform
import re
import shlex
import socket
import string
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
ROCPROFILER_SDK_TESTS_BUILD_PATH = ROCPROFILER_SDK_TESTS_PATH / "build"

_DEFAULT_CDASH_PROJECT = "rocprofiler-sdk-alt"
_DEFAULT_CDASH_BASE_URL = "my.cdash.org"
_DEFAULT_CDASH_GROUP = "TheRock"
_DEFAULT_CDASH_MODEL = "Continuous"

# Determine host triple
host_triple = ""
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


def get_cmake_config_cmd() -> list[str]:
    """Return the installed-test project configure command."""
    cmake_config_cmd = [
        "cmake",
        "-S",
        str(ROCPROFILER_SDK_TESTS_PATH),
        "-B",
        str(ROCPROFILER_SDK_TESTS_BUILD_PATH),
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
    """Return the installed-test project build command."""
    return [
        "cmake",
        "--build",
        str(ROCPROFILER_SDK_TESTS_BUILD_PATH),
        "--parallel",
        "8",
    ]


def get_ctest_cmd() -> list[str]:
    """Return the installed-test project CTest command."""
    ctest_cmd = [
        "ctest",
        "--test-dir",
        str(ROCPROFILER_SDK_TESTS_BUILD_PATH),
        "--parallel",
        "8",
        "--output-on-failure",
    ]
    if is_asan():
        # Exclude tests known to fail/hang in the ASan configuration.
        exclude_regex = "|".join(ASAN_EXCLUDED_TESTS)
        ctest_cmd += ["--exclude-regex", exclude_regex]
    return ctest_cmd


def _run_command(command: list[str]) -> None:
    logging.info(f"++ Exec [{ROCPROFILER_SDK_TESTS_PATH}]$ {shlex.join(command)}")
    subprocess.run(
        command,
        cwd=ROCPROFILER_SDK_TESTS_PATH,
        check=True,
        env=environ_vars,
    )


def cmake_config() -> None:
    _run_command(get_cmake_config_cmd())


# SDK test binaries must be built for the GPU architecture under test. Building
# them against the install tree also validates the packaged developer surface.
def cmake_build() -> None:
    _run_command(get_cmake_build_cmd())


def execute_tests() -> None:
    _run_command(get_ctest_cmd())


class _CMakeTemplate(string.Template):
    """Template that leaves CMake's ``${...}`` expressions untouched."""

    delimiter = "@"


def _cmake_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _command_for_cmake(command: list[str]) -> str:
    return _cmake_escape(shlex.join(command))


def _sanitize_build_name_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def _cdash_build_name() -> str:
    """Return a stable, unique CDash build name for local and CI runs."""
    override = os.getenv("THEROCK_CDASH_BUILD_NAME")
    if override:
        return override

    parts = ["ROCm", "TheRock", "rocprofiler-sdk"]
    pull_request = re.match(r"refs/pull/(\d+)/", os.getenv("GITHUB_REF", ""))
    if pull_request:
        parts.append(f"PR-{pull_request.group(1)}")
    elif ref_name := os.getenv("GITHUB_REF_NAME"):
        parts.append(_sanitize_build_name_part(ref_name))
    else:
        parts.append("CI" if os.getenv("CI") else "local")

    parts.append(
        _sanitize_build_name_part((os.getenv("RUNNER_OS") or platform.system()).lower())
    )
    for env_name in ("AMDGPU_FAMILIES", "BUILD_VARIANT"):
        if value := os.getenv(env_name):
            parts.append(_sanitize_build_name_part(value))

    artifact_run_id = os.getenv("ARTIFACT_RUN_ID")
    test_run_id = os.getenv("GITHUB_RUN_ID")
    if artifact_run_id:
        parts.append(f"artifact-run-{_sanitize_build_name_part(artifact_run_id)}")
    if test_run_id and test_run_id != artifact_run_id:
        parts.append(f"test-run-{_sanitize_build_name_part(test_run_id)}")
    return "/".join(parts)


def _dashboard_notes() -> str:
    """Describe both the tested artifact and the workflow performing the test."""
    fields = {
        "artifact_run_id": os.getenv("ARTIFACT_RUN_ID", "unknown"),
        "test_run_id": os.getenv("GITHUB_RUN_ID", "unknown"),
        "checkout_repository": os.getenv("GITHUB_REPOSITORY", "unknown"),
        "checkout_sha": os.getenv("GITHUB_SHA", "unknown"),
        "checkout_ref": os.getenv("GITHUB_REF", "unknown"),
        "amdgpu_families": os.getenv("AMDGPU_FAMILIES", "unknown"),
        "build_variant": os.getenv("BUILD_VARIANT", "unknown"),
    }
    return "\n".join(f"{key}: {value}" for key, value in fields.items()) + "\n"


def _generate_dashboard(
    *,
    model: str,
    group: str,
    require_cdash_submission: bool,
    notes_file: Path,
) -> str:
    """Generate a CTest dashboard for the installed rocprofiler-sdk tests."""
    project_name = os.getenv("THEROCK_CDASH_PROJECT", _DEFAULT_CDASH_PROJECT)
    submit_url = os.getenv(
        "THEROCK_CDASH_SUBMIT_URL",
        f"https://{_DEFAULT_CDASH_BASE_URL}/submit.php?project={project_name}",
    )
    site = (
        os.getenv("THEROCK_CDASH_SITE")
        or os.getenv("RUNNER_NAME")
        or os.getenv("HOSTNAME")
        or socket.gethostname()
    )
    test_options = ""
    if is_asan():
        exclude_regex = _cmake_escape("|".join(ASAN_EXCLUDED_TESTS))
        test_options = f'EXCLUDE "{exclude_regex}"'

    return _CMakeTemplate(
        """cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

set(CTEST_PROJECT_NAME "@project_name")
set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")
set(CTEST_SUBMIT_URL "@submit_url")
set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE_CDASH TRUE)

set(CTEST_SITE "@site")
set(CTEST_BUILD_NAME "@build_name")
set(CTEST_SOURCE_DIRECTORY "@source_directory")
set(CTEST_BINARY_DIRECTORY "@binary_directory")
set(CTEST_CONFIGURE_COMMAND "@configure_command")
set(CTEST_BUILD_COMMAND "@build_command")
set(CTEST_NOTES_FILES "@notes_file")
set(CTEST_OUTPUT_ON_FAILURE TRUE)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS 100)
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS 100)
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 51200)
set(_require_cdash_submission @require_cdash_submission)

macro(dashboard_submit)
  set(_submit_result 0)
  set(_submit_cmake_error 0)
  ctest_submit(${ARGN}
    RETRY_COUNT 0
    RETRY_DELAY 5
    RETURN_VALUE _submit_result
    CAPTURE_CMAKE_ERROR _submit_cmake_error
  )
  if(NOT _submit_cmake_error EQUAL 0 OR NOT _submit_result EQUAL 0)
    if(_require_cdash_submission)
      message(FATAL_ERROR "CDash submission failed")
    else()
      message(WARNING "CDash submission failed; test results remain available locally")
    endif()
  endif()
endmacro()

macro(dashboard_fail_if_needed stage result_var)
  if(NOT ${${result_var}} EQUAL 0)
    dashboard_submit(PARTS Notes Done)
    message(FATAL_ERROR "${stage} failed: ${${result_var}}")
  endif()
endmacro()

ctest_start(@model GROUP "@group")

ctest_configure(
  BUILD "@binary_directory"
  RETURN_VALUE _configure_result
)
dashboard_submit(PARTS Start Configure)
dashboard_fail_if_needed("Configure" _configure_result)

ctest_build(
  BUILD "@binary_directory"
  RETURN_VALUE _build_result
)
dashboard_submit(PARTS Build)
dashboard_fail_if_needed("Build" _build_result)

ctest_test(
  BUILD "@binary_directory"
  PARALLEL_LEVEL 8
  @test_options
  RETURN_VALUE _test_result
)
dashboard_submit(PARTS Test)
dashboard_fail_if_needed("Test" _test_result)
dashboard_submit(PARTS Notes Done)
"""
    ).substitute(
        project_name=_cmake_escape(project_name),
        submit_url=_cmake_escape(submit_url),
        site=_cmake_escape(site),
        build_name=_cmake_escape(_cdash_build_name()),
        source_directory=_cmake_escape(str(ROCPROFILER_SDK_TESTS_PATH)),
        binary_directory=_cmake_escape(str(ROCPROFILER_SDK_TESTS_BUILD_PATH)),
        configure_command=_command_for_cmake(get_cmake_config_cmd()),
        build_command=_command_for_cmake(get_cmake_build_cmd()),
        notes_file=_cmake_escape(str(notes_file)),
        model=model,
        group=_cmake_escape(group),
        require_cdash_submission="TRUE" if require_cdash_submission else "FALSE",
        test_options=test_options,
    )


def run_cdash(
    *,
    model: str,
    group: str,
    require_cdash_submission: bool,
) -> None:
    """Run configure, build, test, and optional strict submission via CTest."""
    ROCPROFILER_SDK_TESTS_BUILD_PATH.mkdir(parents=True, exist_ok=True)
    dashboard_path = ROCPROFILER_SDK_TESTS_BUILD_PATH / "dashboard.cmake"
    notes_path = ROCPROFILER_SDK_TESTS_BUILD_PATH / "dashboard-notes.txt"
    notes_path.write_text(_dashboard_notes(), encoding="utf-8")
    dashboard_path.write_text(
        _generate_dashboard(
            model=model,
            group=group,
            require_cdash_submission=require_cdash_submission,
            notes_file=notes_path,
        ),
        encoding="utf-8",
    )
    _run_command(
        [
            "ctest",
            "-S",
            str(dashboard_path),
            "--output-on-failure",
            "--no-tests=error",
        ]
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the installed rocprofiler-sdk package from TheRock."
    )
    parser.add_argument(
        "--enable-cdash",
        action="store_true",
        help="Run as a CTest dashboard and submit results to CDash.",
    )
    parser.add_argument(
        "--require-cdash-submission",
        action="store_true",
        help="Fail the job if CDash submission fails.",
    )
    parser.add_argument(
        "--cdash-model",
        choices=("Continuous", "Nightly", "Experimental"),
        default=os.getenv("THEROCK_CDASH_MODEL", _DEFAULT_CDASH_MODEL),
    )
    parser.add_argument(
        "--cdash-group",
        default=os.getenv("THEROCK_CDASH_GROUP", _DEFAULT_CDASH_GROUP),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_env()

    if args.enable_cdash:
        run_cdash(
            model=args.cdash_model,
            group=args.cdash_group,
            require_cdash_submission=args.require_cdash_submission,
        )
    else:
        cmake_config()
        cmake_build()
        execute_tests()
    return 0


if __name__ == "__main__":
    sys.exit(main())
