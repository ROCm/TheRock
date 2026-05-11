#!/usr/bin/env python3
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""This script is used to test the rocprofiler-sdk within TheRock, and optionally
uploads the results to CDash so they can be tracked alongside other CI
runs.
"""

import argparse
import logging
import os
import platform
import re
import shutil
import socket
import string
import subprocess
import sys
from pathlib import Path

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
SOURCE_DIR = str(ROCPROFILER_SDK_TESTS_PATH)
BINARY_DIR = str(ROCPROFILER_SDK_TESTS_PATH / "build")

# Define default project name and CDash base URL
_DEFAULT_PROJECT_NAME = "rocprofiler-sdk-alt"
_DEFAULT_BASE_URL = "my.cdash.org"

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
    environ_vars["ROCM_PATH"] = os.path.realpath(str(THEROCK_PATH))
    environ_vars["HIP_PATH"] = os.path.realpath(str(THEROCK_PATH))
    environ_vars["ROCPROFILER_METRICS_PATH"] = str(ROCPROFILER_SDK_PATH)
    environ_vars["HIP_PLATFORM"] = "amd"
    environ_vars["THEROCK_BIN_DIR"] = str(THEROCK_BIN_PATH)

    ld_lib_paths = [str(THEROCK_LIB_PATH), str(THEROCK_SYSDEPS_LIB_PATH)]

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


class _CMakeTemplate(string.Template):
    """Generated CMake snippets: only ``@python_key`` is expanded by ``.substitute()``.

    CMake ``${VAR}`` / ``${{VAR}}`` text stays literal (avoids Python f-string
    interpolation turning those into empty strings).
    """

    delimiter = "@"


def _os_release_id_version() -> str:
    """Short OS tag for CDash labels, e.g. ``ubuntu-22.04``, ``rhel-8.8``."""
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            data: dict[str, str] = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                data[k] = v.strip().strip('"')
        id_ = (data.get("ID") or "unknown").lower()
        ver = (data.get("VERSION_ID") or data.get("VERSION_CODENAME") or "").lower()
        if ver:
            return f"{id_}-{ver}"
        return id_
    except OSError:
        return platform.system().lower()


def _default_cdash_matrix_label() -> str:
    """Label format: ``ROCm/rocm-systems-<os>-<gpu>``

    * OS segment from ``/etc/os-release`` (or ``platform.system()``).
    * Optional GPU segment from ``THEROCK_CDASH_GPU`` (default empty). When set, e.g.
      ``ROCm/rocm-systems-rhel-8.8-mi325-core``; when empty, no trailing hyphen.
    """
    gpu = os.getenv("THEROCK_CDASH_GPU") or os.getenv("ARTIFACT_GROUP")
    os_part = _os_release_id_version()
    base = f"ROCm/TheRock/rocm-systems-{os_part}"
    if not gpu:
        return base
    return f"{base}-{gpu}"


def _cdash_build_name() -> str:
    """CDash build name: ``<label>`` or ``<label> [RUN_ID: <id>]`` when set.
    * Label from :func:`_default_cdash_matrix_label`.
    * If ``ARTIFACT_RUN_ID`` is non-empty, append `` [RUN_ID: ...]``.
    * If ``GITHUB_REF`` is a pull request, prefix the label with ``PR_<n>_``.
    * Otherwise, for a non-empty ``GITHUB_REF_NAME`` (e.g. manual ``workflow_dispatch``),
      use prefix ``Manual_`` and suffix ``[Branch: <sanitized>]`` (not used when
      ``GITHUB_REF`` is a pull request — PR builds only get the ``PR_<n>_`` prefix).

    Example::

        PR_4946_ROCm/TheRock/rocm-systems-rhel-8.8-mi325-core [RUN_ID: 24378824659]

    """
    ref = os.getenv("GITHUB_REF", "")
    m = re.match(r"refs/pull/(\d+)/", ref)
    prefix = f"PR_{m.group(1)}_" if m else ""
    safe = ""
    if not prefix:
        refname = os.getenv("GITHUB_REF_NAME", "").strip()
        if refname:
            safe = re.sub(r"[^\w.\-]+", "-", refname).strip("-")
            safe = f" [Branch: {safe}]"
            prefix = f"Manual_"
        else:
            prefix = "CI_" if os.getenv("CI") else "Local_"
    label = _default_cdash_matrix_label()
    run_key = (
        os.getenv("GITHUB_RUN_ID")
        or os.getenv("THEROCK_RUN_ID")
        or os.getenv("ARTIFACT_RUN_ID")
    )
    if not run_key:
        return f"{prefix}{label}{safe}"
    return f"{prefix}{label} [RUN_ID: {run_key}]{safe}"


def _which_cmake() -> str:
    """Return path to cmake executable, or 'cmake' if not found in PATH."""
    return shutil.which("cmake") or "cmake"


def _which_ctest() -> str:
    """Return path to ctest executable, or 'ctest' if not found in PATH."""
    return shutil.which("ctest") or "ctest"


def _generate_ctest_custom(cmake_cmd: str) -> str:
    """Generate CTestCustom.cmake: settings and configure/build commands.

    Uses four namespaces (script, configure, build, test). For
    ``CTEST_CONFIGURE_COMMAND``, CMake runs the command with cwd set to the
    *binary* directory; pass ``cmake -S <src> -B <build>`` with absolute paths.

    Args:
        cmake_cmd: Path or command name for the CMake executable.

    Returns:
        CMake script content for CTestCustom.cmake.
    """

    def _esc(s: str) -> str:
        """Escape special characters in a string for use in a CMake script."""
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # Must use explicit -S/-B: CTest runs this with cwd=binary dir (CMake 3.14+).
    configure_cmd = (
        f"{cmake_cmd} -S {SOURCE_DIR} -B {BINARY_DIR} --fresh -G Ninja "
        f"-DCMAKE_PREFIX_PATH={THEROCK_PATH};{THEROCK_SYSDEPS_PATH} "
        f"-DCMAKE_HIP_COMPILER={THEROCK_CLANG_PLUS_PATH} "
        f"-DCMAKE_C_COMPILER={THEROCK_CLANG_PATH} "
        f"-DCMAKE_CXX_COMPILER={THEROCK_CLANG_PLUS_PATH} "
        f"-DPython3_EXECUTABLE={sys.executable}"
    )
    if is_asan():
        # Preload ASan for standalone tests loading instrumented ROCm libraries.
        asan_runtime_library = get_asan_runtime_library()
        configure_cmd += (
            " -DROCPROFILER_MEMCHECK=AddressSanitizer "
            f"-DROCPROFILER_MEMCHECK_PRELOAD_ENV=LD_PRELOAD={asan_runtime_library} "
            f"-DROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE={asan_runtime_library}"
        )
    build_cmd = f"{cmake_cmd} --build {BINARY_DIR} --parallel 8"
    exclude_regex = "|".join(ASAN_EXCLUDED_TESTS) if is_asan() else "^$"

    return _CMakeTemplate(
        """# CTestCustom.cmake content for ROCProfiler SDK tests. Generated by run-therock-ci.py.
# This file is the single source of truth for ctest behavior; the Python
# wrapper invokes `ctest -S dashboard.cmake` and lets the dashboard script
# pick up these variables (parallelism, output, paths, etc.) automatically.
set(CTEST_PROJECT_NAME "@project_name")
set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_SUBMIT_URL "@submit_url")

set(CTEST_UPDATE_TYPE git)
set(CTEST_UPDATE_VERSION_ONLY TRUE)
set(CTEST_GIT_COMMAND "@git_command")
set(CTEST_GIT_INIT_SUBMODULES FALSE)

set(CTEST_OUTPUT_ON_FAILURE TRUE)
set(CTEST_USE_LAUNCHERS TRUE)
set(CTEST_PARALLEL_LEVEL 8)
set(CTEST_EXCLUDE_REGEX "@exclude_regex")

set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/usr/.*;/opt/.*;external/.*;samples/.*;tests/.*;.*/external/.*;.*/samples/.*;.*/tests/.*;.*/details/.*;.*/counters/parser/.*")

set(CTEST_MEMORYCHECK_TYPE "")
set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "")
set(CTEST_MEMORYCHECK_SANITIZER_OPTIONS "")

set(CTEST_SITE "@site")
set(CTEST_BUILD_NAME "@build_name")

set(CTEST_SOURCE_DIRECTORY "@source_directory")
set(CTEST_BINARY_DIRECTORY "@binary_directory")

set(CTEST_CONFIGURE_COMMAND "@configure_command")
set(CTEST_BUILD_COMMAND "@build_command")
set(CTEST_COVERAGE_COMMAND "@gcov_command")
"""
    ).substitute(
        project_name=_DEFAULT_PROJECT_NAME,
        submit_url=f"https://{_DEFAULT_BASE_URL}/submit.php?project={_DEFAULT_PROJECT_NAME}",
        git_command=shutil.which("git") or "git",
        gcov_command=shutil.which("gcov") or "gcov",
        exclude_regex=_esc(exclude_regex),
        site=os.getenv("RUNNER_NAME") or os.getenv("HOSTNAME") or socket.gethostname(),
        build_name=_esc(_cdash_build_name()),
        source_directory=SOURCE_DIR,
        binary_directory=BINARY_DIR,
        configure_command=_esc(configure_cmd),
        build_command=_esc(build_cmd),
    )


def _generate_dashboard(enable_cdash: bool) -> str:
    """Generate dashboard.cmake for CDash.

    Script includes CTestCustom.cmake, then runs configure, build, test,
    and (optionally) submit stages.

    Args:
        enable_cdash: When True, ``ctest_submit`` is invoked to upload results
            to CDash. When False, ``dashboard_submit`` is a no-op and ctest
            output remains local (terminal + ``Testing/Temporary/`` log files).

    Returns:
        CMake script content for dashboard.cmake.
    """

    return _CMakeTemplate(
        """
    cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

    macro(dashboard_submit)
        if("@submit" GREATER 0)
            ctest_submit(${ARGN}
                            RETRY_COUNT 0
                            RETRY_DELAY 10
                            CAPTURE_CMAKE_ERROR _cdash_submit_err)
        endif()
    endmacro()

    include("${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")

    macro(handle_error _message _ret)
        if(NOT ${${_ret}} EQUAL 0)
            dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
            message(FATAL_ERROR "${_message} failed: ${${_ret}}")
        endif()
    endmacro()

    set(STAGES "START;UPDATE;CONFIGURE;BUILD;TEST;SUBMIT")

    ctest_start(@model GROUP @group)
    ctest_update(SOURCE "@repo_source_dir" RETURN_VALUE _update_ret
                    CAPTURE_CMAKE_ERROR _update_err)
    ctest_configure(BUILD "@BINARY_DIR" RETURN_VALUE _configure_ret)
    dashboard_submit(PARTS Start Update Configure RETURN_VALUE _submit_ret)

    if(NOT _update_err EQUAL 0)
        message(WARNING "ctest_update failed")
    endif()

    handle_error("Configure" _configure_ret)

    if("BUILD" IN_LIST STAGES)
        ctest_build(BUILD "@BINARY_DIR" RETURN_VALUE _build_ret)
        dashboard_submit(PARTS Build RETURN_VALUE _submit_ret)
        handle_error("Build" _build_ret)
    endif()

    if("TEST" IN_LIST STAGES)
        ctest_test(BUILD "@BINARY_DIR"
                   EXCLUDE "${CTEST_EXCLUDE_REGEX}"
                   RETURN_VALUE _test_ret)
        dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)
        handle_error("Testing" _test_ret)
    endif()

    dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
    """
    ).substitute(
        submit="1" if enable_cdash else "0",
        model="Experimental",
        group="TheRock",
        repo_source_dir=THEROCK_BIN_PATH,
        BINARY_DIR=BINARY_DIR,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the rocprofiler SDK test runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run rocprofiler SDK ctest dashboard. By default, results stay "
            "local (terminal output + Testing/Temporary/ log files). Pass "
            "--enable-cdash to upload results to CDash."
        ),
    )
    parser.add_argument(
        "--enable-cdash",
        action="store_true",
        default=False,
        help=(
            "Submit ctest results to CDash. When omitted, ctest_submit is a "
            "no-op and output is kept local only."
        ),
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        default=True,
        help=(
            "Pass -V to ctest for verbose output (shows configure/build/test "
            "command output as it runs, not only on failure)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate CTest/dashboard scripts and run ctest for ROCProfiler SDK tests.

    Writes CTestCustom.cmake and dashboard.cmake into the binary dir, then
    runs ctest -S to configure, build, test, and (when ``--enable-cdash`` is
    set) submit to CDash.

    Returns:
        Exit code from ctest (0 on success).
    """
    args = _parse_args(argv)

    setup_env()

    cmake_cmd = _which_cmake()

    os.makedirs(BINARY_DIR, exist_ok=True)

    ctest_custom = _generate_ctest_custom(cmake_cmd)
    dashboard = _generate_dashboard(enable_cdash=args.enable_cdash)

    ctest_custom_path = os.path.join(BINARY_DIR, "CTestCustom.cmake")
    dashboard_path = os.path.join(BINARY_DIR, "dashboard.cmake")

    with open(ctest_custom_path, "w") as f:
        f.write(ctest_custom)

    with open(dashboard_path, "w") as f:
        f.write(dashboard)

    ctest_cmd = _which_ctest()

    # All ctest behavior (parallelism, output-on-failure, test/binary dirs,
    # etc.) is defined in CTestCustom.cmake, which dashboard.cmake includes.
    ctest_argv = [ctest_cmd, "-S", dashboard_path]

    if args.verbose:
        ctest_argv.append("-V")

    try:
        r = subprocess.run(ctest_argv, cwd=SOURCE_DIR, check=True, env=environ_vars)
        return r.returncode

    except subprocess.CalledProcessError as e:
        logging.error(f"ctest failed: {e}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
