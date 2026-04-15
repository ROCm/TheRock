#!/usr/bin/env python3
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""
- Report rocprofiler SDK tests within TheRock to CDash.
- Generate CTestCustom.cmake (settings + configure/build commands) and dashboard.cmake (build, test, report to CDash).
"""

import argparse
import logging
import multiprocessing
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

# Paths match test_rocprofiler_sdk.py (resolved from THEROCK_BIN_DIR / default install).
from test_rocprofiler_sdk import (
    ROCPROFILER_SDK_PATH,
    ROCPROFILER_SDK_TESTS_PATH,
    THEROCK_BIN_PATH,
    THEROCK_CLANG_PATH,
    THEROCK_CLANG_PLUS_PATH,
    THEROCK_LIB_PATH,
    THEROCK_SYSDEPS_LIB_PATH,
    THEROCK_SYSDEPS_PATH,
    THEROCK_PATH,
)

logging.basicConfig(level=logging.INFO)

# Define default project name and CDash base URL
_DEFAULT_PROJECT_NAME = "rocprofiler-sdk-alt"
_DEFAULT_BASE_URL = "my.cdash.org"

# Define paths to output artifacts directory and script directory
OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")

# Defaults; overridden by --source-dir/--binary-dir when provided
SOURCE_DIR = str(ROCPROFILER_SDK_TESTS_PATH)
BINARY_DIR = str(ROCPROFILER_SDK_TESTS_PATH / "build")

# Set up environment variables
environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = os.path.realpath(str(THEROCK_PATH))
environ_vars["HIP_PATH"] = os.path.realpath(str(THEROCK_PATH))
environ_vars["ROCPROFILER_METRICS_PATH"] = str(ROCPROFILER_SDK_PATH)
environ_vars["HIP_PLATFORM"] = "amd"
environ_vars["THEROCK_BIN_DIR"] = str(THEROCK_BIN_PATH)

# Set up LD_LIBRARY_PATH (same layout as test_rocprofiler_sdk.setup_env).
old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "").split(":")
environ_vars["LD_LIBRARY_PATH"] = ":".join(
    [str(THEROCK_LIB_PATH), str(THEROCK_SYSDEPS_LIB_PATH)] + old_ld_lib_path
)


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
    """``ROCm/rocm-systems-<os>`` or ``ROCm/rocm-systems-<os>-<gpu>`` when ``ARTIFACT_GROUP`` is unset.

    * OS segment from ``/etc/os-release`` (or ``platform.system()``).
    * Optional GPU segment from ``ARTIFACT_GROUP`` (default empty). When set, e.g.
      ``ROCm/rocm-systems-rhel-8.8-mi325-core``; when empty, no trailing hyphen.
    """
    gpu = os.getenv("ARTIFACT_GROUP", "")
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

    Example::

        PR_4946_ROCm/TheRock/rocm-systems-rhel-8.8-mi325-core [RUN_ID: 24378824659]

    """
    ref = os.getenv("GITHUB_REF", "")
    m = re.match(r"refs/pull/(\d+)/", ref)
    prefix = f"PR_{m.group(1)}_" if m else ""
    label = _default_cdash_matrix_label()
    run_key = (
        os.getenv("GITHUB_RUN_ID")
        or os.getenv("THEROCK_RUN_ID")
        or os.getenv("ARTIFACT_RUN_ID")
    )
    if not run_key:
        return f"{prefix}{label}"
    return f"{prefix}{label} [RUN_ID: {run_key}]"


def _therock_git_repo_root_for_ctest_update() -> str:
    """Local root of the ROCm/TheRock clone for ``ctest_update`` (a Git work tree). TESTING
    GIT COMPARISON HERE
    The default ``SOURCE_DIR`` points at installed ``share/rocprofiler-sdk/tests``,
    which is not the superproject git checkout. This resolves the TheRock repo root.
    """
    for env_var in ("GITHUB_WORKSPACE", "THEROCK_SOURCE_DIR"):
        raw = os.getenv(env_var, "").strip()
        if not raw:
            continue
        candidate = Path(raw).resolve()
        if (candidate / ".git").exists():
            return str(candidate)

    script_superproject = Path(__file__).resolve().parents[3]
    if (script_superproject / ".git").exists():
        return str(script_superproject)

    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(Path(__file__).resolve().parent),
                "rev-parse",
                "--show-toplevel",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0 and (top := proc.stdout.strip()):
            return str(Path(top).resolve())
    except (OSError, subprocess.TimeoutExpired):
        pass

    return str(script_superproject)


def _which_cmake() -> str:
    """Return path to cmake executable, or 'cmake' if not found in PATH."""
    return shutil.which("cmake") or "cmake"


def _which_ctest() -> str:
    """Return path to ctest executable, or 'ctest' if not found in PATH."""
    return shutil.which("ctest") or "ctest"


def _generate_ctest_custom(
    cmake_cmd: str,
    *,
    configure_cmd: str | None = None,
    build_cmd: str | None = None,
    ctest_args_str: str | None = None,
) -> str:
    """Generate CTestCustom.cmake: settings and configure/build commands.

    Uses four namespaces (script, configure, build, test). For
    ``CTEST_CONFIGURE_COMMAND``, CMake runs the command with cwd set to the
    *binary* directory; pass ``cmake -S <src> -B <build>`` with absolute paths
    (or ``test_rocprofiler_sdk.get_cmake_config_cmd()``).

    Args:
        cmake_cmd: Path or command name for the CMake executable.
        configure_cmd: Full configure shell command; if None, built from cmake_cmd.
        build_cmd: Full build shell command; if None, built from cmake_cmd.
        ctest_args_str: Arguments for ctest (CMAKE_CTEST_ARGUMENTS); if None, default.

    Returns:
        CMake script content for CTestCustom.cmake.
    """

    def _esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # Configure cmake commands and ctest arguments
    if configure_cmd is None:
        # Must use explicit -S/-B: CTest runs this with cwd=binary dir (CMake 3.14+).
        configure_cmd = (
            f"{cmake_cmd} -S {SOURCE_DIR} -B {BINARY_DIR} --fresh -G Ninja "
            f"-DCMAKE_PREFIX_PATH={THEROCK_PATH};{THEROCK_SYSDEPS_PATH} "
            f"-DCMAKE_HIP_COMPILER={THEROCK_CLANG_PLUS_PATH} "
            f"-DCMAKE_C_COMPILER={THEROCK_CLANG_PATH} "
            f"-DCMAKE_CXX_COMPILER={THEROCK_CLANG_PLUS_PATH} "
            f"-DPython3_EXECUTABLE={sys.executable}"
        )
    if build_cmd is None:
        build_cmd = f"{cmake_cmd} --build {BINARY_DIR} -j"
    if ctest_args_str is None:
        ctest_args_str = (
            f"--test-dir {BINARY_DIR} --output-on-failure -j {os.cpu_count() or 1}"
        )

    # CDash build name: <label> or <label> [RUN_ID: ...] (see _cdash_build_name).
    NAME = _cdash_build_name()

    # Specify dashboard URL and site/host name for CDash submission
    URL = f"https://{_DEFAULT_BASE_URL}/submit.php?project={_DEFAULT_PROJECT_NAME}"
    SITE = socket.gethostname()

    return f"""# CTestCustom.cmake content for ROCProfiler SDK tests. Generated by run-therock-ci.py.
set(CTEST_PROJECT_NAME "{_DEFAULT_PROJECT_NAME}")
set(CTEST_NIGHTLY_START_TIME "05:00:00 UTC")

set(CTEST_DROP_METHOD "https")
set(CTEST_DROP_SITE_CDASH TRUE)
set(CTEST_SUBMIT_URL "{URL}")

set(CTEST_UPDATE_TYPE git)
set(CTEST_UPDATE_VERSION_ONLY TRUE)
set(CTEST_GIT_COMMAND "{shutil.which('git') or 'git'}")
set(CTEST_GIT_INIT_SUBMODULES FALSE)

set(CTEST_OUTPUT_ON_FAILURE TRUE)
set(CTEST_USE_LAUNCHERS TRUE)
set(CMAKE_CTEST_ARGUMENTS "{_esc(ctest_args_str)}")

set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/usr/.*;/opt/.*;external/.*;samples/.*;tests/.*;.*/external/.*;.*/samples/.*;.*/tests/.*;.*/details/.*;.*/counters/parser/.*")

set(CTEST_MEMORYCHECK_TYPE "")
set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "")
set(CTEST_MEMORYCHECK_SANITIZER_OPTIONS "")

set(CTEST_SITE "{SITE}")
set(CTEST_BUILD_NAME "{_esc(NAME)}")

set(CTEST_SOURCE_DIRECTORY "{SOURCE_DIR}")
set(CTEST_BINARY_DIRECTORY "{BINARY_DIR}")

set(CTEST_CONFIGURE_COMMAND "{_esc(configure_cmd)}")
set(CTEST_BUILD_COMMAND "{_esc(build_cmd)}")
set(CTEST_COVERAGE_COMMAND "{shutil.which('gcov') or 'gcov'}")
"""


def _generate_dashboard(cmake_cmd: str) -> str:
    """Generate dashboard.cmake for CDash.

    Script includes CTestCustom.cmake, then runs configure, build, test,
    and submit stages.

    Args:
        cmake_cmd: Path or command name for the CMake executable.

    Returns:
        CMake script content for dashboard.cmake.
    """

    # Define variables for dashboard submission
    submit = "1"  # Submit to CDash or not
    model = "Experimental"  # Mode (Nightly, Experimental, Continuous, etc.)
    group = "TheRock"  # Group for the dashboard
    ARGN = "${ARGN}"  # Arguments for dashboard submission

    REPO_SOURCE_DIR = _therock_git_repo_root_for_ctest_update()

    # Generate initial dashboard.cmake content with minimum necessary version and dashboard_submit macro
    _script = f"""
    cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

    macro(dashboard_submit)
        if("{submit}" GREATER 0)
            ctest_submit({ARGN}
                            RETRY_COUNT 0
                            RETRY_DELAY 10
                            CAPTURE_CMAKE_ERROR _cdash_submit_err)
        endif()
    endmacro()
    """

    # Include custom CTest config and define a macro for standardized error handling and submit on error
    _script += """
    include("${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")

    macro(handle_error _message _ret)
        if(NOT ${${_ret}} EQUAL 0)
            dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
            message(FATAL_ERROR "${_message} failed: ${${_ret}}")
        endif()
    endmacro()
    """

    # Run stages for configure, build, test, and submit to CDash
    _script += f"""
    set(STAGES "START;UPDATE;CONFIGURE;BUILD;TEST;SUBMIT")

    ctest_start({model} GROUP {group})
    ctest_update(SOURCE "{REPO_SOURCE_DIR}" RETURN_VALUE _update_ret
                    CAPTURE_CMAKE_ERROR _update_err)
    ctest_configure(BUILD "{BINARY_DIR}" RETURN_VALUE _configure_ret)
    dashboard_submit(PARTS Start Update Configure RETURN_VALUE _submit_ret)

    if(NOT _update_err EQUAL 0)
        message(WARNING "ctest_update failed")
    endif()

    handle_error("Configure" _configure_ret)

    if("BUILD" IN_LIST STAGES)
        ctest_build(BUILD "{BINARY_DIR}" RETURN_VALUE _build_ret)
        dashboard_submit(PARTS Build RETURN_VALUE _submit_ret)
        handle_error("Build" _build_ret)
    endif()

    if("TEST" IN_LIST STAGES)
        ctest_test(BUILD "{BINARY_DIR}" RETURN_VALUE _test_ret)
        dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)
        if(NOT _test_ret EQUAL 0)
            if(_test_ret EQUAL -1)
                message(WARNING "ctest_test finished with code ${{_test_ret}} (test failures); not treating as dashboard failure")
            else()
                handle_error("Testing" _test_ret)
            endif()
        endif()
    endif()

    dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
    """

    return _script


def main(argv: list[str] | None = None) -> int:
    """Generate CTest/dashboard scripts and run ctest for ROCProfiler SDK tests.

    Writes CTestCustom.cmake and dashboard.cmake into the binary dir, then
    runs ctest -S to configure, build, test, and submit to CDash.

    Args:
        argv: Optional command-line arguments. When passed from test_rocprofiler_sdk,
            use --configure-cmd, --build-cmd, and --ctest-args (shell-joined strings).

    Returns:
        Exit code from ctest (0 on success).
    """

    parser = argparse.ArgumentParser(
        description="CDash dashboard run for ROCProfiler SDK tests in TheRock.",
    )
    parser.add_argument(
        "--configure-cmd",
        metavar="CMD",
        help="Full configure command line (CTEST_CONFIGURE_COMMAND), shell form",
    )
    parser.add_argument(
        "--build-cmd",
        metavar="CMD",
        help="Full build command line (CTEST_BUILD_COMMAND), shell form",
    )
    parser.add_argument(
        "--ctest-args",
        metavar="ARGS",
        help="Arguments for ctest (CMAKE_CTEST_ARGUMENTS), without leading 'ctest'",
    )
    args = parser.parse_args(argv)

    # Get path to cmake executable
    cmake_cmd = _which_cmake()

    # Create binary directory if it doesn't exist
    os.makedirs(BINARY_DIR, exist_ok=True)

    # Generate CTestCustom.cmake and dashboard.cmake scripts
    ctest_custom = _generate_ctest_custom(
        cmake_cmd,
        configure_cmd=args.configure_cmd,
        build_cmd=args.build_cmd,
        ctest_args_str=args.ctest_args,
    )
    dashboard = _generate_dashboard(cmake_cmd)

    # Write CTestCustom.cmake and dashboard.cmake scripts to binary directory
    ctest_custom_path = os.path.join(BINARY_DIR, "CTestCustom.cmake")
    dashboard_path = os.path.join(BINARY_DIR, "dashboard.cmake")

    with open(ctest_custom_path, "w") as f:
        f.write(ctest_custom)

    with open(dashboard_path, "w") as f:
        f.write(dashboard)

    ctest_cmd = _which_ctest()

    # Configure ctest run commmand
    ctest_argv = [
        ctest_cmd,
        "-S",
        dashboard_path,
        "--test-dir",
        "build",
        "--output-on-failure",
    ]

    # Run ctest with the generated dashboard.cmake script
    try:
        r = subprocess.run(ctest_argv, cwd=SOURCE_DIR, check=True, env=environ_vars)

    # Log error
    except subprocess.CalledProcessError as e:
        logging.error(f"ctest failed: {e}")


if __name__ == "__main__":
    main()
    sys.exit(0)
