#!/usr/bin/env python3
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
"""
- Report rocprofiler SDK tests within TheRock to CDash.
- Generate CTestCustom.cmake (settings + configure/build commands) and dashboard.cmake (build, test, report to CDash).
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
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Define default project name and CDash base URL
_DEFAULT_PROJECT_NAME = "rocprofiler-sdk-alt"
_DEFAULT_BASE_URL = "my.cdash.org"


class _CMakeTemplate(string.Template):
    """Generated CMake snippets: only ``@python_key`` is expanded by ``.substitute()``.

    CMake ``${VAR}`` / ``${{VAR}}`` text stays literal (avoids Python f-string
    interpolation turning those into empty strings).
    """

    delimiter = "@"


@dataclass(frozen=True, slots=True)
class TheRockCiPaths:
    """Resolved install layout + env derived from ``THEROCK_BIN_PATH``."""

    THEROCK_BIN_PATH: Path
    THEROCK_PATH: Path
    THEROCK_LIB_PATH: Path
    THEROCK_SYSDEPS_PATH: Path
    THEROCK_SYSDEPS_LIB_PATH: Path
    THEROCK_LLVM_BIN_PATH: Path
    THEROCK_CLANG_PATH: Path
    THEROCK_CLANG_PLUS_PATH: Path
    ROCPROFILER_SDK_PATH: Path
    ROCPROFILER_SDK_TESTS_PATH: Path
    SOURCE_DIR: str
    BINARY_DIR: str
    environ_vars: dict[str, str]


def therock_ci_paths_from_bin(therock_bin_path: Path) -> TheRockCiPaths:
    """Build :class:`TheRockCiPaths` from the TheRock install *bin* directory."""
    THEROCK_BIN_PATH = Path(therock_bin_path).resolve()
    THEROCK_PATH = THEROCK_BIN_PATH.parent
    THEROCK_LIB_PATH = THEROCK_PATH / "lib"
    THEROCK_SYSDEPS_PATH = THEROCK_LIB_PATH / "rocm_sysdeps"
    THEROCK_SYSDEPS_LIB_PATH = THEROCK_SYSDEPS_PATH / "lib"
    THEROCK_LLVM_BIN_PATH = THEROCK_PATH / "llvm" / "bin"
    THEROCK_CLANG_PATH = THEROCK_LLVM_BIN_PATH / "amdclang"
    THEROCK_CLANG_PLUS_PATH = THEROCK_LLVM_BIN_PATH / "amdclang++"
    ROCPROFILER_SDK_PATH = THEROCK_PATH / "share" / "rocprofiler-sdk"
    ROCPROFILER_SDK_TESTS_PATH = ROCPROFILER_SDK_PATH / "tests"
    SOURCE_DIR = str(ROCPROFILER_SDK_TESTS_PATH)
    BINARY_DIR = str(ROCPROFILER_SDK_TESTS_PATH / "build")

    environ_vars = os.environ.copy()
    environ_vars["ROCM_PATH"] = os.path.realpath(str(THEROCK_PATH))
    environ_vars["HIP_PATH"] = os.path.realpath(str(THEROCK_PATH))
    environ_vars["ROCPROFILER_METRICS_PATH"] = str(ROCPROFILER_SDK_PATH)
    environ_vars["HIP_PLATFORM"] = "amd"
    environ_vars["THEROCK_BIN_DIR"] = str(THEROCK_BIN_PATH)
    old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "").split(":")
    environ_vars["LD_LIBRARY_PATH"] = ":".join(
        [str(THEROCK_LIB_PATH), str(THEROCK_SYSDEPS_LIB_PATH)] + old_ld_lib_path
    )

    return TheRockCiPaths(
        THEROCK_BIN_PATH=THEROCK_BIN_PATH,
        THEROCK_PATH=THEROCK_PATH,
        THEROCK_LIB_PATH=THEROCK_LIB_PATH,
        THEROCK_SYSDEPS_PATH=THEROCK_SYSDEPS_PATH,
        THEROCK_SYSDEPS_LIB_PATH=THEROCK_SYSDEPS_LIB_PATH,
        THEROCK_LLVM_BIN_PATH=THEROCK_LLVM_BIN_PATH,
        THEROCK_CLANG_PATH=THEROCK_CLANG_PATH,
        THEROCK_CLANG_PLUS_PATH=THEROCK_CLANG_PLUS_PATH,
        ROCPROFILER_SDK_PATH=ROCPROFILER_SDK_PATH,
        ROCPROFILER_SDK_TESTS_PATH=ROCPROFILER_SDK_TESTS_PATH,
        SOURCE_DIR=SOURCE_DIR,
        BINARY_DIR=BINARY_DIR,
        environ_vars=environ_vars,
    )


def _build_argument_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--therock-bin-path",
        type=Path,
        required=True,
        help="TheRock install bin directory (THEROCK_BIN_DIR)",
    )
    return parser


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
    """``ROCm/rocm-systems-<os>`` or ``ROCm/rocm-systems-<os>-<gpu>`` when ``THEROCK_CDASH_LABEL`` is unset.

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
            prefix = ""
    label = _default_cdash_matrix_label() or os.getenv("THEROCK_CDASH_LABEL")
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


def _generate_ctest_custom(
    cmake_cmd: str,
    paths: TheRockCiPaths,
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
        paths: Resolved TheRock install paths and ``environ_vars``-related layout.
        configure_cmd: Full configure shell command; if None, built from cmake_cmd.
        build_cmd: Full build shell command; if None, built from cmake_cmd.
        ctest_args_str: Arguments for ctest (CMAKE_CTEST_ARGUMENTS); if None, default.

    Returns:
        CMake script content for CTestCustom.cmake.
    """

    def _esc(s: str) -> str:
        """Escape special characters in a string for use in a CMake script."""
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # Configure cmake commands and ctest arguments
    if configure_cmd is None:
        # Must use explicit -S/-B: CTest runs this with cwd=binary dir (CMake 3.14+).
        configure_cmd = (
            f"{cmake_cmd} -S {paths.SOURCE_DIR} -B {paths.BINARY_DIR} --fresh -G Ninja "
            f"-DCMAKE_PREFIX_PATH={paths.THEROCK_PATH};{paths.THEROCK_SYSDEPS_PATH} "
            f"-DCMAKE_HIP_COMPILER={paths.THEROCK_CLANG_PLUS_PATH} "
            f"-DCMAKE_C_COMPILER={paths.THEROCK_CLANG_PATH} "
            f"-DCMAKE_CXX_COMPILER={paths.THEROCK_CLANG_PLUS_PATH} "
            f"-DPython3_EXECUTABLE={sys.executable}"
        )
    if build_cmd is None:
        build_cmd = f"{cmake_cmd} --build {paths.BINARY_DIR} -j"
    if ctest_args_str is None:
        ctest_args_str = f"--test-dir {paths.BINARY_DIR} --output-on-failure -j {os.cpu_count() or 1}"

    return _CMakeTemplate(
        """# CTestCustom.cmake content for ROCProfiler SDK tests. Generated by run-therock-ci.py.
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
set(CMAKE_CTEST_ARGUMENTS "@ctest_args")

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
        ctest_args=_esc(ctest_args_str),
        site=os.getenv("RUNNER_NAME") or os.getenv("HOSTNAME") or socket.gethostname(),
        build_name=_esc(_cdash_build_name()),
        source_directory=paths.SOURCE_DIR,
        binary_directory=paths.BINARY_DIR,
        configure_command=_esc(configure_cmd),
        build_command=_esc(build_cmd),
    )


def _generate_dashboard(paths: TheRockCiPaths) -> str:
    """Generate dashboard.cmake for CDash.

    Script includes CTestCustom.cmake, then runs configure, build, test,
    and submit stages.

    Args:
        paths: Resolved TheRock install paths (uses ``BINARY_DIR``).

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
        ctest_test(BUILD "@BINARY_DIR" RETURN_VALUE _test_ret)
        dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)
        handle_error("Testing" _test_ret)
    endif()

    dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
    """
    ).substitute(
        submit="1",
        model="Experimental",
        group="TheRock",
        repo_source_dir=paths.THEROCK_BIN_PATH,
        BINARY_DIR=paths.BINARY_DIR,
    )


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

    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    paths = therock_ci_paths_from_bin(args.therock_bin_path)

    # Get path to cmake executable
    cmake_cmd = _which_cmake()

    # Create binary directory if it doesn't exist
    os.makedirs(paths.BINARY_DIR, exist_ok=True)

    # Generate CTestCustom.cmake and dashboard.cmake scripts
    ctest_custom = _generate_ctest_custom(
        cmake_cmd,
        paths,
        configure_cmd=args.configure_cmd,
        build_cmd=args.build_cmd,
        ctest_args_str=args.ctest_args,
    )
    dashboard = _generate_dashboard(paths)

    # Write CTestCustom.cmake and dashboard.cmake scripts to binary directory
    ctest_custom_path = os.path.join(paths.BINARY_DIR, "CTestCustom.cmake")
    dashboard_path = os.path.join(paths.BINARY_DIR, "dashboard.cmake")

    with open(ctest_custom_path, "w") as f:
        f.write(ctest_custom)

    with open(dashboard_path, "w") as f:
        f.write(dashboard)

    ctest_cmd = _which_ctest()

    # Configure ctest run command
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
        r = subprocess.run(
            ctest_argv, cwd=paths.SOURCE_DIR, check=True, env=paths.environ_vars
        )
        return r.returncode

    # Log error
    except subprocess.CalledProcessError as e:
        logging.error(f"ctest failed: {e}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
