#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Generate CTestCustom.cmake (settings + configure/build commands) and
dashboard.cmake (build, test, report to CDash).
"""

import argparse
import multiprocessing
import os
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

_DEFAULT_PROJECT_NAME = "rocprofiler-sdk-alt"
_DEFAULT_BASE_URL = "my.cdash.org"

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent

THEROCK_BIN_DIR = r'/home/tester/TheRock/therock-build/bin'
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
THEROCK_BIN_PATH = Path(THEROCK_BIN_DIR).resolve()
THEROCK_PATH = THEROCK_BIN_PATH.parent
THEROCK_LIB_PATH = str(THEROCK_PATH / "lib")

ROCPROFILER_SDK_DIRECTORY = f"{THEROCK_PATH}/share/rocprofiler-sdk"
ROCPROFILER_SDK_TESTS_DIRECTORY = f"{ROCPROFILER_SDK_DIRECTORY}/tests"

# Defaults; overridden by --source-dir/--binary-dir when provided
SOURCE_DIR = ROCPROFILER_SDK_TESTS_DIRECTORY
BINARY_DIR = f"{ROCPROFILER_SDK_TESTS_DIRECTORY}/build"

environ_vars = os.environ.copy()
environ_vars["ROCM_PATH"] = os.path.realpath(str(THEROCK_PATH))
environ_vars["HIP_PATH"] = os.path.realpath(str(THEROCK_PATH))

# Env setup
environ_vars["HIP_PLATFORM"] = "amd"

# Set up LD_LIBRARY_PATH
old_ld_lib_path = os.getenv("LD_LIBRARY_PATH", "")
sysdeps_path = f"{THEROCK_LIB_PATH}/rocm_sysdeps/lib"
if old_ld_lib_path:
    environ_vars["LD_LIBRARY_PATH"] = (
        f"{THEROCK_LIB_PATH}:{sysdeps_path}:{old_ld_lib_path}"
    )
else:
    environ_vars["LD_LIBRARY_PATH"] = f"{THEROCK_LIB_PATH}:{sysdeps_path}"


def _which_cmake():
    return shutil.which("cmake") or "cmake"


def _which_ctest():
    return shutil.which("ctest") or "ctest"


def _generate_ctest_custom(cmake_cmd):
    """Generate CTestCustom.cmake: settings and configure/build commands.
    Uses four namespaces (script, configure, build, test). Commands run from source dir
    so literal "build" and "." match test_rocprofiler_sdk.
    """

    # Configure cmake commands and ctest arguments (semicolon after CMAKE_PREFIX_PATH
    # so the next -D is not merged into the path)
    configure_cmd = (
        f"{cmake_cmd} -B {BINARY_DIR} -G Ninja "
        f"-DCMAKE_PREFIX_PATH={THEROCK_PATH};{THEROCK_LIB_PATH}/rocm_sysdeps "
        f"-DCMAKE_HIP_COMPILER={THEROCK_PATH}/llvm/bin/amdclang++ "
        f"-DCMAKE_C_COMPILER={THEROCK_PATH}/llvm/bin/amdclang "
        f"-DCMAKE_CXX_COMPILER={THEROCK_PATH}/llvm/bin/amdclang++ ."
    )
    build_cmd = f'{cmake_cmd} --build {BINARY_DIR} -j'
    ctest_args_str = f'--test-dir {BINARY_DIR} --output-on-failure -j {os.cpu_count() or 1}'

    # Specify CDash submission information. Include a unique run ID in the build
    # name so each run appears as a separate build on the dashboard.
    run_id = os.getenv("GITHUB_RUN_ID") or os.getenv("THEROCK_RUN_ID")
    if not run_id:
        run_id = f"local-{uuid.uuid4().hex}"
    NAME = f"ROCProfiler SDK Tests - {run_id}"

    URL = f'https://{_DEFAULT_BASE_URL}/submit.php?project={_DEFAULT_PROJECT_NAME}'
    SITE = socket.gethostname()

    return f"""set(CTEST_PROJECT_NAME "{_DEFAULT_PROJECT_NAME}")
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
set(CMAKE_CTEST_ARGUMENTS "{ctest_args_str}")

set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_ERRORS "100")
set(CTEST_CUSTOM_MAXIMUM_NUMBER_OF_WARNINGS "100")
set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE "51200")
set(CTEST_CUSTOM_COVERAGE_EXCLUDE "/usr/.*;/opt/.*;external/.*;samples/.*;tests/.*;.*/external/.*;.*/samples/.*;.*/tests/.*;.*/details/.*;.*/counters/parser/.*")

set(CTEST_MEMORYCHECK_TYPE "")
set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "")
set(CTEST_MEMORYCHECK_SANITIZER_OPTIONS "")

set(CTEST_SITE "{SITE}")
set(CTEST_BUILD_NAME "{NAME}")

set(CTEST_SOURCE_DIRECTORY "{SOURCE_DIR}")
set(CTEST_BINARY_DIRECTORY "{BINARY_DIR}")

set(CTEST_CONFIGURE_COMMAND "{configure_cmd}")
set(CTEST_BUILD_COMMAND "{build_cmd}")
set(CTEST_COVERAGE_COMMAND "{shutil.which('gcov') or 'gcov'}")
"""

def _generate_dashboard(cmake_cmd):
    """Generate dashboard.cmake: include CTestCustom, then run configure/build/test/submit."""
    submit = "1"
    mode = 'Experimental'
    ARGN = "${ARGN}"

    REPO_SOURCE_DIR = (
        os.path.dirname(os.path.dirname((SOURCE_DIR)))
        if not os.path.exists(os.path.join(SOURCE_DIR, ".git"))
        else SOURCE_DIR
    )

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

    _script += """
    include("${CMAKE_CURRENT_LIST_DIR}/CTestCustom.cmake")

    macro(handle_error _message _ret)
        if(NOT ${${_ret}} EQUAL 0)
            dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
            message(FATAL_ERROR "${_message} failed: ${${_ret}}")
        endif()
    endmacro()
    """

    _script += f"""
    set(STAGES "START;CONFIGURE;BUILD;TEST;SUBMIT")

    ctest_start({mode})
    # ctest_update(SOURCE "{REPO_SOURCE_DIR}" RETURN_VALUE _update_ret
    #                 CAPTURE_CMAKE_ERROR _update_err)
    ctest_configure(BUILD "{BINARY_DIR}" RETURN_VALUE _configure_ret)
    dashboard_submit(PARTS Start Configure RETURN_VALUE _submit_ret)

    # if(NOT _update_err EQUAL 0)
    #     message(WARNING "ctest_update failed")
    # endif()

    handle_error("Configure" _configure_ret)

    if("BUILD" IN_LIST STAGES)
        ctest_build(BUILD "{BINARY_DIR}" RETURN_VALUE _build_ret)
        dashboard_submit(PARTS Build RETURN_VALUE _submit_ret)
        handle_error("Build" _build_ret)
    endif()

    if("TEST" IN_LIST STAGES)
        ctest_test(BUILD "{BINARY_DIR}" RETURN_VALUE _test_ret)
        dashboard_submit(PARTS Test RETURN_VALUE _submit_ret)
    endif()

    # handle_error("Testing" _test_ret)
    dashboard_submit(PARTS Done RETURN_VALUE _submit_ret)
    """

    return _script

def main(argv=None):

    cmake_cmd = _which_cmake()
    os.makedirs(BINARY_DIR, exist_ok=True)
    
    ctest_custom = _generate_ctest_custom(cmake_cmd)

    dashboard = _generate_dashboard(cmake_cmd)


    ctest_custom_path = os.path.join(BINARY_DIR, "CTestCustom.cmake")
    dashboard_path = os.path.join(BINARY_DIR, "dashboard.cmake")
    
    with open(ctest_custom_path, "w") as f:
        f.write(ctest_custom)
    
    with open(dashboard_path, "w") as f:
        f.write(dashboard)

    ctest_cmd = _which_ctest()

    # Run from source dir so --test-dir build matches test_rocprofiler_sdk
    ctest_argv = [
        ctest_cmd, 
        "-S", 
        dashboard_path,
        "--test-dir", 
        "build", 
        "--output-on-failure"]


    r = subprocess.run(ctest_argv, cwd=SOURCE_DIR, check=True, env=environ_vars)
    
    return r.returncode

if __name__ == "__main__":
    sys.exit(main() or 0)
