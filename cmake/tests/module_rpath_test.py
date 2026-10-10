# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and load a native module using the subproject RPATH hook."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="ELF RPATH test")
@pytest.mark.parametrize("opt_out", ["none", "target", "project"])
def test_module_rpath(tmp_path: Path, opt_out: str) -> None:
    source = tmp_path / "source"
    build = tmp_path / "build"
    install = tmp_path / "install"
    source.mkdir()
    (source / "runtime.c").write_text("int value(void) { return 42; }\n")
    (source / "module.c").write_text(
        "extern int value(void); int module_value(void) { return value(); }\n"
    )
    (source / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.25)
project(module_rpath C)
set(THEROCK_SOURCE_DIR "{Path(__file__).resolve().parents[2].as_posix()}")
set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
set(THEROCK_STAGE_STAMP_FILE "${{CMAKE_BINARY_DIR}}/stage.stamp")
set(THEROCK_PRIVATE_BUILD_RPATH_DIRS "${{CMAKE_BINARY_DIR}}/runtime")
set(THEROCK_INSTALL_RPATH_LIBRARY_DIR "${{CMAKE_INSTALL_PREFIX}}/lib")
set(THEROCK_PRIVATE_INSTALL_RPATH_DIRS "${{CMAKE_INSTALL_PREFIX}}/lib/runtime")
add_library(runtime_fixture SHARED runtime.c)
set_target_properties(runtime_fixture PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/runtime")
add_library(module_fixture MODULE module.c)
target_link_libraries(module_fixture PRIVATE runtime_fixture)
if("{opt_out}" STREQUAL "target")
  set_target_properties(module_fixture PROPERTIES THEROCK_NO_INSTALL_RPATH ON)
elseif("{opt_out}" STREQUAL "project")
  set(THEROCK_NO_INSTALL_RPATH ON)
endif()
include("${{THEROCK_SOURCE_DIR}}/cmake/therock_global_post_subproject.cmake")
get_target_property(build_rpath module_fixture BUILD_RPATH)
get_target_property(install_rpath module_fixture INSTALL_RPATH)
file(WRITE "${{CMAKE_BINARY_DIR}}/rpaths.txt" "${{build_rpath}}\n${{install_rpath}}\n")
install(TARGETS runtime_fixture LIBRARY DESTINATION lib/runtime)
install(TARGETS module_fixture LIBRARY DESTINATION lib)
"""
    )
    for args in (
        [
            "-S",
            str(source),
            "-B",
            str(build),
            "-GNinja",
            f"-DCMAKE_INSTALL_PREFIX={install}",
        ],
        ["--build", str(build)],
        ["--install", str(build)],
    ):
        result = subprocess.run(["cmake", *args], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    build_rpath, install_rpath = (build / "rpaths.txt").read_text().splitlines()
    assert (str(build / "runtime") in build_rpath) == (opt_out != "project")
    assert install_rpath == ("$ORIGIN/runtime" if opt_out == "none" else "")
    # Separate processes avoid reusing a dependency loaded from the build tree.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD")
    }
    for module, succeeds in (
        (build / "libmodule_fixture.so", True),
        (install / "lib/libmodule_fixture.so", opt_out == "none"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import ctypes, sys; assert ctypes.CDLL(sys.argv[1]).module_value() == 42",
                str(module),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == succeeds, result.stderr
        if not succeeds:
            assert "libruntime_fixture.so" in result.stderr
