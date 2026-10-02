#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise sanitizer toolchain settings and module RPATHs with small CMake projects."""

import ctypes
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


THEROCK_ROOT = Path(__file__).resolve().parents[2]


def write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents), encoding="utf-8")


def cmake(*args: str) -> None:
    result = subprocess.run(
        ["cmake", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(result.stdout)


def configure(source: Path, build: Path, *args: str) -> None:
    cmake("-S", str(source), "-B", str(build), "-G", "Ninja", *args)


class TsanCMakeTest(unittest.TestCase):
    def test_sanitizer_flags_and_gpu_targets_reach_child_project(self) -> None:
        cases = (
            ("TSAN", "thread", True, "gfx942;gfx950"),
            ("HOST_ASAN", "address", True, "gfx942;gfx950"),
            ("ASAN", "address", False, "gfx942:xnack+;gfx950:xnack+"),
        )
        for sanitizer, kind, host_only, expected_targets in cases:
            with self.subTest(
                sanitizer=sanitizer
            ), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_file(
                    root / "parent" / "CMakeLists.txt",
                    f"""
                    cmake_minimum_required(VERSION 3.25)
                    project(parent NONE)
                    include("{(THEROCK_ROOT / 'cmake' / 'therock_sanitizers.cmake').as_posix()}")
                    set(THEROCK_SANITIZER {sanitizer})
                    therock_sanitizer_configure(stanza selected clang++ TRUE child)
                    file(WRITE "${{CMAKE_BINARY_DIR}}/toolchain.cmake" "${{stanza}}")
                    """,
                )
                write_file(
                    root / "child" / "CMakeLists.txt",
                    """
                    cmake_minimum_required(VERSION 3.25)
                    project(child NONE)
                    set(GPU_TARGETS "gfx942;gfx950")
                    foreach(language C CXX HIP)
                      set(CMAKE_${language}_COMPILER_LAUNCHER "ccache;--verbose")
                    endforeach()
                    include("${TOOLCHAIN_FILE}")
                    get_directory_property(link_options LINK_OPTIONS)
                    file(WRITE "${CMAKE_BINARY_DIR}/observed.txt"
                      "${CMAKE_C_FLAGS_INIT}\n${CMAKE_CXX_FLAGS_INIT}\n"
                      "${CMAKE_HIP_FLAGS_INIT}\n"
                      "${CMAKE_C_COMPILER_LAUNCHER}\n"
                      "${CMAKE_CXX_COMPILER_LAUNCHER}\n"
                      "${CMAKE_HIP_COMPILER_LAUNCHER}\n"
                      "${GPU_TARGETS}\n${link_options}\n")
                    """,
                )
                configure(root / "parent", root / "parent-build")
                configure(
                    root / "child",
                    root / "child-build",
                    f"-DTOOLCHAIN_FILE={root / 'parent-build' / 'toolchain.cmake'}",
                )
                c_flags, cxx_flags, hip_flags, *observed = (
                    (root / "child-build" / "observed.txt")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                *launchers, gpu_targets, link_options = observed
                prefix = "-Xarch_host " if host_only else ""
                for flags in (c_flags, cxx_flags):
                    self.assertIn(f"{prefix}-fsanitize={kind}", flags)
                    self.assertIn(f"{prefix}-fno-omit-frame-pointer", flags)
                if host_only:
                    self.assertIn(f"-Xarch_host -fsanitize={kind}", hip_flags)
                for launcher in launchers:
                    parts = launcher.split(";")
                    self.assertEqual(parts[-2:], ["ccache", "--verbose"])
                    if host_only:
                        self.assertEqual(parts[1:-2], ["-E", "env", "CCACHE_DISABLE=1"])
                    else:
                        self.assertEqual(len(parts), 2)
                self.assertEqual(gpu_targets, expected_targets)
                self.assertIn(f"-fsanitize={kind}", link_options)

    def test_debug_opt_out_preserves_other_toolchain_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_file(
                root / "parent" / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.25)
                project(parent NONE)
                set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
                list(APPEND CMAKE_MODULE_PATH "${{THEROCK_SOURCE_DIR}}/cmake")
                include(therock_globals)
                include(therock_sanitizers)
                include(therock_flag_utils)
                include(therock_default_targets)
                include(therock_subproject)
                set(THEROCK_DEBUG_INFO_FLAGS "${{DEBUG_INFO_FLAGS}}")
                set(CMAKE_C_FLAGS "-O2 -DKEEP -Xarch_host -fsanitize=thread")
                set(CMAKE_CXX_FLAGS "-O2 -DKEEP -Xarch_host -fsanitize=thread")
                foreach(name with_debug without_debug)
                  add_custom_target(${{name}})
                  set_target_properties(${{name}} PROPERTIES THEROCK_AMDGPU_TARGETS gfx942)
                  _therock_cmake_subproject_setup_toolchain(
                    ${{name}} "" "${{CMAKE_BINARY_DIR}}/${{name}}.cmake")
                endforeach()
                """,
            )
            write_file(
                root / "child" / "CMakeLists.txt",
                """
                cmake_minimum_required(VERSION 3.25)
                project(child NONE)
                include("${TOOLCHAIN_FILE}")
                file(WRITE "${CMAKE_BINARY_DIR}/observed.txt"
                  "${CMAKE_C_FLAGS_INIT}\n${CMAKE_CXX_FLAGS_INIT}\n")
                """,
            )
            configure(
                root / "parent",
                root / "parent-build",
                "-DDEBUG_INFO_FLAGS=-g1 -gdwarf-4",
                "-Dwithout_debug_GENERATE_DEBUG_INFO=OFF",
            )
            for name in ("with_debug", "without_debug"):
                with self.subTest(name=name):
                    build = root / f"{name}-build"
                    configure(
                        root / "child",
                        build,
                        f"-DTOOLCHAIN_FILE={root / 'parent-build' / f'{name}.cmake'}",
                    )
                    flags = (build / "observed.txt").read_text(encoding="utf-8")
                    self.assertIn("-O2 -DKEEP", flags)
                    self.assertIn("-Xarch_host -fsanitize=thread", flags)
                    if name == "without_debug":
                        self.assertNotIn("-g1", flags)
                        self.assertNotIn("-gdwarf-4", flags)
                    else:
                        self.assertIn("-g1 -gdwarf-4", flags)

    @unittest.skipUnless(sys.platform.startswith("linux"), "ELF RPATH test")
    def test_native_module_loads_from_install_rpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            build = root / "build"
            write_file(source / "runtime.c", "int value(void) { return 42; }\n")
            write_file(
                source / "module.c",
                "extern int value(void); int module_value(void) { return value(); }\n",
            )
            write_file(
                source / "CMakeLists.txt",
                f"""
                cmake_minimum_required(VERSION 3.25)
                project(module_rpath C)
                set(THEROCK_SOURCE_DIR "{THEROCK_ROOT.as_posix()}")
                set(THEROCK_BINARY_DIR "${{CMAKE_BINARY_DIR}}")
                set(THEROCK_SUBPROJECT_TARGET module_fixture)
                set(THEROCK_STAGE_STAMP_FILE "${{CMAKE_BINARY_DIR}}/stage.stamp")
                set(THEROCK_BUILD_STAMP_FILE "${{CMAKE_BINARY_DIR}}/build.stamp")
                set(THEROCK_PRIVATE_BUILD_RPATH_DIRS "${{CMAKE_BINARY_DIR}}/runtime")
                set(THEROCK_INSTALL_RPATH_LIBRARY_DIR "${{CMAKE_BINARY_DIR}}/install/lib")
                set(THEROCK_PRIVATE_INSTALL_RPATH_DIRS "${{CMAKE_BINARY_DIR}}/install/lib/clang")
                add_library(runtime_fixture SHARED runtime.c)
                set_target_properties(runtime_fixture PROPERTIES
                  LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/runtime")
                add_library(module_fixture MODULE module.c)
                target_link_libraries(module_fixture PRIVATE runtime_fixture)
                include("{(THEROCK_ROOT / 'cmake' / 'therock_global_post_subproject.cmake').as_posix()}")
                get_target_property(build_rpath module_fixture BUILD_RPATH)
                get_target_property(install_rpath module_fixture INSTALL_RPATH)
                file(WRITE "${{CMAKE_BINARY_DIR}}/rpaths.txt"
                  "${{build_rpath}}\n${{install_rpath}}\n")
                install(TARGETS runtime_fixture LIBRARY DESTINATION lib/clang)
                install(TARGETS module_fixture LIBRARY DESTINATION lib)
                """,
            )
            configure(source, build)
            build_rpath, install_rpath = (
                (build / "rpaths.txt").read_text(encoding="utf-8").splitlines()
            )
            self.assertIn(str(build / "runtime"), build_rpath)
            self.assertIn("$ORIGIN/clang", install_rpath)
            cmake("--build", str(build))
            cmake("--install", str(build), "--prefix", str(build / "install"))
            module = ctypes.CDLL(os.fspath(build / "install/lib/libmodule_fixture.so"))
            module.module_value.restype = ctypes.c_int
            self.assertEqual(module.module_value(), 42)


if __name__ == "__main__":
    unittest.main()
