#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configure the real core declarations with recorded subproject interfaces."""

from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

HARNESS = r"""
cmake_minimum_required(VERSION 3.25)
project(provider_graph NONE)
list(APPEND CMAKE_MODULE_PATH "@ROOT@/cmake")
include(therock_runtime_provider)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(ROCM_RUNTIMES_CARGO "${CMAKE_COMMAND}")
set(ROCM_RUNTIMES_RUSTC "${CMAKE_COMMAND}")
set(THEROCK_ENABLE_CORE_RUNTIME ON)
set(THEROCK_ENABLE_HIP_RUNTIME ON)
set(THEROCK_ENABLE_OCL_RUNTIME ON)
set(THEROCK_ENABLE_OCL_ICD ON)
set(THEROCK_BUILD_TESTING ON)
set(THEROCK_ENABLE_CORE_RUNTIME_TESTS ON)
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/systems")
file(MAKE_DIRECTORY "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/runtimes")
file(WRITE "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/runtimes/CMakeLists.txt" "# fixture")
file(WRITE "${CMAKE_BINARY_DIR}/graph.txt" "provider=${THEROCK_HSA_RUNTIME_TARGET}\n")
foreach(dependency amd-llvm amd-comgr hipcc rocm-core rocm-cmake rocm-kpack
    rocprofiler-register therock-simde therock-yaml-cpp)
  add_custom_target(${dependency})
endforeach()
function(therock_cmake_subproject_declare name)
  cmake_parse_arguments(ARG
    "USE_DIST_AMDGPU_TARGETS;USE_TEST_AMDGPU_TARGETS;DISABLE_AMDGPU_TARGETS;NO_MERGE_COMPILE_COMMANDS;BACKGROUND_BUILD"
    "BINARY_DIR;EXTERNAL_SOURCE_DIR;COMPILER_TOOLCHAIN;FPRINT_SOURCE_DIR;INSTALL_DESTINATION;INSTALL_RPATH_EXECUTABLE_DIR;INSTALL_RPATH_LIBRARY_DIR"
    "BUILD_DEPS;RUNTIME_DEPS;CMAKE_ARGS;FPRINT_FILE_GLOBS;INTERFACE_LINK_DIRS;INTERFACE_INSTALL_RPATH_DIRS;INTERFACE_PROGRAM_DIRS;INSTALL_RPATH_DIRS;INTERFACE_INCLUDE_DIRS"
    ${ARGN})
  foreach(dependency ${ARG_BUILD_DEPS} ${ARG_RUNTIME_DEPS})
    if(NOT TARGET "${dependency}")
      message(FATAL_ERROR "Missing dependency ${dependency} for ${name}")
    endif()
  endforeach()
  add_custom_target(${name})
  set_target_properties(${name} PROPERTIES THEROCK_BINARY_DIR "${ARG_BINARY_DIR}/build"
    THEROCK_DIST_DIR "${ARG_BINARY_DIR}/dist" THEROCK_STAGE_DIR "${ARG_BINARY_DIR}/stage"
    THEROCK_STAMP_DIR "${ARG_BINARY_DIR}/stamp")
  file(APPEND "${CMAKE_BINARY_DIR}/graph.txt" "target=${name}|${ARGN}\n")
endfunction()
function(therock_cmake_subproject_provide_package name package directory)
  file(APPEND "${CMAKE_BINARY_DIR}/graph.txt" "package=${name}|${package}\n")
endfunction()
function(therock_provide_artifact name)
  file(APPEND "${CMAKE_BINARY_DIR}/graph.txt" "artifact=${name}|${ARGN}\n")
endfunction()
function(therock_cmake_subproject_activate name)
  if(name STREQUAL "kfdtest")
    get_target_property(stamp kfdtest THEROCK_STAMP_DIR)
    add_custom_command(OUTPUT "${stamp}/stage.stamp" COMMAND "${CMAKE_COMMAND}" -E true)
  endif()
endfunction()
function(therock_cmake_subproject_build_test)
endfunction()
function(therock_cmake_subproject_glob_c_sources)
endfunction()
function(therock_test_validate_shared_lib)
endfunction()
function(therock_check_subproject_targets result)
  set(${result} ON PARENT_SCOPE)
endfunction()
add_subdirectory("@ROOT@/core" core)
"""


class RuntimeProviderTest(unittest.TestCase):
    def configure(self, enabled, replace, *, core=True, system=None, kfdtests=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "CMakeLists.txt").write_text(HARNESS.replace("@ROOT@", ROOT.as_posix()))
        args = [
            "cmake",
            "-S",
            str(root),
            "-B",
            str(root / "build"),
            f"-DTHEROCK_ENABLE_ROCDDI_RUNTIMES={'ON' if enabled else 'OFF'}",
            f"-DTHEROCK_FLAG_USE_ROCDDI_FOR_HSA_RUNTIME={'ON' if replace else 'OFF'}",
            f"-DTHEROCK_ENABLE_CORE_RUNTIME={'ON' if core else 'OFF'}",
            f"-DTHEROCK_ENABLE_CORE_KFDTESTS={'ON' if kfdtests else 'OFF'}",
        ]
        if system:
            args.append(f"-DCMAKE_SYSTEM_NAME={system}")
        result = subprocess.run(args, capture_output=True, text=True)
        graph_path = root / "build/graph.txt"
        return result, graph_path.read_text() if graph_path.exists() else ""

    def assert_configured(self, enabled, replace):
        result, graph = self.configure(enabled, replace)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return graph

    def test_default_retains_rocr_and_existing_tests(self):
        graph = self.assert_configured(False, False)
        self.assertIn("target=ROCR-Runtime|", graph)
        self.assertNotIn("target=rocm-runtimes|", graph)
        self.assertIn("package=ROCR-Runtime|hsa-runtime64", graph)
        self.assertIn("target=kfdtest|", graph)
        self.assertIn("target=rocrtst|", graph)

    def test_build_only_does_not_supply_canonical_hsa(self):
        graph = self.assert_configured(True, False)
        self.assertIn("target=rocm-runtimes|", graph)
        self.assertIn("target=ROCR-Runtime|", graph)
        self.assertNotIn("package=rocm-runtimes|hsa-runtime64", graph)
        self.assertNotIn("target=rocddi-hsa|", graph)

    def test_replacement_routes_consumers_and_retains_kernel_tests(self):
        graph = self.assert_configured(True, True)
        self.assertNotIn("ROCR-Runtime", graph)
        self.assertIn("target=rocddi-hsa|", graph)
        self.assertIn("package=rocddi-hsa|hsa-runtime64", graph)
        self.assertIn("target=hsakmt|", graph)
        for name in ("hip-clr", "ocl-clr", "rocminfo", "rocrtst"):
            declaration = next(
                line
                for line in graph.splitlines()
                if line.startswith(f"target={name}|")
            )
            self.assertIn("rocddi-hsa", declaration)
        self.assertIn("artifact-core-runtime-rocddi.toml", graph)
        self.assertNotIn("artifact-core-runtime.toml", graph)

    def test_invalid_provider_combinations_fail(self):
        for enabled, core, system in (
            (False, True, None),
            (True, False, None),
            (True, True, "Windows"),
        ):
            with self.subTest(enabled=enabled, core=core, system=system):
                result, _ = self.configure(enabled, True, core=core, system=system)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires", result.stdout + result.stderr)

    def test_no_thunk_build_without_enabled_consumers(self):
        result, graph = self.configure(True, True, kfdtests=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("target=hsakmt|", graph)
        self.assertNotIn("target=kfdtest|", graph)


if __name__ == "__main__":
    unittest.main()
