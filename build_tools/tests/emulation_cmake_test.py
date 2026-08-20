#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import subprocess
import tempfile
import unittest
from pathlib import Path


class EmulationCMakeTest(unittest.TestCase):
    def test_hotswap_library_validation_uses_hotswap_dist(self):
        """The split hotswap library must be validated in its owning dist."""
        source_dir = Path(__file__).parents[2]
        emulation_cmake = (source_dir / "emulation" / "CMakeLists.txt").as_posix()

        script = f"""
cmake_minimum_required(VERSION 3.25)
set(THEROCK_ENABLE_ROCJITSU ON)
set(THEROCK_ENABLE_ROCJITSU_HOTSWAP ON)
set(THEROCK_ENABLE_MIRAGE OFF)
set(THEROCK_BUILD_TESTING ON)
set(THEROCK_ROCM_SYSTEMS_SOURCE_DIR /rocm-systems)

function(therock_cmake_subproject_declare)
endfunction()
function(therock_cmake_subproject_glob_c_sources)
endfunction()
function(therock_cmake_subproject_activate)
endfunction()
function(therock_cmake_subproject_build_test)
endfunction()
function(therock_provide_artifact)
endfunction()
function(therock_test_validate_shared_lib)
  cmake_parse_arguments(ARG "" "PATH" "LIB_NAMES" ${{ARGN}})
  foreach(lib_name IN LISTS ARG_LIB_NAMES)
    list(APPEND validation_pairs "${{lib_name}}=${{ARG_PATH}}")
  endforeach()
  set(validation_pairs "${{validation_pairs}}" PARENT_SCOPE)
endfunction()

include("{emulation_cmake}")

list(FIND validation_pairs
  "libhsa_hotswap_rocjitsu.so=rocjitsu-hotswap/dist/lib"
  hotswap_validation_index)
if(hotswap_validation_index EQUAL -1)
  message(FATAL_ERROR
    "libhsa_hotswap_rocjitsu.so validation did not use rocjitsu-hotswap/dist/lib; "
    "registered validations: ${{validation_pairs}}")
endif()
"""

        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "test.cmake"
            script_path.write_text(script, encoding="utf-8")
            result = subprocess.run(
                ["cmake", "-P", str(script_path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
