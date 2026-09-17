# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import unittest
from pathlib import Path


THEROCK_DIR = Path(__file__).resolve().parents[1]


class HostTsanProfilerPreparationTest(unittest.TestCase):
    def test_scoped_components_are_not_forced_uninstrumented(self):
        presets = json.loads((THEROCK_DIR / "CMakePresets.json").read_text())
        host_tsan = next(
            preset
            for preset in presets["configurePresets"]
            if preset["name"] == "linux-host-tsan-base"
        )
        cache = host_tsan["cacheVariables"]
        for component in (
            "rocprofiler-sdk",
            "rocprofiler-compute",
            "rocgdb",
        ):
            self.assertNotIn(f"{component}_SANITIZER", cache)

        # These dependencies remain deliberately outside this phase.
        self.assertEqual(cache["aqlprofile_SANITIZER"], "OFF")
        self.assertEqual(cache["rocprof-trace-decoder_SANITIZER"], "OFF")
        self.assertEqual(cache["roctracer_SANITIZER"], "OFF")

    def test_rocgdb_maps_host_tsan_to_thread_compile_and_linkage(self):
        hook = (THEROCK_DIR / "debug-tools" / "pre_hook_rocgdb.cmake").read_text()
        self.assertGreaterEqual(hook.count('THEROCK_SANITIZER STREQUAL "HOST_TSAN"'), 2)
        self.assertIn('set(_sanitizer_string "thread")', hook)
        self.assertIn("foreach(_var CMAKE_C_FLAGS CMAKE_CXX_FLAGS)", hook)
        self.assertIn(
            '-fsanitize=${_sanitizer_string} -fno-omit-frame-pointer', hook
        )
        self.assertIn("-fsanitize=${_sanitizer_string} -shared-libsan", hook)

    def test_rocgdb_installs_required_tsan_suppressions(self):
        cmake = (THEROCK_DIR / "debug-tools" / "rocgdb" / "CMakeLists.txt").read_text()
        self.assertIn('"gdb/tsan-suppressions.txt"', cmake)
        self.assertIn("DESTINATION tests/rocgdb/${file_dir}", cmake)

    def test_compute_keeps_host_scope_and_gpu_targets(self):
        hook = (
            THEROCK_DIR / "profiler" / "pre_hook_rocprofiler-compute.cmake"
        ).read_text()
        self.assertIn('THEROCK_SANITIZER STREQUAL "HOST_TSAN"', hook)
        self.assertIn('set(THEROCK_SANITIZER "")', hook)
        self.assertIn('set(ENABLE_SANITIZER "OFF"', hook)
        self.assertNotIn("unset(GPU_TARGETS)", hook)
        self.assertNotIn('set(ENABLE_SANITIZER "TSAN"', hook)

    def test_rocprofiler_systems_unit_binary_is_relocatable_and_packaged(self):
        hook = (
            THEROCK_DIR / "profiler" / "post_hook_rocprofiler-systems.cmake"
        ).read_text()
        destination = "share/rocprofiler-systems/tests/unit-tests/bin"
        self.assertIn("if(TARGET rocprof-sys-unit-tests)", hook)
        self.assertIn("THEROCK_INSTALL_RPATH_ORIGIN", hook)
        self.assertGreaterEqual(hook.count(destination), 2)
        self.assertIn("COMPONENT rocprofiler-systems-tests", hook)
        self.assertIn("foreach(_target gtest gtest_main gmock)", hook)
        self.assertIn("LIBRARY DESTINATION lib/rocprofiler-systems", hook)

        artifact = (
            THEROCK_DIR / "profiler" / "artifact-rocprofiler-systems.toml"
        ).read_text()
        self.assertIn('"share/rocprofiler-systems/tests/**"', artifact)
        for library in ("libgtest.so*", "libgtest_main.so*", "libgmock.so*"):
            self.assertIn(library, artifact)


if __name__ == "__main__":
    unittest.main()
