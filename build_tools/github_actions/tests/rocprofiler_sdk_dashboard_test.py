# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["THEROCK_BIN_DIR"] = str(
    Path(tempfile.gettempdir()) / "therock-rocprofiler-sdk-test" / "bin"
)
sys.path.insert(
    0,
    os.fspath(Path(__file__).parent.parent / "test_executable_scripts"),
)

import test_rocprofiler_sdk


class CdashBuildNameTest(unittest.TestCase):
    def setUp(self):
        self._original_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_pull_request_build_name_contains_ci_dimensions(self):
        os.environ.update(
            {
                "GITHUB_REF": "refs/pull/123/merge",
                "GITHUB_RUN_ID": "456",
                "ARTIFACT_RUN_ID": "789",
                "AMDGPU_FAMILIES": "gfx94X-dcgpu",
                "BUILD_VARIANT": "asan",
                "RUNNER_OS": "Linux",
            }
        )

        self.assertEqual(
            test_rocprofiler_sdk._cdash_build_name(),
            "ROCm/TheRock/rocprofiler-sdk/PR-123/linux/gfx94X-dcgpu/asan/"
            "artifact-run-789/test-run-456",
        )

    def test_explicit_build_name_takes_precedence(self):
        os.environ["THEROCK_CDASH_BUILD_NAME"] = "custom-build"
        self.assertEqual(test_rocprofiler_sdk._cdash_build_name(), "custom-build")


class DashboardGenerationTest(unittest.TestCase):
    def _generate(self, *, require_submission=False):
        return test_rocprofiler_sdk._generate_dashboard(
            model="Continuous",
            group="TheRock",
            require_cdash_submission=require_submission,
            notes_file=Path("/tmp/dashboard-notes.txt"),
        )

    def test_dashboard_uses_install_tree_and_artifact_notes(self):
        with patch.object(test_rocprofiler_sdk, "is_asan", return_value=False):
            dashboard = self._generate()

        self.assertIn(
            'set(CTEST_SOURCE_DIRECTORY "'
            f'{test_rocprofiler_sdk.ROCPROFILER_SDK_TESTS_PATH}")',
            dashboard,
        )
        self.assertIn(
            'set(CTEST_NOTES_FILES "/tmp/dashboard-notes.txt")',
            dashboard,
        )
        self.assertNotIn("ctest_update(", dashboard)
        self.assertIn('ctest_start(Continuous GROUP "TheRock")', dashboard)
        self.assertIn("PARALLEL_LEVEL 8", dashboard)
        self.assertNotIn('EXCLUDE "', dashboard)
        self.assertNotIn("@", dashboard)

    def test_asan_dashboard_preserves_mainline_configuration_and_exclusions(self):
        with (
            patch.object(test_rocprofiler_sdk, "is_asan", return_value=True),
            patch.object(
                test_rocprofiler_sdk,
                "get_asan_runtime_library",
                return_value="/tmp/libclang_rt.asan-x86_64.so",
            ),
        ):
            dashboard = self._generate()

        self.assertIn("-DROCPROFILER_MEMCHECK=AddressSanitizer", dashboard)
        self.assertIn(
            'EXCLUDE "rocprofiler_sdk.unit.spm_core.check_packet_generation|',
            dashboard,
        )

    def test_strict_submission_is_encoded_in_dashboard(self):
        with patch.object(test_rocprofiler_sdk, "is_asan", return_value=False):
            dashboard = self._generate(require_submission=True)
        self.assertIn("set(_require_cdash_submission TRUE)", dashboard)

    def test_notes_distinguish_artifact_and_test_runs(self):
        with patch.dict(
            os.environ,
            {
                "ARTIFACT_RUN_ID": "111",
                "GITHUB_RUN_ID": "222",
                "GITHUB_SHA": "abc123",
            },
            clear=False,
        ):
            notes = test_rocprofiler_sdk._dashboard_notes()

        self.assertIn("artifact_run_id: 111", notes)
        self.assertIn("test_run_id: 222", notes)
        self.assertIn("checkout_sha: abc123", notes)


class MainTest(unittest.TestCase):
    def test_default_path_keeps_mainline_direct_runner(self):
        with (
            patch.object(test_rocprofiler_sdk, "setup_env") as setup_env,
            patch.object(test_rocprofiler_sdk, "cmake_config") as cmake_config,
            patch.object(test_rocprofiler_sdk, "cmake_build") as cmake_build,
            patch.object(test_rocprofiler_sdk, "execute_tests") as execute_tests,
            patch.object(test_rocprofiler_sdk, "run_cdash") as run_cdash,
        ):
            result = test_rocprofiler_sdk.main([])

        self.assertEqual(result, 0)
        setup_env.assert_called_once_with()
        cmake_config.assert_called_once_with()
        cmake_build.assert_called_once_with()
        execute_tests.assert_called_once_with()
        run_cdash.assert_not_called()

    def test_cdash_path_does_not_run_direct_commands(self):
        with (
            patch.object(test_rocprofiler_sdk, "setup_env"),
            patch.object(test_rocprofiler_sdk, "cmake_config") as cmake_config,
            patch.object(test_rocprofiler_sdk, "cmake_build") as cmake_build,
            patch.object(test_rocprofiler_sdk, "execute_tests") as execute_tests,
            patch.object(test_rocprofiler_sdk, "run_cdash") as run_cdash,
        ):
            result = test_rocprofiler_sdk.main(["--enable-cdash"])

        self.assertEqual(result, 0)
        run_cdash.assert_called_once_with(
            model="Continuous",
            group="TheRock",
            require_cdash_submission=False,
        )
        cmake_config.assert_not_called()
        cmake_build.assert_not_called()
        execute_tests.assert_not_called()


if __name__ == "__main__":
    unittest.main()
