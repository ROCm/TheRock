import json
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_amdgpu_matrix
from configure_amdgpu_matrix import (
    PlatformMask,
    TaskMask,
    get_build_config,
    get_test_config,
    matrix_generator,
    is_ci_run_required,
)
from new_amdgpu_family_matrix_types import BuildConfig, GpuRunners, TestConfig

# Set up test runner dictionary for ROCM_THEROCK_TEST_RUNNERS
# This simulates the ROCm organization-wide test runner configuration
# Keys use lowercase 'x' for partial matching (e.g., "gfx110x" matches "gfx110X-all", "gfx110X-dgpu", etc.)
therock_test_runner_dict = {
    "gfx110x": {
        "linux": "linux-gfx110X-gpu-rocm-test",
        "windows": "windows-gfx110X-gpu-rocm-test",
    },
    "gfx94x": {
        "linux": "linux-mi325-orgwide-runner",
        "windows": "windows-mi325-orgwide-runner",
    },
}

os.environ["ROCM_THEROCK_TEST_RUNNERS"] = json.dumps(therock_test_runner_dict)


class TestGetBuildConfig(unittest.TestCase):
    """Tests for get_build_config() function."""

    def test_valid_release_variant(self):
        """Test getting build config for release variant."""
        config = get_build_config(
            BuildConfig(build_variants=["release", "asan"], expect_failure=False),
            "release",
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNotNone(config)
        self.assertEqual(config["build_variant_label"], "release")
        self.assertEqual(config["build_variant_suffix"], "")
        self.assertEqual(config["artifact_group"], "gfx94X-dcgpu")
        self.assertFalse(config["expect_failure"])

    def test_valid_tsan_variant(self):
        """Test getting build config for TSAN variant."""
        config = get_build_config(
            BuildConfig(build_variants=["release", "tsan"], expect_failure=False),
            "tsan",
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNotNone(config)
        self.assertEqual(config["build_variant_label"], "tsan")
        self.assertEqual(config["build_variant_suffix"], "tsan")
        self.assertEqual(config["artifact_group"], "gfx94X-dcgpu-tsan")
        # TSAN variant has expect_failure: True by default
        self.assertTrue(config["expect_failure"])

    def test_invalid_variant_returns_none(self):
        """Test that requesting unavailable variant returns None."""
        config = get_build_config(
            BuildConfig(build_variants=["release"]),
            "asan",
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNone(config)

    def test_expect_failure_from_variant(self):
        """Test that expect_failure from variant takes precedence (tsan has expect_failure=True)."""
        config = get_build_config(
            BuildConfig(build_variants=["tsan"], expect_failure=False),
            "tsan",
            "linux",
            "gfx94X-dcgpu",
        )

        # tsan variant has expect_failure: True, should override family setting
        self.assertTrue(config["expect_failure"])

    def test_expect_failure_from_family(self):
        """Test that expect_failure from family is respected."""
        config = get_build_config(
            BuildConfig(build_variants=["release"], expect_failure=True),
            "release",
            "linux",
            "gfx1152",
        )

        self.assertTrue(config["expect_failure"])


class TestGetTestConfig(unittest.TestCase):
    """Tests for get_test_config() function."""

    def test_run_tests_false_returns_none(self):
        """Test that run_tests: False returns None."""
        config = get_test_config(
            TestConfig(run_tests=False, runs_on=GpuRunners(test="linux-machine")),
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNone(config)

    def test_basic_test_config(self):
        """Test basic test config with runs_on."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(
                    test="linux-mi325-1gpu", benchmark="linux-mi325-1gpu"
                ),
            ),
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNotNone(config)
        self.assertIsInstance(config["runs_on"], dict)
        self.assertEqual(config["runs_on"]["test"], "linux-mi325-1gpu")
        self.assertEqual(config["runs_on"]["benchmark"], "linux-mi325-1gpu")
        self.assertFalse(config["sanity_check_only_for_family"])
        self.assertFalse(config["expect_pytorch_failure"])

    def test_no_runners_returns_none(self):
        """Test that empty runners returns None."""
        config = get_test_config(
            TestConfig(run_tests=True, runs_on=GpuRunners()),
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNone(config)

    def test_test_runner_only(self):
        """Test config with only test runner, no benchmark."""
        config = get_test_config(
            TestConfig(run_tests=True, runs_on=GpuRunners(test="linux-gfx110X-gpu")),
            "linux",
            "gfx110X-all",
        )

        self.assertIsNotNone(config)
        self.assertEqual(config["runs_on"]["test"], "linux-gfx110X-gpu")
        self.assertEqual(config["runs_on"]["benchmark"], "")

    def test_benchmark_runner_only(self):
        """Test config with only benchmark runner, no test."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(benchmark="linux-benchmark-machine"),
            ),
            "linux",
            "gfx94X-dcgpu",
        )

        self.assertIsNotNone(config)
        self.assertEqual(config["runs_on"]["test"], "")
        self.assertEqual(config["runs_on"]["benchmark"], "linux-benchmark-machine")

    def test_optional_fields(self):
        """Test that optional fields are properly handled."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(test="linux-machine"),
                sanity_check_only_for_family=True,
                expect_pytorch_failure=True,
            ),
            "linux",
            "gfx110X-all",
        )

        self.assertTrue(config["sanity_check_only_for_family"])
        self.assertTrue(config["expect_pytorch_failure"])

    def test_orgwide_test_runner_override(self):
        """Test that ROCM_THEROCK_TEST_RUNNERS overrides runs_on for matching targets."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(
                    test="linux-default-runner",
                    benchmark="linux-default-benchmark",
                ),
            ),
            "linux",
            "gfx110X-all",
            orgwide_test_runner_dict=therock_test_runner_dict,
        )

        self.assertEqual(config["runs_on"]["test"], "linux-gfx110X-gpu-rocm-test")
        self.assertEqual(config["runs_on"]["benchmark"], "linux-default-benchmark")

    def test_orgwide_test_runner_no_match(self):
        """Test that non-matching targets are not affected by orgwide dict."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(
                    test="linux-default-runner",
                    benchmark="linux-default-benchmark",
                ),
            ),
            "linux",
            "gfx1152",
            orgwide_test_runner_dict=therock_test_runner_dict,
        )

        self.assertEqual(config["runs_on"]["test"], "linux-default-runner")
        self.assertEqual(config["runs_on"]["benchmark"], "linux-default-benchmark")

    def test_extra_runners_excluded_from_output(self):
        """Test that extra runners in GpuRunners.extra are excluded from output runs_on."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(
                    test="linux-machine",
                    benchmark="linux-benchmark",
                    extra={"oem": "linux-oem-machine", "custom": "linux-custom"},
                ),
            ),
            "linux",
            "gfx94X-dcgpu",
        )

        # Only standard keys in output: test, test-multi-gpu, benchmark
        self.assertIn("test", config["runs_on"])
        self.assertIn("test-multi-gpu", config["runs_on"])
        self.assertIn("benchmark", config["runs_on"])
        self.assertNotIn("oem", config["runs_on"])
        self.assertNotIn("custom", config["runs_on"])
        self.assertEqual(len(config["runs_on"]), 3)

    def test_overwrite_replaces_test_runner_with_extra(self):
        """Test that test_runner overwrite replaces test runner with an extra runner."""
        config = get_test_config(
            TestConfig(
                run_tests=True,
                runs_on=GpuRunners(
                    test="linux-gfx1151-gpu-rocm",
                    extra={"oem": "linux-strix-halo-gpu-rocm-oem"},
                ),
            ),
            "linux",
            "gfx1151",
            overwrite_values={"test_runner": "oem"},
        )

        self.assertIsNotNone(config)
        self.assertEqual(config["runs_on"]["test"], "linux-strix-halo-gpu-rocm-oem")


class TestMatrixGenerator(unittest.TestCase):
    """Tests for matrix_generator() function."""

    def test_single_family_release_build(self):
        """Test generating matrix for single family with release variant."""
        req_families = {
            PlatformMask.LINUX: ["gfx94X-dcgpu"],
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD,
            req_gpu_families_or_targets=req_families,
            build_variant="release",
        )

        self.assertIn("linux", matrix)
        self.assertEqual(len(matrix["linux"]), 1)

        entry = matrix["linux"][0]
        self.assertEqual(entry["amdgpu_family"], "gfx94X-dcgpu")
        self.assertIn("build", entry)
        self.assertEqual(entry["build"]["build_variant_label"], "release")
        self.assertEqual(entry["build"]["artifact_group"], "gfx94X-dcgpu")

    def test_single_family_asan_build(self):
        """Test generating matrix for single family with ASAN variant."""
        req_families = {
            PlatformMask.LINUX: ["gfx94X-dcgpu"],
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD,
            req_gpu_families_or_targets=req_families,
            build_variant="asan",
        )

        self.assertIn("linux", matrix)
        self.assertEqual(len(matrix["linux"]), 1)

        entry = matrix["linux"][0]
        self.assertEqual(entry["amdgpu_family"], "gfx94X-dcgpu")
        self.assertIn("build", entry)
        self.assertEqual(entry["build"]["build_variant_label"], "asan")
        self.assertEqual(entry["build"]["artifact_group"], "gfx94X-dcgpu-asan")
        self.assertFalse(entry["build"]["expect_failure"])

    def test_build_and_test_tasks(self):
        """Test generating matrix with both build and test tasks."""
        req_families = {
            PlatformMask.LINUX: ["gfx94X-dcgpu"],
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD | TaskMask.TEST,
            req_gpu_families_or_targets=req_families,
            build_variant="release",
            orgwide_test_runner_dict={},
        )

        entry = matrix["linux"][0]
        self.assertIn("build", entry)
        self.assertIn("test", entry)
        self.assertIsInstance(entry["test"]["runs_on"], dict)
        self.assertIn("test", entry["test"]["runs_on"])
        self.assertIn("benchmark", entry["test"]["runs_on"])

    def test_multiple_families(self):
        """Test generating matrix with multiple GPU families."""
        req_families = {
            PlatformMask.LINUX: ["gfx94X-dcgpu", "gfx110X-all"],
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD | TaskMask.TEST,
            req_gpu_families_or_targets=req_families,
            build_variant="release",
            orgwide_test_runner_dict={},
        )

        self.assertEqual(len(matrix["linux"]), 2)
        families = [entry["amdgpu_family"] for entry in matrix["linux"]]
        self.assertIn("gfx94X-dcgpu", families)
        self.assertIn("gfx110X-all", families)

    def test_windows_platform(self):
        """Test generating matrix for Windows platform."""
        req_families = {
            PlatformMask.LINUX: [],
            PlatformMask.WINDOWS: ["gfx110X-all"],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.WINDOWS,
            task_mask=TaskMask.BUILD,
            req_gpu_families_or_targets=req_families,
            build_variant="release",
        )

        self.assertIn("windows", matrix)
        self.assertEqual(len(matrix["windows"]), 1)
        self.assertEqual(matrix["windows"][0]["amdgpu_family"], "gfx110X-all")

    def test_release_task(self):
        """Test generating matrix with release task."""
        req_families = {
            PlatformMask.LINUX: ["gfx94X-dcgpu"],
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD | TaskMask.RELEASE,
            req_gpu_families_or_targets=req_families,
            build_variant="release",
        )

        entry = matrix["linux"][0]
        self.assertIn("build", entry)
        self.assertIn("release", entry)
        self.assertIn("bypass_tests_for_releases", entry["release"])

    def test_invalid_variant_for_family_skipped(self):
        """Test that families without the requested variant are skipped."""
        req_families = {
            PlatformMask.LINUX: ["gfx1152"],  # Only has release, not asan
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD,
            req_gpu_families_or_targets=req_families,
            build_variant="asan",
        )

        # gfx1152 doesn't support asan variant, should be skipped
        self.assertEqual(len(matrix["linux"]), 0)

    def test_orgwide_test_runner_integration(self):
        """Test that orgwide test runners are properly integrated into matrix generation."""
        req_families = {
            PlatformMask.LINUX: ["gfx94X-dcgpu"],
            PlatformMask.WINDOWS: [],
        }

        matrix = matrix_generator(
            platform_mask=PlatformMask.LINUX,
            task_mask=TaskMask.BUILD | TaskMask.TEST,
            req_gpu_families_or_targets=req_families,
            build_variant="release",
            orgwide_test_runner_dict=therock_test_runner_dict,
        )

        entry = matrix["linux"][0]
        self.assertIn("test", entry)
        # The orgwide dict should override the default runner for gfx94X
        self.assertEqual(entry["test"]["runs_on"]["test"], "linux-mi325-orgwide-runner")


class TestShouldCIRun(unittest.TestCase):
    """Tests for is_ci_run_required() function."""

    def test_run_ci_if_source_file_edited(self):
        """Test that CI runs if source files are modified."""
        paths = ["source_file.h", "CMakeLists.txt"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_only_markdown_files_edited(self):
        """Test that CI skips if only markdown files are modified."""
        paths = ["README.md", "docs/setup.md"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_experimental_edited(self):
        """Test that CI skips if only experimental code is modified."""
        paths = ["experimental/test.cpp"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_precommit_workflow_edited(self):
        """Test that CI skips if only pre-commit is modified."""
        paths = [".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_workflow_file_edited(self):
        """Test that CI runs if related workflow files are modified."""
        paths = [".github/workflows/ci.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/build_portable_linux_artifacts.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_run_ci_if_mixed_files_edited(self):
        """Test that CI runs if mix of skippable and non-skippable files."""
        paths = ["README.md", "source_file.cpp"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_no_files_returns_false(self):
        """Test that CI skips if no files are modified."""
        paths = None
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)


if __name__ == "__main__":
    unittest.main()
