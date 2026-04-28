# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import json
import unittest

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_test_configurations


class FetchTestConfigurationsTest(unittest.TestCase):
    def setUp(self):
        # Save environment so tests don't leak state
        self._orig_env = os.environ.copy()
        # Save module-level attributes that tests may change
        self._orig_functional_matrix = fetch_test_configurations.functional_matrix
        self._orig_benchmark_matrix = fetch_test_configurations.benchmark_matrix
        self._orig_get_all_families = (
            fetch_test_configurations.get_all_families_for_trigger_types
        )

        os.environ["RUNNER_OS"] = "Linux"
        os.environ["AMDGPU_FAMILIES"] = "gfx94X-dcgpu"
        os.environ["TEST_TYPE"] = "full"
        os.environ["TEST_LABELS"] = "[]"
        os.environ["PROJECTS_TO_TEST"] = "*"

        # Capture gha_set_output instead of writing to GitHub
        self.gha_output = {}

        def fake_gha_set_output(payload):
            self.gha_output.update(payload)

        fetch_test_configurations.gha_set_output = fake_gha_set_output

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)
        # Restore module-level attributes
        fetch_test_configurations.functional_matrix = self._orig_functional_matrix
        fetch_test_configurations.benchmark_matrix = self._orig_benchmark_matrix
        fetch_test_configurations.get_all_families_for_trigger_types = (
            self._orig_get_all_families
        )

    def _get_components(self):
        self.assertIn("components", self.gha_output)
        return json.loads(self.gha_output["components"])

    # -----------------------
    # Basic selection tests
    # -----------------------

    def test_linux_jobs_selected(self):
        fetch_test_configurations.run()
        components = self._get_components()

        self.assertGreater(len(components), 0)
        for job in components:
            self.assertIn("linux", job["platform"])

    def test_single_project_filter(self):
        os.environ["PROJECTS_TO_TEST"] = "hipblas"

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["job_name"], "hipblas")

    def test_test_labels_filter(self):
        os.environ["TEST_LABELS"] = json.dumps(["rocblas", "hipblas"])

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertEqual(names, {"rocblas", "hipblas"})

    # -----------------------
    # TEST_LABELS handling
    # -----------------------

    def test_empty_test_labels_env_is_handled(self):
        # Regression test: json.loads("") used to crash
        os.environ["TEST_LABELS"] = ""

        # Should not raise
        fetch_test_configurations.run()
        components = self._get_components()

        self.assertGreater(len(components), 0)

    def test_missing_test_labels_env_is_handled(self):
        # Regression test: missing TEST_LABELS should behave like []
        if "TEST_LABELS" in os.environ:
            del os.environ["TEST_LABELS"]

        # Should not raise
        fetch_test_configurations.run()
        components = self._get_components()

        self.assertGreater(len(components), 0)

    # -----------------------
    # Sharding behavior
    # -----------------------

    def test_full_test_uses_all_shards(self):
        fetch_test_configurations.run()
        components = self._get_components()

        hipblaslt = next(j for j in components if j["job_name"] == "hipblaslt")
        self.assertEqual(hipblaslt["total_shards"], 6)
        self.assertEqual(hipblaslt["shard_arr"], [1, 2, 3, 4, 5, 6])

    def test_quick_test_forces_single_shard(self):
        os.environ["TEST_TYPE"] = "quick"

        fetch_test_configurations.run()
        components = self._get_components()

        for job in components:
            self.assertEqual(job["total_shards"], 1)
            self.assertEqual(job["shard_arr"], [1])

    def test_platform_specific_shards(self):
        os.environ["PROJECTS_TO_TEST"] = "hipblaslt"
        fetch_test_configurations.run()
        components = self._get_components()
        hipblaslt_linux = components[0]

        os.environ["RUNNER_OS"] = "Windows"
        fetch_test_configurations.run()
        components = self._get_components()
        hipblaslt_windows = components[0]

        self.assertNotEqual(
            hipblaslt_linux["total_shards"], hipblaslt_windows["total_shards"]
        )

    # -----------------------
    # Exclude-family logic
    # -----------------------

    def test_exclude_family_skips_job(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx1150"

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertNotIn("rocroller", names)

    # -----------------------
    # Functional test merging via run_extended_tests
    # -----------------------

    def _setup_functional_test(self):
        """Common setup for functional tests: fake matrix + isolate from other components."""
        os.environ["PROJECTS_TO_TEST"] = "func1"
        fetch_test_configurations.functional_matrix = {
            "func1": {
                "job_name": "func1",
                "platform": ["linux"],
                "total_shards": 1,
            }
        }

    def test_functional_merged_when_enabled(self):
        os.environ["RUN_EXTENDED_TESTS"] = "true"
        self._setup_functional_test()

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["job_name"], "func1")

    def test_functional_not_merged_when_disabled(self):
        os.environ["RUN_EXTENDED_TESTS"] = "false"
        self._setup_functional_test()

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertNotIn("func1", names)

    # -----------------------
    # Benchmark merging via run_extended_tests
    # -----------------------

    def _setup_benchmark_test(self):
        """Common setup for benchmark tests: fake matrix + isolate from other components."""
        os.environ["PROJECTS_TO_TEST"] = "bench1"
        fetch_test_configurations.benchmark_matrix = {
            "bench1": {
                "job_name": "bench1",
                "platform": ["linux"],
                "total_shards_dict": {"linux": 1},
            }
        }

    def test_benchmarks_merged_when_extended_tests_enabled(self):
        os.environ["RUN_EXTENDED_TESTS"] = "true"
        self._setup_benchmark_test()

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0]["job_name"], "bench1")
        self.assertTrue(components[0]["is_benchmark"])
        self.assertEqual(components[0]["test_type"], "full")

    def test_benchmarks_not_merged_when_extended_tests_disabled(self):
        os.environ["RUN_EXTENDED_TESTS"] = "false"
        self._setup_benchmark_test()

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertNotIn("bench1", names)

    # -----------------------
    # Multi-GPU logic (RCCL)
    # -----------------------

    def test_multi_gpu_job_included_when_supported(self):
        def fake_get_all_families(_):
            return {"gfx94x": {"linux": {"test-runs-on-multi-gpu": "linux-mi300-mgpu"}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        rccl = next(j for j in components if j["job_name"] == "rccl")
        self.assertEqual(rccl["multi_gpu_runner"], "linux-mi300-mgpu")

    def test_multi_gpu_job_excluded_when_not_supported(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx90a"

        def fake_get_all_families(_):
            return {}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertNotIn("rccl", names)

    # -----------------------
    # Output contract
    # -----------------------

    def test_windows_hip_tests_default_emits_pal_only(self):
        """On Windows, hip-tests emits only PAL by default (WINDOWS_HIP_ROCR_TESTS off)."""
        os.environ["RUNNER_OS"] = "Windows"
        os.environ["TEST_LABELS"] = json.dumps(["hip-tests"])

        fetch_test_configurations.run()
        components = self._get_components()

        hip_jobs = [j for j in components if "hip-tests" in j["job_name"]]
        self.assertEqual(len(hip_jobs), 1, "Expected only hip-tests (PAL)")
        self.assertEqual(hip_jobs[0]["job_name"], "hip-tests (PAL)")
        self.assertNotIn("expect_failure", hip_jobs[0])
        self.assertEqual(hip_jobs[0]["total_shards"], 4)
        self.assertEqual(hip_jobs[0]["shard_arr"], [1, 2, 3, 4])

    def test_windows_hip_tests_emits_pal_and_rocr_entries(self):
        """On Windows with WINDOWS_HIP_ROCR_TESTS=true, hip-tests runs PAL and ROCR."""
        os.environ["RUNNER_OS"] = "Windows"
        os.environ["TEST_LABELS"] = json.dumps(["hip-tests"])
        os.environ["WINDOWS_HIP_ROCR_TESTS"] = "true"

        fetch_test_configurations.run()
        components = self._get_components()

        hip_jobs = [j for j in components if "hip-tests" in j["job_name"]]
        self.assertEqual(
            len(hip_jobs), 2, "Expected hip-tests (PAL) and hip-tests (ROCR)"
        )
        names = {j["job_name"] for j in hip_jobs}
        self.assertEqual(names, {"hip-tests (PAL)", "hip-tests (ROCR)"})

        pal = next(j for j in hip_jobs if j["job_name"] == "hip-tests (PAL)")
        self.assertNotIn("expect_failure", pal)
        self.assertEqual(pal["total_shards"], 4)
        self.assertEqual(pal["shard_arr"], [1, 2, 3, 4])

        rocr = next(j for j in hip_jobs if j["job_name"] == "hip-tests (ROCR)")
        self.assertTrue(rocr["expect_failure"])
        self.assertEqual(rocr["total_shards"], 4)
        self.assertEqual(rocr["shard_arr"], [1, 2, 3, 4])

    def test_windows_hip_tests_quick_uses_single_shard(self):
        """On Windows with test_type=quick and ROCR enabled, PAL/ROCR each use 1 shard."""
        os.environ["RUNNER_OS"] = "Windows"
        os.environ["TEST_LABELS"] = json.dumps(["hip-tests"])
        os.environ["TEST_TYPE"] = "quick"
        os.environ["WINDOWS_HIP_ROCR_TESTS"] = "true"

        fetch_test_configurations.run()
        components = self._get_components()

        hip_jobs = [j for j in components if "hip-tests" in j["job_name"]]
        for job in hip_jobs:
            self.assertEqual(job["total_shards"], 1)
            self.assertEqual(job["shard_arr"], [1])

    def test_platform_is_emitted(self):
        fetch_test_configurations.run()
        self.assertEqual(self.gha_output["platform"], "linux")


if __name__ == "__main__":
    unittest.main()
