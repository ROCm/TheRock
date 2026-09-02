# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


from pathlib import Path
import os
import sys
import json
import unittest
from unittest.mock import patch

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_test_configurations


class FetchTestConfigurationsTest(unittest.TestCase):
    def setUp(self):
        # Save environment so tests don't leak state
        self._orig_env = os.environ.copy()
        # Save sys.argv so tests don't leak state
        self._orig_argv = sys.argv.copy()
        # Save module-level attributes that tests may change
        self._orig_functional_matrix = fetch_test_configurations.functional_matrix
        self._orig_benchmark_matrix = fetch_test_configurations.benchmark_matrix
        self._orig_get_all_families = (
            fetch_test_configurations.get_all_families_for_trigger_types
        )
        # Snapshot the matrix keys so tests can inject temporary jobs and have
        # them removed in tearDown.
        self._orig_test_matrix_keys = set(fetch_test_configurations.test_matrix.keys())

        os.environ["AMDGPU_FAMILIES"] = "gfx94X-dcgpu"
        os.environ["TEST_TYPE"] = "full"
        os.environ["TEST_LABELS"] = "[]"
        os.environ["PROJECTS_TO_TEST"] = "*"

        # Default to linux platform
        sys.argv = ["fetch_test_configurations.py", "--platform=linux"]

        # Capture gha_set_output instead of writing to GitHub
        self.gha_output = {}

        def fake_gha_set_output(payload):
            self.gha_output.update(payload)

        fetch_test_configurations.gha_set_output = fake_gha_set_output

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._orig_env)
        sys.argv = self._orig_argv
        # Restore module-level attributes
        fetch_test_configurations.functional_matrix = self._orig_functional_matrix
        fetch_test_configurations.benchmark_matrix = self._orig_benchmark_matrix
        fetch_test_configurations.get_all_families_for_trigger_types = (
            self._orig_get_all_families
        )
        # Remove any temporary jobs injected by a test.
        for key in list(fetch_test_configurations.test_matrix.keys()):
            if key not in self._orig_test_matrix_keys:
                del fetch_test_configurations.test_matrix[key]

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

    def test_windows_jobs_selected(self):
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertGreater(len(components), 0)
        for job in components:
            self.assertIn("windows", job["platform"])

    def test_rocprofiler_sdk_submits_to_cdash(self):
        config = fetch_test_configurations.test_matrix["rocprofiler-sdk"]
        self.assertIn("--enable-cdash", config["test_script"])

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

        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
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
    # include_family / exclude_family matching
    # -----------------------

    def _inject_job(self, name, **extra):
        """Register a temporary linux job and target only it via PROJECTS_TO_TEST."""
        os.environ["PROJECTS_TO_TEST"] = name
        fetch_test_configurations.test_matrix[name] = {
            "job_name": name,
            "platform": ["linux"],
            "total_shards_dict": {"linux": 1},
            **extra,
        }

    def _selected_names(self):
        fetch_test_configurations.run()
        return {job["job_name"] for job in self._get_components()}

    def test_family_matches_by_group_string(self):
        self.assertTrue(
            fetch_test_configurations._family_matches(
                ["gfx94X-dcgpu"], "gfx94X-dcgpu", ["gfx942"]
            )
        )

    def test_family_matches_by_gfx_target(self):
        self.assertTrue(
            fetch_test_configurations._family_matches(
                ["gfx942"], "gfx94X-dcgpu", ["gfx942"]
            )
        )

    def test_family_matches_returns_false_when_absent(self):
        self.assertFalse(
            fetch_test_configurations._family_matches(
                ["gfx1100"], "gfx94X-dcgpu", ["gfx942"]
            )
        )

    def test_family_matches_empty_list_is_false(self):
        self.assertFalse(
            fetch_test_configurations._family_matches([], "gfx94X-dcgpu", ["gfx942"])
        )

    def test_include_family_matches_by_gfx_target(self):
        self._inject_job("inc-match", include_family={"linux": ["gfx942"]})
        self.assertIn("inc-match", self._selected_names())

    def test_include_family_matches_by_group_string(self):
        self._inject_job("inc-group", include_family={"linux": ["gfx94X-dcgpu"]})
        self.assertIn("inc-group", self._selected_names())

    def test_include_family_skips_when_not_listed(self):
        self._inject_job("inc-miss", include_family={"linux": ["gfx1100"]})
        self.assertNotIn("inc-miss", self._selected_names())

    def test_include_family_for_other_platform_does_not_gate(self):
        # An include list scoped to a different platform must not gate this one.
        self._inject_job("inc-otherplat", include_family={"windows": ["gfx1100"]})
        self.assertIn("inc-otherplat", self._selected_names())

    def test_no_include_or_exclude_runs(self):
        self._inject_job("plain")
        self.assertIn("plain", self._selected_names())

    def test_include_and_exclude_both_match_excludes_wins(self):
        self._inject_job(
            "inc-exc",
            include_family={"linux": ["gfx942"]},
            exclude_family={"linux": ["gfx942"]},
        )
        self.assertNotIn("inc-exc", self._selected_names())

    def test_include_matches_and_exclude_does_not(self):
        self._inject_job(
            "inc-noexc",
            include_family={"linux": ["gfx942"]},
            exclude_family={"linux": ["gfx1100"]},
        )
        self.assertIn("inc-noexc", self._selected_names())

    def test_corefile_included_on_gfx942(self):
        # Default AMDGPU_FAMILIES is gfx94X-dcgpu, whose gfx target is gfx942.
        os.environ["PROJECTS_TO_TEST"] = "rocgdb-corefile"
        self.assertIn("rocgdb-corefile", self._selected_names())

    def test_corefile_excluded_on_other_family(self):
        os.environ["PROJECTS_TO_TEST"] = "rocgdb-corefile"
        os.environ["AMDGPU_FAMILIES"] = "gfx1150"
        self.assertNotIn("rocgdb-corefile", self._selected_names())

    # -----------------------
    # Emulation logic
    # -----------------------

    def test_emulate_only_job_routes_to_cpu_runner(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "emulation"

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(len(components), 1)
        emulation_job = components[0]
        self.assertEqual(emulation_job["job_name"], "emulation (emulated mi350x)")
        self.assertTrue(emulation_job["linux_cpu_runner"])
        self.assertNotIn("emulate", emulation_job)
        self.assertNotIn("emulate_only", emulation_job)
        self.assertNotIn("--device /dev/kfd", emulation_job["container_options"])

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

    def test_multi_gpu_job_uses_count_labels_when_available(self):
        """When test-runs-on-multi-gpu-labels is present, select_weighted_label is used."""

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-default",
                        "test-runs-on-labels": [
                            {"label": "linux-gfx942-a", "count": 5},
                            {"label": "linux-gfx942-b", "count": 5},
                        ],
                        "test-runs-on-multi-gpu": "linux-mi300-mgpu-default",
                        "test-runs-on-multi-gpu-labels": [
                            {"label": "linux-mi300-mgpu-a", "count": 5},
                            {"label": "linux-mi300-mgpu-b", "count": 5},
                        ],
                    }
                }
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        # Mock select_weighted_label to verify it's called and return known labels
        original_select_weighted_label = fetch_test_configurations.select_weighted_label
        selected_labels = []

        def fake_select_weighted_label(labels_config, context_name):
            selected_labels.append((labels_config, context_name))
            # Return different labels based on whether it's multi-gpu
            if "multi-gpu" in context_name:
                return "linux-mi300-mgpu-a"
            return "linux-gfx942-a"

        fetch_test_configurations.select_weighted_label = fake_select_weighted_label

        try:
            fetch_test_configurations.run()
            components = self._get_components()

            rccl = next(j for j in components if j["job_name"] == "rccl")
            self.assertEqual(rccl["multi_gpu_runner"], "linux-mi300-mgpu-a")
            # Verify select_weighted_label was called for the multi-gpu jobs.
            # With rocshmem added there is more than one multi-GPU job (rccl and
            # rocshmem), each using a "<job_name>-multi-gpu" context.
            multi_gpu_calls = [c for c in selected_labels if "multi-gpu" in c[1]]
            multi_gpu_contexts = {c[1] for c in multi_gpu_calls}
            self.assertIn("rccl-multi-gpu", multi_gpu_contexts)
            self.assertIn("rocshmem-multi-gpu", multi_gpu_contexts)
        finally:
            fetch_test_configurations.select_weighted_label = (
                original_select_weighted_label
            )

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

    def test_windows_hip_tests_emits_pal_and_rocr_entries(self):
        """On Windows, hip-tests runs with both PAL and ROCR backends."""
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
        os.environ["TEST_LABELS"] = json.dumps(["hip-tests"])

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
        self.assertEqual(rocr["total_shards"], 4)
        self.assertEqual(rocr["shard_arr"], [1, 2, 3, 4])

    def test_windows_hip_tests_quick_uses_single_shard(self):
        """On Windows with test_type=quick, PAL/ROCR each use 1 shard."""
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
        os.environ["TEST_LABELS"] = json.dumps(["hip-tests"])
        os.environ["TEST_TYPE"] = "quick"

        fetch_test_configurations.run()
        components = self._get_components()

        hip_jobs = [j for j in components if "hip-tests" in j["job_name"]]
        for job in hip_jobs:
            self.assertEqual(job["total_shards"], 1)
            self.assertEqual(job["shard_arr"], [1])

    def test_platform_is_emitted(self):
        fetch_test_configurations.run()
        self.assertEqual(self.gha_output["platform"], "linux")

    def test_container_options_on_windows_is_string_not_list(self):
        # Regression: a list value here caused
        # `options: ${{ fromJSON(...).container_options }}` in test_component.yml
        # to evaluate to a YAML sequence, failing template parsing.
        job = {
            "container_options": [
                "--cap-add SYS_MODULE",
                "-v /lib/modules:/lib/modules",
            ]
        }
        out = fetch_test_configurations._build_container_options(job, "windows")
        self.assertIsInstance(out["container_options"], str)

    def test_container_options_on_linux_is_joined_string(self):
        job = {"container_options": ["--cap-add=SYS_PTRACE"]}
        out = fetch_test_configurations._build_container_options(job, "linux")
        self.assertIsInstance(out["container_options"], str)
        self.assertIn("--cap-add=SYS_PTRACE", out["container_options"])

    # -----------------------
    # ASAN sandbox runner selection
    # -----------------------

    def test_asan_family_builds_use_sandbox_runner(self):
        """ASAN-family builds should use test-runs-on-sandbox when available.

        Covers "asan", "host-asan", and their "-debug" (RelWithDebInfo +
        line-number debug info) counterparts — all route to the sandbox
        runner, not the regular runner pool.
        """
        for build_variant in ("asan", "host-asan", "asan-debug", "host-asan-debug"):
            with self.subTest(build_variant=build_variant):
                os.environ["BUILD_VARIANT"] = build_variant
                os.environ["PROJECTS_TO_TEST"] = "hipblas"

                def fake_get_all_families(_):
                    return {
                        "gfx94x": {
                            "linux": {
                                "test-runs-on": "linux-gfx942-prod",
                                "test-runs-on-sandbox": "linux-sandbox-runner",
                            }
                        }
                    }

                fetch_test_configurations.get_all_families_for_trigger_types = (
                    fake_get_all_families
                )

                fetch_test_configurations.run()
                components = self._get_components()

                hipblas = next(j for j in components if j["job_name"] == "hipblas")
                self.assertEqual(hipblas["test_runner"], "linux-sandbox-runner")

    def test_release_build_uses_count_runner(self):
        """Release builds should use count-based runner labels, not sandbox."""
        os.environ["BUILD_VARIANT"] = "release"
        os.environ["PROJECTS_TO_TEST"] = "rocblas"

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-default",
                        "test-runs-on-labels": [
                            {"label": "linux-gfx942-count-runner", "count": 10},
                        ],
                        "test-runs-on-sandbox": "linux-sandbox-runner",
                    }
                }
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        # Mock select_weighted_label to return a known value
        original_select_weighted_label = fetch_test_configurations.select_weighted_label

        def fake_select_weighted_label(labels_config, context_name):
            return "linux-gfx942-count-runner"

        fetch_test_configurations.select_weighted_label = fake_select_weighted_label

        try:
            fetch_test_configurations.run()
            components = self._get_components()

            rocblas = next(j for j in components if j["job_name"] == "rocblas")
            self.assertEqual(rocblas["test_runner"], "linux-gfx942-count-runner")
        finally:
            fetch_test_configurations.select_weighted_label = (
                original_select_weighted_label
            )

    def test_asan_build_without_sandbox_uses_default_runner(self):
        """ASAN builds without sandbox config should fall back to default runner."""
        os.environ["BUILD_VARIANT"] = "asan"
        os.environ["PROJECTS_TO_TEST"] = "hipblas"

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-default",
                        # No test-runs-on-sandbox defined
                    }
                }
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        hipblas = next(j for j in components if j["job_name"] == "hipblas")
        self.assertEqual(hipblas["test_runner"], "linux-gfx942-default")

    # -----------------------
    # TEST_LABEL_GROUPS expansion
    # -----------------------

    def test_all_rocgdb_label_selects_cpu_gpu_and_corefile_jobs(self):
        """test:rocgdb should expand to rocgdb-cpu, rocgdb-gpu, and rocgdb-corefile."""
        with patch.dict(os.environ, {"TEST_LABELS": json.dumps(["test:rocgdb"])}):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocgdb-cpu", names)
        self.assertIn("rocgdb-gpu", names)
        self.assertIn("rocgdb-corefile", names)

    def test_all_rocgdb_label_excludes_unrelated_jobs(self):
        """test:rocgdb should not include jobs outside the rocgdb group."""
        with patch.dict(os.environ, {"TEST_LABELS": json.dumps(["test:rocgdb"])}):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertNotIn("rocblas", names)
        self.assertNotIn("hipblas", names)

    def test_group_label_and_individual_label_combine(self):
        """A group label and a plain label together should union their jobs."""
        with patch.dict(
            os.environ,
            {"TEST_LABELS": json.dumps(["test:rocgdb", "test:rocblas"])},
        ):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocgdb-cpu", names)
        self.assertIn("rocgdb-gpu", names)
        self.assertIn("rocgdb-corefile", names)
        self.assertIn("rocblas", names)

    def test_unknown_group_label_is_treated_as_literal(self):
        """A test: label not in TEST_LABEL_GROUPS should fall through as a literal key."""
        with patch.dict(os.environ, {"TEST_LABELS": json.dumps(["test:rocblas"])}):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocblas", names)
        self.assertNotIn("rocgdb-cpu", names)
        self.assertNotIn("rocgdb-gpu", names)

    def test_group_label_and_individual_member_label_do_not_duplicate(self):
        """Passing test:rocgdb alongside an explicit member label should not produce duplicate jobs."""
        with patch.dict(
            os.environ,
            {
                "TEST_LABELS": json.dumps(
                    [
                        "test:rocgdb",
                        "test:rocgdb-cpu",
                        "test:rocgdb-gpu",
                        "test:rocgdb-corefile",
                    ]
                )
            },
        ):
            fetch_test_configurations.run()
            components = self._get_components()

        job_names = [job["job_name"] for job in components]
        self.assertEqual(job_names.count("rocgdb-cpu"), 1)
        self.assertEqual(job_names.count("rocgdb-gpu"), 1)
        self.assertEqual(job_names.count("rocgdb-corefile"), 1)

    # -----------------------
    # mesa-fork labels
    # -----------------------

    def test_rocdecode_label_selects_rocdecode_job(self):
        """test:rocdecode should select the rocdecode job."""
        with patch.dict(os.environ, {"TEST_LABELS": json.dumps(["test:rocdecode"])}):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocdecode", names)
        self.assertNotIn("rocjpeg", names)

    def test_rocjpeg_label_selects_rocjpeg_job(self):
        """test:rocjpeg should select the rocjpeg job."""
        with patch.dict(os.environ, {"TEST_LABELS": json.dumps(["test:rocjpeg"])}):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocjpeg", names)
        self.assertNotIn("rocdecode", names)

    def test_rocdecode_and_rocjpeg_labels_together(self):
        """test:rocdecode and test:rocjpeg together should select both jobs."""
        with patch.dict(
            os.environ,
            {"TEST_LABELS": json.dumps(["test:rocdecode", "test:rocjpeg"])},
        ):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocdecode", names)
        self.assertIn("rocjpeg", names)

    # -----------------------
    # rocprofiler-sdk SPM
    # -----------------------

    def test_rocprofiler_sdk_spm_pre_pinned_runner_is_not_overwritten(self):
        """Pre-pinned test_runner must survive family runner selection."""
        os.environ["PROJECTS_TO_TEST"] = "rocprofiler-sdk-spm"
        os.environ["BUILD_VARIANT"] = "release"

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-prod",
                        "test-runs-on-labels": [
                            {"label": "linux-gfx942-weighted", "weight": 1.0},
                        ],
                        "test-runs-on-sandbox": "linux-mi325-gpu-rocm-cpu-sandbox",
                    }
                }
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        spm = next(j for j in components if j["job_name"] == "rocprofiler-sdk-spm")
        self.assertEqual(
            spm["test_runner"],
            "linux-gfx942-gpu-rocm-profiler",
        )

    def test_rocprofiler_sdk_excludes_spm_label_in_script(self):
        os.environ["PROJECTS_TO_TEST"] = "rocprofiler-sdk"

        fetch_test_configurations.run()
        components = self._get_components()

        sdk = next(j for j in components if j["job_name"] == "rocprofiler-sdk")
        self.assertIn("--ctest-label-exclude spm", sdk["test_script"])

    def test_rocprofiler_sdk_spm_excluded_outside_include_family(self):
        os.environ["PROJECTS_TO_TEST"] = "rocprofiler-sdk-spm"
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"

        fetch_test_configurations.run()
        components = self._get_components()
        names = {j["job_name"] for j in components}

        self.assertNotIn("rocprofiler-sdk-spm", names)


if __name__ == "__main__":
    unittest.main()
