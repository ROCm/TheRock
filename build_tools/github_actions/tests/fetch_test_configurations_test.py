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
        # Use gfx110x for Windows since gfx94x doesn't have a Windows test runner
        os.environ["AMDGPU_FAMILIES"] = "gfx110X-all"

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
    # kpack debug opt-out
    # -----------------------

    def test_kpack_debug_opt_out_passed_through(self):
        # A component opts out of kpack debug logs by setting
        # "rocm_kpack_debug": "0" on its test_matrix entry. fetch_test_configurations
        # passes the field through verbatim; the "1" default and the debug-re-run
        # override both live in the workflow YAML, not here.
        self._inject_job("kpack-opt-out", rocm_kpack_debug="0")

        fetch_test_configurations.run()
        components = self._get_components()

        job = next(j for j in components if j["job_name"] == "kpack-opt-out")
        self.assertEqual(job["rocm_kpack_debug"], "0")

    def test_kpack_debug_absent_when_not_set(self):
        # When a component omits "rocm_kpack_debug", the field is not emitted and
        # the workflow applies its "1" default via fromJSON(...).rocm_kpack_debug.
        self._inject_job("kpack-default")

        fetch_test_configurations.run()
        components = self._get_components()

        job = next(j for j in components if j["job_name"] == "kpack-default")
        self.assertNotIn("rocm_kpack_debug", job)

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

        # Use gfx110x for Windows since gfx94x doesn't have a Windows test runner
        os.environ["AMDGPU_FAMILIES"] = "gfx110X-all"
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
        fetch_test_configurations.run()
        components = self._get_components()
        hipblaslt_windows = components[0]

        self.assertNotEqual(
            hipblaslt_linux["total_shards"], hipblaslt_windows["total_shards"]
        )

    # -----------------------
    # tensilelite ctest-stage gating (AIHPBLAS-4410)
    # -----------------------

    def test_tensilelite_standard_appends_ctest_stage(self):
        """TEST_TYPE=standard should append the ctest stage and extend the timeout."""
        os.environ["PROJECTS_TO_TEST"] = "tensilelite"
        os.environ["TEST_TYPE"] = "standard"

        fetch_test_configurations.run()
        components = self._get_components()

        tensilelite = next(j for j in components if j["job_name"] == "tensilelite")
        self.assertIn(
            "TEST_COMPONENT=hipblaslt-tensilelite", tensilelite["test_script"]
        )
        self.assertIn("test_runner.py", tensilelite["test_script"])
        self.assertEqual(tensilelite["timeout_minutes"], 30)

    def test_tensilelite_quick_omits_ctest_stage(self):
        """TEST_TYPE=quick should not append the ctest stage or extend the timeout."""
        os.environ["PROJECTS_TO_TEST"] = "tensilelite"
        os.environ["TEST_TYPE"] = "quick"

        fetch_test_configurations.run()
        components = self._get_components()

        tensilelite = next(j for j in components if j["job_name"] == "tensilelite")
        self.assertNotIn(
            "TEST_COMPONENT=hipblaslt-tensilelite", tensilelite["test_script"]
        )
        self.assertEqual(tensilelite["timeout_minutes"], 15)

    # -----------------------
    # tensilelite-common (Tensile/Tests/common on real hardware)
    # -----------------------

    def test_tensilelite_common_runs_only_on_opted_in_families(self):
        """Families without skip-gfxNNNN coverage would run every config, so the job is opt-in."""
        os.environ["PROJECTS_TO_TEST"] = "tensilelite-common"
        expected = {
            "gfx90a": True,
            "gfx94X-dcgpu": True,
            "gfx950-dcgpu": True,
            "gfx120X-all": True,
            "gfx110X-all": False,
            "gfx1151": False,
            "gfx1150": False,
        }
        for family, selected in expected.items():
            with self.subTest(family=family):
                os.environ["AMDGPU_FAMILIES"] = family
                self.assertEqual(
                    "tensilelite-common" in self._selected_names(), selected
                )

    def test_tensilelite_common_pins_hw_common_category(self):
        """The job must run hw-common at every tier, without the tensilelite ctest stage."""
        os.environ["PROJECTS_TO_TEST"] = "tensilelite-common"
        for test_type in ("quick", "standard", "comprehensive", "full"):
            with self.subTest(test_type=test_type):
                os.environ["TEST_TYPE"] = test_type
                fetch_test_configurations.run()
                job = next(
                    j
                    for j in self._get_components()
                    if j["job_name"] == "tensilelite-common"
                )
                self.assertTrue(
                    job["test_script"].startswith("TEST_CATEGORY=hw-common ")
                )
                self.assertIn("pytest_runner.py", job["test_script"])
                self.assertNotIn(
                    "TEST_COMPONENT=hipblaslt-tensilelite", job["test_script"]
                )
                self.assertEqual(job["timeout_minutes"], 180)

    def test_tensilelite_label_selects_unit_and_common_jobs(self):
        os.environ["TEST_LABELS"] = json.dumps(["test:tensilelite"])
        names = self._selected_names()
        self.assertIn("tensilelite", names)
        self.assertIn("tensilelite-common", names)

    def test_tensilelite_common_label_selects_only_common_job(self):
        os.environ["TEST_LABELS"] = json.dumps(["test:tensilelite-common"])
        self.assertEqual(self._selected_names(), {"tensilelite-common"})

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

    # -----------------------
    # test_types tier gating
    # -----------------------

    def test_test_types_excludes_disallowed_tier(self):
        # A component that opts out of the quick tier is not scheduled on quick.
        os.environ["TEST_TYPE"] = "quick"
        self._inject_job("tt-gated", test_types=["standard", "comprehensive", "full"])
        self.assertNotIn("tt-gated", self._selected_names())

    def test_test_types_includes_allowed_tier(self):
        # The same component runs on a tier that is in its list.
        os.environ["TEST_TYPE"] = "standard"
        self._inject_job("tt-gated", test_types=["standard", "comprehensive", "full"])
        self.assertIn("tt-gated", self._selected_names())

    def test_test_types_omitted_runs_on_all_tiers(self):
        # Without "test_types", a component runs on every tier, including quick.
        os.environ["TEST_TYPE"] = "quick"
        self._inject_job("tt-ungated")
        self.assertIn("tt-ungated", self._selected_names())

    def test_miopen_dbsync_declares_non_quick_tiers(self):
        # miopen-dbsync is a slow specialist check gated to standard/comprehensive/full.
        config = fetch_test_configurations.test_matrix["miopen-dbsync"]
        self.assertEqual(config["test_types"], ["standard", "comprehensive", "full"])

    def test_miopen_dbsync_excluded_on_quick(self):
        # Integration: the real entry is not scheduled on the quick tier.
        os.environ["PROJECTS_TO_TEST"] = "miopen-dbsync"
        os.environ["TEST_TYPE"] = "quick"
        self.assertNotIn("miopen-dbsync", self._selected_names())

    def _dbsync_job(self, families, gated_gpu, gfx_target):
        """Run selection for miopen-dbsync on `families`; return its job or None."""
        os.environ["PROJECTS_TO_TEST"] = "miopen-dbsync"
        os.environ["AMDGPU_FAMILIES"] = families
        if gated_gpu:
            # configure_multi_arch_ci blanked the GPU runner (e.g. tests_on_trigger).
            os.environ["TEST_RUNS_ON"] = ""
        target = families.split("-")[0].lower()
        fetch_test_configurations.get_all_families_for_trigger_types = lambda _: {
            target: {
                "linux": {
                    "test-runs-on": "" if gated_gpu else "some-gpu-runner",
                    "fetch-gfx-targets": [gfx_target],
                }
            }
        }
        fetch_test_configurations.run()
        jobs = [
            j for j in self._get_components() if j["job_name"] == "miopen-dbsync"
        ]
        return jobs[0] if jobs else None

    def test_miopen_dbsync_include_family_covers_gfx942_and_gfx950(self):
        config = fetch_test_configurations.test_matrix["miopen-dbsync"]
        self.assertEqual(config["include_family"]["linux"], ["gfx942", "gfx950"])

    def test_miopen_dbsync_gfx950_scheduled_on_cpu_runner_when_gpu_gated(self):
        job = self._dbsync_job("gfx950-dcgpu", True, "gfx950")
        self.assertIsNotNone(job)
        self.assertTrue(job["linux_cpu_runner"])
        self.assertTrue(job["test_runner"])

    def test_miopen_dbsync_gfx950_is_advisory(self):
        job = self._dbsync_job("gfx950-dcgpu", True, "gfx950")
        self.assertTrue(job["expect_failure"])

    def test_miopen_dbsync_gfx942_stays_blocking(self):
        job = self._dbsync_job("gfx94X-dcgpu", False, "gfx942")
        self.assertIsNotNone(job)
        self.assertNotIn("expect_failure", job)

    def test_miopen_dbsync_excluded_on_other_families(self):
        self.assertIsNone(self._dbsync_job("gfx120X-all", False, "gfx1200"))

    def test_gpu_components_not_scheduled_on_gated_gfx950(self):
        # The CPU lane must not leak GPU-requiring components onto gated gfx950.
        os.environ["PROJECTS_TO_TEST"] = "miopen-dbsync,rocblas"
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["TEST_RUNS_ON"] = ""
        fetch_test_configurations.get_all_families_for_trigger_types = lambda _: {
            "gfx950": {
                "linux": {"test-runs-on": "", "fetch-gfx-targets": ["gfx950"]}
            }
        }
        names = self._selected_names()
        self.assertIn("miopen-dbsync", names)
        self.assertNotIn("rocblas", names)

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

    def test_multi_gpu_job_excluded_for_quick_tests(self):
        """Multi-GPU tests are skipped on quick runs (temporary capacity constraint)."""
        os.environ["TEST_TYPE"] = "quick"

        def fake_get_all_families(_):
            return {"gfx94x": {"linux": {"test-runs-on-multi-gpu": "linux-mi300-mgpu"}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # Multi-GPU jobs like rccl/rocshmem should be excluded for quick runs
        self.assertNotIn("rccl", names)
        self.assertNotIn("rocshmem", names)

    def test_multi_gpu_job_included_for_standard_tests(self):
        """Multi-GPU tests run on standard (and higher) tiers."""
        os.environ["TEST_TYPE"] = "standard"

        def fake_get_all_families(_):
            return {"gfx94x": {"linux": {"test-runs-on-multi-gpu": "linux-mi300-mgpu"}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # Both multi-GPU jobs should be included for standard tier
        self.assertIn("rccl", names)
        self.assertIn("rocshmem", names)

    # -----------------------
    # ci:run-multi-gpu label forcing
    # -----------------------

    def test_enable_multi_gpu_by_label_label_overrides_quick_exclusion(self):
        """ci:run-multi-gpu label should include multi-GPU tests even on quick runs."""
        os.environ["TEST_TYPE"] = "quick"
        os.environ["TEST_LABELS"] = json.dumps(["ci:run-multi-gpu"])

        def fake_get_all_families(_):
            return {"gfx94x": {"linux": {"test-runs-on-multi-gpu": "linux-mi300-mgpu"}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # Multi-GPU jobs should be included despite quick test type
        self.assertIn("rccl", names)
        self.assertIn("rocshmem", names)

    def test_enable_multi_gpu_by_label_label_enables_unsupported_family(self):
        """ci:run-multi-gpu should force multi-GPU tests even for families without config."""
        os.environ["AMDGPU_FAMILIES"] = "gfx1150"  # Family not in rccl's multi_gpu list
        os.environ["TEST_LABELS"] = json.dumps(["ci:run-multi-gpu"])

        def fake_get_all_families(_):
            # Only gfx1150 has runner config, but rccl only lists gfx94X/gfx950 in multi_gpu
            return {
                "gfx1150": {"linux": {"test-runs-on-multi-gpu": "linux-gfx1150-mgpu"}}
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # Multi-GPU jobs should be included via force flag
        self.assertIn("rccl", names)
        self.assertIn("rocshmem", names)

    def test_enable_multi_gpu_by_label_label_excluded_when_no_runner(self):
        """ci:run-multi-gpu should not include multi-GPU tests if no runner is configured."""
        os.environ["AMDGPU_FAMILIES"] = "gfx90a"
        os.environ["TEST_LABELS"] = json.dumps(["ci:run-multi-gpu"])

        def fake_get_all_families(_):
            # No multi-GPU runner configured for this family
            return {"gfx90a": {"linux": {"test-runs-on": "linux-gfx90a-runner"}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # Multi-GPU jobs should still be excluded - no runner available
        self.assertNotIn("rccl", names)
        self.assertNotIn("rocshmem", names)

    def test_enable_multi_gpu_by_label_combined_with_test_labels(self):
        """ci:run-multi-gpu should work alongside test:* labels."""
        os.environ["TEST_TYPE"] = "quick"
        os.environ["TEST_LABELS"] = json.dumps(["ci:run-multi-gpu", "test:rccl"])

        def fake_get_all_families(_):
            return {"gfx94x": {"linux": {"test-runs-on-multi-gpu": "linux-mi300-mgpu"}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # rccl should be included (test:rccl selects it, ci:run-multi-gpu enables it)
        self.assertIn("rccl", names)
        # rocshmem not selected by test:rccl label
        self.assertNotIn("rocshmem", names)

    # -----------------------
    # Output contract
    # -----------------------

    def test_additional_requirements_files_are_preserved_in_output(self):
        requirements_files = [
            "share/example/requirements.txt",
            "share/example/requirements-test.txt",
        ]
        self._inject_job(
            "custom-requirements",
            additional_requirements_files=requirements_files,
        )

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(len(components), 1)
        self.assertEqual(
            components[0]["additional_requirements_files"], requirements_files
        )

    def test_windows_hip_tests_emits_pal_and_rocr_entries(self):
        """On Windows, hip-tests runs with both PAL and ROCR backends."""
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
        # Use gfx110x for Windows since gfx94x doesn't have a Windows test runner
        os.environ["AMDGPU_FAMILIES"] = "gfx110X-all"
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

    def test_container_images_are_sha256_pinned(self):
        # Check the full matrix, including jobs filtered out for a given run.
        # Entries without an override use the workflow's default image.
        for job_name, config in fetch_test_configurations.test_matrix.items():
            if "container_image" not in config:
                continue
            with self.subTest(job=job_name):
                self.assertRegex(
                    config["container_image"],
                    r"^[^@\s]+@sha256:[0-9a-f]{64}\Z",
                    "Container image overrides must use a full SHA-256 digest pin",
                )

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

    def test_all_rocgdb_label_selects_cpu_and_gpu_jobs(self):
        """test:rocgdb should expand to rocgdb-cpu and rocgdb-gpu."""
        with patch.dict(os.environ, {"TEST_LABELS": json.dumps(["test:rocgdb"])}):
            fetch_test_configurations.run()
            components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("rocgdb-cpu", names)
        self.assertIn("rocgdb-gpu", names)

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
                    ]
                )
            },
        ):
            fetch_test_configurations.run()
            components = self._get_components()

        job_names = [job["job_name"] for job in components]
        self.assertEqual(job_names.count("rocgdb-cpu"), 1)
        self.assertEqual(job_names.count("rocgdb-gpu"), 1)

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
    # CPU-only components when GPU is gated
    # -----------------------

    def test_cpu_only_components_run_without_gpu_runner(self):
        """CPU-only components (linux_cpu_runner=True) run even without GPU runner."""
        os.environ["TEST_LABELS"] = json.dumps(["test:rocgdb-cpu"])

        def fake_get_all_families(_):
            # No test-runs-on or test-runs-on-labels defined - GPU runner not available
            return {"gfx94x": {"linux": {}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # rocgdb-cpu has linux_cpu_runner=True, should be included
        self.assertIn("rocgdb-cpu", names)

    def test_gpu_components_excluded_without_gpu_runner(self):
        """GPU-requiring components are excluded when no GPU runner is available."""
        os.environ["TEST_LABELS"] = json.dumps(["test:rocblas"])

        def fake_get_all_families(_):
            # No test-runs-on or test-runs-on-labels defined - GPU runner not available
            return {"gfx94x": {"linux": {}}}

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        # rocblas requires GPU, should be excluded when no runner available
        self.assertNotIn("rocblas", names)

    def test_multi_gpu_excluded_when_gpu_tests_gated(self):
        """Multi-GPU tests excluded when TEST_RUNS_ON is empty (trigger gating)."""
        os.environ["TEST_RUNS_ON"] = ""

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-prod",
                        "test-runs-on-multi-gpu": "linux-mi300-mgpu",
                    }
                }
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        names = {job["job_name"] for job in self._get_components()}
        self.assertNotIn("rccl", names)
        self.assertNotIn("rocshmem", names)

    # -----------------------
    # test_type_for_family enforcement
    # -----------------------

    def test_test_type_for_family_strict_override(self):
        """test_type_for_family strictly overrides TEST_TYPE env var."""
        os.environ["TEST_TYPE"] = "full"
        os.environ["PROJECTS_TO_TEST"] = "hipblaslt"

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-prod",
                        "test_type_for_family": "quick",
                        "fetch-gfx-targets": ["gfx942"],
                    }
                }
            }

        fetch_test_configurations.get_all_families_for_trigger_types = (
            fake_get_all_families
        )

        fetch_test_configurations.run()
        components = self._get_components()

        hipblaslt = next(j for j in components if j["job_name"] == "hipblaslt")
        self.assertEqual(hipblaslt["test_type"], "quick")
        self.assertEqual(hipblaslt["total_shards"], 1)  # quick uses single shard


if __name__ == "__main__":
    unittest.main()
