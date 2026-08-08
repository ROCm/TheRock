# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import re
from copy import deepcopy
from pathlib import Path
import os
import sys
import json
import unittest

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import fetch_test_configurations
import workflow_utils


class FetchTestConfigurationsTest(unittest.TestCase):
    def setUp(self):
        # Save environment so tests don't leak state
        self._orig_env = os.environ.copy()
        # Save sys.argv so tests don't leak state
        self._orig_argv = sys.argv.copy()
        # Save module-level attributes that tests may change
        self._orig_functional_matrix = fetch_test_configurations.functional_matrix
        self._orig_benchmark_matrix = fetch_test_configurations.benchmark_matrix
        self._orig_test_matrix = deepcopy(fetch_test_configurations.test_matrix)
        self._orig_get_all_families = (
            fetch_test_configurations.get_all_families_for_trigger_types
        )

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
        # Restored in place: tests that opt a component into emulation edit the
        # shared dict, and a leak turns into failures in unrelated tests.
        fetch_test_configurations.test_matrix.clear()
        fetch_test_configurations.test_matrix.update(self._orig_test_matrix)
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

    def test_windows_hip_tests_default_emits_pal_only(self):
        """On Windows, hip-tests emits only PAL by default (WINDOWS_HIP_ROCR_TESTS off)."""
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
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
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
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
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]
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

    def test_asan_build_uses_sandbox_runner(self):
        """ASAN builds should use test-runs-on-sandbox when available."""
        os.environ["BUILD_VARIANT"] = "asan"
        os.environ["PROJECTS_TO_TEST"] = "rocblas"

        def fake_get_all_families(_):
            return {
                "gfx94x": {
                    "linux": {
                        "test-runs-on": "linux-gfx942-prod",
                        "test-runs-on-labels": [
                            {"label": "linux-gfx942-a", "count": 5},
                            {"label": "linux-gfx942-b", "count": 5},
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

        rocblas = next(j for j in components if j["job_name"] == "rocblas")
        self.assertEqual(rocblas["test_runner"], "linux-mi325-gpu-rocm-cpu-sandbox")

    def test_host_asan_build_uses_sandbox_runner(self):
        """host-asan builds should also use test-runs-on-sandbox."""
        os.environ["BUILD_VARIANT"] = "host-asan"
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
    # Emulated test jobs
    # -----------------------

    def _emulated(self, components):
        return [job for job in components if job.get("emulator")]

    def test_mirage_profile_lookup(self):
        get_profile = fetch_test_configurations.get_mirage_profile

        # Family labels as AMDGPU_FAMILIES actually spells them.
        self.assertEqual(get_profile("gfx950-dcgpu"), "mi350x")
        self.assertEqual(get_profile("gfx125X-dcgpu"), "mi450x")
        self.assertEqual(get_profile("gfx94X-dcgpu"), "mi300x")
        # Bare targets.
        self.assertEqual(get_profile("gfx942"), "mi300x")
        self.assertEqual(get_profile("gfx1250"), "mi450x")
        # Unemulated / missing families.
        self.assertIsNone(get_profile("gfx1151"))
        self.assertIsNone(get_profile("gfx110X-all"))
        self.assertIsNone(get_profile(""))
        self.assertIsNone(get_profile(None))

    def test_mirage_profile_prefers_longest_prefix(self):
        # No key in the real map is a prefix of another, so use a map where two
        # nested prefixes disagree: only longest-prefix matching can produce
        # "specific" here. Guards against a future entry that splits a family
        # label (e.g. a "gfx951" that is not the same profile as "gfx95X").
        original = fetch_test_configurations._MIRAGE_PROFILE_BY_FAMILY_PREFIX
        fetch_test_configurations._MIRAGE_PROFILE_BY_FAMILY_PREFIX = {
            "gfx99": "family-label",
            "gfx999": "specific",
        }
        try:
            get_profile = fetch_test_configurations.get_mirage_profile
            self.assertEqual(get_profile("gfx999-dcgpu"), "specific")
            self.assertEqual(get_profile("gfx990-dcgpu"), "family-label")
        finally:
            fetch_test_configurations._MIRAGE_PROFILE_BY_FAMILY_PREFIX = original

    def test_emulated_variant_added_for_emulated_family(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocrtst"

        fetch_test_configurations.run()
        components = self._get_components()

        names = [job["job_name"] for job in components]
        self.assertCountEqual(names, ["rocrtst", "rocrtst (emulated mi350x)"])

        emulated = next(j for j in components if j["emulator_profile"] == "mi350x")
        hardware = next(j for j in components if j["job_name"] == "rocrtst")

        self.assertEqual(emulated["emulator"], "rocjitsu")
        # TEST_COMPONENT must stay undecorated so test scripts still resolve
        # the component's test directory.
        self.assertEqual(emulated["test_component"], "rocrtst")
        self.assertTrue(emulated["linux_cpu_runner"])
        self.assertEqual(
            emulated["timeout_minutes"],
            min(
                hardware["timeout_minutes"]
                * fetch_test_configurations._EMULATION_TIMEOUT_MULTIPLIER,
                fetch_test_configurations._EMULATION_MAX_TIMEOUT_MINUTES,
            ),
        )
        self.assertEqual(emulated["total_shards"], 1)
        self.assertEqual(emulated["shard_arr"], [1])
        # Emulated jobs additionally fetch the emulator itself.
        self.assertEqual(
            emulated["fetch_artifact_args"], "--rocrtst --tests --mirage --rocjitsu"
        )
        self.assertIn("test_runner.py", emulated["test_script"])
        # The hardware variant is untouched.
        self.assertNotIn("emulator", hardware)
        self.assertNotIn("linux_cpu_runner", hardware)
        self.assertIn("test_runner.py", hardware["test_script"])

    def test_emulated_test_script_is_wrapped_in_mirage_run(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocrtst"

        fetch_test_configurations.run()
        script = self._emulated(self._get_components())[0]["test_script"]

        prefix, workload = script.split(" -- ", 1)
        # The whole test script is the workload, so everything it does happens
        # inside one mirage session -- and it is the *same* entry point the
        # hardware job uses, so emulation does not fork the test definition.
        self.assertEqual(
            workload,
            "python build_tools/github_actions/test_executable_scripts/test_runner.py",
        )
        self.assertTrue(prefix.startswith("$THEROCK_BIN_DIR/mirage run "), prefix)
        self.assertIn("--profile mi350x", prefix)
        self.assertIn("--emulator rocjitsu", prefix)

    def test_emulated_test_script_forwards_the_ci_environment(self):
        # mirage's host clears the environment, so anything the test script
        # reads has to cross the session boundary by name.
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocrtst"

        fetch_test_configurations.run()
        prefix = self._emulated(self._get_components())[0]["test_script"].split(" -- ")[
            0
        ]

        for name in fetch_test_configurations._EMULATION_FORWARDED_ENV:
            self.assertIn(f"--env {name}=${name}", prefix, name)
        # The emulator identity is baked in as a literal, not read from the
        # environment, so the command reproduces a run on its own.
        self.assertIn("--env TEST_EMULATOR=rocjitsu", prefix)
        self.assertIn("--env TEST_EMULATOR_PROFILE=mi350x", prefix)

    def test_forwarded_env_covers_what_the_scripts_read(self):
        # A variable an emulated test script reads but that is not forwarded
        # silently becomes empty inside the mirage session. Read the scripts
        # rather than restating the constant, so this can actually fail when
        # someone starts reading a new variable.
        scripts_dir = (
            Path(fetch_test_configurations.__file__).resolve().parent
            / "test_executable_scripts"
        )
        # Names a script reads on purpose without forwarding them.
        not_forwarded = {
            # Baked into the wrapper as literals so the command reproduces a
            # run on its own.
            "TEST_EMULATOR",
            "TEST_EMULATOR_PROFILE",
            # test_runner.py derives ROCM_PATH from THEROCK_BIN_DIR and then
            # sets it for the tests it launches; it never reads an inherited
            # value.
            "ROCM_PATH",
            # mirage re-inherits PATH into the session itself, along with
            # HOME/USER/LANG/LC_ALL/TERM/TMPDIR.
            "PATH",
            # Deliberately not forwarded. mirage owns the loader environment
            # inside a session -- it sets LD_PRELOAD to the rocjitsu shim that
            # interposes the HSA runtime, and sets no LD_LIBRARY_PATH of its
            # own. Carrying the host's value in risks shadowing that shim with
            # the real runtime from the artifacts. test_runner.py only ever
            # prepends to it, so an empty inherited value is the safe default.
            "LD_LIBRARY_PATH",
        }
        read_pattern = re.compile(
            r"""(?:os\.getenv\(|os\.environ\[|env\.get\()["']([A-Z][A-Z0-9_]*)["']"""
        )
        emulated_scripts = {"emulation.py"}
        for config in fetch_test_configurations.test_matrix.values():
            if "emulate" not in config:
                continue
            script = config["test_script"]
            emulated_scripts.add(Path(script.rsplit(" ", 1)[-1]).name)

        for script in sorted(emulated_scripts):
            source = (scripts_dir / script).read_text()
            for name in sorted(set(read_pattern.findall(source)) - not_forwarded):
                self.assertIn(
                    name,
                    fetch_test_configurations._EMULATION_FORWARDED_ENV,
                    f"{script} reads {name}, which would be empty inside the "
                    "mirage session unless it is forwarded (or added to the "
                    "not_forwarded set here with a reason)",
                )

    def test_no_test_script_is_quoted(self):
        # test_script is passed through two layers of shell quoting:
        # test_component.yml wraps it in single quotes (so $THEROCK_BIN_DIR
        # reaches the user unexpanded) and reproduce_test_failure.py re-wraps
        # it in double quotes. Either kind of quote in the value ends one of
        # those early and the rest is re-parsed as shell words, so keep both
        # out. Checked across the whole matrix, not just the emulated jobs,
        # since the quoting is shared -- and on the generated jobs too, which
        # is where the mirage wrapper would show up.
        for matrix in (
            fetch_test_configurations.test_matrix,
            fetch_test_configurations.functional_matrix,
            fetch_test_configurations.benchmark_matrix,
        ):
            for key, config in matrix.items():
                for quote in ("'", '"'):
                    self.assertNotIn(quote, config["test_script"], f"{key}.test_script")

        for family in ("gfx950-dcgpu", "gfx125X-dcgpu", "gfx94X-dcgpu"):
            with self.subTest(family=family):
                os.environ["AMDGPU_FAMILIES"] = family
                self.gha_output.clear()
                fetch_test_configurations.run()
                for job in self._get_components():
                    for quote in ("'", '"'):
                        self.assertNotIn(quote, job["test_script"], job["job_name"])

    def test_every_emulated_job_runs_on_a_cpu_node(self):
        # rocjitsu emulates the GPU in software, so no emulated job may be
        # routed to GPU hardware: it would occupy a scarce runner for the whole
        # (10x) emulated timeout and use none of it. This is the invariant the
        # `linux_cpu_runner` flag exists to express; test_artifacts.yml checks
        # it ahead of every other routing rule.
        for family in ("gfx950-dcgpu", "gfx125X-dcgpu"):
            with self.subTest(family=family):
                os.environ["AMDGPU_FAMILIES"] = family
                self.gha_output.clear()
                fetch_test_configurations.run()
                emulated = self._emulated(self._get_components())
                self.assertGreater(len(emulated), 0)
                for job in emulated:
                    self.assertIs(job["linux_cpu_runner"], True, job["job_name"])
                    # No multi-GPU routing, which is checked before
                    # linux_cpu_runner in some consumers' chains.
                    self.assertNotIn("multi_gpu_runner", job)
                    # And no GPU devices mapped into the container, so even a
                    # misrouted job cannot reach a real device.
                    for gpu_option in ("/dev/kfd", "/dev/dri", "--group-add video"):
                        self.assertNotIn(gpu_option, job["container_options"])
                    self.assertIn("--ipc host", job["container_options"])
                    # The pod's CPU budget still has to reach the container:
                    # these are the most CPU-hungry jobs in the matrix and
                    # would otherwise size themselves from the node's cores.
                    self.assertIn("-e KUBE_CPU_REQUEST", job["container_options"])

    def test_no_emulated_variant_for_unemulated_family(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx1151"

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(self._emulated(components), [])
        # Components that only exist to run emulated are dropped entirely.
        names = {job["job_name"] for job in components}
        self.assertNotIn("emulation-smoke", names)

    def test_no_emulated_variant_for_mapped_but_unscheduled_profile(self):
        # gfx94X maps to the mi300x profile, but we do not schedule emulated
        # jobs for it -- that family has plenty of hardware.
        os.environ["AMDGPU_FAMILIES"] = "gfx94X-dcgpu"

        fetch_test_configurations.run()
        components = self._get_components()

        self.assertEqual(self._emulated(components), [])
        self.assertIn("rocrtst", {job["job_name"] for job in components})

    def test_no_emulated_variant_on_windows(self):
        # The emulator is Linux-only.
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        sys.argv = ["fetch_test_configurations.py", "--platform=windows"]

        fetch_test_configurations.run()

        self.assertEqual(self._emulated(self._get_components()), [])

    def test_emulate_only_components_selected(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx125X-dcgpu"

        fetch_test_configurations.run()
        components = self._get_components()

        names = {job["job_name"] for job in components}
        self.assertIn("emulation-smoke (emulated mi450x)", names)
        # emulate_only components have no hardware variant.
        self.assertNotIn("emulation-smoke", names)

    def test_emulated_variant_respects_exclude_family(self):
        # rocroller excludes gfx950-dcgpu here; the emulated variant must be
        # excluded with it rather than sneaking the component back in.
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocroller"
        # tearDown restores test_matrix, so mutate it directly.
        fetch_test_configurations.test_matrix["rocroller"]["emulate"] = "rocjitsu"
        fetch_test_configurations.test_matrix["rocroller"]["exclude_family"] = {
            "linux": ["gfx950-dcgpu"]
        }

        fetch_test_configurations.run()

        self.assertEqual(self._get_components(), [])

    def test_emulation_keys_never_reach_the_matrix(self):
        # These four keys configure the generator; leaking them into the
        # emitted JSON would make workflow expressions like `component.emulate`
        # silently truthy on hardware. Spelled out rather than read from
        # _EMULATION_MATRIX_KEYS: production strips keys by iterating that same
        # constant, so sharing it would make this test pass for any value of it.
        emulation_keys = (
            "emulate",
            "emulate_only",
            "emulate_test_type",
            "emulate_env",
        )
        self.assertCountEqual(
            emulation_keys,
            fetch_test_configurations._EMULATION_MATRIX_KEYS,
            "a new emulation matrix key must be added to this test too",
        )
        for family in ("gfx950-dcgpu", "gfx125X-dcgpu", "gfx94X-dcgpu"):
            with self.subTest(family=family):
                os.environ["AMDGPU_FAMILIES"] = family
                self.gha_output.clear()
                fetch_test_configurations.run()
                for job in self._get_components():
                    for key in emulation_keys:
                        self.assertNotIn(key, job, f"{job['job_name']} leaked {key}")

    def test_emulated_variant_drops_exclusive_base_only_flag(self):
        # install_rocm_from_artifacts.py treats --base-only as exclusive: it
        # would skip the branch that fetches mirage/rocjitsu entirely.
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "emulation-smoke"

        fetch_test_configurations.run()
        emulated = self._emulated(self._get_components())[0]

        self.assertEqual(
            fetch_test_configurations.test_matrix["emulation-smoke"][
                "fetch_artifact_args"
            ],
            "--base-only",
        )
        self.assertEqual(emulated["fetch_artifact_args"], "--mirage --rocjitsu")

    def test_emulated_timeouts_fit_under_the_job_timeout(self):
        # test_component.yml caps the whole *job*, not just the Test step. The
        # step budget therefore has to leave room for everything ahead of it --
        # two checkouts, the artifact fetch (its own 15 min timeout), health
        # status -- or the job timeout fires first and the "Print test
        # reproduction command" step (gated on the Test step failing) is
        # skipped, so the failure is reported with no diagnostics at all.
        workflow = workflow_utils.load_workflow(
            workflow_utils.WORKFLOWS_DIR / "test_component.yml"
        )
        job_timeout_minutes = workflow_utils.get_workflow_job(
            workflow, "test_component"
        )["timeout-minutes"]
        setup_budget_minutes = 25
        step_budget = job_timeout_minutes - setup_budget_minutes
        for family in ("gfx950-dcgpu", "gfx125X-dcgpu"):
            with self.subTest(family=family):
                os.environ["AMDGPU_FAMILIES"] = family
                self.gha_output.clear()
                fetch_test_configurations.run()
                for job_config in self._emulated(self._get_components()):
                    self.assertLessEqual(
                        job_config["timeout_minutes"],
                        step_budget,
                        f"{job_config['job_name']} would be killed by the "
                        f"{job_timeout_minutes} min job timeout before its own "
                        "step timeout fires",
                    )

    def test_emulated_variant_always_fetches_the_emulator(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"

        fetch_test_configurations.run()
        emulated = self._emulated(self._get_components())
        self.assertGreater(len(emulated), 0)
        for job in emulated:
            args = job["fetch_artifact_args"].split()
            self.assertIn("--mirage", args)
            self.assertIn("--rocjitsu", args)
            self.assertNotIn("--base-only", args)

    def test_emulated_variant_drops_multi_gpu_routing(self):
        # A CPU runner has no GPUs, so multi-GPU routing must not survive into
        # the emulated variant (test_artifacts.yml checks multi_gpu_runner
        # before linux_cpu_runner).
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rccl"
        # tearDown restores test_matrix, so mutate it directly.
        fetch_test_configurations.test_matrix["rccl"]["emulate"] = "rocjitsu"

        fetch_test_configurations.run()

        emulated = self._emulated(self._get_components())[0]
        self.assertNotIn("multi_gpu_runner", emulated)
        self.assertNotIn("multi_gpu", emulated)
        self.assertTrue(emulated["linux_cpu_runner"])

    def test_emulate_test_type_pins_the_tier(self):
        # A pin, not a default: how long a tier takes under emulation is a
        # property of the emulator, so a nightly asking for "comprehensive"
        # must not drag the emulated variant along with it. The hardware
        # variant keeps the tier the run asked for.
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocrtst"
        os.environ["TEST_TYPE"] = "comprehensive"

        fetch_test_configurations.run()
        components = self._get_components()

        pinned = fetch_test_configurations.test_matrix["rocrtst"]["emulate_test_type"]
        emulated = self._emulated(components)[0]
        hardware = next(j for j in components if j["job_name"] == "rocrtst")
        self.assertEqual(emulated["test_type"], pinned)
        self.assertEqual(hardware["test_type"], "comprehensive")

    def test_emulate_test_type_defaults_to_the_hardware_tier(self):
        # Components that do not declare one follow the run's TEST_TYPE, so
        # opting into emulation stays a one-field change.
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocblas"
        os.environ["TEST_TYPE"] = "quick"
        # tearDown restores test_matrix, so mutate it directly.
        fetch_test_configurations.test_matrix["rocblas"]["emulate"] = "rocjitsu"

        fetch_test_configurations.run()

        self.assertEqual(
            self._emulated(self._get_components())[0]["test_type"], "quick"
        )

    def test_emulated_timeout_is_capped(self):
        # The 10x multiplier is a blunt default; without a ceiling a component
        # with a generous hardware budget would let a wedged emulator sit on a
        # CPU-cluster slot for hours.
        cap = fetch_test_configurations._EMULATION_MAX_TIMEOUT_MINUTES
        for family in ("gfx950-dcgpu", "gfx125X-dcgpu"):
            with self.subTest(family=family):
                os.environ["AMDGPU_FAMILIES"] = family
                self.gha_output.clear()
                fetch_test_configurations.run()
                emulated = self._emulated(self._get_components())
                self.assertGreater(len(emulated), 0)
                for job in emulated:
                    self.assertLessEqual(job["timeout_minutes"], cap, job["job_name"])

    def test_emulate_env_is_set_inside_the_session(self):
        os.environ["AMDGPU_FAMILIES"] = "gfx950-dcgpu"
        os.environ["PROJECTS_TO_TEST"] = "rocrtst"

        fetch_test_configurations.run()
        prefix = self._emulated(self._get_components())[0]["test_script"].split(" -- ")[
            0
        ]

        emulate_env = fetch_test_configurations.test_matrix["rocrtst"]["emulate_env"]
        for name, value in emulate_env.items():
            # A literal, like TEST_EMULATOR: the printed command has to
            # reproduce the run on its own, and mirage clears the environment
            # anyway so there would be nothing to expand from.
            self.assertIn(f"--env {name}={value}", prefix)

    def test_emulate_env_needs_no_shell_quoting(self):
        # emulate_env is spliced into a shell command word-by-word, and
        # test_component.yml then wraps the whole test_script in single quotes.
        # Whitespace would split a value into extra mirage arguments and a
        # quote would end the wrapping early, so keep both out rather than
        # growing a quoting scheme no caller needs yet.
        for key, config in fetch_test_configurations.test_matrix.items():
            for name, value in (config.get("emulate_env") or {}).items():
                with self.subTest(component=key, variable=name):
                    self.assertNotIn(" ", f"{name}{value}")
                    self.assertNotIn("'", f"{name}{value}")
                    self.assertNotIn('"', f"{name}{value}")

    def test_emulate_test_type_is_a_known_category(self):
        # test_runner.py validates TEST_TYPE against its own list and silently
        # falls back to "quick" on a typo, which would quietly change what the
        # emulated job covers.
        valid = {"quick", "standard", "comprehensive", "full"}
        for key, config in fetch_test_configurations.test_matrix.items():
            if "emulate_test_type" in config:
                with self.subTest(component=key):
                    self.assertIn(config["emulate_test_type"], valid)

    def test_every_emulated_matrix_entry_is_wired_up(self):
        # test_script values are repo-root-relative POSIX paths.
        repo_root = Path(fetch_test_configurations.__file__).resolve().parents[2]
        for key, config in fetch_test_configurations.test_matrix.items():
            if "emulate" not in config:
                continue
            with self.subTest(component=key):
                # Only Linux runs the emulator.
                self.assertEqual(config["platform"], ["linux"])
                # Every referenced script must exist.
                script = config["test_script"].rsplit(" ", 1)[-1]
                self.assertTrue(
                    (repo_root / script).is_file(),
                    f"{key}.test_script points at a missing script: {script}",
                )


if __name__ == "__main__":
    unittest.main()
