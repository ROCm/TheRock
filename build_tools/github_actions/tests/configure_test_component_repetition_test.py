# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import amdgpu_family_matrix
import baseline_runs
import configure_test_component_repetition as config
import fetch_test_configurations as test_config


class ComponentRepetitionPlanTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        (self.path / "ci_config_api.py").touch()
        self.env = dict(
            os.environ,
            AMDGPU_FAMILIES="gfx120X-all",
            PLATFORMS="linux",
            TEST_LABELS="test:rocblas",
            REPETITIONS="2",
            CI_CONFIG_PATH=str(self.path),
        )
        self.env.pop("TEST_TYPE", None)
        self.env.pop("AMDGPU_TARGETS", None)
        self.env["BUILD_VARIANT"] = "release"
        self.runners = {
            key: {
                p: {"test-runs-on": f"triage-{p}-{key}"} for p in ("linux", "windows")
            }
            for key in config.REPETITION_FAMILIES
        }
        self.external = {"runner_labels": self.runners}
        self.component = dict(
            job_name="rocblas",
            test_type="comprehensive",
            test_runner="triage-linux-gfx120x",
            shard_arr=[1, 2],
            total_shards=2,
            test_script="python test.py",
            fetch_artifact_args="--blas",
        )
        self.real_fetch = config.fetch_repetition_components
        self.components = self.mock(
            config, "fetch_repetition_components", [self.component]
        )
        self.external_mock = self.mock(
            config, "load_external_runner_config", self.external
        )
        self.mock(amdgpu_family_matrix, "load_external_runner_config", self.external)
        self.mock(config.subprocess, "check_output", "a" * 40)

    def mock(self, module, name, value):
        return self.enterContext(patch.object(module, name, return_value=value))

    def plan(self, **overrides):
        return config.create_repetition_plan(dict(self.env, **overrides))

    def test_plan_preserves_components_iterations_shards_and_pinned_config(self):
        matrices, manifest = self.plan(AMDGPU_TARGETS="gfx1200")
        matrix = matrices["linux_gfx120x"]
        expected_component = dict(self.component)
        expected_component["fetch_artifact_args"] += " --run-github-repo ROCm/rockrel"
        self.assertEqual(
            matrix, {"iteration": [1, 2], "component": [expected_component]}
        )
        rows = manifest["expected"]
        self.assertEqual(
            [(e["iteration"], e["shard"]) for e in rows],
            [(1, 1), (1, 2), (2, 1), (2, 2)],
        )
        self.assertEqual(len({e["sample_id"] for e in rows}), 2)
        self.assertEqual(
            (manifest["ci_config_sha"], manifest["amdgpu_targets"]),
            ("a" * 40, "gfx1200"),
        )
        self.assertEqual(matrices["windows_gfx120x"], {"include": []})

    def test_rejects_invalid_inputs_and_component_configuration(self):
        invalid = dict(
            REPETITIONS=("0", "101", "2.5"),
            AMDGPU_FAMILIES=("", "gfx942", "gfx120X-wrong"),
            PLATFORMS=("wsl",),
            TEST_LABELS=("", "test:typo", "ci:run-all-archs"),
            BUILD_VARIANT=("asan",),
            AMDGPU_TARGETS=("gfx942",),
            TEST_TYPE=("invalid",),
        )
        for name, values in invalid.items():
            for value in values:
                with self.subTest(input=name, value=value):
                    self.assertRaises(ValueError, self.plan, **{name: value})
        self.assertRaises(
            ValueError,
            self.plan,
            AMDGPU_FAMILIES="gfx120X-all,gfx110X-all",
            AMDGPU_TARGETS="gfx1200",
        )
        for name, value in (
            ("test_runner", "production-runner"),
            ("test_type", "quick"),
            ("shard_arr", []),
            ("shard_arr", [2]),
        ):
            with self.subTest(component_field=name, value=value):
                self.components.return_value = [dict(self.component, **{name: value})]
                self.assertRaises(ValueError, self.plan)

    def test_both_platforms_all_families_and_matrix_limit(self):
        def components(*, family, platform, environment):
            runner = f"triage-{platform}-{family.lower().split('-')[0]}"
            return [
                dict(self.component, job_name=f"component{i}", test_runner=runner)
                for i in range(3)
            ]

        self.components.side_effect = components
        matrices, manifest = self.plan(
            AMDGPU_FAMILIES="gfx120X-all,gfx110X-all,gfx1151", PLATFORMS=""
        )
        self.assertEqual(len(manifest["targets"]), 6)
        self.assertEqual(len(manifest["expected"]), 72)
        for target, matrix in matrices.items():
            self.assertEqual(
                matrix["component"][0]["test_runner"],
                f"triage-{target.replace('_', '-')}",
            )
        self.assertRaisesRegex(ValueError, "exceeds 256", self.plan, REPETITIONS="100")
        _, manifest = self.plan(REPETITIONS="85")
        self.assertEqual(len(manifest["expected"]), 510)

    def test_custom_routing_and_missing_configuration(self):
        for overrides, runner in (
            (dict(linux_cpu_runner=True, test_runner="triage-cpu"), "triage-cpu"),
            (dict(multi_gpu_runner="triage-multi-gpu"), "triage-multi-gpu"),
        ):
            with self.subTest(runner=runner):
                self.components.return_value = [dict(self.component, **overrides)]
                self.assertRaisesRegex(ValueError, "not explicitly mapped", self.plan)
                self.external["cpu_test_runners"] = {"linux": runner}
                self.runners["gfx120x"]["linux"]["test-runs-on-multi-gpu"] = runner
                _, manifest = self.plan()
                self.assertEqual(
                    {e["runner_label"] for e in manifest["expected"]}, {runner}
                )
        for external, components, error in (
            (None, [self.component], "refusing local runner fallback"),
            ({"runner_labels": {}}, [self.component], "No explicit custom GPU runner"),
            (self.external, [], "No selected components"),
        ):
            with self.subTest(error=error):
                self.external_mock.return_value = external
                self.components.return_value = components
                self.assertRaisesRegex(ValueError, error, self.plan)

    def test_baseline_selection_and_missing_baseline(self):
        run = dict(
            id=123,
            event="schedule",
            status="completed",
            conclusion="success",
            head_sha="b" * 40,
            html_url="https://example/run/123",
            created_at="2026-11-26T00:00:00Z",
        )
        runs = self.mock(
            baseline_runs,
            "query_successful_workflow_runs",
            [dict(run, id=456, event="workflow_dispatch"), run],
        )
        outputs = self.mock(config, "gha_set_output", None)
        path = self.path / "manifest.json"
        config.write_repetition_outputs(self.env, path)
        manifest = json.loads(path.read_text())
        values = outputs.call_args.args[0]
        self.assertEqual(
            (manifest["baseline"]["run_id"], values["artifact_run_id"]), ("123", "123")
        )
        targets = json.loads(values["target_matrix"])["include"]
        self.assertEqual(len(targets), 1)
        self.assertEqual(
            (targets[0]["platform"], targets[0]["family"], targets[0]["family_key"]),
            ("linux", "gfx120X-all", "gfx120x"),
        )
        self.assertEqual(targets[0]["repetition_matrix"]["iteration"], [1, 2])
        self.assertEqual(
            targets[0]["repetition_matrix"]["component"][0]["test_runner"],
            "triage-linux-gfx120x",
        )
        runs.return_value = []
        self.assertRaisesRegex(
            ValueError,
            "No successful scheduled",
            config.write_repetition_outputs,
            self.env,
            path,
        )

    def write_custom_config(self):
        # The subprocess reads this minimal custom CI API.
        (self.path / "ci_config_api.py").write_text(f"""class Config:
    build_runners = {{}}
    _raw = {{}}
    def get_gpu_runner_labels(self):
        return {self.runners!r}
def load_config(version, config_path):
    return Config()
""")

    def test_real_configuration_helper_preserves_selected_tier_runner_and_shards(self):
        self.write_custom_config()
        self.components.side_effect = self.real_fetch
        for tier in (None, "quick", "standard", "full"):
            with self.subTest(tier=tier):
                matrices, manifest = self.plan(**({"TEST_TYPE": tier} if tier else {}))
                component = matrices["linux_gfx120x"]["component"][0]
                self.assertEqual(component["job_name"], "rocblas")
                self.assertEqual(component["test_runner"], "triage-linux-gfx120x")
                self.assertEqual(
                    component["shard_arr"],
                    [1] if tier == "quick" else [1, 2, 3, 4, 5, 6],
                )
                self.assertEqual(
                    (component["test_type"], manifest["test_type"]),
                    (tier or "comprehensive",) * 2,
                )

    def test_empty_platform_selects_only_linux_for_real_amdsmi_configuration(self):
        for runners in self.runners.values():
            runners.pop("windows")
        self.write_custom_config()
        self.components.side_effect = self.real_fetch
        matrices, manifest = self.plan(
            AMDGPU_FAMILIES="gfx120x,gfx110x",
            TEST_LABELS="test:amdsmi",
            PLATFORMS="",
            REPETITIONS="20",
        )
        self.assertEqual(
            {(t["platform"], t["family"]) for t in manifest["targets"]},
            {("linux", "gfx120X-all"), ("linux", "gfx110X-all")},
        )
        self.assertEqual(len(manifest["expected"]), 40)
        self.assertEqual({row["platform"] for row in manifest["expected"]}, {"linux"})
        self.assertEqual({row["component"] for row in manifest["expected"]}, {"amdsmi"})
        self.assertEqual(manifest["omitted_platforms"], ["windows"])
        self.assertEqual(matrices["windows_gfx120x"], {"include": []})
        self.assertEqual(matrices["windows_gfx110x"], {"include": []})
        self.assertEqual(
            [call.kwargs["platform"] for call in self.components.call_args_list],
            ["linux", "linux"],
        )

    def test_explicit_unsupported_platform_explains_component_support(self):
        for platforms in ("windows", "linux,windows"):
            with self.subTest(platforms=platforms):
                self.assertRaisesRegex(
                    ValueError,
                    r"test:amdsmi.*platform\(s\): windows.*Supported pilot platforms: linux",
                    self.plan,
                    TEST_LABELS="test:amdsmi",
                    PLATFORMS=platforms,
                )
        self.components.assert_not_called()

    def test_empty_platform_does_not_accept_no_supported_pilot_platform(self):
        with patch.dict(
            test_config.test_matrix,
            amdsmi=dict(test_config.test_matrix["amdsmi"], platform=["wsl"]),
        ):
            self.assertRaisesRegex(
                ValueError,
                "no configured tests on the pilot platforms",
                self.plan,
                TEST_LABELS="test:amdsmi",
                PLATFORMS="",
            )

    def test_empty_platform_expands_label_groups_and_mixed_component_support(self):
        def components(*, family, platform, environment):
            key = family.lower().split("-")[0]
            return [dict(self.component, test_runner=f"triage-{platform}-{key}")]

        self.components.side_effect = components
        for labels, platforms, omitted in (
            ("test:tensilelite", {"linux"}, ["windows"]),
            ("test:amdsmi,test:rocblas", {"linux", "windows"}, []),
        ):
            with self.subTest(labels=labels):
                _, manifest = self.plan(TEST_LABELS=labels, PLATFORMS="")
                self.assertEqual(
                    {target["platform"] for target in manifest["targets"]},
                    platforms,
                )
                self.assertEqual(manifest["omitted_platforms"], omitted)

    def test_selected_target_output_preserves_each_nested_matrix(self):
        def components(*, family, platform, environment):
            key = family.lower().split("-")[0]
            return [dict(self.component, test_runner=f"triage-{platform}-{key}")]

        self.components.side_effect = components
        self.mock(
            baseline_runs,
            "query_successful_workflow_runs",
            [
                dict(
                    id=123,
                    event="schedule",
                    status="completed",
                    conclusion="success",
                    head_sha="b" * 40,
                    html_url="https://example/run/123",
                    created_at="2026-11-26T00:00:00Z",
                )
            ],
        )
        outputs = self.mock(config, "gha_set_output", None)
        path = self.path / "manifest.json"
        config.write_repetition_outputs(
            dict(
                self.env,
                AMDGPU_FAMILIES="gfx120X-all,gfx110X-all,gfx1151",
                PLATFORMS="",
            ),
            path,
        )
        targets = json.loads(outputs.call_args.args[0]["target_matrix"])["include"]
        self.assertEqual(
            {(target["platform"], target["family_key"]) for target in targets},
            {
                (platform, key)
                for platform in ("linux", "windows")
                for key in config.REPETITION_FAMILIES
            },
        )
        samples = set()
        for target in targets:
            matrix = target["repetition_matrix"]
            self.assertEqual(matrix["iteration"], [1, 2])
            component = matrix["component"][0]
            self.assertEqual(component["shard_arr"], [1, 2])
            self.assertEqual(
                component["test_runner"],
                f"triage-{target['platform']}-{target['family_key']}",
            )
            self.assertEqual(
                target["family"], config.REPETITION_FAMILIES[target["family_key"]]
            )
            for iteration in matrix["iteration"]:
                samples.add(
                    f"repeat-{target['platform']}-{target['family_key']}"
                    f"-{iteration}-{component['job_name']}"
                )
        manifest = json.loads(path.read_text())
        self.assertEqual(samples, {row["sample_id"] for row in manifest["expected"]})


if __name__ == "__main__":
    unittest.main()
