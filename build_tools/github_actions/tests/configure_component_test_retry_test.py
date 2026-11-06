# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import configure_component_test_retry as configure


class ConfigureRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.families = {
            "gfx94x": {
                "linux": {
                    "family": "gfx94X-dcgpu",
                    "test-runs-on": "linux-gpu",
                    "test-runs-on-wsl": "windows-wsl-gpu",
                    "fetch-gfx-targets": ["gfx942", "gfx942:xnack+"],
                    "tests_on_trigger": ["nightly"],
                    "build_variants": ["release"],
                },
            },
        }
        self.cpu = patch.object(
            configure,
            "get_cpu_test_runner",
            side_effect=lambda platform: platform + "-cpu",
        )
        self.cpu.start()
        self.addCleanup(self.cpu.stop)

    def test_family_alias_and_targets(self) -> None:
        targets = configure.select_targets(
            self.families, "gfx94X-dcgpu", "linux", "standard"
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["family"], "gfx94X-dcgpu")
        self.assertEqual(targets[0]["targets"], "gfx942,gfx942:xnack+")
        self.assertEqual(targets[0]["cpu_runner"], "linux-cpu")

    def test_wsl_uses_linux_artifacts_and_wsl_runner(self) -> None:
        targets = configure.select_targets(self.families, "", "wsl", "standard")
        self.assertEqual(targets[0]["runner"], "windows-wsl-gpu")
        self.assertEqual(targets[0]["cpu_runner"], "windows-wsl-gpu")
        self.assertEqual(targets[0]["family"], "gfx94X-dcgpu")

    def test_unknown_selection_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown GPU"):
            configure.select_targets(self.families, "gfx999", "linux", "standard")
        with self.assertRaisesRegex(ValueError, "Platforms"):
            configure.select_targets(self.families, "", "macos", "standard")

    def test_disabled_runner_is_not_dispatched(self) -> None:
        self.families["gfx94x"]["linux"]["test-runs-on"] = ""
        with self.assertRaisesRegex(ValueError, "No requested"):
            configure.select_targets(self.families, "", "linux", "standard")

    def test_disabled_tests_are_not_dispatched(self) -> None:
        self.families["gfx94x"]["linux"]["tests_on_trigger"] = []
        with self.assertRaises(ValueError):
            configure.select_targets(self.families, "", "linux", "standard")

    def test_family_tier_override(self) -> None:
        self.families["gfx94x"]["linux"]["test_type_for_family"] = "quick"
        targets = configure.select_targets(self.families, "", "linux", "full")
        self.assertEqual(targets[0]["test_type"], "quick")

    def test_quick_disabled_by_policy(self) -> None:
        self.families["gfx94x"]["linux"]["run-full-tests-only"] = True
        with self.assertRaises(ValueError):
            configure.select_targets(self.families, "", "linux", "quick")

    def test_absent_platform_does_not_duplicate_targets(self) -> None:
        targets = configure.select_targets(
            self.families, "gfx94x,gfx94X-dcgpu", "linux,windows", "standard"
        )
        self.assertEqual(len(targets), 1)

    def test_baseline_requires_completed_success(self) -> None:
        api = Mock()
        api.send_request.return_value = {"status": "completed", "conclusion": "failure"}
        with self.assertRaisesRegex(ValueError, "successful"):
            configure.validate_baseline(api, "ROCm/rockrel", "123")
        api.send_request.return_value = {
            "status": "completed",
            "conclusion": "success",
            "html_url": "https://example.com/run",
        }
        configure.validate_baseline(api, "ROCm/rockrel", "123")

    def test_invalid_baseline_identity_does_not_call_api(self) -> None:
        api = Mock()
        with self.assertRaises(ValueError):
            configure.validate_baseline(api, "ROCm/rockrel", "../123")
        api.send_request.assert_not_called()

    def test_explicit_baseline_skips_nightly_lookup(self) -> None:
        api = Mock()
        self.assertEqual(
            configure.resolve_baseline(api, "ROCm/rockrel", " 123 "), "123"
        )
        api.send_request.assert_not_called()

    def test_empty_baseline_selects_successful_main_schedule(self) -> None:
        api = Mock()
        api.send_request.return_value = {
            "workflow_runs": [
                {
                    "id": 456,
                    "head_branch": "main",
                    "event": "schedule",
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        }
        self.assertEqual(configure.resolve_baseline(api, "ROCm/rockrel", " "), "456")
        url = urlsplit(api.send_request.call_args.args[0])
        self.assertEqual(
            url.path,
            "/repos/ROCm/rockrel/actions/workflows/multi_arch_release.yml/runs",
        )
        self.assertEqual(
            parse_qs(url.query),
            {
                "branch": ["main"],
                "event": ["schedule"],
                "status": ["success"],
                "per_page": ["1"],
            },
        )

    def test_no_successful_nightly_fails(self) -> None:
        api = Mock()
        api.send_request.return_value = {"workflow_runs": []}
        with self.assertRaisesRegex(ValueError, "No successful scheduled"):
            configure.resolve_baseline(api, "ROCm/rockrel", "")

    def test_invalid_nightly_response_fails(self) -> None:
        api = Mock()
        for event, branch, conclusion in [
            ("workflow_dispatch", "main", "success"),
            ("schedule", "feature", "success"),
            ("schedule", "main", "failure"),
        ]:
            with self.subTest(event=event, branch=branch, conclusion=conclusion):
                api.send_request.return_value = {
                    "workflow_runs": [
                        {
                            "id": 456,
                            "head_branch": branch,
                            "event": event,
                            "status": "completed",
                            "conclusion": conclusion,
                        }
                    ]
                }
                with self.assertRaisesRegex(
                    ValueError, "not a successful main nightly"
                ):
                    configure.resolve_baseline(api, "ROCm/rockrel", "")

    def test_configuration_emits_real_matrix_with_escaped_labels(self) -> None:
        labels = '["test:rocblas"]'
        with (
            patch.dict(
                os.environ,
                {"GITHUB_REPOSITORY": "ROCm/TheRock", "GITHUB_SHA": "a" * 40},
            ),
            patch.object(configure, "load_external_runner_config", return_value={}),
            patch.object(
                configure,
                "get_all_families_for_trigger_types",
                return_value=self.families,
            ),
            patch.object(configure, "validate_baseline") as baseline,
            patch.object(configure, "gha_set_output") as output,
        ):
            self.assertEqual(
                configure.main(
                    [
                        "--artifact-run-id",
                        "123",
                        "--platforms",
                        "linux",
                        "--test-labels",
                        labels,
                    ]
                ),
                0,
            )
        baseline.assert_called_once()
        self.assertEqual(output.call_args.args[0]["artifact_run_id"], "123")
        targets = json.loads(output.call_args.args[0]["matrix"])["include"]
        self.assertEqual(len(targets), 1)
        self.assertEqual(json.loads('"' + targets[0]["test_labels"] + '"'), labels)

    def test_auto_selected_id_is_emitted_for_downstream_dispatch(self) -> None:
        api = Mock()
        api.send_request.return_value = {
            "workflow_runs": [
                {
                    "id": 456,
                    "head_branch": "main",
                    "event": "schedule",
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        }
        with (
            patch.dict(
                os.environ,
                {"GITHUB_REPOSITORY": "ROCm/TheRock", "GITHUB_SHA": "a" * 40},
            ),
            patch.object(configure, "GitHubAPI", return_value=api),
            patch.object(configure, "load_external_runner_config", return_value={}),
            patch.object(
                configure,
                "get_all_families_for_trigger_types",
                return_value=self.families,
            ),
            patch.object(configure, "validate_baseline") as baseline,
            patch.object(configure, "gha_set_output") as output,
        ):
            self.assertEqual(configure.main(["--platforms", "linux"]), 0)
        baseline.assert_called_once_with(api, "ROCm/rockrel", "456")
        self.assertEqual(output.call_args.args[0]["artifact_run_id"], "456")

    def test_determine_step_runs_without_runner_configuration(self) -> None:
        api = Mock()
        api.send_request.return_value = {
            "workflow_runs": [
                {
                    "id": 456,
                    "head_branch": "main",
                    "event": "schedule",
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        }
        with (
            patch.object(configure, "GitHubAPI", return_value=api),
            patch.object(configure, "load_external_runner_config") as runner_config,
            patch.object(configure, "validate_baseline") as baseline,
            patch.object(configure, "gha_set_output") as output,
        ):
            self.assertEqual(configure.main(["--resolve-baseline-only"]), 0)
        baseline.assert_called_once_with(api, "ROCm/rockrel", "456")
        output.assert_called_once_with({"artifact_run_id": "456"})
        runner_config.assert_not_called()

    def test_bad_retry_input_fails_before_baseline_lookup(self) -> None:
        with patch.object(configure, "validate_baseline") as baseline:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    configure.main(["--artifact-run-id", "123", "--max-retries", "-1"])
            self.assertEqual(error.exception.code, 2)
            baseline.assert_not_called()

    def test_missing_external_configuration_refuses_fallback(self) -> None:
        with (
            patch.object(configure, "load_external_runner_config", return_value=None),
            patch.object(configure, "validate_baseline") as baseline,
        ):
            self.assertEqual(configure.main(["--artifact-run-id", "123"]), 1)
            baseline.assert_not_called()

    def test_dispatch_inputs_preserve_quoted_labels_and_existing_interface(
        self,
    ) -> None:
        # Render the real workflow input template and verify its runtime contract.
        import yaml

        root = Path(__file__).resolve().parents[3]
        workflow = yaml.safe_load(
            (root / ".github/workflows/component_test_retry.yml").read_text()
        )
        steps = workflow["jobs"]["trigger_monitor_retry"]["steps"]
        dispatch = next(step for step in steps if step.get("id") == "dispatch")
        template = dispatch["with"]["inputs"]
        labels = '["test:rocblas", "test:hipblas"]'
        target = configure.select_targets(self.families, "", "linux", "standard")[0]
        target["test_labels"] = json.dumps(labels)[1:-1]
        for key, value in target.items():
            template = template.replace("${{ matrix." + key + " }}", value)
        template = template.replace(
            "${{ needs.configure.outputs.artifact_run_id }}", "123"
        )
        template = template.replace("${{ github.repository }}", "ROCm/TheRock")
        template = template.replace("${{ github.sha }}", "a" * 40)
        inputs = json.loads(template)
        self.assertEqual(inputs["test_labels"], labels)
        self.assertNotIn("override_test_runs_on", inputs)
        self.assertEqual(inputs["artifact_run_id"], "123")
        self.assertEqual(inputs["test_runs_on_cpu"], "linux-cpu")


if __name__ == "__main__":
    unittest.main()
