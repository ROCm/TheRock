# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Guard/configuration and actual workflow closure; no build/cache experiment."""

import copy
import importlib.util
import argparse
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "normal_control", ROOT / "build_tools/normal_ci_validation.py"
)
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)


def event():
    return {
        "inputs": {
            "profile_normal_validation": "true",
            "profile_normal_case": "baseline",
            "profile_normal_cache_regime": "ci-readonly",
            "linux_amdgpu_families": "gfx94x,gfx103x",
            "windows_amdgpu_families": "",
            "build_stages": "",
            "prebuilt_stages": "",
            "baseline_run_id": "",
            "stage_reuse_mode": "off",
            "changed_projects": "",
            "pr_labels": "test_filter:comprehensive",
            "build_python_packages": "true",
            "build_native_linux": "true",
            "build_native_windows": "false",
            "build_pytorch": "false",
            "build_jax": "false",
        }
    }


def workflow(name):
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "ROCm/TheRock",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "RELEASE_TYPE": "ci",
        }

    def test_both_case_labels_share_controls_and_both_regimes_are_explicit(self):
        a, b = event(), event()
        b["inputs"]["profile_normal_case"] = "candidate"
        aa = CONTROL.validate_context(self.env, a)
        bb = CONTROL.validate_context(self.env, b)
        self.assertEqual(
            {k: v for k, v in aa.items() if k != "profile_normal_case"},
            {k: v for k, v in bb.items() if k != "profile_normal_case"},
        )
        b["inputs"]["profile_normal_cache_regime"] = "private-cold"
        CONTROL.validate_context(self.env, b)

    def test_foreign_repository_local_release_or_wrong_trigger_rejected(self):
        for key, value in [
            ("GITHUB_REPOSITORY", "ROCm/rockrel"),
            ("GITHUB_ACTIONS", "false"),
            ("RELEASE_TYPE", "nightly"),
            ("GITHUB_EVENT_NAME", "push"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                CONTROL.validate_context(dict(self.env, **{key: value}), event())

    def test_scope_changes_rejected(self):
        for key, value in [
            ("linux_amdgpu_families", "gfx103x"),
            ("windows_amdgpu_families", "gfx103x"),
            ("build_stages", "math-libs"),
            ("prebuilt_stages", "compiler-runtime"),
            ("baseline_run_id", "41147801272"),
            ("stage_reuse_mode", "reuse-stage"),
            ("build_python_packages", "false"),
            ("pr_labels", "test_filter:quick"),
            ("profile_normal_validation", "false"),
            ("profile_normal_cache_regime", "merge-trust-domains"),
        ]:
            modified = event()
            modified["inputs"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CONTROL.validate_context(self.env, modified)

    def test_runner_override_preserves_gpu_and_unrelated_configuration(self):
        original = {
            "build_runners": {
                "linux": {
                    "default": [],
                    "small": [],
                    "medium": [],
                    "sanitizer": [{"label": "large"}],
                },
                "windows": {"default": [{"label": "windows"}]},
            },
            "gpu_runner_labels": {"gfx103x": {"linux": {"test-runs-on": ""}}},
        }
        frozen = copy.deepcopy(original)
        updated = CONTROL.fixed_runner_config(original)
        self.assertEqual(updated["gpu_runner_labels"], original["gpu_runner_labels"])
        self.assertEqual(
            updated["build_runners"]["windows"], original["build_runners"]["windows"]
        )
        self.assertEqual(
            updated["build_runners"]["linux"]["sanitizer"],
            original["build_runners"]["linux"]["sanitizer"],
        )
        self.assertEqual(original, frozen)

    def test_cache_regimes_cannot_write_shared_backend_or_change_compiler_check(self):
        text = f"namespace = therock-v4\nremote_storage = {CONTROL.CI_CACHE_REMOTE}\nremote_only = true\ncache_dir = shared\ncompiler_check = content\nlog_file = build/logs/ccache.log\n"
        for regime in ["ci-readonly", "private-cold"]:
            result = CONTROL.controlled_cache_text(text, Path("private"), regime)
            rows = dict(line.split(" = ", 1) for line in result.splitlines())
            self.assertEqual(rows["remote_only"], "false")
            self.assertEqual(rows["cache_dir"], "private")
            self.assertEqual(rows["compiler_check"], "content")
            self.assertEqual(rows["log_file"], "build/logs/ccache.log")
            if regime == "ci-readonly":
                self.assertEqual(rows["read_only"], "true")
                self.assertEqual(
                    rows["remote_storage"], CONTROL.CI_CACHE_REMOTE + "|read-only=true"
                )
            else:
                self.assertEqual(rows["remote_storage"], "")
                self.assertEqual(rows["read_only"], "false")
        with self.assertRaises(ValueError):
            CONTROL.controlled_cache_text(
                "remote_storage = wrong-trust-domain", Path("private"), "ci-readonly"
            )
        clean = CONTROL.child_environment(
            {
                "PATH": "path",
                "CCACHE_CONFIGPATH": "private.conf",
                "CCACHE_REMOTE_STORAGE": "foreign",
                "CCACHE_PREFIX": "foreign",
                "CCACHE_RECACHE": "1",
            }
        )
        self.assertEqual(clean, {"PATH": "path", "CCACHE_CONFIGPATH": "private.conf"})

    def test_setup_checks_source_and_config_before_writing_controls(self):
        owned = ROOT / ".normal-unit-fixtures"
        owned.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="config-", dir=owned) as directory:
            path = Path(directory).resolve()
            self.assertIn(owned.resolve(), path.parents)
            config = path / "runner-config-v2.json"
            original = {
                "build_runners": {"linux": {"default": [], "small": [], "medium": []}},
                "gpu_runner_labels": {"gfx103x": {"linux": {"test-runs-on": ""}}},
            }
            config.write_text(json.dumps(original), encoding="utf-8")
            args = argparse.Namespace(config=path, output=path / "controls.json")
            rows = "\n".join(
                f"160000 commit {sha}\t{name}" for name, sha in CONTROL.GITLINKS.items()
            )
            before = config.read_bytes()
            for results in [["160000 commit bad\tforeign"], [rows, "wrong-config-sha"]]:
                with patch.object(
                    CONTROL.subprocess, "check_output", side_effect=results
                ), self.assertRaises(ValueError):
                    CONTROL.setup(args)
                self.assertEqual(config.read_bytes(), before)
                self.assertFalse(args.output.exists())
            outputs = path / "github-output.txt"
            with patch.object(
                CONTROL.subprocess,
                "check_output",
                side_effect=[rows, CONTROL.CONFIG_SHA],
            ), patch.dict(os.environ, {"GITHUB_OUTPUT": str(outputs)}):
                CONTROL.setup(args)
            self.assertEqual(
                json.loads(config.read_text())["gpu_runner_labels"],
                original["gpu_runner_labels"],
            )
            self.assertIn(CONTROL.VERSIONS["rocm_package_version"], outputs.read_text())
            receipt = json.loads(args.output.read_text())
            self.assertEqual(receipt["product_sha"], CONTROL.PRODUCT_SHA)
            self.assertEqual(receipt["version_origin_sha"], CONTROL.VERSION_ORIGIN_SHA)


class WorkflowTest(unittest.TestCase):
    def test_normal_graph_and_correctness_paths_retained(self):
        jobs = workflow("multi_arch_ci_linux.yml")["jobs"]
        expected = {
            "copy_prebuilt_stages",
            "build_multi_arch_stages",
            "validate_artifact_structure",
            "test_artifacts_per_family",
            "test_artifacts_wsl_per_family",
            "build_native_deb_packages",
            "build_native_rpm_packages",
            "test_native_packages_ubuntu2404",
            "test_native_packages_rhel8",
            "test_native_packages_rhel10",
            "test_native_packages_sles16",
            "build_python_packages",
            "test_python_packages_per_family",
            "build_pytorch_wheel_fat",
            "build_jax_wheels",
        }
        self.assertEqual(set(jobs), expected)
        for name in [
            "build_native_deb_packages",
            "build_native_rpm_packages",
            "validate_artifact_structure",
            "build_python_packages",
        ]:
            self.assertEqual(jobs[name]["needs"], ["build_multi_arch_stages"])
        stages = workflow("multi_arch_build_portable_linux.yml")["jobs"]
        self.assertEqual(
            stages["comm-libs"]["needs"], ["compiler-runtime", "emulation"]
        )
        self.assertEqual(stages["math-libs"]["needs"], "compiler-runtime")
        self.assertEqual(stages["comm-libs"]["with"]["amdgpu_family"], "")
        native = workflow("multi_arch_build_native_linux_packages.yml")["jobs"][
            "build_native_packages"
        ]
        for name in [
            "Build Packages",
            "Pre-upload build package verification",
            "Simulated install Test",
            "Build Package repository",
            "Upload Package repo to S3",
        ]:
            self.assertTrue(any(s.get("name") == name for s in native["steps"]))

    def test_transitive_notifiers_off_and_no_product_publication(self):
        queue, seen, notifier_count = [".github/workflows/multi_arch_ci.yml"], set(), 0
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            seen.add(name)
            text = (ROOT / name).read_text(encoding="utf-8")
            parsed = yaml.load(text, Loader=yaml.BaseLoader)
            for forbidden in [
                "configure_aws_product_publication_credentials",
                "uses: benc-uk/workflow-dispatch",
                "uses: ./.github/workflows/multi_arch_release",
            ]:
                self.assertNotIn(forbidden, text)
            for job in parsed.get("jobs", {}).values():
                for step in job.get("steps", []):
                    if "notify_quartz@" in step.get("uses", ""):
                        notifier_count += 1
                        self.assertIn(
                            "fromJSON(toJSON(github.event.inputs)).profile_normal_validation != 'true'",
                            step.get("if", "") + job.get("if", ""),
                            name,
                        )
            for relative in re.findall(
                r"uses:\s*['\"]?(\./\.github/(?:workflows|actions)/[^\s'\"]+)", text
            ):
                path = relative[2:]
                if (ROOT / path).is_dir():
                    path += "/action.yml"
                queue.append(path)
        self.assertGreater(notifier_count, 20)

    def test_default_mode_off_suffix_is_separate_and_install_images_pinned(self):
        inputs = workflow("multi_arch_ci.yml")["on"]["workflow_dispatch"]["inputs"]
        self.assertEqual(inputs["profile_normal_validation"]["default"], "false")
        self.assertEqual(
            inputs["profile_normal_cache_regime"]["default"], "ci-readonly"
        )
        native = workflow("multi_arch_build_native_linux_packages.yml")["jobs"][
            "build_native_packages"
        ]
        self.assertIn("github.run_id", native["env"]["ARTIFACT_RUN_ID"])
        self.assertIn("41147801272", native["env"]["PACKAGE_BUILD_SUFFIX"])
        stage = (
            ROOT / ".github/workflows/multi_arch_build_portable_linux_artifacts.yml"
        ).read_text()
        self.assertIn('if [[ "${NORMAL_VALIDATION}" == "true" ]]', stage)
        self.assertIn("runs-on: ${{ inputs.build_runs_on }}", stage)
        self.assertIn("profile_normal_validation == 'true' && '1' || ''", stage)
        setup = (ROOT / ".github/workflows/setup_multi_arch.yml").read_text()
        self.assertIn("|| 'main'", setup)
        self.assertIn(
            "continue-on-error: ${{ fromJSON(toJSON(github.event.inputs)).profile_normal_validation != 'true' }}",
            setup,
        )
        self.assertEqual(len(CONTROL.GITLINKS), 11)
        self.assertTrue(all("@sha256:" in v for v in CONTROL.INSTALL_IMAGES.values()))

    def test_failed_destination_guard_blocks_always_credential_steps(self):
        action = yaml.load(
            (
                ROOT / ".github/actions/configure_aws_artifacts_credentials/action.yml"
            ).read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )
        steps = action["runs"]["steps"]
        self.assertEqual(steps[0]["id"], "normal_guard")
        for step in steps[1:]:
            self.assertIn("steps.normal_guard.outcome == 'success'", step["if"])


if __name__ == "__main__":
    unittest.main()
