# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_tools"))
spec = importlib.util.spec_from_file_location(
    "recheck", ROOT / "build_tools/profile_profiler_lifecycle.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LifecycleGuardTest(unittest.TestCase):
    def original(self, path):
        command = ["git", "-C", str(ROOT)]
        if os.environ.get("RECHECK_FIXTURE_GIT_DIR"):
            command = ["git", "--git-dir", os.environ["RECHECK_FIXTURE_GIT_DIR"]]
        return subprocess.check_output(
            command + ["show", "60faab8d3d433d009f154c3e30f9fd8ac5cfda59:" + path]
        )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.event = Path(self.temp.name) / "event.json"
        self.event.write_text(
            json.dumps({"inputs": {"profile_profiler_lifecycle": "true"}})
        )
        self.contract = json.loads(module.CONTRACT.read_text())
        self.env = dict(
            GITHUB_ACTIONS="true",
            GITHUB_REPOSITORY="ROCm/TheRock",
            GITHUB_ACTOR="sa-faizal",
            GITHUB_EVENT_NAME="workflow_dispatch",
            GITHUB_REF="refs/heads/" + self.contract["branch"],
            GITHUB_WORKFLOW_REF="ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/"
            + self.contract["branch"],
            GITHUB_SHA="1" * 40,
            GITHUB_WORKFLOW_SHA="1" * 40,
            GITHUB_EVENT_PATH=str(self.event),
        )

    def test_context_and_changed_identity(self):
        with patch.object(
            module.subprocess, "check_output", return_value="1" * 40 + "\n"
        ) as query, patch.object(
            module,
            "fingerprint",
            side_effect=lambda path: self.contract["source_files"][
                path.relative_to(ROOT).as_posix()
            ],
        ):
            self.assertEqual(module.guard(self.env, self.contract, ROOT), "1" * 40)
            self.assertEqual(
                query.call_args.args[0][1:3],
                ["-c", f"safe.directory={ROOT.as_posix()}"],
            )
            for key in self.env:
                if key == "GITHUB_EVENT_PATH":
                    continue
                changed = dict(self.env, **{key: "wrong"})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module.guard(changed, self.contract, ROOT)
            with self.assertRaises(ValueError):
                module.guard(
                    dict(self.env, INPUT_PROFILE_NORMAL_VALIDATION="true"),
                    self.contract,
                    ROOT,
                )

    def test_original_driver_binding(self):
        changed = copy.deepcopy(self.contract)
        changed["source_files"][next(iter(changed["source_files"]))] = "0" * 64
        with patch.object(
            module.subprocess, "check_output", return_value="1" * 40 + "\n"
        ), self.assertRaises(ValueError):
            module.guard(self.env, changed, ROOT)

    def test_mode_conflicts_fail_before_git(self):
        for payload in [
            {},
            {"profile_profiler_lifecycle": True},
            {"profile_profiler_lifecycle": "true", "profile_normal_validation": "true"},
            {
                "profile_profiler_lifecycle": "true",
                "profile_native_fixture_only": "true",
            },
        ]:
            self.event.write_text(json.dumps({"inputs": payload}))
            with patch.object(
                module.subprocess, "check_output"
            ) as query, self.assertRaises(ValueError):
                module.guard(self.env, self.contract, ROOT)
            query.assert_not_called()

    def test_original_driver_lf_git_blobs(self):
        import hashlib

        for path, digest in self.contract["source_files"].items():
            data = self.original(path)
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_default_off_projection_preserves_original_graph_and_tests(self):
        import yaml

        for path in (
            ".github/workflows/multi_arch_ci.yml",
            ".github/workflows/test_component.yml",
        ):
            original = yaml.safe_load(self.original(path))
            current = yaml.safe_load((ROOT / path).read_text())
            trigger = current.get("on", current.get(True))
            if path.endswith("multi_arch_ci.yml"):
                trigger["workflow_dispatch"]["inputs"].pop("profile_profiler_lifecycle")
                current["jobs"].pop("profiler_lifecycle")
                for name in ("setup", "ci_summary", "native_fixture_probe"):
                    current["jobs"][name]["if"] = current["jobs"][name]["if"].replace(
                        " && !inputs.profile_profiler_lifecycle", ""
                    )
            else:
                trigger["workflow_call"]["inputs"].pop("profile_lifecycle_diagnostic")
                steps = current["jobs"]["test_component"]["steps"]
                steps[:] = [
                    step
                    for step in steps
                    if step.get("name")
                    not in (
                        "Capture profiler lifecycle identity",
                        "Verify profiler lifecycle identity",
                        "Retain profiler lifecycle receipts",
                    )
                ]
                for step in steps:
                    if "notify_quartz" in step.get("uses", ""):
                        step["if"] = step["if"].replace(
                            "!inputs.profile_lifecycle_diagnostic && ", ""
                        )
            self.assertEqual(current, original)
