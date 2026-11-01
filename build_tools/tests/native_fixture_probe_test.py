# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Real inert subprocess/guard checks; never run local Debian packaging tools."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "build_tools/packaging/linux"
sys.path.insert(0, str(SCRIPTS))
import hipify_deb_fixture as fixture


def workflow():
    return yaml.load(
        (ROOT / ".github/workflows/multi_arch_ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


class ProbeWorkflowTests(unittest.TestCase):
    def test_default_normal_workflow_is_exact_original_projection(self):
        data = workflow()
        baseline = json.loads(
            (
                ROOT / "build_tools/tests/fixtures/native_fixture_probe_baseline.json"
            ).read_text()
        )
        option = data["on"]["workflow_dispatch"]["inputs"].pop(
            "profile_native_fixture_only"
        )
        self.assertEqual((option["type"], option["default"]), ("boolean", "false"))
        self.assertNotIn(
            "profile_native_fixture_only", data["on"]["workflow_call"].get("inputs", {})
        )
        data["jobs"].pop("native_fixture_probe")
        self.assertEqual(
            data["jobs"]["setup"].pop("if"),
            "${{ !inputs.profile_native_fixture_only }}",
        )
        self.assertEqual(
            data["jobs"]["ci_summary"]["if"],
            "${{ always() && !inputs.profile_native_fixture_only }}",
        )
        data["jobs"]["ci_summary"]["if"] = "always()"
        self.assertEqual(list(data["jobs"]), baseline["existing_jobs"])
        actual = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(actual, baseline["canonical_workflow_sha256"])

    def test_actual_native_dependencies_context_and_retention_are_preserved(self):
        data = workflow()
        probe = data["jobs"]["native_fixture_probe"]
        native = yaml.load(
            (
                ROOT / ".github/workflows/multi_arch_build_native_linux_packages.yml"
            ).read_text(encoding="utf-8"),
            Loader=yaml.BaseLoader,
        )["jobs"]["build_native_packages"]
        self.assertEqual(probe["container"], native["container"])
        self.assertEqual(probe["runs-on"], "aws-linux-scale-rocm-small")
        self.assertEqual(probe["permissions"], {"contents": "read"})
        self.assertEqual(probe["defaults"]["run"]["shell"], "bash")
        self.assertIn("github.repository == 'ROCm/TheRock'", probe["if"])
        self.assertIn("github.event_name == 'workflow_dispatch'", probe["if"])
        self.assertIn("inputs.profile_native_fixture_only", probe["if"])
        steps = {s.get("name"): s for s in probe["steps"]}
        original = {s.get("name"): s for s in native["steps"]}
        for name in [
            "Install Python requirements",
            "Install System requirements",
            "Require HIPIFY Debian symlink preservation",
        ]:
            self.assertEqual(steps[name]["run"], original[name]["run"])
            self.assertNotIn("continue-on-error", steps[name])
        self.assertEqual(probe["timeout-minutes"], "15")
        self.assertNotIn("needs", probe)
        for step in probe["steps"]:
            self.assertNotIn("configure-aws", step.get("uses", "").lower())
            self.assertNotIn("quartz", step.get("uses", "").lower())
        upload = steps["Retain functional fixture diagnostics"]
        self.assertEqual(upload["if"], "always()")
        self.assertEqual(upload["with"]["path"], "output/normal-validation/")


class ProbeGuardTests(unittest.TestCase):
    def run_guard(self, changes=None, inputs=None, uid=1001):
        job = workflow()["jobs"]["native_fixture_probe"]
        step = next(
            s
            for s in job["steps"]
            if s.get("name") == "Require isolated functional fixture context"
        )
        lines = step["run"].splitlines()
        self.assertEqual((lines[0], lines[-1]), ("python - <<'PY'", "PY"))
        script = "\n".join(lines[1:-1])
        branch = "users/sa-faizal/build-profile-41147801272-normal-validation-fakeroot"
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "ROCm/TheRock",
            "GITHUB_ACTOR": "sa-faizal",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/" + branch,
            "GITHUB_WORKFLOW_REF": "ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/"
            + branch,
            "GITHUB_SHA": "f" * 40,
        }
        env.update(changes or {})
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            event = directory / "event.json"
            event.write_text(
                json.dumps(
                    {
                        "inputs": (
                            inputs
                            if inputs is not None
                            else {
                                "profile_native_fixture_only": True,
                                "profile_normal_validation": False,
                            }
                        )
                    }
                )
            )
            env["GITHUB_EVENT_PATH"] = str(event)
            with contextlib.chdir(directory), mock.patch.dict(
                os.environ, env
            ), mock.patch.object(
                os, "geteuid", return_value=uid, create=True
            ), mock.patch.object(
                subprocess, "check_output", return_value="f" * 40 + "\n"
            ):
                exec(compile(script, "<actual workflow guard>", "exec"), {})
                return json.loads(
                    (
                        directory / "output/normal-validation/native-fixture-scope.json"
                    ).read_text()
                )

    def test_exact_context_and_omitted_false_default_are_diagnostic_only(self):
        for inputs in [
            {"profile_native_fixture_only": True},
            {
                "profile_native_fixture_only": "true",
                "profile_normal_validation": "false",
            },
        ]:
            result = self.run_guard(inputs=inputs)
            self.assertTrue(result["fixture_only"])
            self.assertFalse(result["normal_performance_validation"])
            self.assertEqual(result["uid"], 1001)

    def test_wrong_scope_checkout_root_or_conflicting_baseline_fail_before_work(self):
        for changes in [
            {"GITHUB_REPOSITORY": "ROCm/rockrel"},
            {"GITHUB_ACTOR": "other"},
            {"GITHUB_EVENT_NAME": "workflow_call"},
            {"GITHUB_REF": "refs/heads/main"},
            {"GITHUB_WORKFLOW_REF": "other"},
            {"GITHUB_SHA": "e" * 40},
        ]:
            with self.subTest(changes=changes), self.assertRaises(SystemExit):
                self.run_guard(changes=changes)
        with self.assertRaises(SystemExit):
            self.run_guard(uid=0)
        for inputs in [
            {},
            {"profile_native_fixture_only": 1},
            {"profile_native_fixture_only": True, "profile_normal_validation": True},
            {"profile_native_fixture_only": True, "profile_normal_validation": None},
        ]:
            with self.subTest(inputs=inputs), self.assertRaises(SystemExit):
                self.run_guard(inputs=inputs)


class MetadataContextTests(unittest.TestCase):
    def test_missing_fakeroot_fails_before_metadata_or_case_work(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "fixture.json"
            with mock.patch.object(sys, "platform", "linux"), mock.patch.object(
                sys, "argv", ["fixture", "--output", str(output)]
            ), mock.patch.dict(
                os.environ,
                {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "ROCm/TheRock"},
            ), mock.patch.object(
                fixture.shutil,
                "which",
                side_effect=lambda name: (
                    None if name == "fakeroot" else "/inert/" + name
                ),
            ), mock.patch.object(
                fixture.subprocess,
                "check_output",
                side_effect=AssertionError("No commands before presence gate"),
            ), mock.patch.object(
                fixture,
                "run_case",
                side_effect=AssertionError("No case work without fakeroot"),
            ):
                with self.assertRaisesRegex(RuntimeError, "fakeroot"):
                    fixture.main()
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        os.name == "posix",
        "Inert POSIX executable forwarding fixture; no local Debian tools",
    )
    def test_actual_metadata_path_adds_context_preserves_argv_and_failure(self):
        with tempfile.TemporaryDirectory(prefix="metadata context ") as raw:
            directory = Path(raw)
            wrapper, control = directory / "fakeroot", directory / "dh_gencontrol"
            wrapper.write_text(
                "#!"
                + sys.executable
                + "\nimport os, subprocess, sys\nos.environ['INERT_METADATA_CONTEXT']='yes'\nraise SystemExit(subprocess.call(sys.argv[1:]))\n"
            )
            control.write_text(
                "#!"
                + sys.executable
                + "\nimport json,os,sys\nfrom pathlib import Path\nif os.environ.get('INERT_METADATA_CONTEXT')!='yes':\n sys.stderr.write('chown(0,0,control): Operation not permitted')\n raise SystemExit(17)\nPath(os.environ['ARGV_RECORD']).write_text(json.dumps({'argv':sys.argv[1:],'cwd':os.getcwd(),'literal':os.environ['LITERAL']}))\nraise SystemExit(int(os.environ.get('CONTROL_EXIT','0')))\n"
            )
            wrapper.chmod(0o755)
            control.chmod(0o755)
            record = directory / "argv.json"
            env = dict(
                os.environ, ARGV_RECORD=str(record), LITERAL="a literal with spaces"
            )
            with self.assertRaisesRegex(RuntimeError, "Operation not permitted"):
                fixture.run([str(control), "-pfixture"], directory, env)
            self.assertFalse(record.exists())
            tools = {"fakeroot": str(wrapper), "dh_gencontrol": str(control)}
            result = fixture.generate_control(directory, tools, env)
            self.assertEqual(result.args, [str(wrapper), str(control), "-pfixture"])
            self.assertEqual(
                json.loads(record.read_text()),
                {
                    "argv": ["-pfixture"],
                    "cwd": str(directory),
                    "literal": "a literal with spaces",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "Fixture command failed"):
                fixture.generate_control(directory, tools, dict(env, CONTROL_EXIT="23"))


if __name__ == "__main__":
    unittest.main()
