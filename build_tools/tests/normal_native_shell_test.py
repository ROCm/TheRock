# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the actual package shell script with an inert argv recorder, never a build."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/multi_arch_build_native_linux_packages.yml"
FIXTURE = ROOT / "build_tools/tests/fixtures/normal_native_shell_failure.json"


def selected_step():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["build_native_packages"]
    step = next(step for step in job["steps"] if step.get("name") == "Build Packages")
    return workflow, job, step


def rendered_script(step):
    # These are the two actual runner substitutions in the retained failed step.
    return (
        step["run"]
        .replace("${{ env.PACKAGE_DIST_DIR }}", "output/packages")
        .replace("${{ env.ARTIFACTS_DIR }}", "output/artifacts")
    )


class NativeShellSourceTest(unittest.TestCase):
    def test_explicit_shell_and_unchanged_actual_workload_body(self):
        _, job, step = selected_step()
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertTrue(job["container"])
        self.assertEqual(step.get("shell"), "bash")
        script = rendered_script(step)
        self.assertEqual(script, fixture["script"])
        self.assertEqual(
            hashlib.sha256(script.encode()).hexdigest(), fixture["script_sha256"]
        )
        self.assertEqual(
            hashlib.sha256(step["run"].encode()).hexdigest(), fixture["run_body_sha256"]
        )
        self.assertEqual(
            {row["effective_shell"] for row in fixture["records"]}, {"sh -e {0}"}
        )


@unittest.skipUnless(
    os.name == "posix" and shutil.which("bash") and shutil.which("sh"),
    "Requires real POSIX sh and Bash; no ROCm build is executed",
)
class NativeShellExecutionTest(unittest.TestCase):
    def test_relevant_normal_snippets_parse_under_their_effective_container_shell(self):
        workflow, job, _ = selected_step()
        with tempfile.TemporaryDirectory() as raw:
            script = Path(raw) / "step.sh"
            checked = []
            for step in job["steps"]:
                if "run" not in step or not (
                    step.get("name") == "Build Packages"
                    or "profile_normal_validation" in str(step.get("if", ""))
                ):
                    continue
                default = (
                    job.get("defaults", {}).get("run", {}).get("shell")
                    or workflow.get("defaults", {}).get("run", {}).get("shell")
                    or "sh"
                )
                shell = step.get("shell") or default
                self.assertIn(shell, ["sh", "bash"])
                script.write_text(rendered_script(step), encoding="utf-8")
                result = subprocess.run(
                    [shell, "-n", str(script)], capture_output=True, text=True
                )
                self.assertEqual(
                    result.returncode, 0, step["name"] + ": " + result.stderr
                )
                checked.append((step["name"], shell))
            self.assertEqual(dict(checked)["Build Packages"], "bash")
            self.assertGreaterEqual(len(checked), 4)

    def test_actual_sh_failure_and_bash_preserve_profiled_unprofiled_argv(self):
        _, _, step = selected_step()
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="native-shell-") as raw:
            directory = Path(raw)
            script, record = directory / "step.sh", directory / "argv.jsonl"
            script.write_text(fixture["script"], encoding="utf-8")
            bindir = directory / "bin"
            bindir.mkdir()
            recorder = bindir / "python"
            recorder.write_text(
                "#!" + sys.executable + "\n"
                "import json, os, sys\n"
                "with open(os.environ['ARGV_RECORD'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            recorder.chmod(0o755)
            env = dict(
                os.environ,
                PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                ARGV_RECORD=str(record),
                ROCM_VERSION="10.3.0 fixture version",
                PACKAGE_BUILD_SUFFIX="41147801272",
                BUILD_VARIANT="release",
            )
            for enabled in ["false", "true"]:
                before = subprocess.run(
                    ["sh", "-e", str(script)],
                    env={
                        **env,
                        "NORMAL_VALIDATION": enabled,
                        "NATIVE_PACKAGE_TYPE": "deb",
                    },
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(before.returncode, 0)
                self.assertIn("Syntax error", before.stderr)
                self.assertFalse(
                    record.exists(), "sh failed before any Python/build command"
                )
            script.write_text(rendered_script(step), encoding="utf-8")
            for enabled in ["false", "true"]:
                for package_type in ["deb", "rpm"]:
                    result = subprocess.run(
                        [
                            step["shell"],
                            "--noprofile",
                            "--norc",
                            "-e",
                            "-o",
                            "pipefail",
                            str(script),
                        ],
                        env={
                            **env,
                            "NORMAL_VALIDATION": enabled,
                            "NATIVE_PACKAGE_TYPE": package_type,
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    observed = json.loads(record.read_text().splitlines()[-1])
                    workload = [
                        "./build_tools/packaging/linux/build_package.py",
                        "--dest-dir",
                        "output/packages",
                        "--rocm-version",
                        env["ROCM_VERSION"],
                        "--artifacts-dir",
                        "output/artifacts",
                        "--pkg-type",
                        package_type,
                        "--version-suffix",
                        "41147801272",
                        "--build-variant",
                        "release",
                    ]
                    prefix = (
                        [
                            "build_tools/profile_build.py",
                            "--output",
                            "output/normal-validation/profile",
                            "--",
                            "python",
                        ]
                        if enabled == "true"
                        else []
                    )
                    self.assertEqual(observed, prefix + workload)
            self.assertEqual(len(record.read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
