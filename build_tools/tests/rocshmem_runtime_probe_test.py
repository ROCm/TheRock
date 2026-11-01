# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import yaml

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "probe", ROOT / "build_tools/profile_rocshmem_runtime.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RuntimeProbeTest(unittest.TestCase):
    def test_actual_sanity_context_and_separate_runtime_dependency(self):
        recorded = json.loads(
            (
                ROOT
                / "build_tools/tests/fixtures/rocshmem-runtime/actual-40-sanity.json"
            ).read_text()
        )
        expected = recorded["component"]
        workflow = yaml.load(
            (ROOT / ".github/workflows/rocshmem_runtime_probe.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        sanity = workflow["jobs"]["sanity"]
        runtime = workflow["jobs"]["rocshmem_runtime"]
        self.assertEqual(recorded["source_run"], 41261635745)
        self.assertEqual(sanity["container"]["image"], expected["container_image"])
        self.assertEqual(sanity["container"]["options"], expected["container_options"])
        step = next(
            x for x in sanity["steps"] if x.get("name") == "Original sanity policy"
        )
        self.assertEqual(step["run"], expected["test_script"])
        self.assertEqual(sanity["env"]["TEST_TYPE"], expected["test_type"])
        fetch = next(
            x
            for x in sanity["steps"]
            if x.get("uses") == "./.github/actions/setup_test_environment"
        )
        self.assertEqual(
            fetch["with"]["FETCH_ARTIFACT_ARGS"], expected["fetch_artifact_args"]
        )
        self.assertEqual(runtime["needs"], ["guard", "sanity"])
        self.assertIn(
            "sha256:a13e8370a2d1d33a910e47669cdf075dbe9554685e78592f22935ec7c7c35e39",
            runtime["container"]["image"],
        )
        self.assertNotIn(
            "Original sanity policy", [x.get("name") for x in runtime["steps"]]
        )

    def test_actions_pins_match_actual_stock_component_workflow(self):
        stock = yaml.load(
            (ROOT / ".github/workflows/test_component.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        probe_workflow = yaml.load(
            (ROOT / ".github/workflows/rocshmem_runtime_probe.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        stock_pins = {
            step["uses"].split("@")[0]: step["uses"]
            for job in stock["jobs"].values()
            for step in job.get("steps", [])
            if step.get("uses", "").startswith("actions/")
        }
        for job in probe_workflow["jobs"].values():
            for step in job.get("steps", []):
                if step.get("uses", "").startswith("actions/"):
                    self.assertEqual(
                        step["uses"], stock_pins[step["uses"].split("@")[0]]
                    )

    def test_installed_hash_mismatch_fails_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "native"
            file.write_bytes(b"original")
            contract = {
                "installed_files": [
                    dict(
                        path="native",
                        bytes=8,
                        sha256=hashlib.sha256(b"original").hexdigest(),
                    )
                ]
            }
            with patch.object(probe, "ROOT", root):
                self.assertEqual(probe.identities(contract)["files"][0]["bytes"], 8)
                file.write_bytes(b"mutated!")
                with self.assertRaises(ValueError):
                    probe.identities(contract)

    def test_guard(self):
        inputs = dict(
            profile_rocshmem_runtime_probe="true",
            profile_rocshmem_expected_sha="a" * 40,
        )
        env = dict(
            GITHUB_REPOSITORY="ROCm/TheRock",
            GITHUB_ACTOR="sa-faizal",
            GITHUB_EVENT_NAME="workflow_dispatch",
            GITHUB_SHA="a" * 40,
            GITHUB_WORKFLOW_SHA="a" * 40,
            GITHUB_REF="refs/heads/" + probe.BRANCH,
            GITHUB_WORKFLOW_REF="ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/"
            + probe.BRANCH,
        )
        probe.guard(inputs, env)
        for key in env:
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.guard(inputs, {**env, key: "wrong"})
        for value in (None, True, "true"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.guard({**inputs, "profile_normal_validation": value}, env)

    def test_zero_and_incomplete_fail(self):
        for text in [
            "100% tests passed, 0 tests failed out of 2\n***Skipped 0.01 sec\n***Skipped 0.01 sec",
            "100% tests passed, 0 tests failed out of 2\nPassed 0.01 sec",
            "99% tests passed, 1 tests failed out of 2\nPassed 0.01 sec",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                probe.parse_ctest(text)

    @unittest.skipUnless(
        os.name == "posix" and shutil.which("bash"), "Inert POSIX shell fixture"
    )
    def test_real_pinned_wrapper_path_regression(self):
        wrapper = ROOT / "build_tools/tests/fixtures/rocshmem-runtime/test_wrapper.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tools = root / "tools"
            tools.mkdir()
            sdk = root / "sdk"
            sdk.mkdir()
            for name in ["dirname", "grep", "awk", "wc", "mkdir", "cat"]:
                (tools / name).symlink_to(shutil.which(name))
            binary = sdk / "rocshmem_functional_tests"
            binary.write_text('#!/bin/sh\necho executed >> "$FIXTURE_MARKER"\n')
            binary.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(tools),
                "ROCSHMEM_BACKEND": "ipc",
                "ROCSHMEM_SLR_NP": "2",
                "FIXTURE_MARKER": str(root / "executed"),
            }
            argv = [
                shutil.which("bash"),
                str(wrapper),
                "put_n2_w1_z1_uuid",
                str(binary),
            ]
            result = subprocess.run(
                argv, cwd=root, env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 125)
            self.assertIn("only 0 available", result.stdout)
            self.assertFalse((root / "executed").exists())
            tool = sdk / "amd-smi"
            tool.write_text(
                '#!/bin/sh\nif [ "$1" = list ]; then printf "GPU: 0\\nGPU: 1\\n"; fi\n'
            )
            tool.chmod(0o755)
            result = subprocess.run(
                argv,
                cwd=root,
                env={**env, "PATH": str(sdk) + ":" + str(tools)},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "executed").read_text(), "executed\n")

    @unittest.skipUnless(
        shutil.which("cmake") and shutil.which("ctest"), "Real host CTest fixture"
    )
    def test_real_ctest_pass_skip_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.20)\nproject(inert NONE)\nenable_testing()\nadd_test(NAME pass COMMAND "${CMAKE_COMMAND}" -E true)\nadd_test(NAME skip COMMAND "${CMAKE_COMMAND}" -E false)\nset_tests_properties(skip PROPERTIES SKIP_RETURN_CODE 1)\n'
            )
            subprocess.run(
                ["cmake", "-S", str(root), "-B", str(root / "build")],
                check=True,
                capture_output=True,
            )
            result = subprocess.run(
                ["ctest", "--test-dir", str(root / "build"), "-V"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                probe.parse_ctest(result.stdout), dict(selected=2, passed=1, skipped=1)
            )

    def test_default_off_original_graph(self):
        path = ".github/workflows/multi_arch_ci.yml"
        snapshot = (
            ROOT / "build_tools/tests/fixtures/rocshmem-runtime/multi_arch_ci-60fa.yml"
        )
        self.assertEqual(
            hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            "185dd54c58dcf81cdf2e15bfaae1739fab21782211e2d5f44019ac28f05bd218",
        )
        before = yaml.load(snapshot.read_text(), Loader=yaml.BaseLoader)
        after = yaml.load((ROOT / path).read_text(), Loader=yaml.BaseLoader)
        for field in [
            "profile_rocshmem_runtime_probe",
            "profile_rocshmem_expected_sha",
        ]:
            value = after["on"]["workflow_dispatch"]["inputs"].pop(field)
            if field.endswith("probe"):
                self.assertEqual(value["default"], "false")
        after["jobs"].pop("rocshmem_runtime_probe")
        for name in ["setup", "ci_summary", "native_fixture_probe"]:
            after["jobs"][name]["if"] = after["jobs"][name]["if"].replace(
                " && !inputs.profile_rocshmem_runtime_probe", ""
            )
        self.assertEqual(before, after)
        workflow = (ROOT / ".github/workflows/rocshmem_runtime_probe.yml").read_text()
        self.assertNotIn("secrets:", workflow)
        self.assertNotIn("Quartz", workflow)
        self.assertNotIn("id-token", workflow)
        self.assertIn("--security-opt seccomp=unconfined", workflow)
        self.assertIn("if: ${{ always() }}", workflow)


if __name__ == "__main__":
    unittest.main()
