# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Source/guard/context fixtures only; never run local container/SDK probes."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_tools"))
import profile_consumer_source_probe as PROBE


def workflow(name):
    return yaml.load(
        (ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader
    )


def project_default(data):
    """Remove exactly the default-off additions; other changes remain visible."""
    data = copy.deepcopy(data)
    options = data["on"]["workflow_dispatch"]["inputs"]
    if options.pop("profile_consumer_source_probe")["default"] != "false":
        raise AssertionError("Probe must default off")
    if options.pop("profile_consumer_expected_sha")["default"] != "":
        raise AssertionError("Expected SHA is not a default source selection")
    data["jobs"].pop("consumer_source_probe")
    for name in ["setup", "ci_summary", "native_fixture_probe"]:
        condition = data["jobs"][name]["if"]
        marker = " && !inputs.profile_consumer_source_probe"
        if condition.count(marker) != 1:
            raise AssertionError("Probe isolation guard is required")
        data["jobs"][name]["if"] = condition.replace(marker, "", 1)
    return data


class ConsumerProbeTests(unittest.TestCase):
    def context(self):
        sha = "a" * 40
        env = dict(
            GITHUB_ACTIONS="true",
            GITHUB_REPOSITORY="ROCm/TheRock",
            GITHUB_ACTOR="sa-faizal",
            GITHUB_EVENT_NAME="workflow_dispatch",
            GITHUB_REF="refs/heads/" + PROBE.BRANCH,
            GITHUB_WORKFLOW_REF="ROCm/TheRock/.github/workflows/multi_arch_ci.yml@refs/heads/"
            + PROBE.BRANCH,
            GITHUB_SHA=sha,
            GITHUB_WORKFLOW_SHA=sha,
            GITHUB_RUN_ID="123",
            GITHUB_RUN_ATTEMPT="1",
        )
        event = dict(
            inputs=dict(
                profile_consumer_source_probe="true", profile_consumer_expected_sha=sha
            )
        )
        return env, event, sha

    def test_default_projection_and_all_normal_entrypoints_are_preserved(self):
        data = workflow("multi_arch_ci.yml")
        baseline = json.loads(
            (
                ROOT / "build_tools/tests/fixtures/consumer_source_probe_baseline.json"
            ).read_text()
        )
        actual = hashlib.sha256(
            json.dumps(
                project_default(data), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        self.assertEqual(actual, baseline["canonical_workflow_sha256"])
        self.assertEqual(
            data["jobs"]["consumer_source_probe"]["uses"],
            "./.github/workflows/normal_consumer_source_probe.yml",
        )
        for name in ["linux_build_and_test", "windows_build_and_test"]:
            self.assertEqual(data["jobs"][name]["needs"], "setup")
            self.assertNotIn("always()", data["jobs"][name]["if"])
        self.assertNotIn("workflow_dispatch", data["jobs"]["manifest_diff"]["if"])
        self.assertEqual(
            data["jobs"]["consumer_source_probe"]["permissions"], {"contents": "read"}
        )

    def test_context_sha_actor_and_conflict_guards_reject(self):
        env, event, sha = self.context()
        PROBE.validate(env, event, sha)
        for key, value in {
            "GITHUB_REPOSITORY": "ROCm/rockrel",
            "GITHUB_ACTOR": "other",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_WORKFLOW_REF": "wrong",
            "GITHUB_WORKFLOW_SHA": "b" * 40,
            "GITHUB_SHA": "b" * 40,
            "GITHUB_EVENT_NAME": "push",
        }.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                PROBE.validate(dict(env, **{key: value}), event, sha)
        for updates in [
            dict(profile_normal_validation="true"),
            dict(profile_native_fixture_only="true"),
            dict(profile_normal_case="candidate"),
            dict(profile_normal_product="leading-build-stack"),
            dict(linux_amdgpu_families="gfx94x"),
            dict(prebuilt_stages="all"),
            dict(profile_consumer_expected_sha="main"),
        ]:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                PROBE.validate(env, dict(inputs=dict(event["inputs"], **updates)), sha)

    def test_exact_images_security_venv_and_consumer_ids_no_sdk_or_credentials(self):
        data = workflow("normal_consumer_source_probe.yml")
        self.assertEqual(data["permissions"], {"contents": "read"})
        self.assertEqual(
            set(data["jobs"]),
            {"guard", "test_wheels", "test_native_linux_packages_install"},
        )
        wheel = data["jobs"]["test_wheels"]
        original = workflow("test_rocm_wheels.yml")["jobs"]["test_wheels"]
        self.assertEqual(
            wheel["container"]["options"], original["container"]["options"]
        )
        self.assertEqual(wheel["env"]["VENV_DIR"], "${{ github.workspace }}/.venv")
        self.assertEqual(
            {r["image"] for r in wheel["strategy"]["matrix"]["include"]},
            set(PROBE.WHEEL_IMAGES.values()),
        )
        native = data["jobs"]["test_native_linux_packages_install"]
        self.assertEqual(native["env"]["TEST_TYPE"], "sanity")
        self.assertEqual(
            native["container"]["options"], "--device /dev/kfd --device /dev/dri"
        )
        self.assertEqual(
            {r["image"] for r in native["strategy"]["matrix"]["include"]},
            set(PROBE.INSTALL_IMAGES.values()),
        )
        for name in ["test_wheels", "test_native_linux_packages_install"]:
            job = data["jobs"][name]
            self.assertEqual(job["needs"], "guard")
            self.assertEqual(job["runs-on"], "linux-gfx942-1gpu-ccs-csp-ossci-rocm")
            for step in job["steps"]:
                run = step.get("run", "")
                self.assertNotIn("rocm-sdk", run)
                self.assertNotIn("--packages", run)
                self.assertNotIn("apt-get install git", run)
                self.assertNotIn("dnf install git", run)
                self.assertNotIn("configure_aws", step.get("uses", ""))
                self.assertNotIn("Quartz", step.get("uses", ""))
            self.assertEqual(job["steps"][-1]["if"], "always()")
        self.assertIn(
            "/tmp/test-venv/bin/python",
            "\n".join(s.get("run", "") for s in native["steps"]),
        )

    @unittest.skipUnless(os.name == "posix", "Actual inert UID/Git fixture")
    def test_real_git_receipt_labels_synthetic_producer_and_rejects_unexercised_rpm_mode(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README").write_text("inert source\n")
            env = dict(
                os.environ,
                GIT_CONFIG_NOSYSTEM="1",
                GIT_CONFIG_GLOBAL=str(root / "isolated-global"),
            )
            for argv in [
                ["init"],
                ["add", "README"],
                [
                    "-c",
                    "user.name=Fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit",
                    "-m",
                    "inert",
                ],
            ]:
                subprocess.run(
                    ["git", "-C", str(root), *argv],
                    env=env,
                    check=True,
                    capture_output=True,
                )
            head = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], env=env, text=True
            ).strip()
            env.update(
                GITHUB_JOB="test_native_linux_packages_install",
                GITHUB_SHA=head,
                NORMAL_IMAGE_PIN=PROBE.INSTALL_IMAGES["rpm-rhel8"],
            )
            with patch.dict(os.environ, env, clear=True):
                receipt = PROBE.run_probe(root, "rpm-rhel8", env)
            self.assertFalse(receipt["validated"])
            self.assertFalse(receipt["expected_source_mode_exercised"])
            self.assertFalse(receipt["expected_git_available"])
            self.assertTrue(receipt["producer_refs_synthetic"])
            self.assertFalse(receipt["real_producer_sdk_output_identity_proven"])
            self.assertEqual(receipt["source"]["measured_git_head"], head)
            self.assertIn("neither published nor fetched", receipt["scope"])
            self.assertGreaterEqual(receipt["capture_cost"]["wall_s"], 0)


if __name__ == "__main__":
    unittest.main()
