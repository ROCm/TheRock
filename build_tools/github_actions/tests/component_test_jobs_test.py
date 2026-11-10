# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import shlex
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

import component_test_jobs as ctj
from _therock_utils.workflow_outputs import WorkflowOutputRoot

MANIFEST = """
schema = 1

[defaults]
shards = { comprehensive = 2 }
timeout_minutes = 40

[remainder]
timeout_minutes = { standard = 45 }

[[group]]
name = "mlops"
labels = ["^HIP_MLOPS_ENGINE$"]
shards = { standard = 2, full = { linux = 4, windows = 3 } }

[[group]]
name = "flydsl"
labels = ["^hipkernel_flydsl_"]
allow_empty = true

[[group.exclude_family]]
platform = "windows"
families = ["gfx110X-all"]
reason = "no gfx11 content"
"""

BASE = {
    "job_name": "comp",
    "test_script": "python run.py",
    "total_shards": 4,
    "shard_arr": [1, 2, 3, 4],
    "timeout_minutes": 30,
    "fetch_artifact_args": "--comp --tests",
}


def _env_of(job):
    """The environment assignments test_component.yml's bash step would apply."""
    env = {}
    for token in shlex.split(job["test_script"]):
        key, sep, value = token.partition("=")
        if not sep or not key.isupper():
            break
        env[key] = value
    return env


def _expand(platform="linux", test_type="standard", excluded=()):
    return {
        j["job_name"]: j
        for j in ctj.expand_jobs(
            BASE,
            ctj.parse_manifest(MANIFEST),
            platform=platform,
            test_type=test_type,
            is_excluded=lambda families: any(f in excluded for f in families),
        )
    }


class ResolutionTest(unittest.TestCase):
    def test_shards_precedence(self):
        manifest = ctj.parse_manifest(MANIFEST)
        mlops, flydsl = manifest.groups
        cases = [
            (mlops, "full", "windows", 3),  # group tier+platform
            (mlops, "standard", "linux", 2),  # group tier
            (mlops, "comprehensive", "linux", 2),  # defaults tier
            (flydsl, "quick", "linux", 1),  # built-in default
        ]
        for job, tier, platform, expected in cases:
            with self.subTest(job=job.name, tier=tier, platform=platform):
                self.assertEqual(
                    ctj.job_shards(manifest, job, tier, platform), expected
                )

    def test_timeout_precedence(self):
        manifest = ctj.parse_manifest(MANIFEST)
        self.assertEqual(
            ctj.job_timeout_minutes(manifest, manifest.remainder, "standard", "linux"),
            45,
        )
        self.assertEqual(
            ctj.job_timeout_minutes(manifest, manifest.remainder, "full", "linux"), 40
        )
        no_defaults = ctj.parse_manifest(
            'schema = 1\n[[group]]\nname = "a"\nlabels = ["^A$"]\n'
        )
        self.assertEqual(
            ctj.job_timeout_minutes(
                no_defaults, no_defaults.groups[0], "quick", "linux"
            ),
            ctj.DEFAULT_TIMEOUT_MINUTES,
        )


class ValidationTest(unittest.TestCase):
    def assertInvalid(self, text, message):
        with self.assertRaises(ctj.ManifestError) as ctx:
            ctj.parse_manifest(text)
        self.assertIn(message, str(ctx.exception))

    def test_schema_version_required(self):
        self.assertInvalid('[[group]]\nname = "a"\nlabels = ["^A$"]\n', "schema")

    def test_unknown_key_rejected(self):
        self.assertInvalid(
            'schema = 1\n[[group]]\nname = "a"\nlabels = ["^A$"]\nshard = 2\n',
            "unknown key",
        )

    def test_unknown_tier_rejected(self):
        self.assertInvalid(
            'schema = 1\n[[group]]\nname = "a"\nlabels = ["^A$"]\n'
            "shards = { nightly = 2 }\n",
            "unknown key",
        )

    def test_shards_per_job_capped(self):
        self.assertInvalid(
            'schema = 1\n[[group]]\nname = "a"\nlabels = ["^A$"]\n'
            f"shards = {ctj.MAX_SHARDS_PER_JOB + 1}\n",
            "outside",
        )

    def test_shards_per_component_capped(self):
        groups = "".join(
            f'[[group]]\nname = "g{i}"\nlabels = ["^G{i}$"]\nshards = 4\n'
            for i in range(3)
        )
        # 3 groups x 4 shards + the remainder's 1 = 13.
        self.assertInvalid("schema = 1\n" + groups, "exceeds")

    def test_label_regex_limited_to_portable_subset(self):
        self.assertInvalid(
            'schema = 1\n[[group]]\nname = "a"\nlabels = ["^A\\\\d$"]\n',
            "must use only",
        )

    def test_exclusion_requires_reason(self):
        self.assertInvalid(
            'schema = 1\n[[group]]\nname = "a"\nlabels = ["^A$"]\n'
            '[[group.exclude_family]]\nplatform = "linux"\nfamilies = ["gfx942"]\n',
            "reason",
        )

    def test_group_cannot_reuse_remainder_name(self):
        self.assertInvalid(
            'schema = 1\n[[group]]\nname = "unit"\nlabels = ["^A$"]\n', "duplicate"
        )


class ExpandTest(unittest.TestCase):
    def test_remainder_and_groups_become_jobs(self):
        jobs = _expand()
        self.assertEqual(list(jobs), ["comp-unit", "comp-mlops", "comp-flydsl"])
        self.assertEqual(jobs["comp-mlops"]["total_shards"], 2)
        self.assertEqual(jobs["comp-mlops"]["shard_arr"], [1, 2])
        self.assertEqual(jobs["comp-unit"]["timeout_minutes"], 45)
        # Everything else is inherited from the component's entry.
        self.assertEqual(jobs["comp-flydsl"]["fetch_artifact_args"], "--comp --tests")

    def test_label_filters_reach_the_test_script(self):
        jobs = _expand()
        self.assertEqual(
            _env_of(jobs["comp-unit"]),
            {
                "TEST_COMPONENT": "comp",
                "TEST_LABEL_EXCLUDE": "^HIP_MLOPS_ENGINE$|^hipkernel_flydsl_",
            },
        )
        self.assertEqual(
            _env_of(jobs["comp-mlops"]),
            {
                "TEST_COMPONENT": "comp",
                "TEST_LABEL_INCLUDE": "^HIP_MLOPS_ENGINE$",
                "TEST_LABEL_DISJOINT": "^hipkernel_flydsl_",
            },
        )
        self.assertEqual(_env_of(jobs["comp-flydsl"])["TEST_LABEL_ALLOW_EMPTY"], "1")
        self.assertTrue(jobs["comp-unit"]["test_script"].endswith(" python run.py"))

    def test_excluded_group_is_dropped_but_still_excluded_from_remainder(self):
        jobs = _expand(platform="windows", excluded={"gfx110X-all"})
        self.assertNotIn("comp-flydsl", jobs)
        self.assertIn(
            "^hipkernel_flydsl_", _env_of(jobs["comp-unit"])["TEST_LABEL_EXCLUDE"]
        )

    def test_exclusion_is_scoped_to_its_platform(self):
        self.assertIn(
            "comp-flydsl", _expand(platform="linux", excluded={"gfx110X-all"})
        )


class LoadManifestTest(unittest.TestCase):
    SPEC = {"submodule": "rocm-libraries", "path": "x/ci_jobs.toml"}

    def setUp(self):
        self._env = os.environ.copy()
        os.environ.pop("ARTIFACT_RUN_ID", None)
        os.environ.pop(ctj.override_env_var("comp"), None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _load(self):
        return ctj.load_manifest(
            "comp", self.SPEC, platform="linux", artifact_group=None
        )

    def test_outside_ci_keeps_one_job_without_fetching(self):
        with patch.object(ctj, "fetch_manifest_text") as fetch:
            self.assertIsNone(self._load())
        fetch.assert_not_called()

    def test_fetch_failure_keeps_one_job(self):
        os.environ["ARTIFACT_RUN_ID"] = "123"
        with patch.object(ctj, "fetch_manifest_text", side_effect=OSError("down")):
            self.assertIsNone(self._load())

    def test_fetched_invalid_manifest_raises(self):
        os.environ["ARTIFACT_RUN_ID"] = "123"
        with patch.object(ctj, "fetch_manifest_text", return_value="schema = 2\n"):
            with self.assertRaises(ctj.ManifestError):
                self._load()

    def test_reads_manifest_at_built_submodule_commit(self):
        root = WorkflowOutputRoot.for_local(run_id="123", platform="linux", bucket="b")
        therock_manifest = {
            "submodules": [
                {
                    "submodule_path": "rocm-libraries",
                    "submodule_url": "https://github.com/ROCm/rocm-libraries",
                    "pin_sha": "abc123",
                }
            ]
        }
        responses = {
            root.manifest_root().https_url: json.dumps(therock_manifest).encode(),
            "https://raw.githubusercontent.com/ROCm/rocm-libraries/abc123/x/ci_jobs.toml": MANIFEST.encode(),
        }
        with patch.object(
            WorkflowOutputRoot, "from_workflow_run", return_value=root
        ), patch.object(ctj, "_http_get", side_effect=responses.get):
            text = ctj.fetch_manifest_text(
                self.SPEC,
                run_id="123",
                platform="linux",
                artifact_group=None,
                github_repository="ROCm/rocm-libraries",
            )
        self.assertEqual(text, MANIFEST)


if __name__ == "__main__":
    unittest.main()
