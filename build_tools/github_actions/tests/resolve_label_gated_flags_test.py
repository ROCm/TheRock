#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for resolve_label_gated_flags.py"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

import resolve_label_gated_flags as resolver
from resolve_label_gated_flags import repo_name_from_external_repo

FAKE_MAP: dict[str, dict[str, list[str]]] = {
    "rocm-libraries": {
        "ci:alpha": ["-DTHEROCK_FLAG_ALPHA=ON"],
        "ci:beta": ["-DTHEROCK_FLAG_BETA=OFF"],
    },
}


class RepoNameFromExternalRepoTest(unittest.TestCase):
    def test_full_name(self):
        self.assertEqual(
            repo_name_from_external_repo('{"repository": "ROCm/rocm-libraries"}'),
            "rocm-libraries",
        )

    def test_case_insensitive(self):
        self.assertEqual(
            repo_name_from_external_repo('{"repository": "ROCm/ROCgdb"}'), "rocgdb"
        )

    def test_bare_name(self):
        self.assertEqual(
            repo_name_from_external_repo('{"repository": "rocm-systems"}'),
            "rocm-systems",
        )

    def test_absent_or_junk(self):
        for value in ["", "   ", "not json", "[]", '"a string"', "{}", '{"ref": "x"}']:
            with self.subTest(value=value):
                self.assertEqual(repo_name_from_external_repo(value), "")

    def test_non_string_repository(self):
        self.assertEqual(repo_name_from_external_repo('{"repository": 42}'), "")


class ResolverMainTest(unittest.TestCase):
    """End-to-end over a temp GITHUB_EVENT_PATH and GITHUB_OUTPUT."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.output_path = self.tmp / "github_output"
        self.output_path.touch()
        self.summary_path = self.tmp / "step_summary.md"

    def tearDown(self):
        self._tmpdir.cleanup()

    def _event(self, *label_names: str) -> Path:
        path = self.tmp / "event.json"
        path.write_text(
            json.dumps(
                {"pull_request": {"labels": [{"name": n} for n in label_names]}}
            ),
            encoding="utf-8",
        )
        return path

    def _run(self, *, event_name, event_path, external_repo, mapping=FAKE_MAP):
        env = {
            "GITHUB_EVENT_NAME": event_name,
            "GITHUB_EVENT_PATH": str(event_path),
            "EXTERNAL_REPO": external_repo,
            "GITHUB_OUTPUT": str(self.output_path),
            "GITHUB_STEP_SUMMARY": str(self.summary_path),
        }
        # collect_flags() and validate_label_gated_flags() read the module
        # global at call time, so patching it here exercises real behavior
        # without depending on whatever entries happen to be shipped.
        with patch.dict("os.environ", env, clear=False), patch(
            "label_gated_flags.LABEL_GATED_FLAGS", mapping
        ):
            rc = resolver.main()
        self.assertEqual(rc, 0)
        outputs = {}
        for line in self.output_path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            outputs[key] = value
        summary = (
            self.summary_path.read_text(encoding="utf-8")
            if self.summary_path.exists()
            else ""
        )
        return outputs, summary

    def test_pull_request_with_gating_label(self):
        outputs, summary = self._run(
            event_name="pull_request",
            event_path=self._event("ci:alpha", "documentation"),
            external_repo='{"repository": "ROCm/rocm-libraries", "ref": "abc"}',
        )
        self.assertEqual(outputs["flags"], "-DTHEROCK_FLAG_ALPHA=ON")
        self.assertEqual(outputs["flags_active"], "true")
        self.assertEqual(outputs["matched_labels"], "ci:alpha")
        self.assertIn("`ci:alpha`", summary)

    def test_pull_request_without_gating_label(self):
        outputs, summary = self._run(
            event_name="pull_request",
            event_path=self._event("documentation"),
            external_repo='{"repository": "ROCm/rocm-libraries"}',
        )
        self.assertEqual(outputs["flags"], "")
        self.assertEqual(outputs["flags_active"], "false")
        self.assertEqual(outputs["matched_labels"], "")
        self.assertIn("_none_", summary)

    def test_non_pull_request_events_never_read_labels(self):
        for event_name in ["workflow_dispatch", "push", "schedule", "release"]:
            with self.subTest(event_name=event_name):
                self.setUp()
                outputs, summary = self._run(
                    event_name=event_name,
                    event_path=self._event("ci:alpha"),
                    external_repo='{"repository": "ROCm/rocm-libraries"}',
                )
                self.assertEqual(outputs["flags_active"], "false")
                self.assertEqual(outputs["flags"], "")
                self.assertIn("not `pull_request`", summary)

    def test_unknown_repo_key_matches_nothing(self):
        outputs, _ = self._run(
            event_name="pull_request",
            event_path=self._event("ci:alpha"),
            external_repo='{"repository": "ROCm/rocm-systems"}',
        )
        self.assertEqual(outputs["flags_active"], "false")

    def test_no_external_repo(self):
        outputs, summary = self._run(
            event_name="pull_request",
            event_path=self._event("ci:alpha"),
            external_repo="",
        )
        self.assertEqual(outputs["flags_active"], "false")
        self.assertIn("No external repository", summary)

    def test_empty_map_is_inert(self):
        outputs, _ = self._run(
            event_name="pull_request",
            event_path=self._event("ci:alpha"),
            external_repo='{"repository": "ROCm/rocm-libraries"}',
            mapping={},
        )
        self.assertEqual(outputs["flags_active"], "false")
        self.assertEqual(outputs["flags"], "")

    def test_missing_event_payload(self):
        outputs, _ = self._run(
            event_name="pull_request",
            event_path=self.tmp / "does-not-exist.json",
            external_repo='{"repository": "ROCm/rocm-libraries"}',
        )
        self.assertEqual(outputs["flags_active"], "false")


if __name__ == "__main__":
    unittest.main()
