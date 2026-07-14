# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for notify_quartz reporting-workflow path attribution."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import notify_quartz
from notify_quartz import _GithubApiResponse, _build_payload, _normalize_reporting_path


class NormalizeReportingPathTest(unittest.TestCase):
    def test_empty_is_passthrough_signal(self):
        self.assertEqual(_normalize_reporting_path(""), "")
        self.assertEqual(_normalize_reporting_path("   "), "")

    def test_bare_filename(self):
        self.assertEqual(
            _normalize_reporting_path("multi_arch_build_portable_linux.yml"),
            ".github/workflows/multi_arch_build_portable_linux.yml",
        )

    def test_full_workflow_ref_strips_owner_and_ref(self):
        self.assertEqual(
            _normalize_reporting_path(
                "ROCm/TheRock/.github/workflows/build_tarballs.yml@refs/heads/main"
            ),
            ".github/workflows/build_tarballs.yml",
        )

    def test_plain_path_passthrough(self):
        self.assertEqual(
            _normalize_reporting_path(".github/workflows/x.yml"),
            ".github/workflows/x.yml",
        )


def _run_obj() -> dict:
    return {
        "id": 999,
        "name": "Multi-Arch Release",
        "path": ".github/workflows/multi_arch_release.yml",
        "workflow_id": 1,
        "status": "in_progress",
    }


class BuildPayloadPathOverrideTest(unittest.TestCase):
    def _build(self, reporting_workflow: str) -> dict:
        with (
            mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "999"}),
            mock.patch.object(
                notify_quartz,
                "_github_api_request",
                return_value=_GithubApiResponse(body=_run_obj(), headers={}),
            ),
        ):
            return _build_payload(
                token="t",
                repo="ROCm/rockrel",
                embedded_inputs={},
                captured_outputs={},
                run_conclusion="",
                run_phase="started",
                reporting_workflow=reporting_workflow,
            )

    def test_override_sets_child_path_but_keeps_shared_run_id(self):
        payload = self._build("multi_arch_build_portable_linux.yml")
        wr = payload["workflow_run"]
        self.assertEqual(
            wr["path"], ".github/workflows/multi_arch_build_portable_linux.yml"
        )
        # run_id stays the shared entry run -- it links the leaf to its parent.
        self.assertEqual(wr["id"], 999)

    def test_empty_reporting_keeps_api_path(self):
        payload = self._build("")
        self.assertEqual(
            payload["workflow_run"]["path"],
            ".github/workflows/multi_arch_release.yml",
        )


if __name__ == "__main__":
    unittest.main()
