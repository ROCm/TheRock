# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workflow_timing as wt


class WorkflowTimingTest(unittest.TestCase):
    @patch("workflow_timing.urllib.request.urlopen")
    def test_collect_timing_records(self, mock_urlopen):
        run_payload = {
            "created_at": "2026-07-09T10:00:00Z",
        }
        jobs_payload = {
            "total_count": 1,
            "jobs": [
                {
                    "name": "build_multi_arch_stages",
                    "runner_name": "ubuntu-24.04",
                    "started_at": "2026-07-09T10:05:00Z",
                    "completed_at": "2026-07-09T10:30:00Z",
                    "conclusion": "success",
                }
            ],
        }

        def _response(payload):
            m = MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(payload).encode(
                "utf-8"
            )
            return m

        # First call: run payload, second call: jobs payload
        mock_urlopen.side_effect = [_response(run_payload), _response(jobs_payload)]

        records = wt.collect_timing_records(
            repository="ROCm/TheRock",
            run_id="123",
            run_attempt="1",
            token="token",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].job_name, "build_multi_arch_stages")
        self.assertEqual(records[0].runner_label, "ubuntu-24.04")
        self.assertAlmostEqual(records[0].queue_seconds, 300.0, places=1)
        self.assertAlmostEqual(records[0].run_seconds, 1500.0, places=1)

    def test_format_timing_summary(self):
        summary = wt.format_timing_summary(
            [
                wt.TimingRecord(
                    workflow_run_id="123",
                    run_attempt="1",
                    job_name="build_multi_arch_stages",
                    runner_label="ubuntu-24.04",
                    queued_at="2026-07-09T10:00:00Z",
                    started_at="2026-07-09T10:05:00Z",
                    completed_at="2026-07-09T10:30:00Z",
                    decision="rebuilt",
                    queue_seconds=300.0,
                    run_seconds=1500.0,
                )
            ]
        )
        self.assertIn("Queue-time / execution-time", summary)
        self.assertIn("build_multi_arch_stages", summary)
        self.assertIn("ubuntu-24.04", summary)
        self.assertIn("rebuilt", summary)

    @patch("workflow_timing.urllib.request.urlopen")
    def test_collect_timing_records_handles_empty_labels(self, mock_urlopen):
        run_payload = {
            "created_at": "2026-07-09T10:00:00Z",
        }
        jobs_payload = {
            "total_count": 1,
            "jobs": [
                {
                    "name": "build_multi_arch_stages",
                    "labels": [],
                    "started_at": "2026-07-09T10:05:00Z",
                    "completed_at": "2026-07-09T10:30:00Z",
                    "conclusion": "success",
                }
            ],
        }

        def _response(payload):
            m = MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(payload).encode(
                "utf-8"
            )
            return m

        mock_urlopen.side_effect = [_response(run_payload), _response(jobs_payload)]

        records = wt.collect_timing_records(
            repository="ROCm/TheRock",
            run_id="123",
            run_attempt="1",
            token="token",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].runner_label, "unknown")

    @patch("workflow_timing.urllib.request.urlopen")
    def test_collect_timing_records_prefers_runner_name_over_empty_labels(
        self, mock_urlopen
    ):
        run_payload = {"created_at": "2026-07-09T10:00:00Z"}
        jobs_payload = {
            "total_count": 1,
            "jobs": [
                {
                    "name": "build_multi_arch_stages",
                    "runner_name": "ubuntu-24.04",
                    "labels": [],
                    "started_at": "2026-07-09T10:05:00Z",
                    "completed_at": "2026-07-09T10:30:00Z",
                    "conclusion": "success",
                }
            ],
        }

        def _response(payload):
            m = MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(payload).encode(
                "utf-8"
            )
            return m

        mock_urlopen.side_effect = [_response(run_payload), _response(jobs_payload)]

        records = wt.collect_timing_records(
            repository="ROCm/TheRock",
            run_id="123",
            run_attempt="1",
            token="token",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].runner_label, "ubuntu-24.04")
