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
                    runner_pool="ubuntu-24.04",
                    runner_instance="runner1",
                    platform="Linux",
                    workflow_phase="Build Stages",
                    component="Compiler Runtime",
                    job_type="Stage",
                    queued_at="2026-07-09T10:00:00Z",
                    started_at="2026-07-09T10:05:00Z",
                    completed_at="2026-07-09T10:30:00Z",
                    decision="rebuilt",
                    queue_seconds=300.0,
                    run_seconds=1500.0,
                    total_seconds=1800.0,
                )
            ]
        )
        self.assertIn("CI Job Timing Summary", summary)
        self.assertIn("Linux", summary)
        self.assertIn("Build Stages", summary)
        self.assertIn("Compiler Runtime", summary)
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

    def test_format_timing_summary_linux_before_windows(self):
        summary = wt.format_timing_summary(
            [
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="windows build",
                    runner_label="azure-windows-scale-rocm",
                    runner_pool="azure-windows-scale-rocm",
                    runner_instance="win1",
                    platform="Windows",
                    workflow_phase="Build Stages",
                    component="Windows Stage",
                    job_type="Stage",
                    queued_at="2026-07-09T10:00:00Z",
                    started_at="2026-07-09T10:01:00Z",
                    completed_at="2026-07-09T10:02:00Z",
                    decision="success",
                    queue_seconds=60,
                    run_seconds=60,
                    total_seconds=120,
                ),
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="linux build",
                    runner_label="aws-linux-scale-rocm-prod",
                    runner_pool="aws-linux-scale-rocm-prod",
                    runner_instance="linux1",
                    platform="Linux",
                    workflow_phase="Build Stages",
                    component="Linux Stage",
                    job_type="Stage",
                    queued_at="2026-07-09T10:00:00Z",
                    started_at="2026-07-09T10:01:00Z",
                    completed_at="2026-07-09T10:02:00Z",
                    decision="success",
                    queue_seconds=60,
                    run_seconds=60,
                    total_seconds=120,
                ),
            ]
        )

        self.assertLess(summary.index("## Linux"), summary.index("## Windows"))

    def test_format_timing_summary_variant_column_removed(self):
        summary = wt.format_timing_summary(
            [
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="build",
                    runner_label="ubuntu-24.04",
                    runner_pool="ubuntu",
                    runner_instance="1",
                    platform="Linux",
                    workflow_phase="Build Stages",
                    component="Compiler Runtime",
                    job_type="Stage",
                    queued_at="2026-07-09T10:00:00Z",
                    started_at="2026-07-09T10:01:00Z",
                    completed_at="2026-07-09T10:02:00Z",
                    decision="success",
                    stage_or_test_family="gfx94X",
                    queue_seconds=60,
                    run_seconds=60,
                    total_seconds=120,
                )
            ]
        )

        self.assertNotIn("Variant", summary)

    def test_format_timing_summary_filters_non_build_and_test_jobs(self):
        summary = wt.format_timing_summary(
            [
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="setup",
                    runner_label="ubuntu",
                    runner_pool="ubuntu",
                    runner_instance="1",
                    platform="Linux",
                    workflow_phase="Setup",
                    component="Setup",
                    job_type="Job",
                    queued_at=None,
                    started_at=None,
                    completed_at=None,
                    decision="success",
                ),
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="manifest",
                    runner_label="ubuntu",
                    runner_pool="ubuntu",
                    runner_instance="1",
                    platform="Linux",
                    workflow_phase="Manifest Diff",
                    component="Manifest Diff",
                    job_type="Job",
                    queued_at=None,
                    started_at=None,
                    completed_at=None,
                    decision="success",
                ),
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="compiler",
                    runner_label="ubuntu",
                    runner_pool="ubuntu",
                    runner_instance="1",
                    platform="Linux",
                    workflow_phase="Build Stages",
                    component="Compiler Runtime",
                    job_type="Stage",
                    queued_at=None,
                    started_at=None,
                    completed_at=None,
                    decision="success",
                ),
                wt.TimingRecord(
                    workflow_run_id="1",
                    run_attempt="1",
                    job_name="rocblas",
                    runner_label="ubuntu",
                    runner_pool="ubuntu",
                    runner_instance="1",
                    platform="Linux",
                    workflow_phase="Tests",
                    component="rocBLAS",
                    job_type="Test",
                    queued_at=None,
                    started_at=None,
                    completed_at=None,
                    decision="success",
                ),
            ]
        )

        self.assertIn("Compiler Runtime", summary)
        self.assertIn("rocBLAS", summary)

        self.assertNotIn("Manifest Diff", summary)
        self.assertNotIn("Setup", summary)
