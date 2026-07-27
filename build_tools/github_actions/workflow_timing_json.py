# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from collections import defaultdict
from typing import Sequence

from workflow_timing import TimingRecord


def format_timing_json(records: Sequence[TimingRecord]) -> str:
    """Render workflow timing information as JSON."""

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    def sort_key(record: TimingRecord) -> tuple[int, float, str]:
        status = record.decision.lower()

        if status in {"failure", "cancelled", "timed_out"}:
            rank = 0
        elif status in {"queued", "in_progress", "waiting", "requested"}:
            rank = 1
        else:
            rank = 2

        total = record.total_seconds
        total_sort = -(total if total is not None else -1.0)

        return (rank, total_sort, record.job_name.lower())

    for record in sorted(records, key=sort_key):
        if record.platform not in ("Linux", "Windows"):
            continue

        if record.workflow_phase not in ("Build Stages", "Tests"):
            continue

        grouped[record.platform][record.workflow_phase].append(
            {
                "component": record.component,
                "job_type": record.job_type,
                "runner_pool": record.runner_pool,
                "runner_instance": record.runner_instance,
                "status": record.decision,
                "queue": {
                    "seconds": record.queue_seconds,
                },
                "run": {
                    "seconds": record.run_seconds,
                },
                "total": {
                    "seconds": record.total_seconds,
                },
                "timestamps": {
                    "queued_at": record.queued_at,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                },
                "metadata": {
                    "workflow_run_id": record.workflow_run_id,
                    "run_attempt": record.run_attempt,
                    "job_name": record.job_name,
                    "runner_label": record.runner_label,
                    "stage_or_test_family": record.stage_or_test_family,
                },
            }
        )

    result = {
        "platforms": [
            {
                "platform": platform,
                "phases": [
                    {
                        "phase": phase,
                        "records": grouped[platform][phase],
                    }
                    for phase in ("Build Stages", "Tests")
                    if phase in grouped[platform]
                ],
            }
            for platform in ("Linux", "Windows")
            if platform in grouped
        ]
    }

    return json.dumps(result, indent=2)
