#!/usr/bin/env python3
"""Select verified test payloads and account for every requested matrix entry."""

from collections import Counter
import json
import os
from pathlib import Path


root = Path(__file__).resolve().parent
producers = json.loads((root / "producer-manifest.json").read_text())["producers"]
expected = json.loads((root / "matrix.json").read_text())["include"]
expected_by_stage = {
    producer["stage"]: [entry for entry in expected if entry["stage"] == producer["stage"]]
    for producer in producers
}
available = []
unavailable = []
reports = []

for stage, entries in expected_by_stage.items():
    found = list(Path("preflight-results").rglob(f"preflight-{stage}.json"))
    if len(found) != 1:
        unavailable.extend(
            {
                "id": entry["id"],
                "component": entry["job_name"],
                "stage": stage,
                "reason": "missing-preflight-record",
                "producer_run_id": entry.get("producer_run_id"),
            }
            for entry in entries
        )
        continue
    report = json.loads(found[0].read_text())
    if report["stage"] != stage:
        raise ValueError(f"Preflight stage mismatch: {found[0]}")
    expected_ids = {entry["id"] for entry in entries}
    reported_ids = [entry["id"] for entry in report["available"] + report["unavailable"]]
    if set(reported_ids) != expected_ids or len(reported_ids) != len(expected_ids):
        raise ValueError(f"Preflight entries incomplete or duplicated for {stage}")
    reports.append(report)
    available.extend(report["available"])
    unavailable.extend(report["unavailable"])

by_reason = dict(sorted(Counter(row["reason"] for row in unavailable).items()))
by_stage = {
    stage: {
        "runnable": sum(row["stage"] == stage for row in available),
        "unavailable": sum(row["stage"] == stage for row in unavailable),
    }
    for stage in expected_by_stage
}
result = {"include": available}
coverage = {
    "requested_jobs": len(expected),
    "available_jobs": len(available),
    "unavailable_jobs": len(unavailable),
    "unavailable_by_reason": by_reason,
    "by_stage": by_stage,
    "unavailable": unavailable,
    "producer_stages": len(reports),
}
if len(available) + len(unavailable) != len(expected):
    raise ValueError("Incomplete selected suite coverage accounting")
Path("dispatch-matrix.json").write_text(json.dumps(result, indent=2) + "\n")
Path("coverage-results.json").write_text(json.dumps(coverage, indent=2) + "\n")

with open(os.environ["GITHUB_OUTPUT"], "a") as output:
    output.write("matrix=" + json.dumps(result, separators=(",", ":")) + "\n")
    output.write(f"count={len(available)}\n")
    output.write(f"missing={len(unavailable)}\n")
    for reason in (
        "artifact-setup-failed",
        "producer-payload-incomplete",
        "missing-preflight-record",
        "missing-test-payload",
        "no-executable-suite",
    ):
        output.write(f"{reason.replace('-', '_')}={by_reason.get(reason, 0)}\n")

with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
    summary.write("## DASan gfx950 selected artifact coverage\n\n")
    summary.write(f"Requested jobs: **{len(expected)}**; runnable: **{len(available)}**; unavailable: **{len(unavailable)}**.\n\n")
    summary.write("| Unavailable reason | Jobs |\n| --- | ---: |\n")
    for reason, count in by_reason.items():
        summary.write(f"| {reason} | {count} |\n")
    summary.write("\nThese are payload and setup outcomes. Test results are recorded by the test jobs.\n\n")
    for row in unavailable:
        detail = row.get("setup_error") or ", ".join(row.get("missing", []))
        summary.write(f"- {row['id']}: {row['reason']}" + (f" ({detail})" if detail else "") + "\n")
