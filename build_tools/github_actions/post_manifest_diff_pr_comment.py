#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Posts (or updates) a sticky PR comment linking to the manifest-diff report.

Run as a step in manifest-diff.yml, after "Upload Report to S3", gated to
submodule-bump PRs only (branch names starting with 'bump-', matching
bump_automation.py). Computes the same S3 report URL that
upload_test_report_script.py used to upload the report, so the linked URL is
guaranteed to match what was actually uploaded for this run.

Usage:
    python post_manifest_diff_pr_comment.py \\
        --run-id ${{ github.run_id }} \\
        --pr-number ${{ github.event.pull_request.number }} \\
        --commit-range-summary "${{ steps.gen-report.outputs.commit_range_summary }}"
"""

import argparse
import logging
import os
from pathlib import Path
import platform
import sys

logging.basicConfig(level=logging.INFO)

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions.github_actions_api import gha_update_pr_comment

PLATFORM = platform.system().lower()

# Stable marker so reruns of the same PR (e.g. re-pushed bump commits) update
# the existing comment in place instead of piling up duplicates.
MARKER = "<!-- therock-report-manifest-diff -->"


def build_comment_body(report_url: str, commit_range_summary: str) -> str:
    """Build the sticky comment body: marker + report link + one-line summary."""
    lines = [MARKER, "### TheRock Manifest Diff Report", f"[View report]({report_url})"]
    if commit_range_summary:
        lines.append(commit_range_summary)
    return "\n\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id, platform=args.platform
    )
    report_url = output_root.log_file("manifest-diff", "index.html").https_url
    body = build_comment_body(report_url, args.commit_range_summary)

    gha_update_pr_comment(
        pr_number=args.pr_number,
        marker=MARKER,
        body=body,
        github_repository=args.github_repository,
    )
    logging.info(
        "Posted manifest-diff report link to %s#%d: %s",
        args.github_repository,
        args.pr_number,
        report_url,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Post the manifest-diff report link as a bump-PR comment"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.environ.get("GITHUB_RUN_ID"),
        help="GitHub Actions run ID whose report was uploaded (default: $GITHUB_RUN_ID)",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        required=True,
        help="Pull request number to comment on",
    )
    parser.add_argument(
        "--commit-range-summary",
        type=str,
        default="",
        help=(
            "One-line commit-range/changed-count summary, e.g. the "
            "'commit_range_summary' output of generate_manifest_diff_report.py. "
            "Omitted from the comment body if blank."
        ),
    )
    parser.add_argument(
        "--github-repository",
        type=str,
        default="ROCm/TheRock",
        help="Repository in 'owner/repo' format (default: ROCm/TheRock)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=PLATFORM,
        help=f"Platform for workflow output paths (default: {PLATFORM})",
    )

    args = parser.parse_args(argv)

    if not args.run_id:
        parser.error("--run-id is required (or set $GITHUB_RUN_ID)")

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
