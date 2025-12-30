#!/usr/bin/env python3
"""Track and display status of triggered workflows.

Queries GitHub API for recently triggered workflows and generates a summary table.

Environment Variables:
    GITHUB_TOKEN: GitHub API token
    GITHUB_REPOSITORY: Repository (default: ROCm/TheRock)
"""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone

from github_actions_utils import gha_append_step_summary, gha_query_workflow_runs


def parse_gpu_family_from_run_name(run_name: str) -> str:
    """Extract GPU family from workflow run name."""
    # Match pattern: (FAMILY, ...)
    match = re.search(r"\(([^,\)]+)", run_name)
    if match:
        return match.group(1).strip()
    return "Unknown"


def parse_package_type_from_run_name(run_name: str) -> str:
    """Extract package type (RPM/DEB) from workflow run name."""
    run_name_lower = run_name.lower()
    if "rpm" in run_name_lower:
        return "RPM"
    elif "deb" in run_name_lower:
        return "DEB"
    return "Unknown"


def main():
    """Generate triggered workflows summary table."""
    parser = argparse.ArgumentParser(description="Track triggered workflows status")
    parser.add_argument(
        "--cutoff-minutes",
        type=int,
        default=10,
        help="Show workflows triggered in last N minutes (default: 10)",
    )
    parser.add_argument(
        "--repository",
        type=str,
        default=None,
        help="GitHub repository 'owner/repo' (default: GITHUB_REPOSITORY env)",
    )
    parser.add_argument(
        "--workflows",
        type=str,
        required=True,
        help="Comma-separated list of workflows to track, format: 'DisplayName:filename.yml:has_pkg_type,...'",
    )

    args = parser.parse_args()

    github_repository = args.repository or os.getenv(
        "GITHUB_REPOSITORY", "ROCm/TheRock"
    )
    cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=args.cutoff_minutes)
    cutoff_time = cutoff_dt.isoformat().replace("+00:00", "Z")

    # Parse workflow specifications
    workflows = []
    for workflow_spec in args.workflows.split(","):
        parts = workflow_spec.strip().split(":")
        if len(parts) == 3:
            display_name, filename, has_pkg_type = parts
            workflows.append((display_name, filename, has_pkg_type.lower() == "true"))
        else:
            print(
                f"Warning: Invalid workflow spec '{workflow_spec}', expected format 'DisplayName:filename.yml:true/false'"
            )

    print(f"Querying: {github_repository} ({args.cutoff_minutes} min ago)")
    print(
        f"Tracking {len(workflows)} workflow(s): {', '.join(w[1] for w in workflows)}"
    )

    summary = "## Triggered workflows status\n\n"
    summary += "This workflow triggered the following child workflows. Click the links to view their progress.\n\n"
    summary += "| Workflow | GPU Family | Package Type | Run ID | Link |\n"
    summary += "|----------|------------|--------------|--------|------|\n"

    total_runs = 0
    for workflow_name, workflow_file, has_package_type in workflows:
        runs = gha_query_workflow_runs(
            github_repository=github_repository,
            workflow_name=workflow_file,
            per_page=40 if has_package_type else 20,
            created_after=cutoff_time,
        )

        for run in runs:
            gpu_family = parse_gpu_family_from_run_name(run.get("name", ""))
            run_id = run.get("id", "")
            run_url = run.get("html_url", "#")

            pkg_type = (
                parse_package_type_from_run_name(run.get("name", ""))
                if has_package_type
                else "-"
            )
            summary += f"| {workflow_name} | {gpu_family} | {pkg_type} | {run_id} | [View Run]({run_url}) |\n"
            total_runs += 1

    if total_runs == 0:
        summary += "| - | - | - | - | No runs triggered |\n"

    summary += "\n---\n\n"
    summary += f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_\n\n"
    summary += (
        "**Tip:** Click the workflow links above to view real-time status and logs.\n"
    )

    gha_append_step_summary(summary)
    print(f"Generated summary ({total_runs} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
