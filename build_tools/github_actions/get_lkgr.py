#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Get the Last Known Good Run (LKGR) from a GitHub Actions workflow."""

import argparse
import json
import sys

from github_actions_api import gha_query_last_successful_workflow_run, gha_set_output


def main():
    parser = argparse.ArgumentParser(description="Get LKGR from a GitHub Actions workflow")
    parser.add_argument("--repo", required=True, help="Repository (e.g., ROCm/TheRock)")
    parser.add_argument("--workflow", required=True, help="Workflow file (e.g., multi_arch_ci.yml)")
    parser.add_argument("--branch", default="main", help="Branch to search (default: main)")
    parser.add_argument("--output", choices=["run_id", "head_sha", "html_url", "all"], default="all")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--gha-output", action="store_true", help="Write to GITHUB_OUTPUT")

    args = parser.parse_args()

    run = gha_query_last_successful_workflow_run(
        github_repository=args.repo,
        workflow_name=args.workflow,
        branch=args.branch,
    )

    if not run:
        print(f"No successful run found for {args.workflow} on {args.branch}", file=sys.stderr)
        sys.exit(1)

    result = {
        "run_id": run["id"],
        "head_sha": run["head_sha"],
        "html_url": run["html_url"],
        "created_at": run["created_at"],
    }

    if args.gha_output:
        gha_set_output({f"lkgr_{k}": str(v) for k, v in result.items()})

    if args.output != "all":
        result = {args.output: result[args.output]}

    if args.json:
        print(json.dumps(result, indent=2))
    elif args.output == "all":
        for k, v in result.items():
            print(f"{k}: {v}")
    else:
        print(list(result.values())[0])


if __name__ == "__main__":
    main()
