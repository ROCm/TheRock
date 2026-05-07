#!/usr/bin/env python3

"""
PR Category Label Script
--------------------
This script analyzes the file paths changed in a pull request and determines which
category labels should be added or removed based on the modified files.

It uses GitHub's cli to fetch the changed files and the existing labels on the pull request.
Then, it computes the desired labels based on file paths, compares them to the existing labels,
and applies the necessary additions and removals unless in dry-run mode.

Arguments:
    --repo      : Full repository name (e.g., org/repo)
    --pr        : Pull request number
    --dry-run   : If set, will only log actions without making changes.
    --debug     : If set, enables detailed debug logging.

Outputs:
    Writes 'add' and 'remove' keys to the GitHub Actions $GITHUB_OUTPUT file, which
    the workflow reads to apply label changes using the GitHub CLI.

Example Usage:
    To run in debug mode and perform a dry-run (no changes made):
        python pr_auto_label.py --repo ROCm/rocm-libraries --pr <pr-number> --dry-run --debug
    To run in debug mode and apply label changes:
        python pr_auto_label.py --repo ROCm/rocm-libraries --pr <pr-number> --debug
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
from github_cli_client import GitHubCLIClient

logger = logging.getLogger(__name__)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply labels based on PR's changed files."
    )
    parser.add_argument(
        "--repo", required=True, help="Full repository name (e.g., org/repo)"
    )
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to GITHUB_OUTPUT.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def compute_desired_labels(file_paths: list) -> set:
    """Determine the desired labels based on the changed files."""
    desired_labels = set()
    for path in file_paths:
        parts = Path(path).parts
        if len(parts) >= 2:
            if parts[0] == "projects":
                desired_labels.add(f"project: {parts[1]}")
            elif parts[0] == "shared":
                desired_labels.add(f"shared: {parts[1]}")
    logger.debug(f"Desired labels based on changes: {desired_labels}")
    return desired_labels


def output_labels(
    existing_labels: List[str], desired_labels: List[str], dry_run: bool
) -> None:
    """Output the labels to add/remove to GITHUB_OUTPUT or log them in dry-run mode."""
    to_add = sorted(desired_labels - set(existing_labels))
    logger.debug(f"Labels to add: {to_add}")
    if dry_run:
        logger.info("Dry run enabled. Labels will not be applied.")
    else:
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                print(f"label_add={','.join(to_add)}", file=f)
            logger.info(f"Wrote to GITHUB_OUTPUT: add={','.join(to_add)}")
        else:
            print(
                "GITHUB_OUTPUT environment variable not set. Outputs cannot be written."
            )
            sys.exit(1)


def main(argv=None) -> None:
    """Main function to execute the PR auto label logic."""
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    client = GitHubCLIClient()
    changed_files = [file for file in client.get_changed_files(args.repo, int(args.pr))]

    if not changed_files:
        logger.warning(
            "REST API failed or returned no changed files. Falling back to SHA-based Git diff..."
        )
        try:
            pr_data = os.popen(f"gh api repos/{args.repo}/pulls/{args.pr}").read()
            pr = json.loads(pr_data)
            base_sha = pr["base"]["sha"]
            head_sha = pr["head"]["sha"]
            logger.debug(f"Base SHA: {base_sha}, Head SHA: {head_sha}")
            os.system(f"git fetch origin {base_sha} {head_sha}")
            result = os.popen(f"git diff --name-only {base_sha} {head_sha}").read()
            changed_files = result.strip().splitlines()
            logger.info(f"Fallback changed files (SHA-based): {changed_files}")
        except Exception as e:
            logger.error(f"SHA-based Git CLI fallback failed: {e}")
            sys.exit(1)

    existing_labels = client.get_existing_labels_on_pr(args.repo, int(args.pr))
    desired_labels = compute_desired_labels(changed_files)
    output_labels(existing_labels, desired_labels, args.dry_run)


if __name__ == "__main__":
    main()
