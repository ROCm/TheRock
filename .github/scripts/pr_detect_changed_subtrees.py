#!/usr/bin/env python3

"""
PR Detect Changed Subtrees Script
---------------------------------
This script analyzes a pull request's changed files and determines which subtrees
(defined in .github/repos-config.json by category/name) were affected.

Steps:
    1. Fetch the changed files in the PR using the GitHub API.
    2. Load the subtree mapping from repos-config.json.
    3. Match changed paths against known category/name prefixes.
    4. Emit a new-line separated list of changed subtrees to GITHUB_OUTPUT as 'subtrees'.

Arguments:
    --repo              : Full repository name (e.g., org/repo)
    --pr                : Pull request number
    --config            : OPTIONAL, path to the repos-config.json file.
    --require-auto-pull : If set, only include entries with auto_subtree_pull=true.
    --require-auto-push : If set, only include entries with auto_subtree_push=true.
    --require-monorepo-source : If set, only include entries with monorepo_source_of_truth=true.
    --dry-run           : If set, will only log actions without making changes.
    --debug             : If set, enables detailed debug logging.

Outputs:
    Writes 'subtrees' key to the GitHub Actions $GITHUB_OUTPUT file, which
    the workflow reads to pass paths to the checkout stages.
    The output is a new-line separated list of subtrees in `category/name` format.

Example Usage:
    To run in auto-push situations in debug mode and perform a dry-run (no changes made):
        python pr_detect_changed_subtrees.py --repo ROCm/rocm-libraries --pr 123 --require-auto-push --debug --dry-run
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Set
from github_cli_client import GitHubCLIClient
from repo_config_model import RepoEntry
from config_loader import load_repo_config

logger = logging.getLogger(__name__)


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect changed subtrees in a PR.")
    parser.add_argument(
        "--repo", required=True, help="Full repository name (e.g., org/repo)"
    )
    parser.add_argument("--pr", required=True, type=int, help="Pull request number")
    parser.add_argument(
        "--config",
        required=False,
        default=".github/repos-config.json",
        help="Path to the repos-config.json file",
    )
    parser.add_argument(
        "--require-auto-pull",
        action="store_true",
        help="Only include entries with auto_subtree_pull=true",
    )
    parser.add_argument(
        "--require-auto-push",
        action="store_true",
        help="Only include entries with auto_subtree_push=true",
    )
    parser.add_argument(
        "--require-monorepo-source",
        action="store_true",
        help="Only include entries with monorepo_source_of_truth=true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to GITHUB_OUTPUT.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def get_valid_prefixes(
    config: List[RepoEntry],
    require_auto_pull: bool = False,
    require_auto_push: bool = False,
    require_monorepo_source: bool = False,
) -> Set[str]:
    """Extract valid subtree prefixes from the configuration based on filters."""
    valid_prefixes = set()
    for entry in config:
        if require_auto_pull and not getattr(entry, "auto_subtree_pull", False):
            continue
        if require_auto_push and not getattr(entry, "auto_subtree_push", False):
            continue
        if require_monorepo_source and not getattr(
            entry, "monorepo_source_of_truth", False
        ):
            continue
        valid_prefixes.add(f"{entry.category}/{entry.name}")
    logger.debug("Valid subtrees:\n" + "\n".join(sorted(valid_prefixes)))
    return valid_prefixes


def find_matched_subtrees(
    changed_files: List[str], valid_prefixes: Set[str]
) -> List[str]:
    """Find subtrees that match the changed files."""
    changed_subtrees = {
        "/".join(path.split("/", 2)[:2])
        for path in changed_files
        if len(path.split("/")) >= 2
    }
    matched = sorted(changed_subtrees & valid_prefixes)
    skipped = sorted(changed_subtrees - valid_prefixes)
    if skipped:
        logger.debug(f"Skipped subtrees: {skipped}")
    logger.debug(f"Matched subtrees: {matched}")
    return matched


def output_subtrees(matched_subtrees: List[str], dry_run: bool) -> None:
    """Output the matched subtrees to GITHUB_OUTPUT or log them in dry-run mode."""
    newline_separated = "\n".join(matched_subtrees)
    if dry_run:
        logger.info(f"[Dry-run] Would output:\n{newline_separated}")
    else:
        output_file = os.environ.get("GITHUB_OUTPUT")
        if output_file:
            with open(output_file, "a") as f:
                print(f"subtrees<<EOF\n{newline_separated}\nEOF", file=f)
            logger.info("Wrote matched subtrees to GITHUB_OUTPUT.")
        else:
            logger.error(
                "GITHUB_OUTPUT environment variable not set. Outputs cannot be written."
            )
            sys.exit(1)


def main(argv=None) -> None:
    """Main function to determine changed subtrees in PR."""
    args = parse_arguments(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    client = GitHubCLIClient()
    config = load_repo_config(args.config)
    changed_files = client.get_changed_files(args.repo, int(args.pr))

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

    valid_prefixes = get_valid_prefixes(
        config,
        args.require_auto_pull,
        args.require_auto_push,
        args.require_monorepo_source,
    )
    matched_subtrees = find_matched_subtrees(changed_files, valid_prefixes)
    output_subtrees(matched_subtrees, args.dry_run)


if __name__ == "__main__":
    main()
