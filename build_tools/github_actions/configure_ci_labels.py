#!/usr/bin/env python3
"""
Manage CI labels for GitHub repositories.

This script creates and cleans up GitHub labels for repos. It is intended to be
used LOCALLY only to sync CI-related labels across repositories.

Usage:
    # List labels that would be created (dry run)
    python manage_ci_labels.py --repo OWNER/REPO --dry-run

    # Create missing labels
    python manage_ci_labels.py --repo OWNER/REPO --create

    # Delete labels not in the defined list
    python manage_ci_labels.py --repo OWNER/REPO --cleanup

    # Sync labels (create missing + delete extra)
    python manage_ci_labels.py --repo OWNER/REPO --sync
"""

import argparse
import subprocess
import json
import sys
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# CI Labels Definition
# =============================================================================
# Labels from ROCm/TheRock with prefixes: ci:, gfx, test:, test_filter:
# Format: (name, color, description)

CI_LABELS: list[tuple[str, str, str]] = [
    # ci: labels
    ("ci:skip", "FFFF00", "Skip all CI builds/tests for this PR"),
    ("ci:run-all-archs", "FFFF00", "Opt-in to building for all architectures on a pull request"),
    ("ci:run-multi-arch", "FFFF00", "Opt-in to running multi-arch CI on a pull request"),
    ("ci:run-non-multi-arch", "FFFF00", "Opt-in to running non-multi-arch CI on a pull request"),
    ("ci:build-jax", "FFFF00", "Enable Jax Build"),
    ("ci:asan", "FFFF00", "Opt-in to building ASAN"),
    ("ci:host-asan", "FFFF00", "Opt-in to running multi-arch host-asan CI on a pull request"),
    # gfx labels
    ("gfx103X-linux", "5A4D41", ""),
    ("gfx103X", "f9d0c4", ""),
    ("gfx90X-dcgpu", "5A4D41", ""),
    ("gfx94X-dcgpu", "5A4D41", "Issue/PR relates to gfx94X-dcgpu family."),
    ("gfx950-dcgpu", "5A4D41", "Issue/PR relates to gfx950-dcgpu family."),
    ("gfx110X-dgpu", "5A4D41", "Issue/PR relates to gfx110X-dgpu family."),
    ("gfx110X-all", "b04f4c", "Issue/PR related to gfx110X-all family"),
    ("gfx1103", "795816", "Issue/PR relates to gfx1103"),
    ("gfx1150", "5A4D41", ""),
    ("gfx1151", "5A4D41", "Issue/PR relates to gfx1151."),
    ("gfx1152", "5A4D41", ""),
    ("gfx1153", "5A4D41", ""),
    ("gfx120X-all", "5A4D41", "Issue/PR relates to gfx120X-all family"),
    ("gfx125x", "5A4D41", "Issue/PR relates to gfx125x family"),
    ("gfx125X-dcgpu", "5A4D41", "Issue/PR relates to gfx125X-dcgpu family."),
    ("gfx900", "5A4D41", ""),
    ("gfx906", "5A4D41", ""),
    ("gfx908", "5A4D41", ""),
    ("gfx90a", "5A4D41", ""),
    ("gfx90c", "5A4D41", ""),
    # test: labels
    ("test:hipblaslt", "3FA7D6", "For pull requests, runs full tests for only hipblaslt and other labeled projects."),
    ("test:hipcub", "3FA7D6", "For pull requests, runs full tests for only hipcub and other labeled projects."),
    ("test:miopen", "3FA7D6", "For pull requests, runs full tests for only miopen and other labeled projects."),
    ("test:rocblas", "3FA7D6", "For pull requests, runs full tests for only rocblas and other labeled projects."),
    ("test:hipblas", "3FA7D6", "For pull requests, runs full tests for only hipblas and other labeled projects."),
    ("test:rocprim", "3FA7D6", "For pull requests, runs full tests for only rocprim and other labeled projects."),
    ("test:rocsolver", "3FA7D6", "For pull requests, runs full tests for only rocsolver and other labeled projects."),
    ("test:rocthrust", "3FA7D6", "For pull requests, runs full tests for only rocthrust and other labeled projects."),
    ("test:rocsparse", "3FA7D6", "For pull requests, runs full tests for only rocsparse and other labeled projects."),
    ("test:hipsparse", "3FA7D6", "For pull requests, runs full tests for only hipsparse and other labeled projects."),
    ("test:hipfft", "3FA7D6", "For pull requests, runs full tests for only hipfft and other labeled projects."),
    ("test:hipsolver", "3FA7D6", "For pull requests, runs full tests for only hipsolver and other labeled projects."),
    ("test:rocfft", "3FA7D6", "For pull requests, runs full tests for only rocfft and other labeled projects."),
    ("test:hipsparselt", "3FA7D6", "For pull requests, runs full tests for only hipsparselt and other labeled projects."),
    ("test:rccl", "3FA7D6", "For pull requests, runs full tests for only rccl and other labeled projects."),
    ("test:hipdnn", "3FA7D6", "For pull requests, runs full tests for only hipdnn and other labeled projects."),
    ("test:rocroller", "3FA7D6", "For pull requests, runs full tests for only rocroller and other labeled projects."),
    ("test:composablekernel", "3FA7D6", "For pull requests, runs full tests for only composable_kernel and other labeled projects."),
    ("test:libhipcxx_hipcc", "3FA7D6", "For pull requests, runs full tests for only libhipcxx_hipcc and other labeled projects."),
    ("test:libhipcxx_hiprtc", "3FA7D6", "For pull requests, runs full tests for only libhipcxx_hiprtc and other labeled projects."),
    ("test:ocltst", "03B8ED", "For pull requests, this label executes the ocltst and other labeled projects."),
    ("test:hip-tests", "20566E", "For pull requests, this label executes the hip-tests"),
    ("test:rocrtst", "3FA7D6", "For pull requests, this label executes the rocrtst and other labeled projects."),
    ("test:origami", "3FA7D6", "For pull requests, this label executes the origami and other labeled projects."),
    ("test:rocdecode", "3FA7D6", "For pull requests, this label executes the rocdecode and other labeled projects."),
    ("test:rocjpeg", "3FA7D6", "For pull requests, this label executes the rocjpeg and other labeled projects."),
    ("test:rocprofiler-systems", "3FA7D6", "For pull requests, runs full tests for only rocprofiler-systems and other labeled projects."),
    ("test:rocprofiler-sdk", "3FA7D6", "For pull requests, runs full tests for only rocprofiler-sdk and other labeled projects."),
    ("test:hipkernelprovider", "20566E", "For pull requests, runs full tests for only hipkernelprovider and other labeled projects."),
    ("test:amdsmi", "277804", ""),
    ("test:rocgdb-cpu", "20566E", "Run ROCgdb cpu tests only"),
    ("test:rocgdb-gpu", "20566E", "Run ROCgdb gpu tests only"),
    ("test:rocgdb", "20566E", "Test all test:rocgdb* labels"),
    ("test:rocprofiler-sdk-spm", "3FA7D6", "To run rocprofier-sdk-spm jobs"),
    ("test:rpp", "3FA7D6", ""),
    ("test:miopen-dbsync", "3FA7D6", "For pull requests, runs the GPU-free miopen-dbsync (StaticFDBSync/rocjitsu) test component."),
    # test_filter: labels
    ("test_filter:quick", "a2fab4", "If enabled, the PR will run quick tests"),
    ("test_filter:standard", "a2fab4", "If enabled, the PR will run standard tests"),
    ("test_filter:comprehensive", "a2fab4", "If enabled, the PR will run comprehensive tests"),
    ("test_filter:full", "a2fab4", "If enabled, the PR will run full tests"),
    # test_runner: labels
    ("test_runner:oem", "23edeb", "If added, the tests will run on a machine configured with `oem` kernel"),
    # build_variant: labels
    ("build_variant:asan", "4b398c", "If enabled, the pull request will run ASAN builds"),
]


@dataclass
class Label:
    name: str
    color: str
    description: str


def run_gh_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    cmd = ["gh"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_repo_labels(repo: str) -> list[Label]:
    """Get all labels from a repository."""
    result = run_gh_command(
        ["label", "list", "--repo", repo, "--limit", "500", "--json", "name,color,description"]
    )
    labels_data = json.loads(result.stdout)
    return [Label(name=l["name"], color=l["color"], description=l["description"]) for l in labels_data]


def create_label(repo: str, name: str, color: str, description: str, dry_run: bool = False) -> bool:
    """Create a label in the repository."""
    if dry_run:
        print(f"  [DRY RUN] Would create label: {name}")
        return True

    args = ["label", "create", name, "--repo", repo, "--color", color, "--force"]
    if description:
        args.extend(["--description", description])

    result = run_gh_command(args, check=False)
    if result.returncode == 0:
        print(f"  Created label: {name}")
        return True
    else:
        print(f"  Failed to create label {name}: {result.stderr}")
        return False


def delete_label(repo: str, name: str, dry_run: bool = False) -> bool:
    """Delete a label from the repository."""
    if dry_run:
        print(f"  [DRY RUN] Would delete label: {name}")
        return True

    result = run_gh_command(["label", "delete", name, "--repo", repo, "--yes"], check=False)
    if result.returncode == 0:
        print(f"  Deleted label: {name}")
        return True
    else:
        print(f"  Failed to delete label {name}: {result.stderr}")
        return False


def get_ci_label_names() -> set[str]:
    """Get set of CI label names."""
    return {label[0] for label in CI_LABELS}


# Prefixes that are safe to clean up (delete if not in CI_LABELS)
# We only cleanup these specific prefixes to avoid touching other labels
CLEANUP_PREFIXES = ("gfx", "test:", "test_filter:")


def filter_cleanable_labels(labels: list[Label]) -> list[Label]:
    """Filter labels to only those with prefixes safe to cleanup."""
    return [l for l in labels if any(l.name.startswith(p) for p in CLEANUP_PREFIXES)]


def list_labels(repo: str) -> None:
    """List all CI labels defined and their status in the repo."""
    print(f"\nCI Labels defined in this script: {len(CI_LABELS)}")
    print("-" * 60)

    existing_labels = get_repo_labels(repo)
    existing_names = {l.name for l in existing_labels}

    print("\nLabels that would be created/updated:")
    for name, color, description in CI_LABELS:
        status = "exists" if name in existing_names else "MISSING"
        print(f"  [{status}] {name} (#{color})")

    ci_label_names = get_ci_label_names()
    extra_cleanable_labels = [l for l in filter_cleanable_labels(existing_labels) if l.name not in ci_label_names]

    if extra_cleanable_labels:
        print(f"\nCleanable labels in repo not in this script ({len(extra_cleanable_labels)}):")
        print(f"(Only labels with prefixes {CLEANUP_PREFIXES} are subject to cleanup)")
        for label in extra_cleanable_labels:
            print(f"  {label.name} (#{label.color})")


def create_labels(repo: str, dry_run: bool = False) -> None:
    """Create missing CI labels in the repository."""
    print(f"\nCreating CI labels in {repo}...")
    if dry_run:
        print("(DRY RUN - no changes will be made)\n")

    existing_labels = get_repo_labels(repo)
    existing_names = {l.name for l in existing_labels}

    created = 0
    skipped = 0
    for name, color, description in CI_LABELS:
        if name in existing_names:
            skipped += 1
            continue
        if create_label(repo, name, color, description, dry_run):
            created += 1

    print(f"\nSummary: {created} created, {skipped} already existed")


def cleanup_labels(repo: str, dry_run: bool = False) -> None:
    """Delete labels with cleanable prefixes that are not in the defined list.

    Only cleans up labels with prefixes: gfx, test:, test_filter:
    Other labels (ci:, test_runner:, build_variant:, etc.) are NOT touched.
    """
    print(f"\nCleaning up labels in {repo}...")
    print(f"(Only cleaning labels with prefixes: {CLEANUP_PREFIXES})")
    if dry_run:
        print("(DRY RUN - no changes will be made)\n")

    existing_labels = get_repo_labels(repo)
    ci_label_names = get_ci_label_names()

    # Only clean up labels with cleanable prefixes that aren't in our list
    extra_labels = [l for l in filter_cleanable_labels(existing_labels) if l.name not in ci_label_names]

    if not extra_labels:
        print("No extra labels to clean up.")
        return

    deleted = 0
    for label in extra_labels:
        if delete_label(repo, label.name, dry_run):
            deleted += 1

    print(f"\nSummary: {deleted} labels deleted")


def sync_labels(repo: str, dry_run: bool = False) -> None:
    """Sync labels: create missing and optionally delete extra."""
    create_labels(repo, dry_run)
    print()
    cleanup_labels(repo, dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Manage CI labels for GitHub repositories (LOCAL USE ONLY)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository in OWNER/REPO format (e.g., ROCm/TheRock)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List CI labels and their status",
    )
    action_group.add_argument(
        "--create",
        action="store_true",
        help="Create missing CI labels",
    )
    action_group.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete labels with prefixes (gfx, test:, test_filter:) not in the defined list",
    )
    action_group.add_argument(
        "--sync",
        action="store_true",
        help="Sync labels (create missing + delete extra)",
    )

    args = parser.parse_args()

    # Validate repo format
    if "/" not in args.repo:
        print(f"Error: Repository must be in OWNER/REPO format, got: {args.repo}")
        sys.exit(1)

    try:
        if args.list:
            list_labels(args.repo)
        elif args.create:
            create_labels(args.repo, args.dry_run)
        elif args.cleanup:
            cleanup_labels(args.repo, args.dry_run)
        elif args.sync:
            sync_labels(args.repo, args.dry_run)
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
