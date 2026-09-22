#!/usr/bin/env python3
"""
Create CI labels for GitHub repositories.

This script creates CI-related GitHub labels for repos. It is intended to be
used LOCALLY only to sync CI-related labels across repositories.

Usage:
    # List labels and their status in the repo
    python configure_ci_labels.py --repo OWNER/REPO --list

    # Create missing labels (dry run)
    python configure_ci_labels.py --repo OWNER/REPO --create --dry-run

    # Create missing labels
    python configure_ci_labels.py --repo OWNER/REPO --create

    # Create missing labels and update existing ones to match color/description
    python configure_ci_labels.py --repo OWNER/REPO --create --force
"""

import argparse
import subprocess
import json
import sys
from dataclasses import dataclass

# =============================================================================
# CI Labels Definition
# =============================================================================
# Labels from ROCm/TheRock for CI behavior manipulation.
# Format: (name, color, description)

CI_LABELS: list[tuple[str, str, str]] = [
    # ci: general labels
    ("ci:skip", "FFFF00", "Skip all CI builds/tests for this PR"),
    ("ci:run-all-archs", "FFFF00", "Opt-in to building for all architectures on a pull request"),
    ("ci:run-multi-arch", "FFFF00", "Opt-in to running multi-arch CI on a pull request"),
    ("ci:run-non-multi-arch", "FFFF00", "Opt-in to running non-multi-arch CI on a pull request"),
    ("ci:build-jax", "FFFF00", "Enable Jax Build"),
    ("ci:asan", "FFFF00", "Opt-in to building ASAN"),
    ("ci:host-asan", "FFFF00", "Opt-in to running multi-arch host-asan CI on a pull request"),
    # ci:gfx labels (GPU architecture opt-in)
    ("ci:gfx103X-linux", "5A4D41", "Opt-in to gfx103X-linux builds/tests"),
    ("ci:gfx103X", "5A4D41", "Opt-in to gfx103X builds/tests"),
    ("ci:gfx90X-dcgpu", "5A4D41", "Opt-in to gfx90X-dcgpu builds/tests"),
    ("ci:gfx94X-dcgpu", "5A4D41", "Opt-in to gfx94X-dcgpu builds/tests"),
    ("ci:gfx950-dcgpu", "5A4D41", "Opt-in to gfx950-dcgpu builds/tests"),
    ("ci:gfx110X-dgpu", "5A4D41", "Opt-in to gfx110X-dgpu builds/tests"),
    ("ci:gfx110X-all", "5A4D41", "Opt-in to gfx110X-all builds/tests"),
    ("ci:gfx1103", "5A4D41", "Opt-in to gfx1103 builds/tests"),
    ("ci:gfx1150", "5A4D41", "Opt-in to gfx1150 builds/tests"),
    ("ci:gfx1151", "5A4D41", "Opt-in to gfx1151 builds/tests"),
    ("ci:gfx1152", "5A4D41", "Opt-in to gfx1152 builds/tests"),
    ("ci:gfx1153", "5A4D41", "Opt-in to gfx1153 builds/tests"),
    ("ci:gfx120X-all", "5A4D41", "Opt-in to gfx120X-all builds/tests"),
    ("ci:gfx125x", "5A4D41", "Opt-in to gfx125x builds/tests"),
    ("ci:gfx125X-dcgpu", "5A4D41", "Opt-in to gfx125X-dcgpu builds/tests"),
    ("ci:gfx900", "5A4D41", "Opt-in to gfx900 builds/tests"),
    ("ci:gfx906", "5A4D41", "Opt-in to gfx906 builds/tests"),
    ("ci:gfx908", "5A4D41", "Opt-in to gfx908 builds/tests"),
    ("ci:gfx90a", "5A4D41", "Opt-in to gfx90a builds/tests"),
    ("ci:gfx90c", "5A4D41", "Opt-in to gfx90c builds/tests"),
    # test: labels (project-specific test opt-in)
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
    ("test:amdsmi", "277804", "For pull requests, runs full tests for only amdsmi and other labeled projects."),
    ("test:rocgdb-cpu", "20566E", "Run ROCgdb cpu tests only"),
    ("test:rocgdb-gpu", "20566E", "Run ROCgdb gpu tests only"),
    ("test:rocgdb", "20566E", "Test all test:rocgdb* labels"),
    ("test:rocprofiler-sdk-spm", "3FA7D6", "To run rocprofiler-sdk-spm jobs"),
    ("test:rpp", "3FA7D6", "For pull requests, runs full tests for only rpp and other labeled projects."),
    ("test:miopen-dbsync", "3FA7D6", "For pull requests, runs the GPU-free miopen-dbsync (StaticFDBSync/rocjitsu) test component."),
    # test_filter: labels (test level override)
    ("test_filter:quick", "a2fab4", "If enabled, the PR will run quick tests"),
    ("test_filter:standard", "a2fab4", "If enabled, the PR will run standard tests"),
    ("test_filter:comprehensive", "a2fab4", "If enabled, the PR will run comprehensive tests"),
    ("test_filter:full", "a2fab4", "If enabled, the PR will run full tests"),
    # test_runner: labels (test machine selection)
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
    """Create a label in the repository. Skips if already exists."""
    if dry_run:
        print(f"  [DRY RUN] Would create label: {name}")
        return True

    args = ["label", "create", name, "--repo", repo, "--color", color]
    if description:
        args.extend(["--description", description])

    result = run_gh_command(args, check=False)
    if result.returncode == 0:
        print(f"  Created label: {name}")
        return True
    else:
        print(f"  Failed to create label {name}: {result.stderr}")
        return False


def update_label(repo: str, name: str, color: str, description: str, dry_run: bool = False) -> bool:
    """Update an existing label's color and description."""
    if dry_run:
        print(f"  [DRY RUN] Would update label: {name}")
        return True

    args = ["label", "edit", name, "--repo", repo, "--color", color]
    if description:
        args.extend(["--description", description])

    result = run_gh_command(args, check=False)
    if result.returncode == 0:
        print(f"  Updated label: {name}")
        return True
    else:
        print(f"  Failed to update label {name}: {result.stderr}")
        return False


def get_ci_label_names() -> set[str]:
    """Get set of CI label names."""
    return {label[0] for label in CI_LABELS}


def list_labels(repo: str) -> None:
    """List all CI labels defined and their status in the repo."""
    print(f"\nCI Labels defined in this script: {len(CI_LABELS)}")
    print("-" * 60)

    existing_labels = get_repo_labels(repo)
    existing_names = {l.name for l in existing_labels}

    missing_count = 0
    exists_count = 0

    print("\nLabel status:")
    for name, color, description in CI_LABELS:
        if name in existing_names:
            status = "exists"
            exists_count += 1
        else:
            status = "MISSING"
            missing_count += 1
        print(f"  [{status}] {name} (#{color})")

    print(f"\nSummary: {exists_count} exist, {missing_count} missing")


def create_labels(repo: str, dry_run: bool = False, force: bool = False) -> None:
    """Create missing CI labels in the repository.

    Args:
        repo: Repository in OWNER/REPO format.
        dry_run: If True, only print what would be done.
        force: If True, update existing labels to match color/description.
    """
    print(f"\nCreating CI labels in {repo}...")
    if force:
        print("(--force: existing labels will be updated to match)")
    if dry_run:
        print("(DRY RUN - no changes will be made)\n")

    existing_labels = get_repo_labels(repo)
    existing_map = {l.name: l for l in existing_labels}

    created = 0
    updated = 0
    skipped = 0
    for name, color, description in CI_LABELS:
        if name in existing_map:
            if force:
                existing = existing_map[name]
                # Check if color or description differs (color comparison is case-insensitive)
                if existing.color.lower() != color.lower() or existing.description != description:
                    if update_label(repo, name, color, description, dry_run):
                        updated += 1
                else:
                    skipped += 1
            else:
                skipped += 1
            continue
        if create_label(repo, name, color, description, dry_run):
            created += 1

    summary_parts = [f"{created} created", f"{skipped} already matched"]
    if force:
        summary_parts.insert(1, f"{updated} updated")
    print(f"\nSummary: {', '.join(summary_parts)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create CI labels for GitHub repositories (LOCAL USE ONLY)",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Update existing labels to match color/description (use with --create)",
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--list",
        action="store_true",
        help="List CI labels and their status in the repo",
    )
    action_group.add_argument(
        "--create",
        action="store_true",
        help="Create missing CI labels (skips existing)",
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
            create_labels(args.repo, args.dry_run, args.force)
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
