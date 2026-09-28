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


@dataclass
class Label:
    name: str
    color: str
    description: str


# =============================================================================
# Color Constants
# =============================================================================
# Named constants for label colors to ensure consistency and easy updates.
COLOR_CI_GENERAL = "FFFF00"  # Yellow - general CI behavior labels
COLOR_CI_GFX = "5A4D41"  # Brown - GPU architecture opt-in labels
COLOR_TEST = "3FA7D6"  # Blue - project-specific test labels
COLOR_TEST_FILTER = "a2fab4"  # Light green - test level override labels
COLOR_TEST_RUNNER = "23edeb"  # Cyan - test machine selection labels
COLOR_BUILD_VARIANT = "4b398c"  # Purple - build variant labels


# =============================================================================
# CI Labels Definition
# =============================================================================
# Labels from ROCm/TheRock for CI behavior manipulation.
# fmt: off
CI_LABELS: list[Label] = [
    # ci: general labels
    Label("ci:skip", COLOR_CI_GENERAL, "Skip all CI builds/tests for this PR"),
    Label("ci:run-all-archs", COLOR_CI_GENERAL, "Opt-in to building for all architectures on a pull request"),
    Label("ci:run-multi-arch", COLOR_CI_GENERAL, "Opt-in to running multi-arch CI on a pull request"),
    Label("ci:run-non-multi-arch", COLOR_CI_GENERAL, "Opt-in to running non-multi-arch CI on a pull request"),
    Label("ci:build-jax", COLOR_CI_GENERAL, "Enable Jax Build"),
    Label("ci:asan", COLOR_CI_GENERAL, "Opt-in to building ASAN"),
    Label("ci:host-asan", COLOR_CI_GENERAL, "Opt-in to running multi-arch host-asan CI on a pull request"),
    # ci:gfx labels (GPU architecture opt-in)
    Label("ci:gfx103X-linux", COLOR_CI_GFX, "Opt-in to gfx103X-linux builds/tests"),
    Label("ci:gfx103X", COLOR_CI_GFX, "Opt-in to gfx103X builds/tests"),
    Label("ci:gfx90X-dcgpu", COLOR_CI_GFX, "Opt-in to gfx90X-dcgpu builds/tests"),
    Label("ci:gfx94X-dcgpu", COLOR_CI_GFX, "Opt-in to gfx94X-dcgpu builds/tests"),
    Label("ci:gfx950-dcgpu", COLOR_CI_GFX, "Opt-in to gfx950-dcgpu builds/tests"),
    Label("ci:gfx110X-dgpu", COLOR_CI_GFX, "Opt-in to gfx110X-dgpu builds/tests"),
    Label("ci:gfx110X-all", COLOR_CI_GFX, "Opt-in to gfx110X-all builds/tests"),
    Label("ci:gfx1103", COLOR_CI_GFX, "Opt-in to gfx1103 builds/tests"),
    Label("ci:gfx1150", COLOR_CI_GFX, "Opt-in to gfx1150 builds/tests"),
    Label("ci:gfx1151", COLOR_CI_GFX, "Opt-in to gfx1151 builds/tests"),
    Label("ci:gfx1152", COLOR_CI_GFX, "Opt-in to gfx1152 builds/tests"),
    Label("ci:gfx1153", COLOR_CI_GFX, "Opt-in to gfx1153 builds/tests"),
    Label("ci:gfx120X-all", COLOR_CI_GFX, "Opt-in to gfx120X-all builds/tests"),
    Label("ci:gfx125x", COLOR_CI_GFX, "Opt-in to gfx125x builds/tests"),
    Label("ci:gfx125X-dcgpu", COLOR_CI_GFX, "Opt-in to gfx125X-dcgpu builds/tests"),
    Label("ci:gfx900", COLOR_CI_GFX, "Opt-in to gfx900 builds/tests"),
    Label("ci:gfx906", COLOR_CI_GFX, "Opt-in to gfx906 builds/tests"),
    Label("ci:gfx908", COLOR_CI_GFX, "Opt-in to gfx908 builds/tests"),
    Label("ci:gfx90a", COLOR_CI_GFX, "Opt-in to gfx90a builds/tests"),
    Label("ci:gfx90c", COLOR_CI_GFX, "Opt-in to gfx90c builds/tests"),
    # test: labels (project-specific test opt-in)
    Label("test:hipblaslt", COLOR_TEST, "Run full tests for hipblaslt"),
    Label("test:hipcub", COLOR_TEST, "Run full tests for hipcub"),
    Label("test:miopen", COLOR_TEST, "Run full tests for miopen"),
    Label("test:rocblas", COLOR_TEST, "Run full tests for rocblas"),
    Label("test:hipblas", COLOR_TEST, "Run full tests for hipblas"),
    Label("test:rocprim", COLOR_TEST, "Run full tests for rocprim"),
    Label("test:rocsolver", COLOR_TEST, "Run full tests for rocsolver"),
    Label("test:rocthrust", COLOR_TEST, "Run full tests for rocthrust"),
    Label("test:rocsparse", COLOR_TEST, "Run full tests for rocsparse"),
    Label("test:hipsparse", COLOR_TEST, "Run full tests for hipsparse"),
    Label("test:hipfft", COLOR_TEST, "Run full tests for hipfft"),
    Label("test:hipsolver", COLOR_TEST, "Run full tests for hipsolver"),
    Label("test:rocfft", COLOR_TEST, "Run full tests for rocfft"),
    Label("test:hipsparselt", COLOR_TEST, "Run full tests for hipsparselt"),
    Label("test:rccl", COLOR_TEST, "Run full tests for rccl"),
    Label("test:hipdnn", COLOR_TEST, "Run full tests for hipdnn"),
    Label("test:rocroller", COLOR_TEST, "Run full tests for rocroller"),
    Label("test:composablekernel", COLOR_TEST, "Run full tests for composable_kernel"),
    Label("test:libhipcxx_hipcc", COLOR_TEST, "Run full tests for libhipcxx_hipcc"),
    Label("test:libhipcxx_hiprtc", COLOR_TEST, "Run full tests for libhipcxx_hiprtc"),
    Label("test:ocltst", COLOR_TEST, "Run ocltst tests"),
    Label("test:hip-tests", COLOR_TEST, "Run hip-tests"),
    Label("test:rocrtst", COLOR_TEST, "Run rocrtst tests"),
    Label("test:origami", COLOR_TEST, "Run origami tests"),
    Label("test:rocdecode", COLOR_TEST, "Run rocdecode tests"),
    Label("test:rocjpeg", COLOR_TEST, "Run rocjpeg tests"),
    Label("test:rocprofiler-systems", COLOR_TEST, "Run full tests for rocprofiler-systems"),
    Label("test:rocprofiler-sdk", COLOR_TEST, "Run full tests for rocprofiler-sdk"),
    Label("test:hipkernelprovider", COLOR_TEST, "Run full tests for hipkernelprovider"),
    Label("test:amdsmi", COLOR_TEST, "Run full tests for amdsmi"),
    Label("test:rocgdb-cpu", COLOR_TEST, "Run ROCgdb cpu tests only"),
    Label("test:rocgdb-gpu", COLOR_TEST, "Run ROCgdb gpu tests only"),
    Label("test:rocgdb", COLOR_TEST, "Test all test:rocgdb* labels"),
    Label("test:rocprofiler-sdk-spm", COLOR_TEST, "Run rocprofiler-sdk-spm tests"),
    Label("test:rpp", COLOR_TEST, "Run full tests for rpp"),
    Label("test:miopen-dbsync", COLOR_TEST, "Run miopen-dbsync (StaticFDBSync/rocjitsu) tests"),
    # test_filter: labels (test level override)
    Label("test_filter:quick", COLOR_TEST_FILTER, "If enabled, the PR will run quick tests"),
    Label("test_filter:standard", COLOR_TEST_FILTER, "If enabled, the PR will run standard tests"),
    Label("test_filter:comprehensive", COLOR_TEST_FILTER, "If enabled, the PR will run comprehensive tests"),
    Label("test_filter:full", COLOR_TEST_FILTER, "If enabled, the PR will run full tests"),
    # test_runner: labels (test machine selection)
    Label("test_runner:oem", COLOR_TEST_RUNNER, "Run tests on a machine configured with `oem` kernel"),
    # build_variant: labels
    Label("build_variant:asan", COLOR_BUILD_VARIANT, "If enabled, the pull request will run ASAN builds"),
]
# fmt: on


def run_gh_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command."""
    cmd = ["gh"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_repo_labels(repo: str) -> list[Label]:
    """Get all labels from a repository."""
    result = run_gh_command(
        [
            "label",
            "list",
            "--repo",
            repo,
            "--limit",
            "500",
            "--json",
            "name,color,description",
        ]
    )
    labels_data = json.loads(result.stdout)
    return [
        Label(name=l["name"], color=l["color"], description=l["description"])
        for l in labels_data
    ]


def create_label(
    repo: str, name: str, color: str, description: str, dry_run: bool = False
) -> bool:
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


def update_label(
    repo: str,
    name: str,
    color: str,
    description: str,
    existing: Label,
    dry_run: bool = False,
) -> bool:
    """Update an existing label's color and description."""
    changes = []
    if existing.color.lower() != color.lower():
        changes.append(f"color: #{existing.color} -> #{color}")
    if existing.description != description:
        old_desc = (
            existing.description[:30] + "..."
            if len(existing.description) > 30
            else existing.description
        )
        new_desc = description[:30] + "..." if len(description) > 30 else description
        changes.append(f'desc: "{old_desc}" -> "{new_desc}"')
    change_str = ", ".join(changes)

    if dry_run:
        print(f"  [DRY RUN] Would update label: {name} ({change_str})")
        return True

    args = ["label", "edit", name, "--repo", repo, "--color", color]
    if description:
        args.extend(["--description", description])

    result = run_gh_command(args, check=False)
    if result.returncode == 0:
        print(f"  Updated label: {name} ({change_str})")
        return True
    else:
        print(f"  Failed to update label {name}: {result.stderr}")
        return False


def get_ci_label_names() -> set[str]:
    """Get set of CI label names."""
    return {label.name for label in CI_LABELS}


def list_labels(repo: str) -> None:
    """List all CI labels defined and their status in the repo."""
    print(f"\nCI Labels defined in this script: {len(CI_LABELS)}")
    print("-" * 60)

    existing_labels = get_repo_labels(repo)
    existing_names = {l.name for l in existing_labels}

    missing_count = 0
    exists_count = 0

    print("\nLabel status:")
    for label in CI_LABELS:
        if label.name in existing_names:
            status = "exists"
            exists_count += 1
        else:
            status = "MISSING"
            missing_count += 1
        print(f"  [{status}] {label.name} (#{label.color})")

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
    for label in CI_LABELS:
        if label.name in existing_map:
            if force:
                existing = existing_map[label.name]
                # Check if color or description differs
                if (
                    existing.color.lower() != label.color.lower()
                    or existing.description != label.description
                ):
                    if update_label(
                        repo,
                        label.name,
                        label.color,
                        label.description,
                        existing,
                        dry_run,
                    ):
                        updated += 1
                else:
                    if dry_run:
                        print(f"  [DRY RUN] Already up to date: {label.name}")
                    skipped += 1
            else:
                skipped += 1
            continue
        if create_label(repo, label.name, label.color, label.description, dry_run):
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
