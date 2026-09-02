#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute sparse checkout paths for a specific build stage.

This script determines which project paths should be checked out for a
given build stage, based on which changed_projects affect that stage.

Usage:
    python compute_stage_sparse_checkout.py \
        --stage math-libs \
        --changed-projects "projects/rocprim,shared/rocroller"

Output (GitHub Actions format):
    sparse_checkout_paths<<EOF
    projects/rocprim
    shared/rocroller
    EOF

If the stage is NOT affected by any changed projects, outputs empty:
    sparse_checkout_paths=

This signals the workflow to do a full checkout instead.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology


def get_stage_source_paths(stage_name: str) -> Set[str]:
    """Get all source_paths that map to artifacts built in this stage.

    Returns a set of source_path values (e.g., {"rocprim", "hipcub", "rocthrust"})
    """
    topology = get_topology()
    source_paths: Set[str] = set()

    stage = topology.build_stages.get(stage_name)
    if not stage:
        return source_paths

    for group_name in stage.artifact_groups:
        for artifact in topology.get_artifacts_in_group(group_name):
            if artifact.source_paths:
                source_paths.update(artifact.source_paths)
            else:
                # Default: use artifact name as source_path
                source_paths.add(artifact.name)

    return source_paths


def extract_source_path_from_project(project_path: str) -> Optional[str]:
    """Extract the source_path name from a project path.

    E.g., "projects/rocprim" -> "rocprim"
         "shared/rocroller" -> "rocroller"
         "dnn-providers/miopen-provider" -> "miopen-provider"
    """
    parts = project_path.strip().split("/")
    if len(parts) >= 2:
        return parts[-1]
    return project_path.strip() if project_path.strip() else None


def compute_stage_sparse_checkout(stage_name: str, changed_projects: str) -> List[str]:
    """Compute sparse checkout paths for a stage based on changed projects.

    Args:
        stage_name: Build stage name (e.g., "math-libs")
        changed_projects: Comma-separated list of changed project paths

    Returns:
        List of project paths to sparse checkout, or empty list if stage
        is not affected (signals full checkout should be used).
    """
    if not changed_projects or not changed_projects.strip():
        return []

    # Get source_paths that this stage builds
    stage_source_paths = get_stage_source_paths(stage_name)
    if not stage_source_paths:
        # Unknown stage or stage with no source_paths - do full checkout
        print(f"Stage '{stage_name}' has no source_paths defined", file=sys.stderr)
        return []

    # Find which changed projects affect this stage
    affected_paths: List[str] = []
    for project in changed_projects.split(","):
        project = project.strip()
        if not project:
            continue

        source_path = extract_source_path_from_project(project)
        if source_path and source_path in stage_source_paths:
            affected_paths.append(project)

    return sorted(affected_paths)


def output_github_actions(paths: List[str]) -> None:
    """Output results in GitHub Actions format."""
    output_file = os.environ.get("GITHUB_OUTPUT")

    if output_file:
        with open(output_file, "a") as f:
            if paths:
                paths_str = "\n".join(paths)
                f.write(f"sparse_checkout_paths<<EOF\n{paths_str}\nEOF\n")
            else:
                f.write("sparse_checkout_paths=\n")
    else:
        # Print for local testing
        if paths:
            paths_str = "\n".join(paths)
            print(f"sparse_checkout_paths<<EOF\n{paths_str}\nEOF")
        else:
            print("sparse_checkout_paths=")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute sparse checkout paths for a build stage",
    )
    parser.add_argument(
        "--stage",
        required=True,
        help="Build stage name (e.g., math-libs, cv-libs)",
    )
    parser.add_argument(
        "--changed-projects",
        required=True,
        help="Comma-separated list of changed project paths",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    paths = compute_stage_sparse_checkout(args.stage, args.changed_projects)

    if paths:
        print(f"Stage '{args.stage}' sparse checkout: {paths}", file=sys.stderr)
    else:
        print(
            f"Stage '{args.stage}' not affected, using full checkout", file=sys.stderr
        )

    output_github_actions(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
