#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute affected build stages from changed projects, expanding TEST_SUBPROJECTS."""

import os
import sys
from pathlib import Path

# Add parent directory to path for _therock_utils imports
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from _therock_utils.build_topology import get_topology

# Add test_tools to path for TEST_SUBPROJECTS parsing
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "test_tools"))
from determine_rocm_test_dependencies import get_subprojects_to_test

from github_actions_api import gha_set_output


def compute_affected(changed_projects: str) -> tuple[str, str]:
    """Return (affected_stages, expanded_projects) for the given changed projects."""
    if not changed_projects or not changed_projects.strip():
        print("No changed_projects specified, building all stages")
        return "all", ""

    raw_projects = [p.strip() for p in changed_projects.split(",") if p.strip()]
    if not raw_projects:
        print("Empty projects list after parsing, building all stages")
        return "all", ""

    # Normalize "projects/hip" -> "hip"
    projects = [p.split("/")[-1] if "/" in p else p for p in raw_projects]

    print(f"Raw changed projects: {raw_projects}")
    print(f"Normalized projects: {projects}")

    # Expand to include TEST_SUBPROJECTS dependencies (e.g., rocprim -> rocsparse)
    expanded_projects = get_subprojects_to_test(projects, REPO_ROOT)
    expanded_list = sorted(expanded_projects)
    print(f"Expanded projects (with TEST_SUBPROJECTS): {expanded_list}")

    topology = get_topology()
    affected = topology.get_stages_for_projects(expanded_list)

    if not affected:
        print("No stages found for projects, building all stages")
        return "all", ""

    stages = ",".join(sorted(affected))
    # Space-separated for --projects arg compatibility
    projects_str = " ".join(expanded_list)
    print(f"Affected stages: {stages}")
    print(f"Expanded projects output: {projects_str}")
    return stages, projects_str


def main():
    changed_projects = os.environ.get("CHANGED_PROJECTS", "")
    affected_stages, expanded_projects = compute_affected(changed_projects)
    gha_set_output(
        {
            "affected_stages": affected_stages,
            "expanded_projects": expanded_projects,
        }
    )


if __name__ == "__main__":
    main()
