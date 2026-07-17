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


def get_affected_stages(changed_projects: str) -> str:
    """Return comma-separated affected stage names, or 'all' for full build."""
    if not changed_projects or not changed_projects.strip():
        print("No changed_projects specified, building all stages")
        return "all"

    raw_projects = [p.strip() for p in changed_projects.split(",") if p.strip()]
    if not raw_projects:
        print("Empty projects list after parsing, building all stages")
        return "all"

    # Normalize "projects/hip" -> "hip"
    projects = [p.split("/")[-1] if "/" in p else p for p in raw_projects]

    print(f"Raw changed projects: {raw_projects}")
    print(f"Normalized projects: {projects}")

    # Expand to include TEST_SUBPROJECTS dependencies (e.g., rocprim -> rocsparse)
    expanded_projects = get_subprojects_to_test(projects, REPO_ROOT)
    print(f"Expanded projects (with TEST_SUBPROJECTS): {sorted(expanded_projects)}")

    topology = get_topology()
    affected = topology.get_stages_for_projects(list(expanded_projects))

    if not affected:
        # No stages found - fall back to full build
        print("No stages found for projects, building all stages")
        return "all"

    result = ",".join(sorted(affected))
    print(f"Affected stages: {result}")
    return result


def main():
    changed_projects = os.environ.get("CHANGED_PROJECTS", "")
    affected_stages = get_affected_stages(changed_projects)
    gha_set_output({"affected_stages": affected_stages})


if __name__ == "__main__":
    main()
