#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute affected build stages from changed projects.

For selective builds (e.g., external repos like rocm-systems or rocm-libraries),
this script determines which build stages need to run based on which projects
have changes.

Inputs:
    CHANGED_PROJECTS: Comma-separated list of changed project paths/names.
                      Empty or unset means full build (all stages).

Outputs (written to GITHUB_OUTPUT):
    affected_stages: Comma-separated list of affected stage names,
                     or "all" for full build.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for _therock_utils imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _therock_utils.build_topology import get_topology

from github_actions_api import gha_set_output


def get_affected_stages(changed_projects: str) -> str:
    """Get build stages affected by the changed projects.

    Args:
        changed_projects: Comma-separated list of changed project paths/names

    Returns:
        Comma-separated list of affected stage names, or "all" if no projects
        specified (full build)
    """
    if not changed_projects or not changed_projects.strip():
        print("No changed_projects specified, building all stages")
        return "all"

    raw_projects = [p.strip() for p in changed_projects.split(",") if p.strip()]
    if not raw_projects:
        print("Empty projects list after parsing, building all stages")
        return "all"

    # Normalize project paths to just the project name
    # e.g., "projects/hip" -> "hip", "projects/rocblas" -> "rocblas"
    projects = []
    for p in raw_projects:
        # Strip common path prefixes used by external repos
        if "/" in p:
            # Take the last component (e.g., "projects/hip" -> "hip")
            name = p.split("/")[-1]
        else:
            name = p
        projects.append(name)

    print(f"Raw changed projects: {raw_projects}")
    print(f"Normalized projects: {projects}")

    topology = get_topology()
    affected = topology.get_stages_for_projects(projects)

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
