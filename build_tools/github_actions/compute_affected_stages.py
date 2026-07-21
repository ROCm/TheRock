#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute affected build stages from changed projects, expanding TEST_SUBPROJECTS.

For external repo builds (rocm-systems, rocm-libraries), this script determines:
- affected_stages: Stages that need to be built (contain changed projects)
- prebuilt_stages: Stages that can be copied from baseline (unaffected)
- expanded_projects: Projects to build/test (includes TEST_SUBPROJECTS deps)
"""

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

# All build stages that can be prebuilt/skipped
ALL_STAGES = [
    "compiler-runtime",
    "runtime-tests",
    "wsl-rocdxg",
    "math-libs",
    "comm-libs",
    "storage-libs",
    "debug-tools",
    "dctools-core",
    "profiler-apps",
    "media-libs",
]


def compute_affected(changed_projects: str) -> tuple[str, str, str]:
    """Return (affected_stages, prebuilt_stages, expanded_projects)."""
    if not changed_projects or not changed_projects.strip():
        print("No changed_projects specified, building all stages")
        return "all", "", ""

    raw_projects = [p.strip() for p in changed_projects.split(",") if p.strip()]
    if not raw_projects:
        print("Empty projects list after parsing, building all stages")
        return "all", "", ""

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
        return "all", "", ""

    # Compute prebuilt_stages = ALL_STAGES - affected_stages
    # These are stages that don't need to be built, artifacts copied from baseline
    prebuilt = [s for s in ALL_STAGES if s not in affected]

    affected_str = ",".join(sorted(affected))
    prebuilt_str = ",".join(prebuilt)
    # Space-separated for --projects arg compatibility
    projects_str = " ".join(expanded_list)

    print(f"Affected stages (to build): {affected_str}")
    print(f"Prebuilt stages (copy from baseline): {prebuilt_str}")
    print(f"Expanded projects output: {projects_str}")
    return affected_str, prebuilt_str, projects_str


def main():
    changed_projects = os.environ.get("CHANGED_PROJECTS", "")
    affected_stages, prebuilt_stages, expanded_projects = compute_affected(
        changed_projects
    )
    gha_set_output(
        {
            "affected_stages": affected_stages,
            "prebuilt_stages": prebuilt_stages,
            "expanded_projects": expanded_projects,
        }
    )


if __name__ == "__main__":
    main()
