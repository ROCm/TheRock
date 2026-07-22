#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute affected build stages from changed projects, expanding TEST_SUBPROJECTS.

For external repo builds (rocm-systems, rocm-libraries), this script determines:
- affected_stages: Stages that need to be built (contain changed projects)
- prebuilt_stages: Stages that can be copied from baseline (unaffected)
- expanded_projects: Projects to build/test (includes TEST_SUBPROJECTS deps)
- baseline_run_id: Workflow run ID to copy prebuilt artifacts from

Baseline selection looks for artifacts in the external repo's own multi-arch CI,
not TheRock. This ensures artifacts match the external repo's source state.
If no baseline is available, the build proceeds without prebuilt artifacts.
"""

import json
import logging
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

from baseline_runs import RequiredArtifact, select_baseline_run
from github_actions_api import gha_set_output, GitHubAPIError

logger = logging.getLogger(__name__)

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


def parse_external_repo_config() -> tuple[str, str] | None:
    """Parse EXTERNAL_REPO_JSON to get repository and branch for baseline lookup.

    Returns (repository, branch) tuple, or None if not configured.
    """
    external_repo_json = os.environ.get("EXTERNAL_REPO_JSON", "")
    if not external_repo_json:
        return None

    try:
        external_repo = json.loads(external_repo_json)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse EXTERNAL_REPO_JSON: %s", exc)
        return None

    repository = external_repo.get("repository", "")
    if not repository:
        logger.warning("EXTERNAL_REPO_JSON missing 'repository' field")
        return None

    # Use the external repo's default branch (usually 'develop' for rocm-* repos)
    # The 'ref' field is the PR ref, not the base branch, so we default to 'develop'
    branch = external_repo.get("base_branch", "develop")

    return (repository, branch)


def select_baseline_for_prebuilt_stages(
    prebuilt_stages: list[str],
    linux_amdgpu_families: list[str],
) -> str | None:
    """Select a baseline workflow run that has artifacts for prebuilt stages.

    For external repos, this finds a healthy baseline run in the external repo's
    own multi-arch CI (not TheRock). This ensures artifacts match the external
    repo's source state. If no baseline is available, returns None and the build
    proceeds without prebuilt artifacts.

    Returns the baseline_run_id, or None if no suitable baseline is found.
    """
    if not prebuilt_stages:
        print("No prebuilt stages, skipping baseline selection")
        return None

    # Get external repo config for baseline lookup
    external_config = parse_external_repo_config()
    if not external_config:
        print("No external repo configured, skipping baseline selection")
        return None

    github_repository, branch = external_config

    topology = get_topology()
    artifacts_by_group = topology.get_artifact_group_to_artifacts()

    # Collect all artifacts needed for prebuilt stages
    required_artifacts: list[RequiredArtifact] = []
    for stage_name in prebuilt_stages:
        stage = topology.build_stages.get(stage_name)
        if stage is None:
            continue
        for group_name in stage.artifact_groups:
            for artifact_name in artifacts_by_group.get(group_name, []):
                # Need artifacts for each GPU family + generic
                for family in linux_amdgpu_families + ["generic"]:
                    req = RequiredArtifact(name=artifact_name, target_family=family)
                    if req not in required_artifacts:
                        required_artifacts.append(req)

    if not required_artifacts:
        print("No required artifacts for prebuilt stages")
        return None

    print(f"Looking for baseline with {len(required_artifacts)} required artifacts")

    # External repos use therock-multi-arch-ci.yml as their workflow name
    workflow_name = "therock-multi-arch-ci.yml"
    max_age_hours_raw = os.environ.get("BASELINE_MAX_AGE_HOURS", "72")
    try:
        max_age_hours = float(max_age_hours_raw)
    except ValueError:
        max_age_hours = 72.0

    print(f"Searching for baseline in {github_repository}/{workflow_name}@{branch}")

    try:
        baseline = select_baseline_run(
            required_artifacts=required_artifacts,
            github_repository=github_repository,
            workflow_name=workflow_name,
            branch=branch,
            platform="linux",  # Primary platform for artifact verification
            max_age_hours=max_age_hours,
        )
    except GitHubAPIError as exc:
        logger.warning("Failed to select baseline run: %s", exc)
        print(f"Baseline lookup failed: {exc} - proceeding without prebuilt artifacts")
        return None

    if baseline is None:
        print("No suitable baseline run found - proceeding without prebuilt artifacts")
        return None

    print(f"Selected baseline run: {baseline.run_id} ({baseline.html_url})")
    return baseline.run_id


def main():
    changed_projects = os.environ.get("CHANGED_PROJECTS", "")
    affected_stages, prebuilt_stages, expanded_projects = compute_affected(
        changed_projects
    )

    # For external repos, select a baseline run to copy prebuilt artifacts from
    baseline_run_id = ""
    if prebuilt_stages:
        prebuilt_list = [s.strip() for s in prebuilt_stages.split(",") if s.strip()]
        # Parse GPU families from environment (comma-separated)
        linux_families_raw = os.environ.get("LINUX_AMDGPU_FAMILIES", "")
        linux_families = [
            f.strip() for f in linux_families_raw.split(",") if f.strip()
        ]
        baseline = select_baseline_for_prebuilt_stages(prebuilt_list, linux_families)
        if baseline:
            baseline_run_id = baseline

    gha_set_output(
        {
            "affected_stages": affected_stages,
            "prebuilt_stages": prebuilt_stages,
            "expanded_projects": expanded_projects,
            "baseline_run_id": baseline_run_id,
        }
    )


if __name__ == "__main__":
    main()
