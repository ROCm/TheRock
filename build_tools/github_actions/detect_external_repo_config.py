#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Detects external repository configuration for TheRock CI workflows.

This script determines build configuration settings based on the external repository
being built (rocm-libraries, rocm-systems, etc.). It outputs GitHub Actions variables
that control checkout steps, patches, and build options.

Usage:
    python detect_external_repo_config.py --repository <repository_name>

Examples:
    # Config for rocm-libraries:
    python build_tools/github_actions/detect_external_repo_config.py --repository rocm-libraries

    # Config for rocm-systems:
    python build_tools/github_actions/detect_external_repo_config.py --repository rocm-systems

    # Include a workspace path to produce an extra_cmake_options entry:
    python build_tools/github_actions/detect_external_repo_config.py --repository rocm-libraries --workspace "$GITHUB_WORKSPACE/source-repo"

Output (GitHub Actions format):
    cmake_source_var=THEROCK_ROCM_LIBRARIES_SOURCE_DIR
    submodule_path=rocm-libraries
    fetch_sources_args=--skip-submodules rocm-libraries
"""

import argparse
import importlib.util
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from _therock_utils.build_topology import get_topology
from github_actions_api import gha_set_output


def get_stage_source_paths(stage_name: str) -> Set[str]:
    """Get all source_paths that map to artifacts built in this stage."""
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
                source_paths.add(artifact.name)

    return source_paths


def extract_source_path_from_project(project_path: str) -> Optional[str]:
    """Extract the last component from a project path (e.g., "projects/rocprim" -> "rocprim")."""
    parts = project_path.strip().split("/")
    if len(parts) >= 2:
        return parts[-1]
    return project_path.strip() if project_path.strip() else None


def get_artifact_for_source_path_in_stage(source_path: str, stage_name: str) -> Optional[str]:
    """Get the artifact name that contains the given source_path in a stage.

    Args:
        source_path: Source path name (e.g., "rocprim")
        stage_name: Build stage name (e.g., "math-libs")

    Returns:
        Artifact name if found, None otherwise
    """
    topology = get_topology()

    stage = topology.build_stages.get(stage_name)
    if not stage:
        return None

    for group_name in stage.artifact_groups:
        for artifact in topology.get_artifacts_in_group(group_name):
            artifact_sources = artifact.source_paths or [artifact.name]
            if source_path in artifact_sources:
                return artifact.name

    return None


def get_artifact_sibling_paths(source_path: str, stage_name: str) -> Set[str]:
    """Get all source_paths from artifacts that contain the given source_path.

    When a project is changed, we need to checkout all sibling projects that are
    built together in the same artifact. For example, if rocprim changes, we also
    need hipcub, rocthrust, primbench (all in the 'prim' artifact).
    """
    topology = get_topology()
    sibling_paths: Set[str] = set()

    stage = topology.build_stages.get(stage_name)
    if not stage:
        return sibling_paths

    for group_name in stage.artifact_groups:
        for artifact in topology.get_artifacts_in_group(group_name):
            artifact_sources = artifact.source_paths or [artifact.name]
            if source_path in artifact_sources:
                # Found the artifact containing this source_path, add all its siblings
                sibling_paths.update(artifact_sources)

    return sibling_paths


def get_source_sets_for_stage(stage_name: str) -> Set[str]:
    """Get all source_set names used by a stage."""
    topology = get_topology()
    source_set_names: Set[str] = set()

    stage = topology.build_stages.get(stage_name)
    if not stage:
        return source_set_names

    for group_name in stage.artifact_groups:
        group = topology.artifact_groups.get(group_name)
        if group:
            source_set_names.update(group.source_sets)

    return source_set_names


def get_artifact_build_dependency_paths(
    artifact_name: str, stage_name: str, stage_source_sets: Set[str]
) -> Set[str]:
    """Get source_paths from build dependencies of an artifact.

    This follows artifact_deps transitively and returns source_paths only for
    artifacts that belong to the same source_sets as the stage (i.e., artifacts
    within the same external repo like rocm-libraries).

    Args:
        artifact_name: Name of the artifact to get dependencies for
        stage_name: Build stage name
        stage_source_sets: Source sets used by this stage (to filter deps)

    Returns:
        Set of source_path names from dependency artifacts in the same source_sets
    """
    topology = get_topology()
    dependency_paths: Set[str] = set()

    artifact = topology.artifacts.get(artifact_name)
    if not artifact:
        return dependency_paths

    # Collect all transitive artifact dependencies
    all_deps: Set[str] = set()
    deps_to_process = list(artifact.artifact_deps)

    while deps_to_process:
        dep_name = deps_to_process.pop()
        if dep_name in all_deps:
            continue
        all_deps.add(dep_name)

        dep_artifact = topology.artifacts.get(dep_name)
        if dep_artifact:
            deps_to_process.extend(dep_artifact.artifact_deps)

    # For each dependency, check if it's in the same source_sets as the stage
    # and if so, include its source_paths
    for dep_name in all_deps:
        dep_artifact = topology.artifacts.get(dep_name)
        if not dep_artifact:
            continue

        # Check if this artifact's group uses any of the stage's source_sets
        dep_group = topology.artifact_groups.get(dep_artifact.artifact_group)
        if not dep_group:
            continue

        # If the dependency's artifact_group uses any of the same source_sets,
        # then its source_paths are in the same external repo
        if stage_source_sets.intersection(dep_group.source_sets):
            dep_sources = dep_artifact.source_paths or [dep_artifact.name]
            dependency_paths.update(dep_sources)

    return dependency_paths


def compute_stage_sparse_checkout(stage_name: str, changed_projects: str) -> List[str]:
    """Return project paths to sparse checkout for a stage, or empty list for full checkout.

    When a project changes, this function expands to include:
    1. All sibling projects from the same artifact (e.g., rocprim -> hipcub, rocthrust)
    2. All source_paths from transitive build dependencies that are in the same
       external repo (determined by shared source_sets)

    For example, if projects/rocprim changes:
    - rocprim is in the 'prim' artifact with source_paths [rocprim, hipcub, rocthrust]
    - 'prim' has artifact_deps including 'rand' (source_paths: [rocrand, hiprand])
    - Both are in the 'rocm-libraries' source_set
    - So we checkout: rocprim, hipcub, rocthrust, rocrand, hiprand

    This ensures the build has all necessary source code from the external repo.
    """
    if not changed_projects or not changed_projects.strip():
        return []

    stage_source_paths = get_stage_source_paths(stage_name)
    if not stage_source_paths:
        return []

    # Get source_sets for this stage (used to filter dependencies to same repo)
    stage_source_sets = get_source_sets_for_stage(stage_name)

    # First, find which source_paths in this stage are affected
    affected_source_paths: Set[str] = set()
    project_prefixes: Dict[str, str] = {}  # source_path -> project prefix (e.g., "rocprim" -> "projects/")

    for project in changed_projects.split(","):
        project = project.strip()
        if not project:
            continue

        source_path = extract_source_path_from_project(project)
        if source_path and source_path in stage_source_paths:
            affected_source_paths.add(source_path)
            # Remember the prefix (e.g., "projects/" from "projects/rocprim")
            prefix = project[: len(project) - len(source_path)]
            project_prefixes[source_path] = prefix

    if not affected_source_paths:
        return []

    # Expand to include all sibling source_paths from affected artifacts
    # AND their transitive build dependencies within the same source_sets
    all_needed_paths: Set[str] = set()
    affected_artifacts: Set[str] = set()

    for source_path in affected_source_paths:
        # Get sibling paths from the same artifact
        siblings = get_artifact_sibling_paths(source_path, stage_name)
        all_needed_paths.update(siblings)

        # Track which artifacts are affected so we can get their dependencies
        artifact_name = get_artifact_for_source_path_in_stage(source_path, stage_name)
        if artifact_name:
            affected_artifacts.add(artifact_name)

    # Get transitive build dependency paths for all affected artifacts
    for artifact_name in affected_artifacts:
        dep_paths = get_artifact_build_dependency_paths(
            artifact_name, stage_name, stage_source_sets
        )
        all_needed_paths.update(dep_paths)

    # Filter to only include paths that exist in this stage's source_paths
    # (dependencies from other stages/repos are handled by artifact fetching)
    all_needed_paths = all_needed_paths.intersection(stage_source_paths)

    # Convert source_paths back to project paths using the prefix
    # Use the first known prefix for paths we haven't seen directly
    default_prefix = next(iter(project_prefixes.values()), "projects/")
    result_paths: List[str] = []
    for sp in all_needed_paths:
        prefix = project_prefixes.get(sp, default_prefix)
        result_paths.append(f"{prefix}{sp}")

    return sorted(result_paths)


def compute_all_stage_sparse_checkouts(changed_projects: str) -> Dict[str, str]:
    """Pre-compute sparse checkout paths for all stages (stage_name -> newline-separated paths)."""
    if not changed_projects or not changed_projects.strip():
        return {}

    topology = get_topology()
    result: Dict[str, str] = {}

    for stage_name in topology.build_stages:
        paths = compute_stage_sparse_checkout(stage_name, changed_projects)
        # Convert list to newline-separated string for actions/checkout sparse-checkout
        result[stage_name] = "\n".join(paths)

    return result


# Repository configuration map
REPO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "rocm-libraries": {
        "cmake_source_var": "THEROCK_ROCM_LIBRARIES_SOURCE_DIR",
        "submodule_path": "rocm-libraries",
        "skip_submodules": ["rocm-libraries"],
    },
    "rocm-systems": {
        "cmake_source_var": "THEROCK_ROCM_SYSTEMS_SOURCE_DIR",
        "submodule_path": "rocm-systems",
        "skip_submodules": ["rocm-systems"],
        "dvc_projects": ["external-rocm-systems"],
    },
    "rocgdb": {
        "cmake_source_var": "THEROCK_ROCGDB_SOURCE_DIR",
        "submodule_path": "debug-tools/rocgdb/source",
        "skip_submodules": ["rocgdb"],
    },
    # Future repos can be added here:
    # "llvm-project": {...},
    # "hipify": {...},
    # "libhipcxx": {...},
}


def _log_warning(message: str) -> None:
    """Helper to log warning messages to stderr."""
    print(f"WARNING: {message}", file=sys.stderr)


def normalize_changed_projects(changed_projects: str) -> str:
    """Normalize changed_projects into a consistent format.

    Args:
        changed_projects: Comma-separated list of changed project paths
            (e.g., "projects/rocprim,shared/rocroller")

    Returns:
        Comma-separated string of paths (normalized, deduplicated, sorted).
        Empty string if changed_projects is empty.

    Note: The sparse checkout paths are pre-computed for all stages using
    compute_all_stage_sparse_checkouts() and embedded in config_json.
    """
    if not changed_projects or not changed_projects.strip():
        return ""

    # Parse changed_projects into a set - just use the paths directly
    paths = set()
    for project in changed_projects.split(","):
        project = project.strip()
        if project:
            paths.add(project)

    if not paths:
        return ""

    # Return as comma-separated string (will be parsed per-stage)
    return ",".join(sorted(paths))


def get_repo_config(repo_name: str) -> Dict[str, Any]:
    """Returns config for a known external repo name."""
    if repo_name not in REPO_CONFIGS:
        raise ValueError(
            f"Unknown external repository: {repo_name}\n"
            f"Known repositories: {', '.join(REPO_CONFIGS.keys())}"
        )

    return REPO_CONFIGS[repo_name]


@lru_cache()
def get_external_repo_path(repo_name: str) -> Path:
    """Determines the path to the external repository checkout.

    This function is cached (using functools.lru_cache) to avoid repeated
    filesystem lookups. The path for a given repo_name is computed once
    and reused for all subsequent calls during program execution.

    This function encapsulates the logic for finding where an external repo
    is checked out in different scenarios (external repo calling TheRock,
    test integration workflows, TheRock CI, etc.).

    Args:
        repo_name (str): Repository name (e.g., "rocm-libraries", "rocm-systems")

    Returns:
        Path: Path to the external repository root directory (cached after first call)

    Raises:
        ValueError: If the external repo path cannot be determined
    """
    try:
        repo_config = get_repo_config(repo_name)
    except ValueError as e:
        raise ValueError(f"Unknown repository: {repo_name}") from e

    # Priority order for determining external repo location:

    # 1. EXTERNAL_SOURCE_PATH environment variable
    #    Set in test integration workflows where TheRock is main checkout
    external_source_env = os.environ.get("EXTERNAL_SOURCE_PATH")
    if external_source_env:
        workspace_env = os.environ.get("GITHUB_WORKSPACE")
        if workspace_env:
            base_path = Path(workspace_env)
        else:
            base_path = Path.cwd()
            _log_warning(
                "EXTERNAL_SOURCE_PATH set but GITHUB_WORKSPACE not set, using CWD as base"
            )
        repo_path = base_path / external_source_env
        # Validate that the path ends with the expected repo name
        if (
            repo_path.exists()
            and _is_valid_repo_path(repo_path)
            and repo_path.name == repo_name
        ):
            print(
                f"Found external repo via EXTERNAL_SOURCE_PATH: {repo_path}",
                file=sys.stderr,
            )
            return repo_path

    # 2. Current directory (external repo calling TheRock CI)
    #    Most common case when external repos use TheRock workflows
    if _is_valid_repo_path(Path.cwd()):
        print(f"Found external repo at CWD: {Path.cwd()}", file=sys.stderr)
        return Path.cwd()

    raise ValueError(
        f"Could not find external repo '{repo_name}'. Checked:\n"
        f"  - EXTERNAL_SOURCE_PATH: {external_source_env or 'not set'}\n"
        f"  - CWD: {Path.cwd()}"
    )


def _is_valid_repo_path(path: Path) -> bool:
    """Validate that a path is a git repository with required TheRock integration scripts.

    External repositories that integrate with TheRock must have:
    - A .github/scripts/ directory
    - therock_matrix.py: Defines project build matrix and test lists
    - therock_configure_ci.py: CI configuration including skippable path patterns

    Args:
        path: Path to check

    Returns:
        True if path is a valid external repo with all required integration scripts
    """
    # Check for git repository (can be file or directory for worktrees)
    git_path = path / ".git"
    if not git_path.exists():
        return False

    # Check for .github/scripts directory (external repo structure)
    scripts_dir = path / ".github" / "scripts"
    if not scripts_dir.exists() or not scripts_dir.is_dir():
        return False

    # Check for required TheRock integration scripts
    required_scripts = ["therock_matrix.py", "therock_configure_ci.py"]
    for script_name in required_scripts:
        script_path = scripts_dir / script_name
        if not script_path.exists():
            return False

    return True


def import_external_repo_module(repo_name: str, module_name: str) -> Optional[Any]:
    """Dynamically import a module from an external repo's .github/scripts directory.

    Args:
        repo_name (str): Repository name (e.g., "rocm-libraries", "rocm-systems")
        module_name (str): Module name without .py extension (e.g., "therock_matrix")

    Returns:
        Optional[Any]: The imported module, or None if import fails

    Raises:
        ValueError: If the external repo path cannot be determined
    """
    # Get the validated repo path (will raise ValueError if invalid)
    repo_path = get_external_repo_path(repo_name)

    # All external repos follow the same convention: .github/scripts/
    script_path = repo_path / ".github" / "scripts" / f"{module_name}.py"

    if not script_path.exists():
        _log_warning(f"Could not find {module_name}.py at {script_path}")
        return None

    print(f"Importing {module_name} from: {script_path}", file=sys.stderr)

    try:
        spec = importlib.util.spec_from_file_location(
            f"{repo_name}.{module_name}", script_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            _log_warning(
                f"Could not create module spec for {module_name} at {script_path}"
            )
            return None
    except ImportError as e:
        _log_warning(f"Failed to import {module_name} from {repo_name}: {e}")
        return None
    except Exception as e:
        print(
            f"ERROR: Unexpected error importing {module_name} from {repo_name}: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        raise


def get_skip_patterns(repo_name: str) -> list[str]:
    """Get skip patterns from external repo's therock_configure_ci.py.

    These are file path patterns that, when ALL modified files in a PR match at least
    one pattern, indicate the changes have no impact on CI workflows. When this occurs,
    TheRock CI will skip build/test jobs to save resources.

    Example patterns: ["*.md", "docs/*", ".github/workflows/*"]

    See: https://github.com/ROCm/rocm-libraries/blob/develop/.github/scripts/therock_configure_ci.py

    Args:
        repo_name (str): Repository name (e.g., "rocm-libraries", "rocm-systems")

    Returns:
        list[str]: List of skip patterns, or empty list if not found
    """
    configure_module = import_external_repo_module(repo_name, "therock_configure_ci")
    if configure_module and hasattr(configure_module, "SKIPPABLE_PATH_PATTERNS"):
        patterns = configure_module.SKIPPABLE_PATH_PATTERNS
        print(
            f"Loaded {len(patterns)} skip patterns from {repo_name}",
            file=sys.stderr,
        )
        return patterns
    return []


def get_test_list(repo_name: str) -> list[str]:
    """Get test list from external repo's therock_matrix.py project_map.

    Args:
        repo_name (str): Repository name (e.g., "rocm-libraries", "rocm-systems")

    Returns:
        list[str]: List of test names, or empty list if not found
    """
    matrix_module = import_external_repo_module(repo_name, "therock_matrix")
    if not matrix_module or not hasattr(matrix_module, "project_map"):
        return []

    # Collect all unique tests from all projects
    # NOTE: We ignore their cmake_options since we're doing full builds
    all_tests = set()
    project_map = matrix_module.project_map

    for project_config in project_map.values():
        tests = project_config.get("project_to_test", [])
        # Handle both list and comma-separated string formats
        if isinstance(tests, str):
            tests = [t.strip() for t in tests.split(",")]
        all_tests.update(tests)

    if all_tests:
        test_list = sorted(all_tests)
        print(f"Loaded {len(test_list)} tests from {repo_name}", file=sys.stderr)
        return test_list

    return []


def output_github_actions_vars(config: Dict[str, Any]) -> None:
    """Writes config as GitHub Actions outputs using the standard utility.

    Args:
        config: Configuration dictionary with keys like 'cmake_source_var',
            'submodule_path', etc. Values should be strings or booleans.

    Note:
        Uses gha_set_output() from github_actions_api.py which handles
        writing to GITHUB_OUTPUT file or stdout for local testing.
        Booleans are converted to lowercase strings for bash compatibility.
    """
    # Convert booleans to lowercase strings for bash compatibility
    normalized_config = {}
    for key, value in config.items():
        if isinstance(value, bool):
            normalized_config[key] = str(value).lower()
        else:
            normalized_config[key] = str(value)

    gha_set_output(normalized_config)


def main(argv=None):
    """Main entry point for the script.

    Args:
        argv: Command line arguments (defaults to sys.argv if None)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Detect external repository configuration for TheRock CI workflows.\n\n"
            "This script determines build configuration settings based on the external\n"
            "repository being built (rocm-libraries, rocm-systems, etc.). It outputs\n"
            "GitHub Actions variables that control checkout steps, patches, and build options.\n\n"
            "Output Format (GitHub Actions):\n"
            "  cmake_source_var=THEROCK_ROCM_LIBRARIES_SOURCE_DIR\n"
            "  submodule_path=rocm-libraries\n"
            "  fetch_sources_args=--skip-submodules rocm-libraries"
        ),
        epilog=(
            "Examples:\n"
            "  # Config for rocm-libraries:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            "    --repository rocm-libraries\n\n"
            "  # Config for rocm-systems:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            "    --repository rocm-systems\n\n"
            "  # Include workspace path for CMake options:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            '    --repository rocm-libraries --workspace "$GITHUB_WORKSPACE/source-repo"\n\n'
            "  # List all known repositories:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--external-repo-json",
        type=str,
        help="JSON object with 'repository' (e.g., 'ROCm/rocm-libraries') and 'ref' fields.",
    )
    parser.add_argument(
        "--repository",
        help="Repository name (e.g., rocm-libraries, rocm-systems). "
        "Alternative to --external-repo-json for direct invocation.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all known repository configurations",
    )
    parser.add_argument(
        "--changed-projects",
        type=str,
        default="",
        help="Comma-separated list of changed project paths (e.g., 'projects/rocprim,shared/rocroller'). "
        "Used to compute sparse checkout paths for the external repo.",
    )

    args = parser.parse_args(argv)

    if args.list:
        print("Known external repositories:")
        for repo_name in REPO_CONFIGS.keys():
            print(f"  - {repo_name}")
        return 0

    # Parse external repo JSON if provided, extract repository name
    source_repository = None
    source_ref = ""
    if args.external_repo_json:
        try:
            external_repo = json.loads(args.external_repo_json)
            source_repository = external_repo.get("repository", "")
            source_ref = external_repo.get("ref", "")
            # Extract repo name from full name (e.g., "rocm-libraries" from "ROCm/rocm-libraries").
            # Lowercase so REPO_CONFIGS lookup is case-insensitive (e.g. "ROCgdb" -> "rocgdb").
            if "/" in source_repository:
                repo_name = source_repository.split("/")[-1].lower()
            else:
                repo_name = source_repository.lower()
            args.repository = repo_name
            print(
                f"Parsed external_repo: repository={source_repository}, ref={source_ref}",
                file=sys.stderr,
            )
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --external-repo-json: {e}", file=sys.stderr)
            return 1

    if not args.repository:
        print(
            "ERROR: --repository or --external-repo-json is required",
            file=sys.stderr,
        )
        return 1

    try:
        config = get_repo_config(args.repository)

        # Log to stderr for visibility in CI logs
        print(f"Detected repository: {args.repository}", file=sys.stderr)
        print(f"Configuration: {config}", file=sys.stderr)

        # checkout_path is relative (for actions/checkout path: parameter)
        # Use "external-" prefix to avoid collisions with submodule paths
        checkout_path = f"external-{args.repository}"

        # Generate fetch_sources_args from skip_submodules and dvc_projects
        fetch_args_parts = []
        if "skip_submodules" in config and config["skip_submodules"]:
            skip_args = " ".join(config["skip_submodules"])
            fetch_args_parts.append(f"--skip-submodules {skip_args}")
        if "dvc_projects" in config and config["dvc_projects"]:
            dvc_args = " ".join(config["dvc_projects"])
            fetch_args_parts.append(f"--dvc-projects {dvc_args}")

        if fetch_args_parts:
            config["fetch_sources_args"] = " ".join(fetch_args_parts)
            print(
                f"Generated fetch_sources_args: {config['fetch_sources_args']}",
                file=sys.stderr,
            )

        # Build config_json with all fields needed by workflows
        final_source_repo = source_repository or f"ROCm/{args.repository}"
        source_package = (
            config["cmake_source_var"]
            .replace("THEROCK_", "")
            .replace("_SOURCE_DIR", "")
        )

        # Extract caller-supplied fields from external_repo JSON if provided
        projects = ""
        family_overrides = {}
        extra_cmake_options = ""

        if args.external_repo_json:
            try:
                external_repo = json.loads(args.external_repo_json)
                projects = external_repo.get("projects", "")

                # family_overrides allows external repos to specify per-family config
                # (e.g., test runners, test_labels_for_family) for their CI runs only
                family_overrides = external_repo.get("family_overrides", {})

                extra_cmake_options = external_repo.get("extra_cmake_options", "")
            except json.JSONDecodeError as e:
                print(
                    f"Warning: failed to parse external_repo_json: {e}",
                    file=sys.stderr,
                )

        # Normalize changed_projects for sparse checkout computation
        changed_projects_normalized = normalize_changed_projects(args.changed_projects)
        print(
            f"Changed projects (normalized): {changed_projects_normalized}",
            file=sys.stderr,
        )

        # Pre-compute sparse checkout paths for all stages
        # This enables per-stage sparse checkout without calling a separate script
        sparse_checkout_by_stage = compute_all_stage_sparse_checkouts(
            changed_projects_normalized
        )
        if sparse_checkout_by_stage:
            affected_stages = [s for s, p in sparse_checkout_by_stage.items() if p]
            print(
                f"Sparse checkout computed for {len(sparse_checkout_by_stage)} stages, "
                f"{len(affected_stages)} affected: {affected_stages}",
                file=sys.stderr,
            )

        config_json = {
            "repository": final_source_repo,
            "ref": source_ref,
            "checkout_path": checkout_path,
            "source_package": source_package,
            "fetch_sources_args": config.get("fetch_sources_args", ""),
            "extra_cmake_options": extra_cmake_options,
            "projects": projects,
            "family_overrides": family_overrides,
            "changed_projects": changed_projects_normalized,
            "sparse_checkout_by_stage": sparse_checkout_by_stage,
        }
        config["config_json"] = json.dumps(config_json)
        print(
            f"Generated config_json:\n{json.dumps(config_json, indent=2)}",
            file=sys.stderr,
        )

        output_github_actions_vars(config)
        return 0

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
