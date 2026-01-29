#!/usr/bin/env python3
"""
Detects external repository configuration for TheRock CI workflows.

This script determines build configuration settings based on the external repository
being built (rocm-libraries, rocm-systems, etc.). It outputs GitHub Actions variables
that control checkout steps, patches, and build options.

Usage:
    python detect_external_repo_config.py --repository <repository_name>

Examples:
    # Config for rocm-libraries:
    python build_tools/github_actions/detect_external_repo_config.py --repository ROCm/rocm-libraries

    # Config for rocm-systems:
    python build_tools/github_actions/detect_external_repo_config.py --repository rocm-systems

    # Include a workspace path to produce an extra_cmake_options entry:
    python build_tools/github_actions/detect_external_repo_config.py --repository ROCm/rocm-libraries --workspace "$GITHUB_WORKSPACE/source-repo"

Output (GitHub Actions format):
    cmake_source_var=THEROCK_ROCM_LIBRARIES_SOURCE_DIR
    submodule_path=rocm-libraries
    fetch_exclusion=--no-include-rocm-libraries
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

from github_actions_utils import gha_set_output


# Repository configuration map
REPO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "rocm-libraries": {
        "cmake_source_var": "THEROCK_ROCM_LIBRARIES_SOURCE_DIR",
        "submodule_path": "rocm-libraries",
        "fetch_exclusion": "--no-include-rocm-libraries",
    },
    "rocm-systems": {
        "cmake_source_var": "THEROCK_ROCM_SYSTEMS_SOURCE_DIR",
        "submodule_path": "rocm-systems",
        "fetch_exclusion": "--no-include-rocm-systems",
    },
    # Future repos can be added here:
    # "composable_kernel": {...},
    # "rccl": {...},
}


def detect_repo_name(repo_full_name: str) -> str:
    """Returns the repo name from `owner/repo` or `repo`."""
    # Handle both "ROCm/rocm-libraries" and "rocm-libraries" formats
    if "/" in repo_full_name:
        return repo_full_name.split("/")[-1]
    return repo_full_name


def get_repo_config(repo_name: str) -> Dict[str, Any]:
    """Returns config for a known external repo name."""
    if repo_name not in REPO_CONFIGS:
        raise ValueError(
            f"Unknown external repository: {repo_name}\n"
            f"Known repositories: {', '.join(REPO_CONFIGS.keys())}"
        )

    return REPO_CONFIGS[repo_name]


def get_external_repo_path(repo_name: str) -> Path:
    """Determines the path to the external repository checkout.

    This function encapsulates the logic for finding where an external repo
    is checked out in different scenarios (external repo calling TheRock,
    test integration workflows, TheRock CI, etc.).

    Args:
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")

    Returns:
        Path to the external repository root directory

    Raises:
        ValueError: If the external repo path cannot be determined
    """
    from pathlib import Path

    try:
        normalized_name = detect_repo_name(repo_name)
        repo_config = get_repo_config(normalized_name)
    except (ValueError, KeyError):
        raise ValueError(f"Unknown repository: {repo_name}")

    # Priority order for determining external repo location:

    # 1. EXTERNAL_SOURCE_PATH environment variable
    #    Set in test integration workflows where TheRock is main checkout
    external_source_env = os.environ.get("EXTERNAL_SOURCE_PATH")
    if external_source_env:
        base_path = Path(os.environ.get("GITHUB_WORKSPACE", "."))
        repo_path = base_path / external_source_env
        # Validate that the path ends with the repo name we're looking for
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

    # 3. GITHUB_WORKSPACE (when different from cwd)
    if os.environ.get("GITHUB_WORKSPACE"):
        workspace_path = Path(os.environ["GITHUB_WORKSPACE"])
        if workspace_path.exists() and _is_valid_repo_path(workspace_path):
            print(
                f"Found external repo at GITHUB_WORKSPACE: {workspace_path}",
                file=sys.stderr,
            )
            return workspace_path

    # 4. TheRock submodule (TheRock's own CI)
    therock_root = Path(__file__).parent.parent.parent
    submodule_path = therock_root / repo_config.get("submodule_path", repo_name)
    if submodule_path.exists() and _is_valid_repo_path(submodule_path):
        print(f"Found external repo as submodule: {submodule_path}", file=sys.stderr)
        return submodule_path

    raise ValueError(
        f"Could not find external repo '{repo_name}'. Tried:\n"
        f"  - EXTERNAL_SOURCE_PATH: {external_source_env}\n"
        f"  - CWD: {Path.cwd()}\n"
        f"  - GITHUB_WORKSPACE: {os.environ.get('GITHUB_WORKSPACE')}\n"
        f"  - Submodule: {submodule_path}"
    )


def _is_valid_repo_path(path: Path) -> bool:
    """Validate that a path is a git repository with .github/scripts structure.

    Args:
        path: Path to check

    Returns:
        True if path appears to be a valid external repo checkout
    """
    # Check for git repository
    if not (path / ".git").exists():
        return False

    # Check for .github/scripts directory (external repo structure)
    if not (path / ".github" / "scripts").exists():
        return False

    return True


def import_external_repo_module(
    repo_name: str, module_name: str, repo_path: Path = None
):
    """Dynamically import a module from an external repo's .github/scripts directory.

    Args:
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")
        module_name: Module name without .py extension (e.g., "therock_matrix")
        repo_path: Optional path to the external repo. If not provided,
                  calls get_external_repo_path() to determine it.

    Returns:
        The imported module, or None if import fails
    """
    import importlib.util
    from pathlib import Path

    # Determine repo path if not provided
    if repo_path is None:
        try:
            repo_path = get_external_repo_path(repo_name)
        except ValueError as e:
            print(f"WARNING: {e}", file=sys.stderr)
            return None

    # All external repos follow the same convention: .github/scripts/
    script_path = repo_path / ".github" / "scripts" / f"{module_name}.py"

    if not script_path.exists():
        print(
            f"WARNING: Could not find {module_name}.py at {script_path}",
            file=sys.stderr,
        )
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
    except Exception as e:
        print(
            f"WARNING: Failed to import {module_name} from {repo_name}: {e}",
            file=sys.stderr,
        )
        return None

    return None


def get_skip_patterns(repo_name: str, repo_path: Path = None) -> list:
    """Get skip patterns from external repo's therock_configure_ci.py.

    Args:
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")
        repo_path: Optional path to the external repo. If not provided,
                  will be determined automatically.

    Returns:
        List of skip patterns, or empty list if not found
    """
    configure_module = import_external_repo_module(
        repo_name, "therock_configure_ci", repo_path
    )
    if configure_module and hasattr(configure_module, "SKIPPABLE_PATH_PATTERNS"):
        patterns = configure_module.SKIPPABLE_PATH_PATTERNS
        print(
            f"Loaded {len(patterns)} skip patterns from {repo_name}",
            file=sys.stderr,
        )
        return patterns
    return []


def get_test_list(repo_name: str, repo_path: Path = None) -> list:
    """Get test list from external repo's therock_matrix.py project_map.

    Args:
        repo_name: Repository name (e.g., "rocm-libraries", "rocm-systems")
        repo_path: Optional path to the external repo. If not provided,
                  will be determined automatically.

    Returns:
        List of test names, or empty list if not found
    """
    matrix_module = import_external_repo_module(repo_name, "therock_matrix", repo_path)
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
        Uses gha_set_output() from github_actions_utils.py which handles
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect external repository configuration for TheRock CI workflows.\n\n"
            "This script determines build configuration settings based on the external\n"
            "repository being built (rocm-libraries, rocm-systems, etc.). It outputs\n"
            "GitHub Actions variables that control checkout steps, patches, and build options.\n\n"
            "Output Format (GitHub Actions):\n"
            "  cmake_source_var=THEROCK_ROCM_LIBRARIES_SOURCE_DIR\n"
            "  submodule_path=rocm-libraries\n"
            "  fetch_exclusion=--no-include-rocm-libraries"
        ),
        epilog=(
            "Examples:\n"
            "  # Config for rocm-libraries:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            "    --repository ROCm/rocm-libraries\n\n"
            "  # Config for rocm-systems:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            "    --repository rocm-systems\n\n"
            "  # Include workspace path for CMake options:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            '    --repository ROCm/rocm-libraries --workspace "$GITHUB_WORKSPACE/source-repo"\n\n'
            "  # List all known repositories:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Full repository name (e.g., ROCm/rocm-libraries) or short name (e.g., rocm-libraries). Defaults to $GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="GitHub workspace path for formatting CMake options",
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["linux", "windows"],
        default="linux" if os.name == "posix" else "windows",
        help="Platform for platform-specific configuration. Defaults to current platform.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all known repository configurations",
    )

    args = parser.parse_args()

    if args.list:
        print("Known external repositories:")
        for repo_name in REPO_CONFIGS.keys():
            print(f"  - {repo_name}")
        return 0

    try:
        repo_name = detect_repo_name(args.repository)
        config = get_repo_config(repo_name)

        # Log to stderr for visibility in CI logs
        print(f"Detected repository: {repo_name}", file=sys.stderr)
        print(f"Configuration: {config}", file=sys.stderr)

        # Format the full CMake option if workspace path provided
        if args.workspace:
            cmake_var = config["cmake_source_var"]
            config["extra_cmake_options"] = f"-D{cmake_var}={args.workspace}"
            print(
                f"Generated CMake option: {config['extra_cmake_options']}",
                file=sys.stderr,
            )

        output_github_actions_vars(config)
        return 0

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
