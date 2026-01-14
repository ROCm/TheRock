#!/usr/bin/env python3
"""
Detects external repository configuration for TheRock CI workflows.

This script determines build configuration settings based on the external repository
being built (rocm-libraries, rocm-systems, etc.). It outputs GitHub Actions variables
that control checkout steps, patches, and build options.

Usage:
    python detect_external_repo_config.py --repository <repository_name>

Examples:
    # Linux config for rocm-libraries:
    python build_tools/github_actions/detect_external_repo_config.py --repository ROCm/rocm-libraries --platform linux

    # Windows config for rocm-systems:
    python build_tools/github_actions/detect_external_repo_config.py --repository rocm-systems --platform windows

    # Include a workspace path to produce an extra_cmake_options entry:
    python build_tools/github_actions/detect_external_repo_config.py --repository ROCm/rocm-libraries --workspace "$GITHUB_WORKSPACE/source-repo" --platform linux

Output (GitHub Actions format):
    cmake_source_var=THEROCK_ROCM_LIBRARIES_SOURCE_DIR
    patches_dir=rocm-libraries
    fetch_exclusion=--no-include-rocm-libraries
    enable_dvc=true
    enable_ck=true
"""

import argparse
import os
import sys
from typing import Dict, Any


# Repository configuration map
REPO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "rocm-libraries": {
        "cmake_source_var": "THEROCK_ROCM_LIBRARIES_SOURCE_DIR",
        "patches_dir": "rocm-libraries",
        "fetch_exclusion": "--no-include-rocm-libraries --no-include-ml-frameworks",
        # DVC is required on both platforms for rocm-libraries
        "enable_dvc": {
            "linux": True,
            "windows": True,
        },
    },
    "rocm-systems": {
        "cmake_source_var": "THEROCK_ROCM_SYSTEMS_SOURCE_DIR",
        "patches_dir": "rocm-systems",
        "fetch_exclusion": "--no-include-rocm-systems --no-include-rocm-libraries --no-include-ml-frameworks",
        # DVC is required on Windows but not Linux for rocm-systems
        "enable_dvc": {
            "linux": False,
            "windows": True,
        },
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


def output_github_actions_vars(config: Dict[str, Any]) -> None:
    """Writes config as GitHub Actions outputs (to `GITHUB_OUTPUT` or stdout).

    Args:
        config: Configuration dictionary with keys like 'cmake_source_var',
            'patches_dir', etc. Values should be strings or booleans (already
            resolved for platform-specific values).

    Returns:
        None. Outputs are written as side effects:
        - If GITHUB_OUTPUT env var is set: Appends to that file
        - Otherwise: Prints to stdout for local testing

    Note:
        Boolean values are converted to lowercase strings ('true'/'false')
        for bash compatibility. Platform-specific values should be resolved
        by the caller before passing to this function.
    """
    github_output = os.environ.get("GITHUB_OUTPUT")

    # Convert boolean values to lowercase strings for bash compatibility
    output_lines = []
    for key, value in config.items():
        value_str = str(value).lower() if isinstance(value, bool) else str(value)
        output_lines.append(f"{key}={value_str}")

    # Write to GITHUB_OUTPUT file if available, otherwise print to stdout
    if github_output:
        with open(github_output, "a") as f:
            f.write("\n".join(output_lines) + "\n")
    else:
        # Fallback for local testing
        print("\n".join(output_lines))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect external repository configuration for TheRock CI workflows.\n\n"
            "This script determines build configuration settings based on the external\n"
            "repository being built (rocm-libraries, rocm-systems, etc.). It outputs\n"
            "GitHub Actions variables that control checkout steps, patches, and build options.\n\n"
            "Output Format (GitHub Actions):\n"
            "  cmake_source_var=THEROCK_ROCM_LIBRARIES_SOURCE_DIR\n"
            "  patches_dir=rocm-libraries\n"
            "  fetch_exclusion=--no-include-rocm-libraries\n"
            "  enable_dvc=true"
        ),
        epilog=(
            "Examples:\n"
            "  # Linux config for rocm-libraries:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            "    --repository ROCm/rocm-libraries --platform linux\n\n"
            "  # Windows config for rocm-systems:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            "    --repository rocm-systems --platform windows\n\n"
            "  # Include workspace path for CMake options:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py \\\n"
            '    --repository ROCm/rocm-libraries --workspace "$GITHUB_WORKSPACE/source-repo" \\\n'
            "    --platform linux\n\n"
            "  # List all known repositories:\n"
            "  python build_tools/github_actions/detect_external_repo_config.py --list"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repository",
        required=True,
        help="Full repository name (e.g., ROCm/rocm-libraries) or short name (e.g., rocm-libraries)",
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
        help="Platform for platform-specific configuration (linux or windows)",
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

        # Some config values are platform-specific (dict keyed by linux/windows).
        # Require --platform in that case so downstream output formatting is simple.
        if any(isinstance(v, dict) for v in config.values()) and not args.platform:
            raise ValueError("--platform is required for this repository configuration")

        # Log to stderr for visibility in CI logs
        print(f"Detected repository: {repo_name}", file=sys.stderr)
        print(f"Platform: {args.platform or 'not specified'}", file=sys.stderr)
        print(f"Configuration: {config}", file=sys.stderr)

        # Resolve platform-specific values
        resolved_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                resolved_config[key] = value[args.platform]
            else:
                resolved_config[key] = value

        # Format the full CMake option if workspace path provided
        if args.workspace:
            cmake_var = resolved_config["cmake_source_var"]
            resolved_config["extra_cmake_options"] = f"-D{cmake_var}={args.workspace}"
            print(
                f"Generated CMake option: {resolved_config['extra_cmake_options']}",
                file=sys.stderr,
            )

        output_github_actions_vars(resolved_config)
        return 0

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
