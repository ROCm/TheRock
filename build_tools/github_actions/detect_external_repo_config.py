#!/usr/bin/env python3
"""
Detects external repository configuration for TheRock CI workflows.

This script determines build configuration settings based on the external repository
being built (rocm-libraries, rocm-systems, etc.). It outputs GitHub Actions variables
that control checkout steps, patches, and build options.

Usage:
    python detect_external_repo_config.py <repository_name>

Example:
    python detect_external_repo_config.py ROCm/rocm-libraries

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
    """
    Extract the repository name from a full GitHub repository identifier.

    Args:
        repo_full_name: Full repository name (e.g., "ROCm/rocm-libraries")

    Returns:
        Repository name (e.g., "rocm-libraries")
    """
    # Handle both "ROCm/rocm-libraries" and "rocm-libraries" formats
    if "/" in repo_full_name:
        return repo_full_name.split("/")[-1]
    return repo_full_name


def get_repo_config(repo_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific repository.

    Args:
        repo_name: Repository name (e.g., "rocm-libraries")

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If repository is not recognized
    """
    if repo_name not in REPO_CONFIGS:
        raise ValueError(
            f"Unknown external repository: {repo_name}\n"
            f"Known repositories: {', '.join(REPO_CONFIGS.keys())}"
        )

    return REPO_CONFIGS[repo_name]


def output_github_actions_vars(config: Dict[str, Any], platform: str = None) -> None:
    """
    Output configuration as GitHub Actions environment variables.

    Args:
        config: Configuration dictionary
        platform: Platform name ('linux' or 'windows') for platform-specific values
    """
    github_output = os.environ.get("GITHUB_OUTPUT")

    # Convert boolean values to lowercase strings for bash compatibility
    output_lines = []
    for key, value in config.items():
        # Handle platform-specific values (dict with 'linux'/'windows' keys)
        if isinstance(value, dict) and platform and platform in value:
            value = value[platform]

        if isinstance(value, bool):
            value_str = str(value).lower()
        elif isinstance(value, dict):
            # If still a dict after platform resolution, skip or use a default
            print(
                f"WARNING: {key} has platform-specific config but no platform specified",
                file=sys.stderr,
            )
            value_str = "false"  # Default to false for safety
        else:
            value_str = str(value)
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
        description="Detect external repository configuration for TheRock CI"
    )
    parser.add_argument(
        "repository",
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

        # Log to stderr for visibility in CI logs
        print(f"Detected repository: {repo_name}", file=sys.stderr)
        print(f"Platform: {args.platform or 'not specified'}", file=sys.stderr)
        print(f"Configuration: {config}", file=sys.stderr)

        # Format the full CMake option if workspace path provided
        if args.workspace:
            cmake_var = config["cmake_source_var"]
            config["extra_cmake_options"] = f"-D{cmake_var}={args.workspace}"
            print(
                f"Generated CMake option: {config['extra_cmake_options']}",
                file=sys.stderr,
            )

        output_github_actions_vars(config, platform=args.platform)
        return 0

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
