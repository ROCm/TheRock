#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Detect external repository configuration for TheRock CI workflows.

This script identifies the repository being built (rocm-libraries, rocm-systems,
etc.) and its normal submodule path. It emits a config_json GitHub Actions output
containing the source identity and caller-supplied build options. Build workflows
pass this JSON as EXTERNAL_REPO_CONFIG to fetch_sources.py and manifest generation.
fetch_sources.py selects sources by stage and checks out the requested commit at
the normal submodule path, using its standard DVC project defaults.

Usage:
    python build_tools/github_actions/detect_external_repo_config.py --repository <repository_name>

Examples:
    # Inspect configuration for rocm-libraries:
    python build_tools/github_actions/detect_external_repo_config.py --repository rocm-libraries

    # Configure a rocm-systems CI build with its exact tested commit:
    python build_tools/github_actions/detect_external_repo_config.py \\
        --external-repo-json '{"repository": "ROCm/rocm-systems", "ref": "0123456789abcdef0123456789abcdef01234567"}'

    # List supported external repositories:
    python build_tools/github_actions/detect_external_repo_config.py --list

Output (GitHub Actions format; emitted as one line):
    config_json={"repository": "ROCm/rocm-systems", "ref": "0123456789abcdef0123456789abcdef01234567", "submodule_path": "rocm-systems", "extra_cmake_options": "", "projects": "", "family_overrides": {}}

The --repository form leaves ref empty for inspecting configuration. CI callers
use --external-repo-json with a full commit SHA. Optional extra_cmake_options,
projects, and family_overrides fields are forwarded from that input unchanged.
"""

import argparse
import json
import sys

from github_actions_api import gha_set_output


# Map supported external repositories to their normal submodule locations.
SUBMODULE_PATHS = {
    "rocm-libraries": "rocm-libraries",
    "rocm-systems": "rocm-systems",
    "rocgdb": "debug-tools/rocgdb/source",
}


def main(argv=None):
    """Main entry point for the script.

    Args:
        argv: Command line arguments (defaults to sys.argv if None)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Resolve external source identity and build options for TheRock CI."
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

    args = parser.parse_args(argv)

    if args.list:
        print("Known external repositories:")
        for repo_name in SUBMODULE_PATHS.keys():
            print(f"  - {repo_name}")
        return 0

    # Parse external repo JSON if provided, extract repository name
    external_repo = {}
    source_repository = None
    source_ref = ""
    if args.external_repo_json:
        try:
            external_repo = json.loads(args.external_repo_json)
            source_repository = external_repo.get("repository", "")
            source_ref = external_repo.get("ref", "")
            # Extract repo name from full name (e.g., "rocm-libraries" from "ROCm/rocm-libraries").
            # Lowercase so SUBMODULE_PATHS lookup is case-insensitive (e.g. "ROCgdb" -> "rocgdb").
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
        if args.repository not in SUBMODULE_PATHS:
            raise ValueError(
                f"Unknown external repository: {args.repository}\n"
                f"Known repositories: {', '.join(SUBMODULE_PATHS)}"
            )
        submodule_path = SUBMODULE_PATHS[args.repository]

        # Log to stderr for visibility in CI logs
        print(f"Detected repository: {args.repository}", file=sys.stderr)

        final_source_repo = source_repository or f"ROCm/{args.repository}"

        # Keep caller build/test options separate from source fetching.
        projects = external_repo.get("projects", "")
        family_overrides = external_repo.get("family_overrides", {})
        extra_cmake_options = external_repo.get("extra_cmake_options", "")

        config_json = {
            "repository": final_source_repo,
            "ref": source_ref,
            "submodule_path": submodule_path,
            "extra_cmake_options": extra_cmake_options,
            "projects": projects,
            "family_overrides": family_overrides,
        }
        print(
            f"Generated config_json:\n{json.dumps(config_json, indent=2)}",
            file=sys.stderr,
        )

        gha_set_output({"config_json": json.dumps(config_json)})
        return 0

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
