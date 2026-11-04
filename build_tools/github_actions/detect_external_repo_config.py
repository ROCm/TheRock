#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Detects external repository configuration for TheRock CI workflows.

This script identifies the repository for being built (rocm-libraries, rocm-systems,
etc.) and its normal submodule path within TheRock. It then emits a config_json
GitHub Actions output containing the source identity and caller-supplied build
options. Build workflows can then pass this JSON as EXTERNAL_REPO_CONFIG to
scripts like fetch_sources.py and generate_therock_manifest.py to override the
default git submodule state.

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
from dataclasses import asdict, dataclass, field
import json
import sys

from github_actions_api import gha_set_output


# Map supported external repositories to their normal submodule locations.
SUBMODULE_PATHS = {
    "rocm-libraries": "rocm-libraries",
    "rocm-systems": "rocm-systems",
    "rocgdb": "debug-tools/rocgdb/source",
}


@dataclass
class ExternalRepoConfig:
    """Source identity and caller build options forwarded to build workflows."""

    repository: str
    ref: str
    submodule_path: str
    extra_cmake_options: str = ""
    projects: str = ""
    family_overrides: dict[str, dict[str, object]] = field(default_factory=dict)


def build_external_repo_config(
    repository: str,
    *,
    ref: str = "",
    extra_cmake_options: str = "",
    projects: str = "",
    family_overrides: dict[str, dict[str, object]] | None = None,
) -> ExternalRepoConfig:
    """Build source checkout configuration and attach caller-supplied options.

    Args:
        repository: Supported component name or GitHub owner/repo, including a
            fork. The component name determines the submodule path, ignoring
            case. Bare names become ROCm/<lowercase-name>; explicit owner/repo
            values are preserved.
        ref: Commit to test from repository. CI callers supply a full commit
            SHA; an empty string allows inspecting the configuration without
            selecting a commit. Commit validation happens during source fetching.
        extra_cmake_options: Additional CMake arguments for the build, forwarded
            unchanged as a string.
        projects: Caller-supplied project selection string, forwarded unchanged
            without parsing or validation.
        family_overrides: CI configuration overrides keyed by GPU family, then
            platform. Forwarded unchanged; None becomes an empty dictionary.

    Returns:
        Configuration containing the resolved repository and submodule path,
        together with the supplied ref and build options. The caller can
        serialize it as JSON for build jobs using dataclasses.asdict().

    Raises:
        ValueError: The repository name is missing or unsupported.

    Examples:
        Inspect the default repository and submodule path without a commit:

        >>> config = build_external_repo_config("rocm-systems")
        >>> (config.repository, config.submodule_path, config.ref)
        ('ROCm/rocm-systems', 'rocm-systems', '')

        Test a fork's commit at the same submodule path with extra build options:

        >>> config = build_external_repo_config(
        ...     "my-fork/rocm-systems",
        ...     ref="0123456789abcdef0123456789abcdef01234567",
        ...     projects="rccl",
        ...     extra_cmake_options="-DTHEROCK_ENABLE_RCCL=ON",
        ... )
        >>> (config.repository, config.submodule_path)
        ('my-fork/rocm-systems', 'rocm-systems')
    """
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("A repository name or owner/repo is required")
    repo_name = repository.rsplit("/", 1)[-1].lower()
    if repo_name not in SUBMODULE_PATHS:
        raise ValueError(
            f"Unknown external repository: {repo_name}\n"
            f"Known repositories: {', '.join(SUBMODULE_PATHS)}"
        )
    return ExternalRepoConfig(
        repository=repository if "/" in repository else f"ROCm/{repo_name}",
        ref=ref,
        submodule_path=SUBMODULE_PATHS[repo_name],
        extra_cmake_options=extra_cmake_options,
        projects=projects,
        family_overrides=family_overrides if family_overrides is not None else {},
    )


def main(argv: list[str] | None = None) -> None:
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
        return

    # Parse external repo JSON if provided, extract repository name
    external_repo = {}
    if args.external_repo_json:
        external_repo = json.loads(args.external_repo_json)
        if not isinstance(external_repo, dict):
            raise ValueError("--external-repo-json must contain a JSON object")
    repository = (
        external_repo.get("repository", "")
        if args.external_repo_json
        else args.repository or ""
    )
    config = build_external_repo_config(
        repository,
        ref=external_repo.get("ref", ""),
        extra_cmake_options=external_repo.get("extra_cmake_options", ""),
        projects=external_repo.get("projects", ""),
        family_overrides=external_repo.get("family_overrides"),
    )
    config_json = asdict(config)
    print(
        f"Generated config_json:\n{json.dumps(config_json, indent=2)}", file=sys.stderr
    )
    gha_set_output({"config_json": json.dumps(config_json)})


if __name__ == "__main__":
    main(sys.argv[1:])
