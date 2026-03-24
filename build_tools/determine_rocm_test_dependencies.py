# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute package-level test dependencies for changed components.

This module reads BUILD_TOPOLOGY.toml to extract artifact dependencies and
determines which packages need testing when a specific package is updated.

By default, we test the changed package plus its direct downstream dependents
(computed from the reverse artifact_deps graph). Artifacts can optionally specify
a test_deps field to override this behavior and limit testing to specific consumers.

usage: determine_rocm_test_dependencies.py [-h] [--therock-dir THEROCK_DIR]
                                           [--changed PACKAGE [PACKAGE ...]]
                                           [--list-packages]

options:
  -h, --help            show this help message and exit
  --therock-dir THEROCK_DIR
                        Path to TheRock directory (default: current directory)
  --changed PACKAGE [PACKAGE ...]
                        Package(s) that have changed (e.g., blas, miopen)
  --list-packages       List all available packages with their artifact groups
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]




class ArtifactInfo:
    """Information about an artifact from BUILD_TOPOLOGY.toml."""

    def __init__(self, name: str, artifact_group: str):
        self.name = name
        self.artifact_group = artifact_group
        self.artifact_deps: Set[str] = set()
        self.test_deps: Optional[Set[str]] = None  # If set, overrides reverse deps


class ArtifactDependencyAnalyzer:
    """Analyzes artifact-level test dependencies from BUILD_TOPOLOGY.toml."""

    def __init__(self, therock_dir: Path):
        self.therock_dir = therock_dir
        self.artifacts: Dict[str, ArtifactInfo] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}
        self._load_topology()
        self._build_reverse_dependency_graph()

    def _load_topology(self):
        """Load and parse BUILD_TOPOLOGY.toml."""
        topology_file = self.therock_dir / "BUILD_TOPOLOGY.toml"
        if not topology_file.exists():
            raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found: {topology_file}")

        with open(topology_file, "rb") as f:
            topology = tomllib.load(f)

        # Extract artifacts
        artifacts_section = topology.get("artifacts", {})
        for artifact_name, artifact_data in artifacts_section.items():
            artifact_group = artifact_data.get("artifact_group", "")
            info = ArtifactInfo(artifact_name, artifact_group)

            # Get artifact dependencies
            deps = artifact_data.get("artifact_deps", [])
            info.artifact_deps = set(deps)

            # Get test dependencies (optional override)
            test_deps = artifact_data.get("test_deps")
            if test_deps is not None:
                info.test_deps = set(test_deps)

            self.artifacts[artifact_name] = info

    def _build_reverse_dependency_graph(self):
        """
        Build a reverse dependency graph for fast lookups.

        Maps each artifact to its direct downstream dependents.

        Example: if solver has artifact_deps containing "blas":
            artifacts["solver"].artifact_deps contains "blas"
            reverse_deps["blas"] contains "solver"
        """
        # Initialize empty sets for all artifacts
        for artifact_name in self.artifacts:
            self.reverse_deps[artifact_name] = set()

        # Populate reverse dependencies
        for artifact_name, artifact_info in self.artifacts.items():
            for dep_name in artifact_info.artifact_deps:
                if dep_name in self.artifacts:
                    self.reverse_deps[dep_name].add(artifact_name)

    def get_packages_to_test(self, changed_packages: List[str]) -> Set[str]:
        """
        Get all packages that need testing given a list of changed packages.

        Returns the changed packages PLUS their downstream dependents.
        - If an artifact specifies test_deps, only those artifacts are tested
        - Otherwise, all direct downstream dependents (from reverse artifact_deps) are tested

        Example: blas <- solver <- sparse
            If blas.test_deps = ["solver"]:
                get_packages_to_test(["blas"]) -> {blas, solver}
            If blas has no test_deps:
                get_packages_to_test(["blas"]) -> {blas, solver, sparse, rocwmma, ...}
        """
        packages_to_test = set(changed_packages)

        for changed in changed_packages:
            artifact_info = self.artifacts.get(changed)
            if artifact_info and artifact_info.test_deps is not None:
                # Use explicit test_deps override
                packages_to_test.update(artifact_info.test_deps)
            else:
                # Add all direct dependents from reverse dependency graph
                packages_to_test.update(self.reverse_deps.get(changed, set()))

        return packages_to_test


def create_analyzer(therock_dir: Optional[Path] = None) -> ArtifactDependencyAnalyzer:
    """
    Create an ArtifactDependencyAnalyzer instance.

    This is a convenience function for programmatic use.

    Example:
        >>> from determine_rocm_test_dependencies import create_analyzer
        >>> analyzer = create_analyzer()
        >>> packages_to_test = analyzer.get_packages_to_test(["blas"])
        >>> print(packages_to_test)
        {'blas', 'solver'}
    """
    if therock_dir is None:
        therock_dir = Path.cwd()
    else:
        therock_dir = Path(therock_dir).resolve()

    if not therock_dir.exists():
        raise FileNotFoundError(f"TheRock root directory not found: {therock_dir}")

    return ArtifactDependencyAnalyzer(therock_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compute package-level test dependencies for changed components"
    )
    parser.add_argument(
        "--therock-dir",
        type=str,
        default=".",
        help="Path to TheRock directory (default: current directory)",
    )
    parser.add_argument(
        "--changed",
        type=str,
        nargs="+",
        metavar="PACKAGE",
        help="Package(s) that have changed (e.g., blas, miopen)",
    )
    parser.add_argument(
        "--list-packages",
        action="store_true",
        help="List all available packages with their artifact groups",
    )

    args = parser.parse_args()

    # Find TheRock root
    therock_dir = Path(args.therock_dir).resolve()
    if not therock_dir.exists():
        print(
            f"Error: TheRock root directory not found: {therock_dir}", file=sys.stderr
        )
        sys.exit(1)

    # Create the analyzer
    try:
        analyzer = ArtifactDependencyAnalyzer(therock_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not analyzer.artifacts:
        print(
            "Error: No artifacts found. Check the TheRock root path.", file=sys.stderr
        )
        sys.exit(1)

    # Handle --list-packages
    if args.list_packages:
        # Group artifacts by artifact_group
        packages_by_group: Dict[str, List[str]] = {}
        for artifact_name, artifact_info in analyzer.artifacts.items():
            group = artifact_info.artifact_group or "unknown"
            if group not in packages_by_group:
                packages_by_group[group] = []
            packages_by_group[group].append(artifact_name)

        # Sort package lists within each group
        for group in packages_by_group:
            packages_by_group[group].sort()

        print(json.dumps(packages_by_group, indent=2, sort_keys=True))
        return

    # Require --changed if not listing packages
    if not args.changed:
        parser.error("the following arguments are required: --changed")

    # Normalize input package names to lowercase
    changed_packages = [p.lower() for p in args.changed]

    # Validate package names
    valid_packages = set(analyzer.artifacts.keys())
    invalid = [p for p in changed_packages if p not in valid_packages]
    if invalid:
        print(f"Error: Unknown package(s): {', '.join(invalid)}", file=sys.stderr)
        sys.exit(1)

    # Get packages to test
    packages_to_test = analyzer.get_packages_to_test(changed_packages)

    # Output JSON array
    print(json.dumps(sorted(packages_to_test)))


if __name__ == "__main__":
    main()
