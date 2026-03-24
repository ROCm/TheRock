# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute subproject-level test dependencies for changed components.

This module reads the CMake-generated subproject_test_manifest.json to determine
which subprojects need testing when a specific subproject is updated.

By default, we test the changed subproject plus its direct downstream dependents
(computed from the reverse runtime_deps graph). Subprojects can optionally specify
test_subprojects to override this behavior and limit testing to specific consumers.

usage: determine_rocm_test_dependencies.py [-h] [--therock-dir THEROCK_DIR]
                                           [--build-dir BUILD_DIR]
                                           [--changed SUBPROJECT [SUBPROJECT ...]]
                                           [--list-subprojects]

options:
  -h, --help            show this help message and exit
  --therock-dir THEROCK_DIR
                        Path to TheRock directory (default: current directory)
  --build-dir BUILD_DIR
                        Path to build directory (default: therock-dir/build)
  --changed SUBPROJECT [SUBPROJECT ...]
                        Subproject(s) that have changed (e.g., rocBLAS, MIOpen)
  --list-subprojects    List all available subprojects
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


class SubprojectInfo:
    """Information about a subproject from the CMake manifest."""

    def __init__(self, name: str):
        self.name = name
        self.runtime_deps: Set[str] = set()
        self.test_subprojects: Optional[Set[str]] = None  # If set, overrides reverse deps


class SubprojectDependencyAnalyzer:
    """Analyzes subproject-level test dependencies from CMake manifest."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.subprojects: Dict[str, SubprojectInfo] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}
        self._load_manifest()
        self._build_reverse_dependency_graph()

    def _load_manifest(self):
        """Load and parse the CMake-generated manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Subproject test manifest not found: {self.manifest_path}\n"
                f"Please run CMake configure first to generate the manifest."
            )

        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        # Extract subprojects
        subprojects_section = manifest.get("subprojects", {})
        for subproject_name, subproject_data in subprojects_section.items():
            info = SubprojectInfo(subproject_name)

            # Get runtime dependencies
            runtime_deps = subproject_data.get("runtime_deps", [])
            info.runtime_deps = set(runtime_deps)

            # Get test subprojects override (optional)
            test_subprojects = subproject_data.get("test_subprojects")
            if test_subprojects is not None:
                info.test_subprojects = set(test_subprojects)

            self.subprojects[subproject_name] = info

    def _build_reverse_dependency_graph(self):
        """
        Build a reverse dependency graph for fast lookups.

        Maps each subproject to its direct downstream dependents.

        Example: if rocSOLVER has runtime_deps containing "rocBLAS":
            subprojects["rocSOLVER"].runtime_deps contains "rocBLAS"
            reverse_deps["rocBLAS"] contains "rocSOLVER"
        """
        # Initialize empty sets for all subprojects
        for subproject_name in self.subprojects:
            self.reverse_deps[subproject_name] = set()

        # Populate reverse dependencies
        for subproject_name, subproject_info in self.subprojects.items():
            for dep_name in subproject_info.runtime_deps:
                if dep_name in self.subprojects:
                    self.reverse_deps[dep_name].add(subproject_name)

    def get_subprojects_to_test(self, changed_subprojects: List[str]) -> Set[str]:
        """
        Get all subprojects that need testing given a list of changed subprojects.

        Returns the changed subprojects PLUS their downstream dependents.
        - If a subproject specifies test_subprojects, only those subprojects are tested
        - Otherwise, all direct downstream dependents (from reverse runtime_deps) are tested

        Example: rocBLAS <- rocSOLVER <- rocSPARSE
            If rocBLAS.test_subprojects = ["rocSOLVER", "hipBLAS"]:
                get_subprojects_to_test(["rocBLAS"]) -> {rocBLAS, rocSOLVER, hipBLAS}
            If rocBLAS has no test_subprojects:
                get_subprojects_to_test(["rocBLAS"]) -> {rocBLAS, rocSOLVER, hipBLAS, rocSPARSE, ...}
        """
        subprojects_to_test = set(changed_subprojects)

        for changed in changed_subprojects:
            subproject_info = self.subprojects.get(changed)
            if subproject_info and subproject_info.test_subprojects is not None:
                # Use explicit test_subprojects override
                subprojects_to_test.update(subproject_info.test_subprojects)
            else:
                # Add all direct dependents from reverse dependency graph
                subprojects_to_test.update(self.reverse_deps.get(changed, set()))

        return subprojects_to_test


def create_analyzer(
    therock_dir: Optional[Path] = None, build_dir: Optional[Path] = None
) -> SubprojectDependencyAnalyzer:
    """
    Create a SubprojectDependencyAnalyzer instance.

    This is a convenience function for programmatic use.

    Example:
        >>> from determine_rocm_test_dependencies import create_analyzer
        >>> analyzer = create_analyzer()
        >>> subprojects_to_test = analyzer.get_subprojects_to_test(["rocBLAS"])
        >>> print(subprojects_to_test)
        {'rocBLAS', 'hipBLAS', 'rocSOLVER'}
    """
    if therock_dir is None:
        therock_dir = Path.cwd()
    else:
        therock_dir = Path(therock_dir).resolve()

    if not therock_dir.exists():
        raise FileNotFoundError(f"TheRock root directory not found: {therock_dir}")

    if build_dir is None:
        build_dir = therock_dir / "build"
    else:
        build_dir = Path(build_dir).resolve()

    manifest_path = build_dir / "subproject_test_manifest.json"
    return SubprojectDependencyAnalyzer(manifest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute subproject-level test dependencies for changed components"
    )
    parser.add_argument(
        "--therock-dir",
        type=str,
        default=".",
        help="Path to TheRock directory (default: current directory)",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        help="Path to build directory (default: therock-dir/build)",
    )
    parser.add_argument(
        "--changed",
        type=str,
        nargs="+",
        metavar="SUBPROJECT",
        help="Subproject(s) that have changed (e.g., rocBLAS, MIOpen)",
    )
    parser.add_argument(
        "--list-subprojects",
        action="store_true",
        help="List all available subprojects",
    )

    args = parser.parse_args()

    # Find TheRock root
    therock_dir = Path(args.therock_dir).resolve()
    if not therock_dir.exists():
        print(
            f"Error: TheRock root directory not found: {therock_dir}", file=sys.stderr
        )
        sys.exit(1)

    # Find build directory
    if args.build_dir:
        build_dir = Path(args.build_dir).resolve()
    else:
        build_dir = therock_dir / "build"

    if not build_dir.exists():
        print(
            f"Error: Build directory not found: {build_dir}\n"
            f"Please run CMake configure first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create the analyzer
    try:
        analyzer = create_analyzer(therock_dir, build_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not analyzer.subprojects:
        print(
            "Error: No subprojects found in manifest. Check the build directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Handle --list-subprojects
    if args.list_subprojects:
        subproject_list = sorted(analyzer.subprojects.keys())
        print(json.dumps(subproject_list, indent=2))
        return

    # Require --changed if not listing subprojects
    if not args.changed:
        parser.error("the following arguments are required: --changed")

    # Validate subproject names (case-sensitive for CMake targets)
    valid_subprojects = set(analyzer.subprojects.keys())
    invalid = [p for p in args.changed if p not in valid_subprojects]
    if invalid:
        print(f"Error: Unknown subproject(s): {', '.join(invalid)}", file=sys.stderr)
        print(f"\nAvailable subprojects:", file=sys.stderr)
        for sp in sorted(valid_subprojects)[:20]:
            print(f"  {sp}", file=sys.stderr)
        if len(valid_subprojects) > 20:
            print(f"  ... and {len(valid_subprojects) - 20} more", file=sys.stderr)
        sys.exit(1)

    # Get subprojects to test
    subprojects_to_test = analyzer.get_subprojects_to_test(args.changed)

    # Output JSON array
    print(json.dumps(sorted(subprojects_to_test)))


if __name__ == "__main__":
    main()
