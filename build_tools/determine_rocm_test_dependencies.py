# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute package-level test dependencies for changed components.

This module analyzes CMakeLists.txt files to extract RUNTIME_DEPS and determines
which packages need testing when a specific package is updated.

We only test the changed package plus its direct downstream dependents.

Example dependency chain: rocblas ← hipblas ← hipblaslt ← miopen
If rocblas was updated, we test: rocblas, hipblas

usage: determine_rocm_test_dependencies.py [-h] [--therock-dir THEROCK_DIR] --changed PACKAGE [PACKAGE ...]
options:
  -h, --help            show this help message and exit
  --therock-dir THEROCK_DIR
                        Path to TheRock directory (default: current directory)
  --changed PACKAGE [PACKAGE ...]
                        Package(s) that have changed (e.g., rocBLAS, hipBLAS)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class PackageInfo:
    """Information about a CMake subproject package."""

    def __init__(self, name: str, cmake_file: Path):
        self.name = name
        self.cmake_file = cmake_file
        self.runtime_deps: Set[str] = set()
        self.artifact: Optional[str] = None


class PackageDependencyAnalyzer:
    """Analyzes package-level test dependencies from CMakeLists.txt files."""

    def __init__(self, therock_dir: Path):
        self.therock_dir = therock_dir
        self.packages: Dict[str, PackageInfo] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}
        self._discover_packages()
        self._build_reverse_dependency_graph()

    def _discover_packages(self):
        """Discover all CMake subprojects and their dependencies."""
        # Search in known directories
        search_dirs = [
            self.therock_dir / "math-libs",
            self.therock_dir / "ml-libs",
            self.therock_dir / "comm-libs",
            self.therock_dir / "core",
            self.therock_dir / "compiler",
            self.therock_dir / "base",
            self.therock_dir / "profiler",
            self.therock_dir / "debug-tools",
            self.therock_dir / "dctools",
            self.therock_dir / "media-libs",
            self.therock_dir / "iree-libs",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for cmake_file in search_dir.rglob("CMakeLists.txt"):
                try:
                    content = cmake_file.read_text()
                    self._parse_cmake_file(cmake_file, content)
                except Exception as e:
                    # Skip files that can't be read
                    pass

    def _parse_cmake_file(self, cmake_file: Path, content: str):
        """Parse a CMakeLists.txt file to extract package declarations."""
        # Pattern to match therock_cmake_subproject_declare
        declare_pattern = re.compile(
            r'therock_cmake_subproject_declare\((\w+(?:-\w+)*)',
            re.MULTILINE
        )

        for match in declare_pattern.finditer(content):
            package_name = match.group(1).lower()  # Normalize to lowercase

            # Find the corresponding closing parenthesis for this declaration
            start_pos = match.start()
            # Extract the full declaration block (simplified - find next standalone ')')
            # This is a heuristic; proper parsing would require a CMake parser
            decl_end = content.find('\n)', start_pos)
            if decl_end == -1:
                decl_end = len(content)

            decl_block = content[start_pos:decl_end + 2]

            pkg_info = PackageInfo(package_name, cmake_file)

            # Extract RUNTIME_DEPS
            # Match until we hit another keyword or closing paren
            runtime_deps_match = re.search(
                r'RUNTIME_DEPS\s+((?:(?!\b(?:BUILD_DEPS|CMAKE_ARGS|CMAKE_INCLUDES|COMPILER_TOOLCHAIN|EXTERNAL_SOURCE_DIR|BINARY_DIR|BACKGROUND_BUILD|DEFAULT_GPU_TARGETS)\b)[^\n)]+(?:\n\s+)?)+)',
                decl_block,
                re.MULTILINE
            )
            if runtime_deps_match:
                deps_text = runtime_deps_match.group(1)
                for line in deps_text.split('\n'):
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    dep = line.split('#')[0].strip()
                    if dep and not dep.startswith('${'):
                        pkg_info.runtime_deps.add(dep.lower())  # Normalize to lowercase

            # Determine artifact from file path
            rel_path = cmake_file.relative_to(self.therock_dir)
            if len(rel_path.parts) > 0:
                pkg_info.artifact = rel_path.parts[0]

            self.packages[package_name] = pkg_info

    def _build_reverse_dependency_graph(self):
        """
        Build a reverse dependency graph for fast lookups.

        Maps each package to its direct downstream dependents (packages that depend on it).

        Example: if hipblas has RUNTIME_DEPS on rocblas:
            packages["hipblas"].runtime_deps contains "rocblas"
            reverse_deps["rocblas"] contains "hipblas"
        """
        # Initialize empty sets for all packages
        for package_name in self.packages:
            self.reverse_deps[package_name] = set()

        # Populate reverse dependencies based on RUNTIME_DEPS
        for package_name, pkg_info in self.packages.items():
            for dep_name in pkg_info.runtime_deps:
                # Only track dependencies on packages we know about
                if dep_name in self.packages:
                    self.reverse_deps[dep_name].add(package_name)


    def get_packages_to_test(self, changed_packages: List[str]) -> Set[str]:
        """
        Get all packages that need testing given a list of changed packages.

        Returns the changed packages PLUS their direct downstream dependents.

        Example: rocblas ← hipblas ← hipblaslt
            get_packages_to_test(["rocblas"]) → {rocblas, hipblas}
            get_packages_to_test(["hipblas"]) → {hipblas, hipblaslt}
            get_packages_to_test(["hipblaslt"]) → {hipblaslt}
        """
        packages_to_test = set(changed_packages)

        for changed in changed_packages:
            # Add only DIRECT dependents
            packages_to_test.update(self.reverse_deps.get(changed, set()))

        return packages_to_test



def create_analyzer(therock_dir: Optional[Path] = None) -> PackageDependencyAnalyzer:
    """
    Create a PackageDependencyAnalyzer instance.

    This is a convenience function for programmatic use.

    Example:
        >>> from determine_rocm_test_dependencies import create_analyzer
        >>> analyzer = create_analyzer()
        >>> packages_to_test = analyzer.get_packages_to_test(["rocblas"])
        >>> print(packages_to_test)
        {'rocblas', 'hipblas'}
    """
    if therock_dir is None:
        therock_dir = Path.cwd()
    else:
        therock_dir = Path(therock_dir).resolve()

    if not therock_dir.exists():
        raise FileNotFoundError(f"TheRock root directory not found: {therock_dir}")

    return PackageDependencyAnalyzer(therock_dir)


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
        required=True,
        help="Package(s) that have changed (e.g., rocBLAS, hipBLAS)",
    )

    args = parser.parse_args()

    # Find TheRock root
    therock_dir = Path(args.therock_dir).resolve()
    if not therock_dir.exists():
        print(f"Error: TheRock root directory not found: {therock_dir}", file=sys.stderr)
        sys.exit(1)

    # Create the analyzer
    analyzer = PackageDependencyAnalyzer(therock_dir)

    if not analyzer.packages:
        print("Error: No packages found. Check the TheRock root path.", file=sys.stderr)
        sys.exit(1)

    # Normalize input package names to lowercase
    changed_packages = [p.lower() for p in args.changed]

    # Validate package names
    valid_packages = set(analyzer.packages.keys())
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
