#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute fine-grained package-level test dependencies for changed components.

This script analyzes CMakeLists.txt files to extract package-level dependencies
(e.g., rocBLAS, hipBLAS, rocSOLVER) and determines which packages need testing
when a specific package is updated.

TESTING PRINCIPLE:
    Only test DOWNSTREAM components that CAN be affected by changes.
    DO NOT test UPSTREAM components that CANNOT be affected by changes.

    Example dependency chain: rocblas ← hipblas ← hipblaslt
    (arrows show dependency direction: hipblas depends on rocblas)

    If rocblas changes:
        ✓ Test rocblas (changed)
        ✓ Test hipblas (downstream - depends on rocblas, could break)
        ✓ Test hipblaslt (downstream - transitively depends on rocblas, could break)

    If hipblas changes:
        ✗ DON'T test rocblas (upstream - cannot be affected by hipblas)
        ✓ Test hipblas (changed)
        ✓ Test hipblaslt (downstream - depends on hipblas, could break)

    If hipblaslt changes:
        ✗ DON'T test rocblas (upstream - cannot be affected)
        ✗ DON'T test hipblas (upstream - cannot be affected)
        ✓ Test hipblaslt (changed, leaf node with no dependents)

Usage:
    # Find what needs testing if rocBLAS changes
    python3 compute_package_test_dependencies.py --changed rocblas

    # Find what needs testing for multiple changes
    python3 compute_package_test_dependencies.py --changed rocblas hipblas

    # Output as JSON for automation
    python3 compute_package_test_dependencies.py --changed rocblas --format json

    # Show full dependency graph
    python3 compute_package_test_dependencies.py --graph

    # List all available packages
    python3 compute_package_test_dependencies.py --list-packages
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
        self.build_deps: Set[str] = set()
        self.runtime_deps: Set[str] = set()
        self.all_deps: Set[str] = set()
        self.artifact: Optional[str] = None


class PackageDependencyAnalyzer:
    """Analyzes package-level test dependencies from CMakeLists.txt files."""

    def __init__(self, therock_root: Path):
        self.therock_root = therock_root
        self.packages: Dict[str, PackageInfo] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}
        self._discover_packages()
        self._build_reverse_dependency_graph()

    def _discover_packages(self):
        """Discover all CMake subprojects and their dependencies."""
        # Pattern to match therock_cmake_subproject_declare
        declare_pattern = re.compile(
            r'therock_cmake_subproject_declare\((\w+(?:-\w+)*)',
            re.MULTILINE
        )

        # Pattern to extract BUILD_DEPS and RUNTIME_DEPS
        build_deps_pattern = re.compile(
            r'BUILD_DEPS\s+((?:[^\n)]+(?:\n\s+)?)+)',
            re.MULTILINE
        )
        runtime_deps_pattern = re.compile(
            r'RUNTIME_DEPS\s+((?:[^\n)]+(?:\n\s+)?)+)',
            re.MULTILINE
        )

        # Search in known directories
        search_dirs = [
            self.therock_root / "math-libs",
            self.therock_root / "ml-libs",
            self.therock_root / "comm-libs",
            self.therock_root / "core",
            self.therock_root / "compiler",
            self.therock_root / "base",
            self.therock_root / "profiler",
            self.therock_root / "debug-tools",
            self.therock_root / "dctools",
            self.therock_root / "media-libs",
            self.therock_root / "iree-libs",
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

            # Extract BUILD_DEPS
            # Match until we hit another keyword (RUNTIME_DEPS, CMAKE_ARGS, etc.) or closing paren
            build_deps_match = re.search(
                r'BUILD_DEPS\s+((?:(?!\b(?:RUNTIME_DEPS|CMAKE_ARGS|CMAKE_INCLUDES|COMPILER_TOOLCHAIN|EXTERNAL_SOURCE_DIR|BINARY_DIR|BACKGROUND_BUILD|DEFAULT_GPU_TARGETS)\b)[^\n)]+(?:\n\s+)?)+)',
                decl_block,
                re.MULTILINE
            )
            if build_deps_match:
                deps_text = build_deps_match.group(1)
                # Extract dependency names (filter out comments and variables)
                for line in deps_text.split('\n'):
                    line = line.strip()
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    # Extract dependency name (before any comment)
                    dep = line.split('#')[0].strip()
                    # Skip variable references like ${...}
                    if dep and not dep.startswith('${'):
                        pkg_info.build_deps.add(dep.lower())  # Normalize to lowercase
                        pkg_info.all_deps.add(dep.lower())

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
                        pkg_info.all_deps.add(dep.lower())

            # Determine artifact from file path
            rel_path = cmake_file.relative_to(self.therock_root)
            if len(rel_path.parts) > 0:
                pkg_info.artifact = rel_path.parts[0]

            self.packages[package_name] = pkg_info

    def _build_reverse_dependency_graph(self):
        """
        Build a reverse dependency graph for fast lookups.

        This maps each package to its DOWNSTREAM dependents (packages that depend on it).

        Example: if hipblas depends on rocblas:
            packages["hipblas"].all_deps contains "rocblas" (forward dep)
            reverse_deps["rocblas"] contains "hipblas" (reverse dep / downstream)
        """
        # Initialize empty sets for all packages
        for package_name in self.packages:
            self.reverse_deps[package_name] = set()

        # Populate reverse dependencies
        for package_name, pkg_info in self.packages.items():
            for dep_name in pkg_info.all_deps:
                # Only track dependencies on packages we know about
                if dep_name in self.packages:
                    self.reverse_deps[dep_name].add(package_name)

    def get_transitive_dependents(self, package_name: str) -> Set[str]:
        """
        Get all DOWNSTREAM packages that transitively depend on the given package.

        This computes the "reverse transitive closure" - all packages that would be
        affected if this package changes. Returns ONLY downstream packages, NOT
        upstream dependencies.

        Example: rocblas ← hipblas ← hipblaslt
            get_transitive_dependents("rocblas") → {hipblas, hipblaslt}
            get_transitive_dependents("hipblas") → {hipblaslt}
            get_transitive_dependents("hipblaslt") → {} (no downstream dependents)

        Args:
            package_name: Name of the changed package

        Returns:
            Set of DOWNSTREAM package names that depend on this package (directly or transitively).
            Does NOT include upstream dependencies (which cannot be affected by changes).
        """
        if package_name not in self.reverse_deps:
            return set()

        visited = set()
        to_visit = [package_name]

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            # Add all direct dependents to the visit queue
            for dependent in self.reverse_deps.get(current, set()):
                if dependent not in visited:
                    to_visit.append(dependent)

        # Remove the original package from the result
        visited.discard(package_name)
        return visited

    def get_packages_to_test(self, changed_packages: List[str]) -> Set[str]:
        """
        Get all packages that need testing given a list of changed packages.

        Returns the changed packages PLUS all downstream packages that depend on them.
        Does NOT include upstream dependencies (which cannot be affected).

        Example: rocblas ← hipblas ← hipblaslt
            get_packages_to_test(["rocblas"]) → {rocblas, hipblas, hipblaslt}
            get_packages_to_test(["hipblas"]) → {hipblas, hipblaslt} (NOT rocblas)
            get_packages_to_test(["hipblaslt"]) → {hipblaslt} (NOT hipblas or rocblas)

        Args:
            changed_packages: List of package names that have changed

        Returns:
            Set of package names that need testing (changed packages + downstream dependents).
        """
        packages_to_test = set(changed_packages)

        for changed in changed_packages:
            # Add the changed package itself
            packages_to_test.add(changed)
            # Add all transitive dependents
            packages_to_test.update(self.get_transitive_dependents(changed))

        return packages_to_test

    def get_dependency_info(self, package_name: str) -> Dict:
        """
        Get detailed dependency information for a package.

        Args:
            package_name: Name of the package

        Returns:
            Dictionary with dependency information
        """
        pkg_info = self.packages.get(package_name)
        if not pkg_info:
            return {"error": f"Package '{package_name}' not found"}

        direct_dependents = self.reverse_deps.get(package_name, set())
        transitive_dependents = self.get_transitive_dependents(package_name)

        return {
            "name": package_name,
            "artifact": pkg_info.artifact,
            "cmake_file": str(pkg_info.cmake_file.relative_to(self.therock_root)),
            "build_dependencies": sorted(pkg_info.build_deps),
            "runtime_dependencies": sorted(pkg_info.runtime_deps),
            "all_dependencies": sorted(pkg_info.all_deps),
            "direct_dependents": sorted(direct_dependents),
            "transitive_dependents": sorted(transitive_dependents),
            "total_dependent_count": len(transitive_dependents),
        }

    def print_dependency_graph(self):
        """Print the full dependency graph in a readable format."""
        print("=== Package Dependency Graph ===\n")

        for package_name in sorted(self.packages.keys()):
            pkg_info = self.packages[package_name]
            reverse_deps = self.reverse_deps.get(package_name, set())

            print(f"{package_name}:")
            if pkg_info.artifact:
                print(f"  Artifact: {pkg_info.artifact}")

            if pkg_info.build_deps:
                print(f"  Build deps: {', '.join(sorted(pkg_info.build_deps))}")
            if pkg_info.runtime_deps:
                print(f"  Runtime deps: {', '.join(sorted(pkg_info.runtime_deps))}")
            if not pkg_info.all_deps:
                print("  Dependencies: (none)")

            if reverse_deps:
                print(f"  Depended on by: {', '.join(sorted(reverse_deps))}")
            else:
                print("  Depended on by: (none)")

            print()

    def print_test_plan(self, changed_packages: List[str], format: str = "text"):
        """
        Print a test plan for the given changed packages.

        Args:
            changed_packages: List of changed package names
            format: Output format ("text" or "json")
        """
        packages_to_test = self.get_packages_to_test(changed_packages)

        if format == "json":
            result = {
                "changed_packages": changed_packages,
                "packages_to_test": sorted(packages_to_test),
                "test_count": len(packages_to_test),
            }
            print(json.dumps(result, indent=2))
            return

        # Text format
        print("=== Package Test Dependency Analysis ===\n")
        print(f"Changed packages: {', '.join(changed_packages)}\n")

        if not packages_to_test:
            print("No packages need testing.")
            return

        print(f"Total packages to test: {len(packages_to_test)}\n")

        # Group by artifact for better readability
        by_artifact: Dict[str, List[str]] = {}
        for package_name in packages_to_test:
            pkg_info = self.packages.get(package_name)
            if pkg_info:
                artifact = pkg_info.artifact or "unknown"
                if artifact not in by_artifact:
                    by_artifact[artifact] = []
                by_artifact[artifact].append(package_name)

        print("Packages to test (grouped by artifact/directory):\n")
        for artifact in sorted(by_artifact.keys()):
            print(f"  {artifact}:")
            for package_name in sorted(by_artifact[artifact]):
                if package_name in changed_packages:
                    status = "(changed)"
                else:
                    status = "(dependent)"
                print(f"    - {package_name} {status}")
            print()

    def print_package_list(self):
        """Print a list of all available packages."""
        print("=== Available Packages ===\n")

        # Group by artifact
        by_artifact: Dict[str, List[str]] = {}
        for package_name, pkg_info in self.packages.items():
            artifact = pkg_info.artifact or "unknown"
            if artifact not in by_artifact:
                by_artifact[artifact] = []
            by_artifact[artifact].append(package_name)

        for artifact in sorted(by_artifact.keys()):
            print(f"{artifact}:")
            for package_name in sorted(by_artifact[artifact]):
                print(f"  - {package_name}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Compute package-level test dependencies for changed components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find what needs testing if rocBLAS changes
  %(prog)s --changed rocBLAS

  # Find what needs testing for multiple changes
  %(prog)s --changed rocBLAS hipBLAS

  # Output as JSON
  %(prog)s --changed rocBLAS --format json

  # Show full dependency graph
  %(prog)s --graph

  # List all available packages
  %(prog)s --list-packages
        """,
    )
    parser.add_argument(
        "--therock-root",
        type=str,
        default=".",
        help="Path to TheRock root directory (default: current directory)",
    )
    parser.add_argument(
        "--changed",
        type=str,
        nargs="+",
        metavar="PACKAGE",
        help="Package(s) that have changed (e.g., rocBLAS, hipBLAS)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Print the full dependency graph",
    )
    parser.add_argument(
        "--list-packages",
        action="store_true",
        help="List all available packages",
    )
    parser.add_argument(
        "--info",
        type=str,
        metavar="PACKAGE",
        help="Show detailed dependency info for a specific package",
    )

    args = parser.parse_args()

    # Find TheRock root
    therock_root = Path(args.therock_root).resolve()
    if not therock_root.exists():
        print(f"Error: TheRock root directory not found: {therock_root}", file=sys.stderr)
        sys.exit(1)

    # Create the analyzer
    analyzer = PackageDependencyAnalyzer(therock_root)

    if not analyzer.packages:
        print("Warning: No packages found. Check the TheRock root path.", file=sys.stderr)
        sys.exit(1)

    # Handle different commands
    if args.list_packages:
        analyzer.print_package_list()
    elif args.graph:
        analyzer.print_dependency_graph()
    elif args.info:
        # Normalize input package name to lowercase
        info = analyzer.get_dependency_info(args.info.lower())
        if args.format == "json":
            print(json.dumps(info, indent=2))
        else:
            if "error" in info:
                print(info["error"], file=sys.stderr)
                sys.exit(1)
            print(f"=== Dependency Info: {info['name']} ===\n")
            print(f"Artifact: {info['artifact']}")
            print(f"CMake file: {info['cmake_file']}")
            print(f"\nBuild dependencies: {', '.join(info['build_dependencies']) or '(none)'}")
            print(f"Runtime dependencies: {', '.join(info['runtime_dependencies']) or '(none)'}")
            print(f"\nDirect dependents: {', '.join(info['direct_dependents']) or '(none)'}")
            print(
                f"\nTransitive dependents ({info['total_dependent_count']} total):"
            )
            if info["transitive_dependents"]:
                for dep in info["transitive_dependents"]:
                    print(f"  - {dep}")
            else:
                print("  (none)")
    elif args.changed:
        # Normalize input package names to lowercase
        changed_packages = [p.lower() for p in args.changed]

        # Validate package names
        valid_packages = set(analyzer.packages.keys())
        invalid = [p for p in changed_packages if p not in valid_packages]
        if invalid:
            print(
                f"Error: Unknown package(s): {', '.join(invalid)}", file=sys.stderr
            )
            print(
                f"\nRun '{sys.argv[0]} --list-packages' to see available packages.",
                file=sys.stderr,
            )
            sys.exit(1)

        analyzer.print_test_plan(changed_packages, args.format)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
