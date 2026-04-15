# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compute subproject test dependencies from CMake manifest."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


class SubprojectInfo:
    def __init__(self, name: str):
        self.name = name
        self.runtime_deps: Set[str] = set()
        self.test_subprojects: Optional[Set[str]] = None


class SubprojectDependencyAnalyzer:
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.subprojects: Dict[str, SubprojectInfo] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}
        self._load_manifest()
        self._build_reverse_dependency_graph()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Subproject test manifest not found: {self.manifest_path}\n"
                f"Please run CMake configure first."
            )

        with open(self.manifest_path) as f:
            manifest = json.load(f)

        for name, data in manifest.get("subprojects", {}).items():
            info = SubprojectInfo(name)
            info.runtime_deps = set(data.get("runtime_deps", []))
            test_subprojects = data.get("test_subprojects")
            if test_subprojects is not None:
                info.test_subprojects = set(test_subprojects)
            self.subprojects[name] = info

    def _build_reverse_dependency_graph(self):
        for name in self.subprojects:
            self.reverse_deps[name] = set()

        for name, info in self.subprojects.items():
            for dep in info.runtime_deps:
                if dep in self.subprojects:
                    self.reverse_deps[dep].add(name)

    def get_subprojects_to_test(self, changed_subprojects: List[str]) -> Set[str]:
        """Get subprojects to test when given subprojects change.

        Returns changed subprojects + their dependents.
        If test_subprojects is set, use those; otherwise use reverse deps.
        """
        result = set(changed_subprojects)

        for changed in changed_subprojects:
            info = self.subprojects.get(changed)
            if info and info.test_subprojects is not None:
                result.update(info.test_subprojects)
            else:
                result.update(self.reverse_deps.get(changed, set()))

        return result


def get_rocm_test_dependencies(
    changed_subprojects: List[str],
    therock_dir: Optional[Path] = None,
    build_dir: Optional[Path] = None,
) -> Set[str]:
    """Get all subprojects to test when specific subprojects change."""
    analyzer = create_analyzer(therock_dir, build_dir)
    return analyzer.get_subprojects_to_test(changed_subprojects)


def create_analyzer(
    therock_dir: Optional[Path] = None, build_dir: Optional[Path] = None
) -> SubprojectDependencyAnalyzer:
    if therock_dir is None:
        therock_dir = Path.cwd()
    else:
        therock_dir = Path(therock_dir).resolve()

    if not therock_dir.exists():
        raise FileNotFoundError(f"TheRock root not found: {therock_dir}")

    if build_dir is None:
        build_dir = therock_dir / "build"
    else:
        build_dir = Path(build_dir).resolve()

    manifest_path = build_dir / "subproject_test_manifest.json"
    return SubprojectDependencyAnalyzer(manifest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compute subproject test dependencies"
    )
    parser.add_argument(
        "--therock-dir", type=str, default=".", help="TheRock directory"
    )
    parser.add_argument("--build-dir", type=str, help="Build directory")
    parser.add_argument(
        "--changed",
        type=str,
        nargs="+",
        metavar="SUBPROJECT",
        help="Changed subproject(s)",
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        metavar="PROJECT",
        help="Alias for --changed (for consistency with fetch_test_configurations.py)",
    )
    parser.add_argument(
        "--list-subprojects", action="store_true", help="List all subprojects"
    )
    parser.add_argument(
        "--format",
        choices=["json", "list"],
        default="json",
        help="Output format: json (default) or list (newline-separated)",
    )

    args = parser.parse_args()

    therock_dir = Path(args.therock_dir).resolve()
    if not therock_dir.exists():
        print(f"Error: TheRock root not found: {therock_dir}", file=sys.stderr)
        sys.exit(1)

    build_dir = (
        Path(args.build_dir).resolve() if args.build_dir else therock_dir / "build"
    )

    if not build_dir.exists():
        print(
            f"Error: Build directory not found: {build_dir}\n"
            f"Please run CMake configure first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        analyzer = create_analyzer(therock_dir, build_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not analyzer.subprojects:
        print("Error: No subprojects found in manifest.", file=sys.stderr)
        sys.exit(1)

    if args.list_subprojects:
        print(json.dumps(sorted(analyzer.subprojects.keys()), indent=2))
        return

    # Support both --changed and --projects (alias)
    changed = args.changed or args.projects
    if not changed:
        parser.error("one of the following arguments is required: --changed, --projects")

    valid = set(analyzer.subprojects.keys())
    invalid = [p for p in changed if p not in valid]
    if invalid:
        print(f"Error: Unknown subproject(s): {', '.join(invalid)}", file=sys.stderr)
        print(f"\nAvailable subprojects:", file=sys.stderr)
        for sp in sorted(valid)[:20]:
            print(f"  {sp}", file=sys.stderr)
        if len(valid) > 20:
            print(f"  ... and {len(valid) - 20} more", file=sys.stderr)
        sys.exit(1)

    result = analyzer.get_subprojects_to_test(changed)

    if args.format == "json":
        print(json.dumps(sorted(result)))
    else:  # list format
        for item in sorted(result):
            print(item)


if __name__ == "__main__":
    main()
