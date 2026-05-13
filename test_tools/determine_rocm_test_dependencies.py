# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute subproject test dependencies by parsing CMakeLists.txt files.

Example:
$ python build_tools/determine_rocm_test_dependencies.py --projects rocSPARSE
["hipsparse", "rocsparse"]
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_cmake_test_subprojects(therock_dir):
    """Parse CMakeLists.txt files to extract TEST_SUBPROJECTS declarations.

    Returns dict mapping subproject name (lowercase) -> list of test dependencies (lowercase).
    """
    test_deps = {}
    cmake_files = list(Path(therock_dir).rglob("CMakeLists.txt"))

    for cmake_file in cmake_files:
        content = cmake_file.read_text()

        # Find all therock_cmake_subproject_declare blocks
        pattern = r"therock_cmake_subproject_declare\s*\(\s*(\w+)(.*?)\)"

        for match in re.finditer(pattern, content, re.DOTALL):
            subproject_name = match.group(1).lower()
            block_content = match.group(2)

            # Look for TEST_SUBPROJECTS within this block
            # Match TEST_SUBPROJECTS with optional list of dependencies
            # Empty TEST_SUBPROJECTS is valid (only tests itself)
            test_subprojects_match = re.search(
                r"TEST_SUBPROJECTS(?:\s+((?:\w+\s*)+))?", block_content
            )

            if test_subprojects_match:
                deps_str = (test_subprojects_match.group(1) or "").strip()
                deps = [d.strip().lower() for d in deps_str.split() if d.strip()]
                test_deps[subproject_name] = deps

    return test_deps


def get_subprojects_to_test(changed_subprojects, therock_dir=None):
    """Get all subprojects to test when given subprojects change."""
    if therock_dir is None:
        therock_dir = Path.cwd()
    else:
        therock_dir = Path(therock_dir)

    test_deps = parse_cmake_test_subprojects(therock_dir)

    # Convert inputs to lowercase
    changed_lower = [p.lower() for p in changed_subprojects]
    result = set(changed_lower)

    for changed in changed_lower:
        if changed in test_deps:
            result.update(test_deps[changed])

    return result


def get_rocm_test_dependencies(changed_subprojects, therock_dir=None):
    """Get all subprojects to test when specific subprojects change."""
    return get_subprojects_to_test(changed_subprojects, therock_dir)


def list_subprojects(therock_dir=None, show_deps=False):
    """List all subprojects with TEST_SUBPROJECTS.

    Args:
        therock_dir: Path to TheRock directory
        show_deps: If True, return dict with deps; if False, return list of names
    """
    if therock_dir is None:
        therock_dir = Path.cwd()

    test_deps = parse_cmake_test_subprojects(therock_dir)

    if show_deps:
        # Return dict with "empty" indicator for subprojects with no test deps
        result = {}
        for name in sorted(test_deps.keys()):
            deps = test_deps[name]
            result[name] = deps if deps else "empty"
        return result

    return sorted(test_deps.keys())


def main():
    parser = argparse.ArgumentParser(
        description="Compute subproject test dependencies by parsing CMakeLists.txt"
    )
    parser.add_argument(
        "--therock-dir", type=str, default=".", help="TheRock directory"
    )
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
        help="Alias for --changed",
    )
    parser.add_argument(
        "--list-subprojects", action="store_true", help="List all subprojects"
    )
    parser.add_argument(
        "--show-deps",
        action="store_true",
        help="With --list-subprojects, show dependencies (or 'empty' if none)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "list"],
        default="json",
        help="Output format: json (default) or list (newline-separated)",
    )

    args = parser.parse_args()

    therock_dir = Path(args.therock_dir).resolve()

    if args.list_subprojects:
        result = list_subprojects(therock_dir, show_deps=args.show_deps)
        print(json.dumps(result, indent=2))
        return

    changed = args.changed or args.projects
    if not changed:
        parser.error(
            "one of the following arguments is required: --changed, --projects"
        )

    result = get_subprojects_to_test(changed, therock_dir)

    if args.format == "json":
        print(json.dumps(sorted(result)))
    else:
        for item in sorted(result):
            print(item)


if __name__ == "__main__":
    main()
