#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Selective Test Filter Tool

Given two git refs (branches or commit IDs), this tool:
- Identifies changed files between the refs
- Loads the enhanced dependency mapping JSON (from enhanced_ninja_parser.py)
- Maps changed files to affected test executables (optionally filtering for "test_" prefix)
- Exports the list of tests to run to tests_to_run.json

Usage:
  python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]

Arguments:
  <depmap_json>   Path to enhanced_dependency_mapping.json
  <ref1>          Source git ref (branch or commit)
  <ref2>          Target git ref (branch or commit)

Options:
  --all           Include all executables (default)
  --test-prefix   Only include executables starting with "test_"
  --output        Output JSON file (default: tests_to_run.json)
"""

import sys
import subprocess
import json
import os


def get_changed_files(ref1, ref2):
    """Return a set of files changed between two git refs."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", ref1, ref2],
            capture_output=True,
            text=True,
            check=True,
        )
        files = set(line.strip() for line in result.stdout.splitlines() if line.strip())
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        sys.exit(1)


def load_depmap(depmap_json):
    """Load the dependency mapping JSON."""
    with open(depmap_json, "r") as f:
        data = json.load(f)
    # Support both old and new formats
    if "file_to_executables" in data:
        return data["file_to_executables"]
    return data


def select_tests(file_to_executables, changed_files, filter_mode):
    """Return a set of test executables affected by changed files."""
    affected = set()
    for f in changed_files:
        if f in file_to_executables:
            for exe in file_to_executables[f]:
                if filter_mode == "all":
                    affected.add(exe)
                elif filter_mode == "test_prefix" and exe.startswith("test_"):
                    affected.add(exe)
    return sorted(affected)


def main():
    if "--audit" in sys.argv:
        if len(sys.argv) < 2:
            print("Usage: python selective_test_filter.py <depmap_json> --audit")
            sys.exit(1)
        depmap_json = sys.argv[1]
        if not os.path.exists(depmap_json):
            print(f"Dependency map JSON not found: {depmap_json}")
            sys.exit(1)
        file_to_executables = load_depmap(depmap_json)
        for f, exes in file_to_executables.items():
            print(f"{f}: {', '.join(exes)}")
        print(f"Total files: {len(file_to_executables)}")
        sys.exit(0)

    if "--optimize-build" in sys.argv:
        if len(sys.argv) < 3:
            print(
                "Usage: python selective_test_filter.py <depmap_json> --optimize-build <changed_file1> [<changed_file2> ...]"
            )
            sys.exit(1)
        depmap_json = sys.argv[1]
        changed_files = set(sys.argv[sys.argv.index("--optimize-build") + 1 :])
        if not os.path.exists(depmap_json):
            print(f"Dependency map JSON not found: {depmap_json}")
            sys.exit(1)
        file_to_executables = load_depmap(depmap_json)
        affected_executables = set()
        for f in changed_files:
            if f in file_to_executables:
                affected_executables.update(file_to_executables[f])
        print("Affected executables:")
        for exe in sorted(affected_executables):
            print(exe)
        print(f"Total affected executables: {len(affected_executables)}")
        sys.exit(0)

    if len(sys.argv) < 4:
        print(
            "Usage: python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]"
        )
        sys.exit(1)

    depmap_json = sys.argv[1]
    ref1 = sys.argv[2]
    ref2 = sys.argv[3]
    filter_mode = "all"
    output_json = "tests_to_run.json"

    if "--test-prefix" in sys.argv:
        filter_mode = "test_prefix"
    if "--all" in sys.argv:
        filter_mode = "all"
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_json = sys.argv[idx + 1]

    if not os.path.exists(depmap_json):
        print(f"Dependency map JSON not found: {depmap_json}")
        sys.exit(1)

    changed_files = get_changed_files(ref1, ref2)
    if not changed_files:
        print("No changed files detected.")
        tests = []
    else:
        file_to_executables = load_depmap(depmap_json)
        tests = select_tests(file_to_executables, changed_files, filter_mode)

    with open(output_json, "w") as f:
        json.dump(
            {"tests_to_run": tests, "changed_files": sorted(changed_files)}, f, indent=2
        )

    print(f"Exported {len(tests)} tests to run to {output_json}")


if __name__ == "__main__":
    main()
