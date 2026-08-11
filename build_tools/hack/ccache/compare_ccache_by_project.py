#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare ccache hit rates per subproject across two log files.

Useful for comparing Linux vs Windows hit rates, or before/after a config
change, to identify which subprojects have cache hit regressions.

Usage:
    python build_tools/hack/ccache/compare_ccache_by_project.py \
        /path/to/linux/ccache.log /path/to/windows/ccache.log
"""

import re
import sys
from collections import Counter


def parse_by_project(log_file: str) -> dict:
    """Parse ccache log and return per-project hit/miss counts for clang."""
    pid_data = {}

    with open(log_file, "r", errors="replace") as f:
        for line in f:
            m = re.match(r"\[.*? (\d+)\s*\]", line)
            if not m:
                continue
            pid = m.group(1)
            if pid not in pid_data:
                pid_data[pid] = {}

            if "Source file:" in line:
                mm = re.search(r"Source file: (.+)", line)
                if mm:
                    pid_data[pid]["source"] = mm.group(1).strip()
            elif "Compiler:" in line and "Compiler type" not in line:
                mm = re.search(r"Compiler: (.+)", line)
                if mm:
                    pid_data[pid]["compiler"] = mm.group(1).strip()
            elif "Result: direct_cache_hit" in line:
                pid_data[pid]["result"] = "hit"
            elif "Result: cache_miss" in line:
                if pid_data[pid].get("result") != "hit":
                    pid_data[pid]["result"] = "miss"

    # Aggregate by project (clang compilations only, skip CMake probes)
    projects = Counter()
    project_hits = Counter()

    for d in pid_data.values():
        src = d.get("source", "")
        comp = d.get("compiler", "")
        result = d.get("result", "")
        if not src or not result:
            continue
        if "clang" not in comp:
            continue
        if "TryCompile" in src or "CMakeScratch" in src or "cmTC_" in src:
            continue

        project = _extract_project(src)
        projects[project] += 1
        if result == "hit":
            project_hits[project] += 1

    return {p: (project_hits[p], projects[p]) for p in projects}


def _extract_project(src: str) -> str:
    """Extract project name from source path."""
    src = src.replace("\\", "/")

    # Source tree: .../projects/{name}/...
    m = re.search(r"/projects/([^/]+)/", src)
    if m:
        return m.group(1).lower()

    # Build tree: .../math-libs/{group}/{name}/... or .../math-libs/{name}/...
    m = re.search(r"(?:math-libs|ml-libs)/(?:BLAS/)?([^/]+)/", src)
    if m:
        return m.group(1).lower()

    # Third-party
    m = re.search(r"third-party/([^/]+)/", src)
    if m:
        return "3p-" + m.group(1).lower()

    return "?"


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} LOG1 LOG2")
        sys.exit(1)

    log1, log2 = sys.argv[1], sys.argv[2]

    print("Parsing log 1...", file=sys.stderr)
    data1 = parse_by_project(log1)
    print("Parsing log 2...", file=sys.stderr)
    data2 = parse_by_project(log2)

    all_projects = sorted(set(data1.keys()) | set(data2.keys()))

    total1_hits = sum(h for h, t in data1.values())
    total1 = sum(t for h, t in data1.values())
    total2_hits = sum(h for h, t in data2.values())
    total2 = sum(t for h, t in data2.values())

    print(f"\n{'Project':<30s} {'Log 1':>18s} {'Log 2':>18s} {'Gap':>8s}")
    print(f"{'':<30s} {'hits/total   rate':>18s} {'hits/total   rate':>18s}")
    print("-" * 78)

    for proj in all_projects:
        h1, t1 = data1.get(proj, (0, 0))
        h2, t2 = data2.get(proj, (0, 0))
        r1 = f"{100 * h1 / t1:.0f}%" if t1 > 0 else "-"
        r2 = f"{100 * h2 / t2:.0f}%" if t2 > 0 else "-"
        gap = ""
        if t1 > 0 and t2 > 0:
            diff = (h2 / t2 - h1 / t1) * 100
            gap = f"{diff:+.0f}%"
        s1 = f"{h1}/{t1} {r1:>5s}" if t1 > 0 else f"{'--':>12s}"
        s2 = f"{h2}/{t2} {r2:>5s}" if t2 > 0 else f"{'--':>12s}"
        print(f"  {proj:<28s} {s1:>18s} {s2:>18s} {gap:>8s}")

    print("-" * 78)
    r1 = f"{100 * total1_hits / total1:.1f}%" if total1 > 0 else "-"
    r2 = f"{100 * total2_hits / total2:.1f}%" if total2 > 0 else "-"
    print(
        f"  {'TOTAL':<28s} {total1_hits}/{total1} {r1:>5s}"
        f"       {total2_hits}/{total2} {r2:>5s}"
    )


if __name__ == "__main__":
    main()
