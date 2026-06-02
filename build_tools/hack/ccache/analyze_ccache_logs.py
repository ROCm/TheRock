#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Analyze ccache logs from TheRock CI to diagnose cache hit rate issues.

Downloads and parses ccache logs from CI artifacts (S3) for a given workflow
run, then produces a summary of hit/miss rates broken down by compiler and
source file patterns.

Usage:
    # Analyze a specific run + stage + gfx family
    python build_tools/hack/ccache/analyze_ccache_logs.py \
        --run-id 25465494022 --stage math-libs --gfx gfx1151

    # Specify platform (default: windows)
    python build_tools/hack/ccache/analyze_ccache_logs.py \
        --run-id 25465494022 --stage math-libs --gfx gfx1151 --platform linux

    # Just parse an already-downloaded log file
    python build_tools/hack/ccache/analyze_ccache_logs.py \
        --log-file /path/to/ccache.log
"""

import argparse
import io
import re
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# Hardcoded for convenience in this debugging script. For production code,
# use _therock_utils.s3_buckets and _therock_utils.workflow_outputs instead.
S3_BASE = "https://therock-ci-artifacts.s3.amazonaws.com"


def download_ccache_logs(
    run_id: str, stage: str, gfx: str, platform: str, output_dir: Path
) -> Path:
    """Download and extract ccache logs from S3."""
    out_dir = output_dir / f"run_{run_id}" / f"{stage}_{gfx}"
    log_file = out_dir / "ccache.log"
    if log_file.exists():
        print(f"  Using cached: {log_file}", file=sys.stderr)
        return log_file

    url = f"{S3_BASE}/{run_id}-{platform}/logs/{stage}/{gfx}/ccache_logs.tar.zst"
    print(f"  Downloading: {url}", file=sys.stderr)

    try:
        import zstandard
    except ImportError:
        print("ERROR: pip install zstandard", file=sys.stderr)
        sys.exit(1)

    try:
        with urllib.request.urlopen(url) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        print(f"ERROR: {e.code} fetching {url}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(io.BytesIO(data))
    with tarfile.open(fileobj=reader, mode="r|") as tf:
        tf.extractall(out_dir, filter="tar")

    if not log_file.exists():
        print("ERROR: ccache.log not found in archive", file=sys.stderr)
        sys.exit(1)

    print(f"  Extracted to: {out_dir}", file=sys.stderr)
    return log_file


def parse_ccache_log(log_path: Path) -> dict:
    """Parse a ccache log file and return structured data."""
    pid_compiler = {}
    pid_source = {}
    compiler_results = Counter()
    result_counts = Counter()
    source_miss_patterns = Counter()
    remote_errors = []
    compiler_hashes = defaultdict(set)
    total_entries = 0
    cant_read_count = 0
    cant_read_guids = set()

    re_compiler = re.compile(r"\[.*? (\d+)\s*\] Compiler: (.+)")
    re_source = re.compile(r"\[.*? (\d+)\s*\] Source file: (.+)")
    re_result = re.compile(
        r"\[.*? (\d+)\s*\] Result: "
        r"(direct_cache_hit|preprocessed_cache_hit|"
        r"cache_miss|direct_cache_miss|preprocessed_cache_miss|"
        r"unsupported_compiler_option|preprocessor_error|"
        r"unsupported_code_directive)"
    )
    re_hash = re.compile(r"\[.*? (\d+)\s*\] Hash of compiler (.+?) is (.+)")
    re_remote_err = re.compile(r"\[.*? \d+\s*\].*(error|timeout|failed).*remote", re.I)
    re_guid = re.compile(
        r"[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}"
    )

    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if "mentioned in a manifest" in line and "can't be read" in line:
                cant_read_count += 1
                m = re_guid.search(line)
                if m:
                    cant_read_guids.add(m.group(0))
                continue

            m = re_compiler.match(line)
            if m:
                pid_compiler[m.group(1)] = m.group(2).strip()
                continue

            m = re_source.match(line)
            if m:
                pid_source[m.group(1)] = m.group(2).strip()
                continue

            m = re_hash.match(line)
            if m:
                compiler_hashes[m.group(2).strip()].add(m.group(3).strip())
                continue

            m = re_result.match(line)
            if m:
                pid = m.group(1)
                result = m.group(2)
                total_entries += 1
                result_counts[result] += 1

                comp_path = pid_compiler.get(pid, "<unknown>")
                comp_short = _shorten_compiler(comp_path)
                compiler_results[(comp_short, result)] += 1

                # Only count cache_miss (the final verdict), not
                # direct_cache_miss/preprocessed_cache_miss (intermediate).
                if result == "cache_miss" and pid in pid_source:
                    src = pid_source[pid]
                    pattern = _source_pattern(src)
                    source_miss_patterns[pattern] += 1

                continue

            if re_remote_err.search(line):
                remote_errors.append(line.strip())

    return {
        "compiler_results": compiler_results,
        "result_counts": result_counts,
        "source_miss_patterns": source_miss_patterns,
        "remote_errors": remote_errors,
        "compiler_hashes": dict(compiler_hashes),
        "total_entries": total_entries,
        "cant_read_count": cant_read_count,
        "cant_read_guids": cant_read_guids,
    }


def _shorten_compiler(path: str) -> str:
    """Shorten compiler path for display."""
    if "cl.exe" in path:
        return "cl.exe (MSVC)"
    if "clr" in path and "clang++" in path:
        return "clr/clang++"
    if "clr" in path and "clang" in path:
        return "clr/clang"
    if "amd-llvm" in path and "clang++" in path:
        return "amd-llvm/clang++"
    if "c++" in path:
        return "gcc (c++)"
    if "/cc" in path:
        return "gcc (cc)"
    return path


def _source_pattern(src: str) -> str:
    """Extract project name from source path."""
    src = src.replace("\\", "/")
    # Normalize to lowercase to merge source-tree (projects/miopen/)
    # and build-tree (ml-libs/MIOpen/) paths for the same project.
    m = re.search(r"/projects/([^/]+)/", src)
    if m:
        return m.group(1).lower()
    m = re.search(r"(?:math-libs|ml-libs)/(?:BLAS/)?([^/]+)/", src)
    if m:
        return m.group(1).lower()
    m = re.search(r"third-party/([^/]+)/", src)
    if m:
        return "3p-" + m.group(1).lower()
    parts = src.split("/")
    return "/".join(parts[-3:-1]).lower() if len(parts) > 2 else src


def print_analysis(data: dict, label: str = ""):
    """Print human-readable analysis."""
    if label:
        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")

    rc = data["result_counts"]
    hits = rc.get("direct_cache_hit", 0) + rc.get("preprocessed_cache_hit", 0)
    misses = rc.get("cache_miss", 0)
    total_cacheable = hits + misses
    uncacheable = (
        rc.get("unsupported_compiler_option", 0)
        + rc.get("preprocessor_error", 0)
        + rc.get("unsupported_code_directive", 0)
    )

    print("\n## Overall Summary")
    print(f"  Total result entries:   {data['total_entries']}")
    print(f"  Cacheable:              {total_cacheable}")
    if total_cacheable > 0:
        print(f"    Hits:                 {hits} ({100 * hits / total_cacheable:.1f}%)")
        print(f"      Direct:             {rc.get('direct_cache_hit', 0)}")
        print(f"      Preprocessed:       {rc.get('preprocessed_cache_hit', 0)}")
        print(
            f"    Misses:               {misses} ({100 * misses / total_cacheable:.1f}%)"
        )
    print(f"  Uncacheable:            {uncacheable}")

    # Can't-be-read diagnostics
    if data["cant_read_count"] > 0:
        print(f"\n## Path Issues")
        print(f"  'Can't be read' entries: {data['cant_read_count']}")
        print(f"  Unique GUIDs in paths:   {len(data['cant_read_guids'])}")
        if data["cant_read_guids"]:
            print("  (Stale entries from runners with different workspace GUIDs)")

    # Per-compiler breakdown
    print("\n## Per-Compiler Breakdown")
    compilers = sorted(set(c for c, _ in data["compiler_results"]))
    for comp in compilers:
        comp_hits = sum(
            data["compiler_results"].get((comp, r), 0)
            for r in ("direct_cache_hit", "preprocessed_cache_hit")
        )
        comp_misses = data["compiler_results"].get((comp, "cache_miss"), 0)
        comp_total = comp_hits + comp_misses
        if comp_total > 0:
            rate = 100 * comp_hits / comp_total
            print(f"  {comp:30s}  {comp_hits:5d} / {comp_total:5d} hits ({rate:.1f}%)")

    # Top miss sources
    if data["source_miss_patterns"]:
        print("\n## Top Projects by Cache Misses")
        for pattern, count in data["source_miss_patterns"].most_common(15):
            print(f"  {pattern:30s}  {count:5d} misses")

    # Remote errors
    if data["remote_errors"]:
        print(f"\n## Remote Storage Errors ({len(data['remote_errors'])} total)")
        for err in data["remote_errors"][:10]:
            print(f"  {err}")

    print()


def main():
    p = argparse.ArgumentParser(
        description="Analyze ccache logs from TheRock CI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run-id", help="Workflow run ID to analyze")
    p.add_argument("--compare-run-id", help="Second run ID for comparison")
    p.add_argument(
        "--stage", default="math-libs", help="Stage name (default: math-libs)"
    )
    p.add_argument("--gfx", default="gfx1151", help="GPU family (default: gfx1151)")
    p.add_argument(
        "--platform", default="windows", help="Platform: linux or windows (default)"
    )
    p.add_argument(
        "--log-file", type=Path, help="Analyze a local log file instead of downloading"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for downloaded logs (default: temp dir)",
    )
    args = p.parse_args()

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="ccache_analysis_"))

    if args.log_file:
        print(f"Analyzing: {args.log_file}", file=sys.stderr)
        data = parse_ccache_log(args.log_file)
        print_analysis(data, label=str(args.log_file))
    elif args.run_id:
        log = download_ccache_logs(
            args.run_id, args.stage, args.gfx, args.platform, output_dir
        )
        data = parse_ccache_log(log)
        print_analysis(data, label=f"Run {args.run_id} / {args.stage} / {args.gfx}")

        if args.compare_run_id:
            log2 = download_ccache_logs(
                args.compare_run_id, args.stage, args.gfx, args.platform, output_dir
            )
            data2 = parse_ccache_log(log2)
            print_analysis(
                data2,
                label=f"Run {args.compare_run_id} / {args.stage} / {args.gfx}",
            )
    else:
        p.error("Provide either --run-id or --log-file")


if __name__ == "__main__":
    main()
