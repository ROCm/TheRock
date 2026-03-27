#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Collect build/test metrics and emit them to the GitHub Actions job summary.

Metrics are written as a hidden HTML comment that notify_quartz.yml extracts:
    <!-- THEROCK_METRICS {"ccache_hit_rate": 87.3, ...} -->

Usage examples:
    # Build job — collect ccache, artifacts, and peak memory:
    python emit_build_metrics.py \
        --ccache \
        --artifact-dir build/artifacts \
        --resource-info-dir build/logs/therock-build-prof

    # Test job — record shard info:
    python emit_build_metrics.py --test-shard 2 4
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from github_actions_api import gha_emit_metrics


def collect_ccache_hit_rate() -> float | None:
    """Parse ccache statistics and return the overall hit rate as a percentage."""
    try:
        result = subprocess.run(
            ["ccache", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            print(f"[WARN] ccache -s exited with {result.returncode}", file=sys.stderr)
            return None
        return _parse_ccache_output(result.stdout)
    except FileNotFoundError:
        print("[WARN] ccache not found", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print("[WARN] ccache -s timed out", file=sys.stderr)
        return None


def _parse_ccache_output(text: str) -> float | None:
    """Extract hit rate from ccache -s output.

    Handles two formats:
      ccache 4.x+:  "Hits:  1234 / 5678"
      older:        "cache hit (direct) N", "cache hit (preprocessed) N", "cache miss N"
    """
    # ccache 4.x+ format: "Hits:  1234 / 5678"
    m = re.search(r"Hits:\s+(\d+)\s*/\s*(\d+)", text)
    if m:
        hits, total = int(m.group(1)), int(m.group(2))
        if total > 0:
            return round(hits / total * 100, 2)
        return 0.0

    # Older ccache format
    direct = _extract_stat(text, r"cache hit \(direct\)\s+(\d+)")
    preprocessed = _extract_stat(text, r"cache hit \(preprocessed\)\s+(\d+)")
    miss = _extract_stat(text, r"cache miss\s+(\d+)")
    total = direct + preprocessed + miss
    if total > 0:
        return round((direct + preprocessed) / total * 100, 2)

    return None


def _extract_stat(text: str, pattern: str) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def collect_artifact_size_mb(artifact_dir: Path) -> float | None:
    """Sum the sizes of *.tar.xz files in the artifact directory."""
    if not artifact_dir.is_dir():
        print(f"[WARN] Artifact directory not found: {artifact_dir}", file=sys.stderr)
        return None

    total_bytes = sum(
        f.stat().st_size for f in artifact_dir.glob("*.tar.xz") if f.is_file()
    )
    if total_bytes == 0:
        return None
    return round(total_bytes / (1024 * 1024), 2)


def collect_peak_memory_gb(resource_info_dir: Path) -> float | None:
    """Read the comp-summary.md table and return the maximum max_rss_gb value."""
    summary_path = resource_info_dir / "comp-summary.md"
    if not summary_path.is_file():
        print(
            f"[WARN] comp-summary.md not found: {summary_path}", file=sys.stderr
        )
        return None

    max_rss_gb = 0.0
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "---" not in line:
            cols = [c.strip() for c in line.split("|")]
            # comp-summary.md columns: component, ..., max_rss_mb, max_rss_gb
            # The last non-empty column is max_rss_gb
            try:
                rss_gb = float(cols[-2])  # last col before trailing empty
                max_rss_gb = max(max_rss_gb, rss_gb)
            except (ValueError, IndexError):
                continue

    return round(max_rss_gb, 4) if max_rss_gb > 0 else None


def main():
    parser = argparse.ArgumentParser(
        description="Emit build/test metrics to GitHub Actions job summary."
    )
    parser.add_argument(
        "--ccache",
        action="store_true",
        help="Collect ccache hit rate from `ccache -s`",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory containing *.tar.xz artifacts to measure",
    )
    parser.add_argument(
        "--resource-info-dir",
        type=Path,
        help="Directory containing comp-summary.md from resource_info.py --finalize",
    )
    parser.add_argument(
        "--test-shard",
        nargs=2,
        type=int,
        metavar=("INDEX", "TOTAL"),
        help="Test shard index and total count (1-based)",
    )
    args = parser.parse_args()

    metrics: dict = {}

    if args.ccache:
        rate = collect_ccache_hit_rate()
        metrics["ccache_hit_rate"] = rate

    if args.artifact_dir:
        size = collect_artifact_size_mb(args.artifact_dir)
        metrics["artifact_size_mb"] = size

    if args.resource_info_dir:
        peak = collect_peak_memory_gb(args.resource_info_dir)
        metrics["peak_memory_gb"] = peak

    if args.test_shard:
        index, total = args.test_shard
        metrics["test_shard"] = {"index": index, "total": total}

    if not metrics:
        print("[INFO] No metrics collected, nothing to emit.", file=sys.stderr)
        return

    print(f"[INFO] Emitting metrics: {metrics}")
    gha_emit_metrics(metrics)


if __name__ == "__main__":
    main()
