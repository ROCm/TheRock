#!/usr/bin/env python3
"""Analyze memory logs from CI builds to identify out-of-memory issues.

This script analyzes the JSON memory logs produced by memory_monitor.py
and generates reports showing which build phases had the highest memory usage.

Usage:
    # Analyze all logs in default directory
    python build_tools/analyze_memory_logs.py

    # Analyze specific log directory
    python build_tools/analyze_memory_logs.py --log-dir build/logs

    # Generate a detailed report
    python build_tools/analyze_memory_logs.py --detailed
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict


class MemoryLogAnalyzer:
    """Analyzes memory logs from build phases."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.phase_data = defaultdict(list)

    def load_logs(self):
        """Load all memory log files."""
        if not self.log_dir.exists():
            print(f"[ERROR] Log directory not found: {self.log_dir}", file=sys.stderr)
            return False

        log_files = list(self.log_dir.glob("*.json")) + list(
            self.log_dir.glob("*.jsonl")
        )

        if not log_files:
            print(f"[!] No log files found in {self.log_dir}", file=sys.stderr)
            return False

        print(f"[*] Found {len(log_files)} log file(s)")

        for log_file in log_files:
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            phase = data.get("phase", "Unknown")
                            self.phase_data[phase].append(data)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"[!] Warning: Failed to read {log_file}: {e}", file=sys.stderr)

        return len(self.phase_data) > 0

    def analyze_phase(
        self, phase: str, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze a single phase's memory usage."""
        if not samples:
            return {}

        memory_percents = [s["memory_percent"] for s in samples]
        memory_gbs = [s["used_memory_gb"] for s in samples]
        swap_percents = [s["swap_percent"] for s in samples]

        # Calculate statistics
        analysis = {
            "phase": phase,
            "num_samples": len(samples),
            "avg_memory_percent": sum(memory_percents) / len(memory_percents),
            "min_memory_percent": min(memory_percents),
            "max_memory_percent": max(memory_percents),
            "peak_memory_gb": max(memory_gbs),
            "avg_swap_percent": sum(swap_percents) / len(swap_percents),
            "max_swap_percent": max(swap_percents),
            "start_time": samples[0]["timestamp"],
            "end_time": samples[-1]["timestamp"],
        }

        # Calculate duration
        try:
            start = datetime.fromisoformat(analysis["start_time"])
            end = datetime.fromisoformat(analysis["end_time"])
            analysis["duration_seconds"] = (end - start).total_seconds()
        except Exception:
            analysis["duration_seconds"] = 0

        # Determine severity
        max_mem = analysis["max_memory_percent"]
        if max_mem >= 95:
            analysis["severity"] = "CRITICAL"
        elif max_mem >= 90:
            analysis["severity"] = "HIGH"
        elif max_mem >= 75:
            analysis["severity"] = "MEDIUM"
        else:
            analysis["severity"] = "LOW"

        return analysis

    def generate_report(self, detailed: bool = False) -> str:
        """Generate a text report of memory usage."""
        if not self.phase_data:
            return "No data to analyze"

        # Analyze each phase
        phase_analyses = []
        for phase, samples in self.phase_data.items():
            analysis = self.analyze_phase(phase, samples)
            if analysis:
                phase_analyses.append(analysis)

        # Sort by max memory usage
        phase_analyses.sort(key=lambda x: x["max_memory_percent"], reverse=True)

        # Build report
        lines = []
        lines.append("=" * 80)
        lines.append("MEMORY USAGE ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total phases analyzed: {len(phase_analyses)}")
        lines.append("")

        # Summary table
        lines.append("SUMMARY (Sorted by Peak Memory Usage)")
        lines.append("-" * 80)
        lines.append(f"{'Phase':<40} {'Peak':<12} {'Avg':<12} {'Severity':<10}")
        lines.append("-" * 80)

        for analysis in phase_analyses:
            phase_name = analysis["phase"][:38]
            peak = f"{analysis['max_memory_percent']:.1f}%"
            avg = f"{analysis['avg_memory_percent']:.1f}%"
            severity = analysis["severity"]

            # Add prefix based on severity
            prefix = {
                "CRITICAL": "[!!]",
                "HIGH": "[!] ",
                "MEDIUM": "[~] ",
                "LOW": "[OK]",
            }.get(severity, "[  ]")

            lines.append(
                f"{prefix} {phase_name:<37} {peak:<12} {avg:<12} {severity:<10}"
            )

        lines.append("-" * 80)
        lines.append("")

        # Detailed breakdown
        if detailed:
            lines.append("DETAILED BREAKDOWN")
            lines.append("=" * 80)

            for analysis in phase_analyses:
                lines.append("")
                lines.append(f"Phase: {analysis['phase']}")
                lines.append(f"  Duration: {analysis['duration_seconds']:.1f}s")
                lines.append(f"  Samples: {analysis['num_samples']}")
                lines.append(f"  Memory Usage:")
                lines.append(f"    Average: {analysis['avg_memory_percent']:.1f}%")
                lines.append(
                    f"    Peak: {analysis['max_memory_percent']:.1f}% ({analysis['peak_memory_gb']:.2f} GB)"
                )
                lines.append(
                    f"    Range: {analysis['min_memory_percent']:.1f}% - {analysis['max_memory_percent']:.1f}%"
                )
                lines.append(f"  Swap Usage:")
                lines.append(f"    Average: {analysis['avg_swap_percent']:.1f}%")
                lines.append(f"    Peak: {analysis['max_swap_percent']:.1f}%")
                lines.append(f"  Time Range:")
                lines.append(f"    Start: {analysis['start_time']}")
                lines.append(f"    End: {analysis['end_time']}")

                if analysis["severity"] in ["CRITICAL", "HIGH"]:
                    lines.append(
                        f"  [!] WARNING: This phase had {analysis['severity']} memory usage!"
                    )

        # Key findings
        lines.append("")
        lines.append("KEY FINDINGS")
        lines.append("=" * 80)

        critical_phases = [
            a for a in phase_analyses if a["severity"] in ["CRITICAL", "HIGH"]
        ]
        if critical_phases:
            lines.append("")
            lines.append("[!] PHASES WITH HIGH MEMORY USAGE:")
            for analysis in critical_phases:
                lines.append(
                    f"  - {analysis['phase']}: {analysis['max_memory_percent']:.1f}% peak"
                )
                lines.append(
                    f"    Consider reducing parallel jobs or increasing available memory"
                )

        high_swap = [a for a in phase_analyses if a["max_swap_percent"] > 50]
        if high_swap:
            lines.append("")
            lines.append("[!] PHASES WITH HIGH SWAP USAGE:")
            for analysis in high_swap:
                lines.append(
                    f"  - {analysis['phase']}: {analysis['max_swap_percent']:.1f}% peak"
                )
                lines.append(
                    f"    Swap usage indicates memory pressure and will slow down builds"
                )

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def write_github_summary(self, phase_analyses: List[Dict[str, Any]]):
        """Write analysis to GitHub Actions step summary."""
        if "GITHUB_STEP_SUMMARY" not in os.environ:
            return

        try:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                f.write("\n## Memory Usage Analysis\n\n")
                f.write("### Peak Memory Usage by Phase\n\n")
                f.write("| Phase | Peak Memory | Avg Memory | Peak Swap | Severity |\n")
                f.write("|-------|-------------|------------|-----------|----------|\n")

                for analysis in phase_analyses[:10]:  # Top 10
                    prefix = {
                        "CRITICAL": ":red_circle:",
                        "HIGH": ":orange_circle:",
                        "MEDIUM": ":yellow_circle:",
                        "LOW": ":green_circle:",
                    }.get(analysis["severity"], ":white_circle:")

                    f.write(
                        f"| {analysis['phase']} | "
                        f"{analysis['max_memory_percent']:.1f}% | "
                        f"{analysis['avg_memory_percent']:.1f}% | "
                        f"{analysis['max_swap_percent']:.1f}% | "
                        f"{prefix} {analysis['severity']} |\n"
                    )

                # Add warnings for critical phases
                critical = [
                    a for a in phase_analyses if a["severity"] in ["CRITICAL", "HIGH"]
                ]
                if critical:
                    f.write("\n### :warning: Phases with High Memory Usage\n\n")
                    for analysis in critical:
                        f.write(
                            f"- **{analysis['phase']}**: {analysis['max_memory_percent']:.1f}% peak memory\n"
                        )

                f.write("\n")
        except Exception as e:
            print(f"Warning: Failed to write GitHub summary: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze memory logs from CI builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("build/logs"),
        help="Directory containing memory log files (default: build/logs)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed report with per-phase breakdowns",
    )

    parser.add_argument(
        "--output", type=Path, help="Write report to file instead of stdout"
    )

    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Write summary to GITHUB_STEP_SUMMARY",
    )

    args = parser.parse_args()

    # Create analyzer and load logs
    analyzer = MemoryLogAnalyzer(args.log_dir)

    if not analyzer.load_logs():
        return 1

    # Generate report
    report = analyzer.generate_report(detailed=args.detailed)

    # Output report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"[LOG] Report written to: {args.output}")
    else:
        print(report)

    # Write GitHub summary if requested
    if args.github_summary:
        phase_analyses = []
        for phase, samples in analyzer.phase_data.items():
            analysis = analyzer.analyze_phase(phase, samples)
            if analysis:
                phase_analyses.append(analysis)
        phase_analyses.sort(key=lambda x: x["max_memory_percent"], reverse=True)
        analyzer.write_github_summary(phase_analyses)

    return 0


if __name__ == "__main__":
    sys.exit(main())
