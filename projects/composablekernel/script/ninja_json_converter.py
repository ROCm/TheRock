#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Converts .ninja_log files into Chrome's about:tracing format.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Iterator


class BuildTarget:
    """Represents a single build target with timing information."""

    def __init__(self, start_time: int, end_time: int, output_name: str, cmd_hash: str):
        self.start_time = int(start_time)
        self.end_time = int(end_time)
        self.cmd_hash = cmd_hash
        self.duration = self.end_time - self.start_time
        self.targets = [output_name]  # List of target names for this command hash

    @property
    def category(self) -> str:
        """Categorize the build target based on file extension."""
        # Use the first target for categorization
        primary_target = self.targets[0] if self.targets else ""
        ext = Path(primary_target).suffix.lower()
        if ext in [".o", ".obj"]:
            return "compile"
        elif ext in [".a", ".lib"]:
            return "archive"
        elif ext in [".so", ".dll", ".dylib"]:
            return "link_shared"
        elif ext in [".exe", ".out"]:
            return "link_executable"
        elif "test" in primary_target.lower():
            return "test"
        else:
            return "other"

    @property
    def output_name(self) -> str:
        """Get the primary output name (for backward compatibility)."""
        return self.targets[0] if self.targets else ""


class ThreadScheduler:
    """Simulates thread allocation for parallelism analysis."""

    def __init__(self, legacy_mode: bool = False):
        self.workers: List[int] = []
        self.legacy_mode = legacy_mode

    def allocate_thread(self, target: BuildTarget) -> int:
        """Allocate a thread for the given target."""
        if self.legacy_mode:
            # Legacy algorithm from old ninjatracer
            for worker in range(len(self.workers)):
                if self.workers[worker] >= target.end_time:
                    self.workers[worker] = target.start_time
                    return worker
            self.workers.append(target.start_time)
            return len(self.workers) - 1
        else:
            # New algorithm
            for i, worker_end_time in enumerate(self.workers):
                if worker_end_time <= target.start_time:
                    self.workers[i] = target.end_time
                    return i

            # No available worker, create a new one
            self.workers.append(target.end_time)
            return len(self.workers) - 1


class NinjaLogParser:
    """Parser for ninja build log files."""

    def __init__(self, show_all_builds: bool = False):
        self.show_all_builds = show_all_builds

    def parse_log_file(self, log_path: str) -> List[BuildTarget]:
        """Parse the ninja log file and return build targets."""
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Ninja log file not found: {log_path}")

        with open(log_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        if not lines:
            raise ValueError("Empty ninja log file")

        # Parse and validate header
        header = lines[0].strip()
        version_match = re.match(r"^# ninja log v(\d+)$", header)
        if not version_match:
            raise ValueError(f"Invalid ninja log header: {header}")

        version = int(version_match.group(1))
        if version < 5:
            raise ValueError(f"Unsupported ninja log version: {version}")

        # Skip additional header line for version 6
        start_line = 2 if version > 5 else 1

        targets: Dict[str, BuildTarget] = {}
        last_end_time = 0

        for line_num, line in enumerate(lines[start_line:], start=start_line + 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 5:
                print(
                    f"Warning: Skipping malformed line {line_num}: {line}",
                    file=sys.stderr,
                )
                continue

            try:
                start_time, end_time, _, output_name, cmd_hash = parts[:5]
                start_time, end_time = int(start_time), int(end_time)

                # Handle incremental builds
                if not self.show_all_builds and end_time < last_end_time:
                    targets.clear()

                last_end_time = end_time

                # Group targets by command hash
                if cmd_hash not in targets:
                    targets[cmd_hash] = BuildTarget(
                        start_time, end_time, output_name, cmd_hash
                    )
                else:
                    # Update with the latest timing and add output
                    existing = targets[cmd_hash]
                    existing.start_time = min(existing.start_time, start_time)
                    existing.end_time = max(existing.end_time, end_time)
                    existing.duration = existing.end_time - existing.start_time
                    existing.targets.append(output_name)

            except (ValueError, IndexError) as e:
                print(f"Warning: Error parsing line {line_num}: {e}", file=sys.stderr)
                continue

        return sorted(targets.values(), key=lambda t: t.end_time, reverse=True)


class FTimeTraceReader:
    """Reads and processes Clang -ftime-trace JSON files."""

    def __init__(self, granularity_us: int = 50000):
        self.granularity_us = granularity_us

    def read_trace_file(self, trace_path: str) -> Optional[Dict]:
        """Read and parse a Clang time trace file."""
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            return None

    def filter_events(self, trace_data: Dict) -> List[Dict]:
        """Filter trace events based on criteria."""
        if "traceEvents" not in trace_data:
            return []

        filtered_events = []
        for event in trace_data["traceEvents"]:
            # Only include complete events (ph=X) that meet duration threshold
            if (
                event.get("ph") == "X"
                and event.get("dur", 0) >= self.granularity_us
                and not event.get("name", "").startswith("Total")
            ):
                filtered_events.append(event)

        return filtered_events

    def adjust_event_timing(
        self, event: Dict, target: BuildTarget, pid: int, tid: int
    ) -> Dict:
        """Adjust event timing to align with ninja build timing."""
        ninja_duration_us = target.duration * 1000

        # Validate event duration against ninja timing
        if event.get("dur", 0) > ninja_duration_us:
            print(
                f"Warning: Clang trace event duration ({event['dur']}μs) exceeds "
                f"ninja duration ({ninja_duration_us}μs) for {target.output_name}",
                file=sys.stderr,
            )
            return None

        # Adjust event timing
        adjusted_event = event.copy()
        adjusted_event["pid"] = pid
        adjusted_event["tid"] = tid
        adjusted_event["ts"] += target.start_time * 1000  # Offset by ninja start time

        return adjusted_event


class ChromeTraceGenerator:
    """Generates Chrome tracing format from build targets."""

    def __init__(
        self,
        process_id: int = 1,
        embed_ftime_traces: bool = False,
        granularity_us: int = 50000,
        ninja_log_dir: Optional[str] = None,
        legacy_format: bool = False,
    ):
        self.process_id = process_id
        self.scheduler = ThreadScheduler(legacy_mode=legacy_format)
        self.embed_ftime_traces = embed_ftime_traces
        self.ninja_log_dir = ninja_log_dir
        self.ftime_reader = (
            FTimeTraceReader(granularity_us) if embed_ftime_traces else None
        )
        self.legacy_format = legacy_format

    def find_ftime_trace_files(self, target: BuildTarget) -> List[str]:
        """Find Clang -ftime-trace files for a build target."""
        if not self.ninja_log_dir:
            return []

        trace_files = []

        # Look for .json files adjacent to object files
        obj_path = Path(self.ninja_log_dir) / target.output_name
        json_path = obj_path.with_suffix(".json")

        if json_path.exists():
            trace_files.append(str(json_path))

        return trace_files

    def generate_ftime_events(self, target: BuildTarget, tid: int) -> Iterator[Dict]:
        """Generate Clang -ftime-trace events for a target."""
        if not self.embed_ftime_traces or not self.ftime_reader:
            return

        trace_files = self.find_ftime_trace_files(target)

        for trace_file in trace_files:
            trace_data = self.ftime_reader.read_trace_file(trace_file)
            if not trace_data:
                continue

            filtered_events = self.ftime_reader.filter_events(trace_data)

            for event in filtered_events:
                adjusted_event = self.ftime_reader.adjust_event_timing(
                    event, target, self.process_id, tid
                )
                if adjusted_event:
                    yield adjusted_event

    def generate_trace_events(self, targets: List[BuildTarget]) -> List[Dict]:
        """Generate Chrome trace events from build targets."""
        events = []

        for target in targets:
            thread_id = self.scheduler.allocate_thread(target)

            # Add main ninja build event
            if self.legacy_format:
                # Legacy format: join multiple targets with commas, use "targets" category, empty args
                target_name = (
                    ", ".join(target.targets)
                    if len(target.targets) > 1
                    else target.output_name
                )
                ninja_event = {
                    "name": target_name,
                    "cat": "targets",
                    "ph": "X",  # Complete event
                    "ts": target.start_time * 1000,  # Convert to microseconds
                    "dur": target.duration * 1000,  # Convert to microseconds
                    "pid": self.process_id,
                    "tid": thread_id,
                    "args": {},
                }
            else:
                # New format: smart categorization, detailed args
                ninja_event = {
                    "name": target.output_name,
                    "cat": target.category,
                    "ph": "X",  # Complete event
                    "ts": target.start_time * 1000,  # Convert to microseconds
                    "dur": target.duration * 1000,  # Convert to microseconds
                    "pid": self.process_id,
                    "tid": thread_id,
                    "args": {
                        "output": target.output_name,
                        "duration_ms": target.duration,
                        "cmd_hash": target.cmd_hash,
                    },
                }
            events.append(ninja_event)

            # Add embedded Clang -ftime-trace events
            if self.embed_ftime_traces:
                ftime_events = list(self.generate_ftime_events(target, thread_id))
                events.extend(ftime_events)

                if ftime_events:
                    print(
                        f"Embedded {len(ftime_events)} -ftime-trace events for {target.output_name}",
                        file=sys.stderr,
                    )

        return events


class BuildAnalyzer:
    """Analyzes build performance and provides statistics."""

    def __init__(self, targets: List[BuildTarget]):
        self.targets = targets

    def get_build_summary(self) -> Dict:
        """Generate build performance summary."""
        if not self.targets:
            return {}

        total_duration = sum(t.duration for t in self.targets)
        total_targets = len(self.targets)

        # Category statistics
        category_stats = {}
        for target in self.targets:
            cat = target.category
            if cat not in category_stats:
                category_stats[cat] = {"count": 0, "total_time": 0}
            category_stats[cat]["count"] += 1
            category_stats[cat]["total_time"] += target.duration

        # Top slowest targets
        slowest_targets = sorted(self.targets, key=lambda t: t.duration, reverse=True)[
            :10
        ]

        return {
            "total_targets": total_targets,
            "total_duration_ms": total_duration,
            "total_duration_sec": total_duration / 1000,
            "average_duration_ms": total_duration / total_targets
            if total_targets > 0
            else 0,
            "category_stats": category_stats,
            "slowest_targets": [
                {
                    "name": t.output_name,
                    "duration_ms": t.duration,
                    "category": t.category,
                }
                for t in slowest_targets
            ],
        }

    def print_summary(self):
        """Print build summary to stderr."""
        summary = self.get_build_summary()
        if not summary:
            print("No build data available", file=sys.stderr)
            return

        print("\n=== Build Summary ===", file=sys.stderr)
        print(f"Total targets: {summary['total_targets']}", file=sys.stderr)
        print(f"Total time: {summary['total_duration_sec']:.2f}s", file=sys.stderr)
        print(
            f"Average time per target: {summary['average_duration_ms']:.2f}ms",
            file=sys.stderr,
        )

        print("\nBy category:", file=sys.stderr)
        for category, stats in summary["category_stats"].items():
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            print(
                f"  {category:15} {stats['count']:6} targets "
                f"{stats['total_time'] / 1000:8.2f}s "
                f"(avg: {avg_time / 1000:.3f}s)",
                file=sys.stderr,
            )

        print("\nSlowest targets:", file=sys.stderr)
        for i, target in enumerate(summary["slowest_targets"][:5], 1):
            print(
                f"  {i:2}. {target['name']} ({target['duration_ms']}ms, {target['category']})",
                file=sys.stderr,
            )


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert ninja build logs to Chrome tracing format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s build/.ninja_log                 # Basic usage
  %(prog)s build/.ninja_log --output trace.json  # Save to file
  %(prog)s build/.ninja_log --summary       # Show build summary
  %(prog)s build/.ninja_log --show-all      # Include all builds
  %(prog)s build/.ninja_log --embed-ftime-trace  # Include Clang timing data
  %(prog)s build/.ninja_log --granularity 10000  # Custom granularity threshold
        """,
    )

    parser.add_argument(
        "ninja_logs",
        nargs="+",  # Accept one or more ninja log files
        help="Path(s) to the .ninja_log file(s)",
    )

    parser.add_argument("-o", "--output", help="Output file (default: stdout)")

    parser.add_argument(
        "--show-all", action="store_true", help="Show all builds, not just the last one"
    )

    parser.add_argument(
        "--summary", action="store_true", help="Print build summary to stderr"
    )

    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )

    parser.add_argument(
        "--embed-ftime-trace",
        action="store_true",
        help="Embed Clang -ftime-trace JSON files found adjacent to targets",
    )

    parser.add_argument(
        "--granularity",
        type=int,
        default=50000,
        help="Minimum duration for -ftime-trace events in microseconds (default: 50000)",
    )

    parser.add_argument(
        "--legacy-format",
        action="store_true",
        help='Output in legacy format compatible with old ninjatracer (simple JSON array, all categories as "targets", empty args)',
    )

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    try:
        # Process multiple ninja log files
        all_events = []

        for pid, ninja_log_path in enumerate(args.ninja_logs):
            # Parse ninja log
            log_parser = NinjaLogParser(show_all_builds=args.show_all)
            targets = log_parser.parse_log_file(ninja_log_path)

            if not targets:
                print(
                    f"No build targets found in ninja log: {ninja_log_path}",
                    file=sys.stderr,
                )
                continue

            # Determine ninja log directory for -ftime-trace files
            ninja_log_dir = (
                os.path.dirname(os.path.abspath(ninja_log_path))
                if args.embed_ftime_trace
                else None
            )

            # Generate trace events for this log file
            trace_generator = ChromeTraceGenerator(
                process_id=pid,  # Use different PID for each log file
                embed_ftime_traces=args.embed_ftime_trace,
                granularity_us=args.granularity,
                ninja_log_dir=ninja_log_dir,
                legacy_format=args.legacy_format,
            )
            events = trace_generator.generate_trace_events(targets)
            all_events.extend(events)

            # Print summary if requested (for each log file)
            if args.summary:
                print(f"\n=== Summary for {ninja_log_path} ===", file=sys.stderr)
                analyzer = BuildAnalyzer(targets)
                analyzer.print_summary()

        if not all_events:
            print("No build targets found in any ninja log files", file=sys.stderr)
            return 1

        # Output format logic
        if args.legacy_format:
            # Legacy format: always output simple JSON array
            json_kwargs = {"indent": 2} if args.pretty else {}
            json_output = json.dumps(all_events, **json_kwargs)
        elif args.output or args.pretty:
            # Enhanced format with metadata (when saving to file or pretty printing)
            trace_data = {
                "traceEvents": all_events,
                "displayTimeUnit": "ms",
                "systemTraceEvents": "SystemTraceData",
                "otherData": {"version": "1.0", "generator": "ninja_json_converter.py"},
            }
            json_kwargs = {"indent": 2} if args.pretty else {}
            json_output = json.dumps(trace_data, **json_kwargs)
        else:
            # Original format (simple JSON array to stdout)
            json_output = json.dumps(all_events)

        if args.output:
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"Trace written to {args.output}", file=sys.stderr)
        else:
            print(json_output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
