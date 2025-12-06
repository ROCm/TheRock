#!/usr/bin/env python3
"""Memory monitoring utility for detecting out-of-memory issues in CI builds.

This script monitors system memory usage at regular intervals and logs detailed
memory statistics. When integrated with GitHub Actions workflows, it helps identify
which build phase is causing out-of-memory errors.

Usage:
    # Monitor a single command:
    python build_tools/memory_monitor.py --phase "Configure Projects" -- cmake ...

    # Run as background monitoring:
    python build_tools/memory_monitor.py --background --interval 5 --phase "Build Phase"

Environment Variables:
    MEMORY_MONITOR_INTERVAL: Override default monitoring interval (seconds)
    MEMORY_MONITOR_LOG_FILE: Path to write detailed memory logs
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import threading

try:
    import psutil
except ImportError:
    print("ERROR: psutil is not installed. Install it with: pip install psutil")
    sys.exit(1)


class MemoryMonitor:
    """Monitors system and process memory usage."""

    def __init__(
        self,
        interval: float = 5.0,
        phase_name: str = "Unknown",
        log_file: Optional[Path] = None,
    ):
        self.interval = interval
        self.phase_name = phase_name
        self.log_file = log_file
        self.running = False
        self.peak_memory = 0
        self.peak_swap = 0
        self.samples = []
        self.start_time = None
        self.end_time = None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Collect current memory statistics."""
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Get current process and its children
        current_process = psutil.Process()
        process_memory = current_process.memory_info().rss

        # Try to get memory of all child processes
        children_memory = 0
        try:
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    children_memory += child.memory_info().rss
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

        total_process_memory = process_memory + children_memory

        # Track peak usage
        self.peak_memory = max(self.peak_memory, vm.used)
        self.peak_swap = max(self.peak_swap, swap.used)

        stats = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.phase_name,
            # System memory
            "total_memory_gb": vm.total / (1024**3),
            "available_memory_gb": vm.available / (1024**3),
            "used_memory_gb": vm.used / (1024**3),
            "memory_percent": vm.percent,
            "free_memory_gb": vm.free / (1024**3),
            # Peak memory
            "peak_memory_gb": self.peak_memory / (1024**3),
            "peak_swap_gb": self.peak_swap / (1024**3),
            # Swap
            "total_swap_gb": swap.total / (1024**3),
            "used_swap_gb": swap.used / (1024**3),
            "swap_percent": swap.percent,
            # Process memory
            "process_memory_gb": process_memory / (1024**3),
            "children_memory_gb": children_memory / (1024**3),
            "total_process_memory_gb": total_process_memory / (1024**3),
        }

        return stats

    def format_memory_stats(self, stats: Dict[str, Any]) -> str:
        """Format memory stats for human-readable output."""
        lines = [
            f"[{stats['timestamp']}] Memory Stats - Phase: {stats['phase']}",
            f"  System Memory: {stats['used_memory_gb']:.2f} GB / {stats['total_memory_gb']:.2f} GB ({stats['memory_percent']:.1f}% used)",
            f"  Available: {stats['available_memory_gb']:.2f} GB | Free: {stats['free_memory_gb']:.2f} GB",
            f"  Swap: {stats['used_swap_gb']:.2f} GB / {stats['total_swap_gb']:.2f} GB ({stats['swap_percent']:.1f}% used)",
            f"  Process Memory: {stats['total_process_memory_gb']:.2f} GB (Self: {stats['process_memory_gb']:.2f} GB, Children: {stats['children_memory_gb']:.2f} GB)",
        ]
        return "\n".join(lines)

    def log_stats(self, stats: Dict[str, Any]):
        """Log memory statistics to console and file."""
        formatted = self.format_memory_stats(stats)

        # Always print to stdout for GitHub Actions logs
        print(formatted, flush=True)

        # Log detailed JSON to file if specified
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(stats) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to log file: {e}", file=sys.stderr)

        # Check for concerning memory levels
        if stats["memory_percent"] > 90:
            print(
                f"[WARNING] Memory usage is critically high ({stats['memory_percent']:.1f}%)",
                file=sys.stderr,
            )
        elif stats["memory_percent"] > 75:
            print(f"[WARNING] Memory usage is high ({stats['memory_percent']:.1f}%)")

        if stats["swap_percent"] > 50:
            print(
                f"[WARNING] Swap usage is high ({stats['swap_percent']:.1f}%), this may slow down builds",
                file=sys.stderr,
            )

    def monitor_loop(self):
        """Main monitoring loop."""
        next_tick = time.monotonic()
        while self.running:
            try:
                stats = self.get_memory_stats()
                self.samples.append(stats)
                self.log_stats(stats)
            except Exception as e:
                print(f"Error collecting memory stats: {e}", file=sys.stderr)

            next_tick += self.interval
            sleep_for = max(0, next_tick - time.monotonic())
            if sleep_for == 0:
                print(
                    f"[WARNING] Stats collection took longer than interval ({self.interval}s)",
                    file=sys.stderr,
                )
            time.sleep(sleep_for)

    def start(self):
        """Start monitoring in a background thread."""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.thread.start()
        print(
            f"[MONITOR] Memory monitoring started for phase: {self.phase_name} (interval: {self.interval}s)"
        )

    def stop(self):
        """Stop monitoring and print summary."""
        self.running = False
        self.end_time = time.time()

        if hasattr(self, "thread"):
            self.thread.join(timeout=self.interval + 1)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print summary statistics."""
        if not self.samples:
            print("No memory samples collected")
            return

        duration = (
            self.end_time - self.start_time if self.end_time and self.start_time else 0
        )

        avg_memory_percent = sum(s["memory_percent"] for s in self.samples) / len(
            self.samples
        )
        max_memory_percent = max(s["memory_percent"] for s in self.samples)
        # Use the tracked peak memory from the last sample (cumulative peak)
        peak_memory_gb = self.samples[-1]["peak_memory_gb"] if self.samples else 0
        peak_swap_gb = self.samples[-1]["peak_swap_gb"] if self.samples else 0

        avg_swap_percent = sum(s["swap_percent"] for s in self.samples) / len(
            self.samples
        )
        max_swap_percent = max(s["swap_percent"] for s in self.samples)

        print("\n" + "=" * 80)
        print(f"[SUMMARY] Memory Monitoring Summary - Phase: {self.phase_name}")
        print("=" * 80)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Samples collected: {len(self.samples)}")
        print(f"")
        print(f"Memory Usage:")
        print(f"  Average: {avg_memory_percent:.1f}%")
        print(f"  Peak: {max_memory_percent:.1f}% ({peak_memory_gb:.2f} GB)")
        print(f"")
        print(f"Swap Usage:")
        print(f"  Average: {avg_swap_percent:.1f}%")
        print(f"  Peak: {max_swap_percent:.1f}% ({peak_swap_gb:.2f} GB)")

        # Warnings
        if max_memory_percent > 90:
            print(f"\n[CRITICAL] Memory usage exceeded 90% during this phase!")
            print(f"   This phase is likely causing out-of-memory issues.")
        elif max_memory_percent > 75:
            print(f"\n[WARNING] Memory usage exceeded 75% during this phase.")

        if max_swap_percent > 50:
            print(
                f"\n[WARNING] Significant swap usage detected ({max_swap_percent:.1f}%)"
            )
            print(f"   Consider increasing available memory or reducing parallel jobs.")

        print("=" * 80 + "\n")

        # GitHub Actions Step Summary
        if "GITHUB_STEP_SUMMARY" in os.environ:
            self.write_github_summary(
                duration,
                avg_memory_percent,
                max_memory_percent,
                peak_memory_gb,
                peak_swap_gb,
                avg_swap_percent,
                max_swap_percent,
            )

    def write_github_summary(
        self,
        duration,
        avg_memory_percent,
        max_memory_percent,
        peak_memory_gb,
        peak_swap_gb,
        avg_swap_percent,
        max_swap_percent,
    ):
        """Write summary to GitHub Actions step summary."""
        try:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
                # Determine status indicator
                if max_memory_percent > 90:
                    status = "CRITICAL"
                elif max_memory_percent > 75:
                    status = "WARNING"
                else:
                    status = "OK"

                f.write(f"\n## [{status}] Memory Stats: {self.phase_name}\n\n")

                # Main statistics table
                f.write("| Metric | Value |\n")
                f.write("|:-------|------:|\n")
                f.write(f"| **Duration** | {duration:.1f}s |\n")
                f.write(f"| **Samples Collected** | {len(self.samples)} |\n")
                f.write(f"| **Average Memory** | {avg_memory_percent:.1f}% |\n")
                f.write(
                    f"| **Peak Memory** | {max_memory_percent:.1f}% ({peak_memory_gb:.2f} GB) |\n"
                )
                f.write(f"| **Average Swap** | {avg_swap_percent:.1f}% |\n")
                f.write(
                    f"| **Peak Swap** | {max_swap_percent:.1f}% ({peak_swap_gb:.2f} GB) |\n"
                )

                # Add warnings as alerts if needed
                if max_memory_percent > 90:
                    f.write("\n> [!CAUTION]\n")
                    f.write(
                        "> Memory usage exceeded 90% during this phase! This phase is likely causing out-of-memory issues.\n"
                    )
                elif max_memory_percent > 75:
                    f.write("\n> [!WARNING]\n")
                    f.write("> Memory usage exceeded 75% during this phase.\n")

                if max_swap_percent > 50:
                    f.write("\n> [!WARNING]\n")
                    f.write(
                        f"> Significant swap usage detected ({max_swap_percent:.1f}%). Consider increasing available memory or reducing parallel jobs.\n"
                    )

                f.write("\n")
        except Exception as e:
            print(f"Warning: Failed to write GitHub summary: {e}", file=sys.stderr)


def run_command_with_monitoring(
    command: list,
    phase_name: str,
    interval: float,
    log_file: Optional[Path],
) -> int:
    """Run a command while monitoring memory usage."""
    monitor = MemoryMonitor(
        interval=interval,
        phase_name=phase_name,
        log_file=log_file,
    )

    monitor.start()

    # Log command start
    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "phase": phase_name,
                            "event": "command_start",
                            "command": " ".join(command),
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            print(
                f"Warning: Failed to write command start to log file: {e}",
                file=sys.stderr,
            )

    try:
        # Run the command
        print(f"[EXEC] Executing command: {' '.join(command)}")
        result = subprocess.run(command)
        return_code = result.returncode
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Interrupted by user")
        return_code = 130
    except Exception as e:
        print(f"[ERROR] Error executing command: {e}", file=sys.stderr)
        return_code = 1
    finally:
        if log_file:
            try:
                with open(log_file, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timestamp": datetime.now().isoformat(),
                                "phase": phase_name,
                                "event": "command_end",
                                "return_code": return_code,
                                "command": " ".join(command),
                            }
                        )
                        + "\n"
                    )
            except Exception as e:
                print(
                    f"Warning: Failed to write command end to log file: {e}",
                    file=sys.stderr,
                )
        monitor.stop()

    return return_code


def main():
    parser = argparse.ArgumentParser(
        description="Monitor memory usage during CI builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--phase",
        type=str,
        default="Build Phase",
        help="Name of the build phase being monitored",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("MEMORY_MONITOR_INTERVAL", "30")),
        help="Monitoring interval in seconds (default: 30)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=os.getenv("MEMORY_MONITOR_LOG_FILE"),
        help="Path to write detailed JSON logs",
    )

    parser.add_argument(
        "--background",
        action="store_true",
        help="Run monitoring in background without executing a command",
    )

    parser.add_argument(
        "command",
        nargs="*",
        help="Command to execute while monitoring (use -- to separate from options)",
    )

    args = parser.parse_args()

    # Handle the -- separator if present
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]

    if args.background:
        # Background monitoring mode
        print("[INFO] Background monitoring mode - press Ctrl+C to stop")
        monitor = MemoryMonitor(
            interval=args.interval,
            phase_name=args.phase,
            log_file=args.log_file,
        )
        monitor.start()

        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[STOP] Stopping background monitoring...")
            monitor.stop()

        return 0

    elif args.command:
        # Command execution mode
        return_code = run_command_with_monitoring(
            command=args.command,
            phase_name=args.phase,
            interval=args.interval,
            log_file=args.log_file,
        )
        return return_code

    else:
        # One-shot monitoring
        monitor = MemoryMonitor(
            interval=args.interval,
            phase_name=args.phase,
            log_file=args.log_file,
        )
        stats = monitor.get_memory_stats()
        monitor.log_stats(stats)
        return 0


if __name__ == "__main__":
    sys.exit(main())
