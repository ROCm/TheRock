#!/usr/bin/env python3
"""Memory monitoring utility for detecting out-of-memory issues in CI builds.

This script monitors system memory usage at regular intervals and logs detailed
memory statistics. When integrated with GitHub Actions workflows, it helps identify
which build phase is causing out-of-memory errors.

Thread Safety:
    Uses threading.Event for stop signaling to ensure thread-safe communication between threads
    Immediate shutdown response via Event.wait()

Graceful Shutdown:
    On Linux/Unix: Responds to SIGTERM/SIGINT via signal handlers
    On Windows: Checks for stop signal file (Windows has limited signal support)
    Both approaches call monitor.stop() which sets the stop_event and prints summary

Usage:
    # Monitor a single command:
    python build_tools/github_actions/memory_monitor.py --phase "Configure Projects" -- cmake ...

    # Run as background monitoring:
    python build_tools/github_actions/memory_monitor.py --background --interval 5 --phase "Build Phase"

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
import psutil

from github_actions_utils import gha_append_step_summary

# Constants
BYTES_TO_GB = 1024**3

# Memory thresholds (percentages)
MEMORY_CRITICAL_PERCENT = 90
MEMORY_WARNING_PERCENT = 75
SWAP_WARNING_PERCENT = 50

# Default intervals (seconds)
DEFAULT_INTERVAL_SECONDS = 5.0
DEFAULT_INTERVAL_ENV_FALLBACK = 30.0

# Timeouts (seconds)
THREAD_JOIN_BUFFER_SECONDS = 1
STOP_CHECK_INTERVAL_SECONDS = 1

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

# Display formatting
SEPARATOR_WIDTH = 80


class MemoryMonitor:
    """Monitors system and process memory usage."""

    def __init__(
        self,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        phase_name: str = "Unknown",
        log_file: Optional[Path] = None,
        stop_signal_file: Optional[Path] = None,
        max_runtime_seconds: Optional[float] = None,
        parent_pid: Optional[int] = None,
    ):
        self.interval_seconds = interval_seconds
        self.phase_name = phase_name
        self.log_file = log_file
        self.stop_signal_file = stop_signal_file
        self.max_runtime_seconds = max_runtime_seconds
        self.parent_pid = parent_pid
        self.stop_event = threading.Event()
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
            "total_memory_gb": vm.total / BYTES_TO_GB,
            "available_memory_gb": vm.available / BYTES_TO_GB,
            "used_memory_gb": vm.used / BYTES_TO_GB,
            "memory_percent": vm.percent,
            "free_memory_gb": vm.free / BYTES_TO_GB,
            # Peak memory
            "peak_memory_gb": self.peak_memory / BYTES_TO_GB,
            "peak_swap_gb": self.peak_swap / BYTES_TO_GB,
            # Swap
            "total_swap_gb": swap.total / BYTES_TO_GB,
            "used_swap_gb": swap.used / BYTES_TO_GB,
            "swap_percent": swap.percent,
            # Process memory
            "process_memory_gb": process_memory / BYTES_TO_GB,
            "children_memory_gb": children_memory / BYTES_TO_GB,
            "total_process_memory_gb": total_process_memory / BYTES_TO_GB,
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
        if stats["memory_percent"] > MEMORY_CRITICAL_PERCENT:
            print(
                f"[WARNING] Memory usage is critically high ({stats['memory_percent']:.1f}%)",
                file=sys.stderr,
            )
        elif stats["memory_percent"] > MEMORY_WARNING_PERCENT:
            print(f"[WARNING] Memory usage is high ({stats['memory_percent']:.1f}%)")

        if stats["swap_percent"] > SWAP_WARNING_PERCENT:
            print(
                f"[WARNING] Swap usage is high ({stats['swap_percent']:.1f}%), this may slow down builds",
                file=sys.stderr,
            )

    def monitor_loop(self):
        """Main monitoring loop."""
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            # Check if we've exceeded max runtime
            if self.max_runtime_seconds and self.start_time:
                elapsed = time.time() - self.start_time
                if elapsed >= self.max_runtime_seconds:
                    print(
                        f"\n[TIMEOUT] Maximum runtime ({self.max_runtime_seconds}s) exceeded, stopping monitoring..."
                    )
                    self.stop_event.set()
                    break

            # Check if parent process is still alive
            if self.parent_pid:
                try:
                    parent = psutil.Process(self.parent_pid)
                    if not parent.is_running():
                        print(
                            f"\n[PARENT_DIED] Parent process (PID {self.parent_pid}) is no longer running, stopping monitoring..."
                        )
                        self.stop_event.set()
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(
                        f"\n[PARENT_DIED] Parent process (PID {self.parent_pid}) no longer exists, stopping monitoring..."
                    )
                    self.stop_event.set()
                    break

            # Check for stop signal file (for Windows compatibility)
            if self.stop_signal_file and self.stop_signal_file.exists():
                print(
                    f"\n[STOP_SIGNAL] Stop signal file detected, stopping monitoring..."
                )
                self.stop_event.set()
                break

            try:
                stats = self.get_memory_stats()
                self.samples.append(stats)
                self.log_stats(stats)
            except Exception as e:
                print(f"Error collecting memory stats: {e}", file=sys.stderr)

            next_tick += self.interval_seconds
            sleep_for = max(0, next_tick - time.monotonic())
            if sleep_for == 0:
                print(
                    f"[WARNING] Stats collection took longer than interval ({self.interval_seconds}s)",
                    file=sys.stderr,
                )

            # Use wait() instead of sleep() for more responsive shutdown
            # It will return immediately if stop_event is set
            self.stop_event.wait(timeout=sleep_for)

    def start(self):
        """Start monitoring in a background thread."""
        self.stop_event.clear()
        self.start_time = time.time()
        self.thread = threading.Thread(target=self.monitor_loop, daemon=False)
        self.thread.start()
        print(
            f"[MONITOR] Memory monitoring started for phase: {self.phase_name} (interval: {self.interval_seconds}s)"
        )

    def stop(self):
        """Stop monitoring and print summary."""
        self.stop_event.set()
        self.end_time = time.time()

        if hasattr(self, "thread"):
            self.thread.join(timeout=self.interval_seconds + THREAD_JOIN_BUFFER_SECONDS)

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

        print("\n" + "=" * SEPARATOR_WIDTH)
        print(f"[SUMMARY] Memory Monitoring Summary - Phase: {self.phase_name}")
        print("=" * SEPARATOR_WIDTH)
        print(f"Duration: {duration / 60:.1f} minutes")
        print(f"Samples collected: {len(self.samples)}")
        print()
        print(f"Memory Usage:")
        print(f"  Average: {avg_memory_percent:.1f}%")
        print(f"  Peak: {max_memory_percent:.1f}% ({peak_memory_gb:.2f} GB)")
        print()
        print(f"Swap Usage:")
        print(f"  Average: {avg_swap_percent:.1f}%")
        print(f"  Peak: {max_swap_percent:.1f}% ({peak_swap_gb:.2f} GB)")

        # Warnings
        if max_memory_percent > MEMORY_CRITICAL_PERCENT:
            print(
                f"\n[CRITICAL] Memory usage exceeded {MEMORY_CRITICAL_PERCENT}% during this phase!"
            )
            print(f"   This phase is likely causing out-of-memory issues.")
        elif max_memory_percent > MEMORY_WARNING_PERCENT:
            print(
                f"\n[WARNING] Memory usage exceeded {MEMORY_WARNING_PERCENT}% during this phase."
            )

        if max_swap_percent > SWAP_WARNING_PERCENT:
            print(
                f"\n[WARNING] Significant swap usage detected ({max_swap_percent:.1f}%)"
            )
            print(f"   Consider increasing available memory or reducing parallel jobs.")

        print("=" * SEPARATOR_WIDTH + "\n")

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
        # Determine status indicator
        if max_memory_percent > MEMORY_CRITICAL_PERCENT:
            status = "CRITICAL"
        elif max_memory_percent > MEMORY_WARNING_PERCENT:
            status = "WARNING"
        else:
            status = "OK"

        # Build the summary markdown
        summary = f"## [{status}] Memory Stats: {self.phase_name}\n\n"

        # Main statistics table
        summary += "| Metric | Value |\n"
        summary += "|:-------|------:|\n"
        summary += f"| **Duration** | {duration / 60:.1f} min |\n"
        summary += f"| **Samples Collected** | {len(self.samples)} |\n"
        summary += f"| **Average Memory** | {avg_memory_percent:.1f}% |\n"
        summary += f"| **Peak Memory** | {max_memory_percent:.1f}% ({peak_memory_gb:.2f} GB) |\n"
        summary += f"| **Average Swap** | {avg_swap_percent:.1f}% |\n"
        summary += (
            f"| **Peak Swap** | {max_swap_percent:.1f}% ({peak_swap_gb:.2f} GB) |\n"
        )

        # Add warnings as alerts if needed
        if max_memory_percent > MEMORY_CRITICAL_PERCENT:
            summary += "\n> [!CAUTION]\n"
            summary += f"> Memory usage exceeded {MEMORY_CRITICAL_PERCENT}% during this phase! This phase is likely causing out-of-memory issues.\n"
        elif max_memory_percent > MEMORY_WARNING_PERCENT:
            summary += "\n> [!WARNING]\n"
            summary += f"> Memory usage exceeded {MEMORY_WARNING_PERCENT}% during this phase.\n"

        if max_swap_percent > SWAP_WARNING_PERCENT:
            summary += "\n> [!WARNING]\n"
            summary += f"> Significant swap usage detected ({max_swap_percent:.1f}%). Consider increasing available memory or reducing parallel jobs.\n"

        # Use the centralized function to append to GitHub Actions step summary
        gha_append_step_summary(summary)


def run_command_with_monitoring(
    command: list,
    phase_name: str,
    interval_seconds: float,
    log_file: Optional[Path],
) -> int:
    """Run a command while monitoring memory usage."""
    monitor = MemoryMonitor(
        interval_seconds=interval_seconds,
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
        return_code = EXIT_INTERRUPTED
    except Exception as e:
        print(f"[ERROR] Error executing command: {e}", file=sys.stderr)
        return_code = EXIT_ERROR
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


def setup_signal_handlers(monitor: MemoryMonitor):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}, stopping monitoring...")
        monitor.stop()
        sys.exit(EXIT_SUCCESS)

    # Register handlers for SIGTERM and SIGINT (Ctrl+C)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # On Windows, also handle SIGBREAK (Ctrl+Break)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal_handler)


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
        dest="interval_seconds",
        default=float(
            os.getenv("MEMORY_MONITOR_INTERVAL", str(DEFAULT_INTERVAL_ENV_FALLBACK))
        ),
        help=f"Monitoring interval in seconds (default: {DEFAULT_INTERVAL_ENV_FALLBACK})",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=os.getenv("MEMORY_MONITOR_LOG_FILE"),
        help="Path to write detailed JSON logs",
    )

    parser.add_argument(
        "--stop-signal-file",
        type=Path,
        help="Path to a file that, if it exists, signals the monitor to stop gracefully (useful for Windows)",
    )

    parser.add_argument(
        "--max-runtime",
        type=float,
        dest="max_runtime_seconds",
        help="Maximum runtime in seconds before automatically stopping",
    )

    parser.add_argument(
        "--parent-pid",
        type=int,
        help="PID of parent process to monitor; exit if parent dies",
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
            interval_seconds=args.interval_seconds,
            phase_name=args.phase,
            log_file=args.log_file,
            stop_signal_file=args.stop_signal_file,
            max_runtime_seconds=args.max_runtime_seconds,
            parent_pid=args.parent_pid,
        )

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(monitor)

        monitor.start()

        try:
            # Keep running until interrupted
            stopped_via_signal_file = False
            while not monitor.stop_event.is_set():
                # Check for stop signal file
                if monitor.stop_signal_file and monitor.stop_signal_file.exists():
                    print("\n[STOP] Stop signal file detected, stopping...")
                    monitor.stop()
                    # Clean up the signal file
                    try:
                        monitor.stop_signal_file.unlink()
                    except:
                        pass
                    stopped_via_signal_file = True
                    break

                # Use wait() for more responsive shutdown
                monitor.stop_event.wait(timeout=STOP_CHECK_INTERVAL_SECONDS)

            # If we exited the loop because stop_event was set by a signal handler
            # (not by the stop signal file check above), we need to call stop() to print the summary
            if monitor.stop_event.is_set() and not stopped_via_signal_file:
                print("\n[STOP] Stop event detected, finalizing...")
                monitor.stop()
                # Clean up the signal file if it exists
                if monitor.stop_signal_file and monitor.stop_signal_file.exists():
                    try:
                        monitor.stop_signal_file.unlink()
                    except:
                        pass
        except KeyboardInterrupt:
            print("\n[STOP] Stopping background monitoring...")
            monitor.stop()

        return EXIT_SUCCESS

    elif args.command:
        # Command execution mode
        return_code = run_command_with_monitoring(
            command=args.command,
            phase_name=args.phase,
            interval_seconds=args.interval_seconds,
            log_file=args.log_file,
        )
        return return_code

    else:
        # One-shot monitoring
        monitor = MemoryMonitor(
            interval_seconds=args.interval_seconds,
            phase_name=args.phase,
            log_file=args.log_file,
        )
        stats = monitor.get_memory_stats()
        monitor.log_stats(stats)
        return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
