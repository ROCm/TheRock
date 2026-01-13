#!/usr/bin/env python3
"""Graceful process shutdown utility.

This script provides a way to gracefully terminate a process on Windows,
allowing it to run cleanup code before exiting. It sends appropriate termination
signals and waits for the process to exit, force killing if the timeout is exceeded.

Usage:
    python graceful_shutdown.py <PID> [--timeout SECONDS]

Examples:
    # Gracefully stop a process with 10 second timeout (default), force kill if needed
    python graceful_shutdown.py 12345 --timeout 10
"""

import argparse
import sys
import time
from pathlib import Path
import psutil


def graceful_shutdown(
    pid: int,
    timeout_seconds: float = 10.0,
    verbose: bool = True,
    stop_signal_file: str = None,
) -> bool:
    """Gracefully shutdown a process by PID.

    Args:
        pid: Process ID to terminate
        timeout_seconds: How long to wait for graceful shutdown (default: 10.0)
        verbose: If True, print status messages to stdout
        stop_signal_file: Optional path to a stop signal file

    Returns:
        True if process was terminated successfully, False otherwise
    """

    def cleanup_stop_file(stop_file: Path) -> None:
        """Helper to safely cleanup stop signal file."""
        try:
            stop_file.unlink()
        except OSError:
            pass

    def update_remaining_timeout(start: float, total: float) -> float:
        """Calculate remaining timeout based on elapsed time."""
        elapsed = time.time() - start
        return max(0, total - elapsed)

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        if verbose:
            print(f"Process {pid} has already terminated (this is fine)")
        return True
    except psutil.AccessDenied:
        print(f"ERROR: Access denied to process {pid}")
        return False

    if verbose:
        print(
            f"Attempting graceful shutdown of process {pid} (timeout: {timeout_seconds}s)"
        )

    start_time = time.time()

    try:
        # If a stop_signal_file is provided, create it first
        # This allows the process to detect it and shutdown gracefully
        if stop_signal_file:
            stop_file = Path(stop_signal_file)
            try:
                stop_file.touch()
                if verbose:
                    print(f"Created stop signal file: {stop_signal_file}")
                    print(f"Waiting for process to detect stop signal file...")

                # Wait for the process to detect the file and exit gracefully
                # Check every 0.5 seconds
                wait_time = 0
                while wait_time < timeout_seconds:
                    try:
                        # Check if process is still running
                        if not process.is_running():
                            if verbose:
                                print(
                                    f"Process {pid} exited gracefully via stop signal file"
                                )
                            cleanup_stop_file(stop_file)
                            return True
                    except psutil.NoSuchProcess:
                        if verbose:
                            print(f"Process {pid} has exited")
                        cleanup_stop_file(stop_file)
                        return True

                    time.sleep(0.5)
                    wait_time += 0.5

                # If we get here, the stop signal file didn't work
                if verbose:
                    print(
                        f"Process did not respond to stop signal file within {timeout_seconds}s"
                    )
                cleanup_stop_file(stop_file)

            except Exception as e:
                if verbose:
                    print(f"Warning: Could not create/use stop signal file: {e}")

        # Send termination signal as fallback (only if process is still running)
        remaining_timeout = update_remaining_timeout(start_time, timeout_seconds)

        try:
            if process.is_running():
                process.terminate()
                if verbose:
                    print(f"Sent termination signal to process {pid}")
        except psutil.NoSuchProcess:
            if verbose:
                print(f"Process {pid} already exited")
            return True

        # Wait for the process to exit gracefully
        try:
            process.wait(timeout=remaining_timeout)
            if verbose:
                print(f"Process {pid} terminated gracefully")
            return True
        except psutil.TimeoutExpired:
            total_elapsed = time.time() - start_time
            if verbose:
                print(
                    f"WARNING: Process {pid} did not terminate within {timeout_seconds}s (total elapsed: {total_elapsed:.1f}s)"
                )
                print(f"Force killing process {pid}")
            process.kill()
            process.wait(timeout=5)
            if verbose:
                print(f"Process {pid} was force killed")
            return True

    except psutil.NoSuchProcess:
        if verbose:
            print(f"Process {pid} terminated before signal could be sent")
        return True
    except psutil.AccessDenied:
        print(f"ERROR: Access denied when trying to terminate process {pid}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error terminating process {pid}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Gracefully terminate a process by PID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "pid",
        type=int,
        help="Process ID to terminate",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds to wait for graceful shutdown (default: 10.0)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (errors still printed)",
    )

    parser.add_argument(
        "--stop-signal-file",
        type=str,
        help="Path to a stop signal file to create before terminating",
    )

    args = parser.parse_args()

    success = graceful_shutdown(
        pid=args.pid,
        timeout_seconds=args.timeout,
        verbose=not args.quiet,
        stop_signal_file=args.stop_signal_file,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
