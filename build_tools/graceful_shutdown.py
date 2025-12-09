#!/usr/bin/env python3
"""Graceful process shutdown utility.

This script provides a cross-platform way to gracefully terminate a process,
allowing it to run cleanup code before exiting. It sends appropriate termination
signals and waits for the process to exit before optionally forcing termination.

Usage:
    python graceful_shutdown.py <PID> [--timeout SECONDS] [--force]

Examples:
    # Gracefully stop a process with 10 second timeout
    python graceful_shutdown.py 12345 --timeout 10

    # Try graceful shutdown, force kill after 5 seconds if needed
    python graceful_shutdown.py 12345 --timeout 5 --force
"""

import argparse
import sys
import time
import platform
import psutil


def graceful_shutdown(
    pid: int,
    timeout_seconds: float = 10.0,
    force_on_timeout: bool = False,
    verbose: bool = True,
) -> bool:
    """Gracefully shutdown a process by PID.

    This function sends a termination signal to the process and waits for it
    to exit gracefully. On Windows, it uses SIGTERM. On Unix, it uses SIGTERM
    which triggers KeyboardInterrupt in Python processes.

    Args:
        pid: Process ID to terminate
        timeout_seconds: How long to wait for graceful shutdown (default: 10.0)
        force_on_timeout: If True, force kill the process if timeout is exceeded
        verbose: If True, print status messages to stdout

    Returns:
        True if process was terminated successfully, False otherwise
    """
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        if verbose:
            print(f"Process {pid} does not exist (already terminated)")
        return True
    except psutil.AccessDenied:
        print(f"ERROR: Access denied to process {pid}")
        return False

    if verbose:
        print(f"Attempting graceful shutdown of process {pid} (timeout: {timeout_seconds}s)")

    try:
        # Send termination signal
        # On Windows, terminate() sends SIGTERM
        # On Unix, terminate() sends SIGTERM which Python's signal handler converts to KeyboardInterrupt
        process.terminate()

        if verbose:
            print(f"Sent termination signal to process {pid}")

        # Wait for the process to exit gracefully
        try:
            process.wait(timeout=timeout_seconds)
            if verbose:
                print(f"Process {pid} terminated gracefully")
            return True
        except psutil.TimeoutExpired:
            if verbose:
                print(f"WARNING: Process {pid} did not terminate within {timeout_seconds}s")

            if force_on_timeout:
                if verbose:
                    print(f"Force killing process {pid}")
                process.kill()
                process.wait(timeout=5)
                if verbose:
                    print(f"Process {pid} was force killed")
                return True
            else:
                if verbose:
                    print(f"Process {pid} is still running")
                return False

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
        "--force",
        action="store_true",
        help="Force kill the process if it doesn't terminate gracefully within timeout",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (errors still printed)",
    )

    args = parser.parse_args()

    success = graceful_shutdown(
        pid=args.pid,
        timeout_seconds=args.timeout,
        force_on_timeout=args.force,
        verbose=not args.quiet,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

