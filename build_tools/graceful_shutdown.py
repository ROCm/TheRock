#!/usr/bin/env python3
"""Graceful process shutdown utility.

This script provides a cross-platform way to gracefully terminate a process,
allowing it to run cleanup code before exiting. It sends appropriate termination
signals and waits for the process to exit, force killing if the timeout is exceeded.

Usage:
    python graceful_shutdown.py <PID> [--timeout SECONDS]

Examples:
    # Gracefully stop a process with 10 second timeout (default), force kill if needed
    python graceful_shutdown.py 12345 --timeout 10
"""

import argparse
import platform
import sys
import time
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
        stop_signal_file: Optional path to a stop signal file (for Windows compatibility)

    Returns:
        True if process was terminated successfully, False otherwise
    """
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

    # Track remaining timeout
    remaining_timeout = timeout_seconds
    start_time = time.time()

    try:
        # On Windows, if a stop_signal_file is provided, create it first
        # This allows the process to detect it and shutdown gracefully
        # On Linux/Unix, we skip this and send SIGTERM directly
        is_windows = platform.system() == "Windows"

        if stop_signal_file and not is_windows and verbose:
            print(f"Skipping stop signal file on Linux - using SIGTERM instead")

        if stop_signal_file and is_windows:
            from pathlib import Path

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
                            # Clean up the signal file
                            try:
                                stop_file.unlink()
                            except OSError:
                                pass
                            return True
                    except psutil.NoSuchProcess:
                        if verbose:
                            print(f"Process {pid} has exited")
                        # Clean up the signal file
                        try:
                            stop_file.unlink()
                        except OSError:
                            pass
                        return True

                    time.sleep(0.5)
                    wait_time += 0.5

                # If we get here, the stop signal file didn't work
                # Update remaining timeout
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout_seconds - elapsed)

                if verbose:
                    print(
                        f"Process did not respond to stop signal file within {timeout_seconds}s"
                    )
                # Clean up the signal file
                try:
                    stop_file.unlink()
                except OSError:
                    pass

            except Exception as e:
                if verbose:
                    print(f"Warning: Could not create/use stop signal file: {e}")
                # Update remaining timeout even if exception occurred
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout_seconds - elapsed)

        # Send termination signal as fallback (only if process is still running)
        try:
            if process.is_running():
                # On Linux/Unix: terminate() sends SIGTERM which triggers signal handlers
                # On Windows: This is a fallback if stop_signal_file didn't work
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

            if verbose:
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
        help="Path to a stop signal file to create before terminating (useful for Windows)",
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
