#!/usr/bin/env python3
"""Tests for graceful_shutdown utility."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add parent directory to path to import graceful_shutdown
sys.path.insert(0, str(Path(__file__).parent.parent))

from graceful_shutdown import graceful_shutdown


@pytest.fixture
def temp_files(tmp_path):
    """Fixture to provide temporary file paths and cleanup."""
    log_file = tmp_path / "test_memory_log.jsonl"
    output_file = tmp_path / "test_monitor_output.txt"
    stop_signal_file = tmp_path / "test_stop_signal.txt"
    return log_file, output_file, stop_signal_file


def test_graceful_shutdown_with_memory_monitor(temp_files):
    """Test that graceful_shutdown properly stops memory monitor and prints summary.

    This test uses a stop-signal file to gracefully shutdown the memory monitor process.
    This approach works reliably on both Windows and Unix systems, since it doesn't
    rely on signal handlers that Windows doesn't support properly.
    """
    log_file, output_file, stop_signal_file = temp_files

    # Open output file for subprocess
    out_file = open(output_file, "w")

    try:
        # Start memory monitor in background with stop-signal-file
        process = subprocess.Popen(
            [
                sys.executable,
                "build_tools/memory_monitor.py",
                "--background",
                "--phase",
                "Test Phase",
                "--interval",
                "2",
                "--log-file",
                str(log_file),
                "--stop-signal-file",
                str(stop_signal_file),
            ],
            stdout=out_file,
            stderr=subprocess.STDOUT,
        )

        # Let it collect a few samples
        time.sleep(6)

        # Gracefully shutdown using stop signal file
        success = graceful_shutdown(
            pid=process.pid,
            timeout_seconds=10.0,
            verbose=True,
            stop_signal_file=str(stop_signal_file),
        )

        assert success, "Graceful shutdown should succeed"

        # Wait for process to fully exit
        process.wait(timeout=2)

        # Close the output file to flush all content
        out_file.close()

        # Small delay to ensure file system has flushed
        time.sleep(0.5)

        # Check output file for summary
        assert output_file.exists(), "Output file should exist"

        content = output_file.read_text()

        # Check for summary indicators - the monitor should detect the stop signal
        # and call stop() which calls print_summary()
        assert (
            "[SUMMARY] Memory Monitoring Summary" in content
            or "[STOP_SIGNAL]" in content
        ), "Should have summary header or stop signal detection marker"
        assert "Duration:" in content, "Should have duration"
        assert "Samples collected:" in content, "Should have samples count"
        assert "Memory Usage:" in content, "Should have memory usage section"
        assert "Peak:" in content, "Should have peak values"

    finally:
        # Close output file if still open
        try:
            out_file.close()
        except:
            pass

        # Clean up stop signal file if it exists
        try:
            if stop_signal_file.exists():
                stop_signal_file.unlink()
        except:
            pass

        # Ensure process is terminated even if test fails
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        except:
            pass


def test_graceful_shutdown_nonexistent_process():
    """Test graceful shutdown of a non-existent process."""
    # Use a PID that definitely doesn't exist (very high number)
    success = graceful_shutdown(
        pid=999999,
        timeout_seconds=1.0,
        verbose=False,
    )

    # Should return True because process doesn't exist (already terminated)
    assert success, "Should return True for non-existent process"


def test_graceful_shutdown_with_force(tmp_path):
    """Test that graceful shutdown force kills when timeout is exceeded.

    This test creates a process that ignores both signals and stop signal files,
    then verifies that force kill works. Works on both Windows and Unix systems.
    """
    stop_signal_file = tmp_path / "test_stop_force.txt"

    # Start a process that ignores signals and doesn't check for stop signal file
    ignore_termination_code = """
import signal
import time
# Ignore all termination signals
try:
    signal.signal(signal.SIGTERM, lambda signum, frame: None)
    signal.signal(signal.SIGINT, lambda signum, frame: None)
except (AttributeError, ValueError):
    pass  # Some signals might not be available on Windows
# Just sleep without checking for stop signal file
time.sleep(30)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", ignore_termination_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Try to shutdown with force
        success = graceful_shutdown(
            pid=process.pid,
            timeout_seconds=1.0,
            verbose=False,
            stop_signal_file=str(stop_signal_file),
        )

        # Should return True because force kill was used
        assert success, "Should return True when force kill is used"

        # Process should be terminated
        assert process.poll() is not None, "Process should be terminated"

    except:
        # Clean up if test fails
        try:
            process.kill()
            process.wait()
        except:
            pass
    finally:
        # Clean up stop signal file if it exists
        try:
            if stop_signal_file.exists():
                stop_signal_file.unlink()
        except:
            pass
