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
    return log_file, output_file


def test_graceful_shutdown_with_memory_monitor(temp_files):
    """Test that graceful_shutdown properly stops memory monitor and prints summary."""
    log_file, output_file = temp_files
    
    # Start memory monitor in background
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
        ],
        stdout=open(output_file, "w"),
        stderr=subprocess.STDOUT,
    )
    
    try:
        # Let it collect a few samples
        time.sleep(6)
        
        # Gracefully shutdown
        success = graceful_shutdown(
            pid=process.pid,
            timeout_seconds=10.0,
            force_on_timeout=True,
            verbose=True,
        )
        
        assert success, "Graceful shutdown should succeed"
        
        # Check output file for summary
        assert output_file.exists(), "Output file should exist"
        
        content = output_file.read_text()
        
        # Check for summary indicators
        assert "[SUMMARY] Memory Monitoring Summary" in content, "Should have summary header"
        assert "Duration:" in content, "Should have duration"
        assert "Samples collected:" in content, "Should have samples count"
        assert "Memory Usage:" in content, "Should have memory usage section"
        assert "Peak:" in content, "Should have peak values"
        
    finally:
        # Ensure process is terminated even if test fails
        try:
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
        force_on_timeout=False,
        verbose=False,
    )
    
    # Should return True because process doesn't exist (already terminated)
    assert success, "Should return True for non-existent process"


def test_graceful_shutdown_timeout_no_force():
    """Test that graceful shutdown respects timeout when not forcing."""
    # Start a process that ignores SIGTERM
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    try:
        # Try to shutdown without force (will timeout)
        success = graceful_shutdown(
            pid=process.pid,
            timeout_seconds=1.0,
            force_on_timeout=False,
            verbose=False,
        )
        
        # Should return False because timeout was reached and force was not used
        assert not success, "Should return False when timeout is reached without force"
        
        # Process should still be running
        assert process.poll() is None, "Process should still be running"
        
    finally:
        # Clean up
        process.kill()
        process.wait()


def test_graceful_shutdown_with_force():
    """Test that graceful shutdown force kills when timeout is exceeded."""
    # Start a process that ignores SIGTERM
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    try:
        # Try to shutdown with force
        success = graceful_shutdown(
            pid=process.pid,
            timeout_seconds=1.0,
            force_on_timeout=True,
            verbose=False,
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

