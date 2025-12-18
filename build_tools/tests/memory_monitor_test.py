#!/usr/bin/env python3
"""Tests for memory monitoring functionality."""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_monitor import MemoryMonitor


def test_memory_stats_collection():
    """Test that memory stats can be collected."""
    monitor = MemoryMonitor(phase_name="Test Phase")
    stats = monitor.get_memory_stats()

    # Verify all expected keys are present
    expected_keys = [
        "timestamp",
        "phase",
        "total_memory_gb",
        "available_memory_gb",
        "used_memory_gb",
        "memory_percent",
        "free_memory_gb",
        "peak_memory_gb",
        "peak_swap_gb",
        "total_swap_gb",
        "used_swap_gb",
        "swap_percent",
        "process_memory_gb",
        "children_memory_gb",
        "total_process_memory_gb",
    ]

    for key in expected_keys:
        assert key in stats, f"Missing key: {key}"

    # Verify reasonable values
    assert stats["total_memory_gb"] > 0, "Total memory should be positive"
    assert 0 <= stats["memory_percent"] <= 100, "Memory percent should be 0-100"
    assert 0 <= stats["swap_percent"] <= 100, "Swap percent should be 0-100"


def test_monitoring_loop():
    """Test that monitoring loop runs and collects samples."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        log_file = Path(f.name)

    try:
        monitor = MemoryMonitor(
            interval_seconds=0.5,  # Fast interval for testing
            phase_name="Test Loop",
            log_file=log_file,
        )

        monitor.start()
        time.sleep(2)  # Let it collect a few samples
        monitor.stop()

        # Verify samples were collected
        assert (
            len(monitor.samples) >= 3
        ), f"Expected at least 3 samples, got {len(monitor.samples)}"

        # Verify log file was written
        assert log_file.exists(), "Log file should exist"

        # Verify log file contains valid JSON
        with open(log_file, "r") as f:
            lines = f.readlines()
            assert len(lines) >= 3, f"Expected at least 3 log lines, got {len(lines)}"

            for line in lines:
                data = json.loads(line)
                assert "phase" in data
                assert data["phase"] == "Test Loop"

    finally:
        if log_file.exists():
            log_file.unlink()


def test_stop_event_mechanism():
    """Test that threading.Event is used for stop signaling."""
    monitor = MemoryMonitor(interval_seconds=1.0, phase_name="Event Test")

    # Event should not be set initially
    assert not monitor.stop_event.is_set(), "Stop event should not be set initially"

    # Start monitoring
    monitor.start()
    time.sleep(0.5)

    # Event should still not be set while running
    assert not monitor.stop_event.is_set(), "Stop event should not be set while running"

    # Stop monitoring
    monitor.stop()

    # Event should be set after stop
    assert monitor.stop_event.is_set(), "Stop event should be set after stop"


def test_stop_event_responsive_shutdown():
    """Test that Event.wait() makes shutdown responsive."""
    monitor = MemoryMonitor(
        interval_seconds=30.0,  # Very long interval
        phase_name="Responsive Test",
    )

    monitor.start()
    time.sleep(0.5)  # Very short wait

    # Measure how long it takes to stop
    start = time.time()
    monitor.stop()
    stop_duration = time.time() - start

    # Should stop quickly, not wait for the full 30s interval
    assert (
        stop_duration < 2.0
    ), f"Stop took {stop_duration:.2f}s, should be < 2s (responsive shutdown)"

    # Verify some samples were collected
    assert len(monitor.samples) >= 1, "Should have collected at least 1 sample"


def test_stop_signal_file_with_event():
    """Test that stop signal file detection works with threading.Event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stop_file = Path(tmpdir) / "stop.signal"

        monitor = MemoryMonitor(
            interval_seconds=1.0,
            phase_name="Signal File Test",
            stop_signal_file=stop_file,
        )

        monitor.start()
        time.sleep(2.5)  # Let it collect a few samples

        # Create stop signal file
        stop_file.touch()

        # Give it time to detect the file
        time.sleep(2)

        # The monitoring thread should have set stop_event
        assert (
            monitor.stop_event.is_set()
        ), "Stop event should be set after signal file detected"

        # Call stop to ensure summary is printed
        monitor.stop()

        # Verify samples were collected
        assert len(monitor.samples) >= 2, "Should have collected multiple samples"


def test_analysis_script():
    """Test that analysis script can process logs."""
    analysis_script = Path(__file__).parent.parent / "analyze_memory_logs.py"
    assert analysis_script.exists(), f"Analysis script not found: {analysis_script}"

    # Create test logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        log_file = log_dir / "test_phase.jsonl"

        # Write some test data
        test_data = [
            {
                "timestamp": "2025-11-21T10:00:00",
                "phase": "Test Phase",
                "total_memory_gb": 32.0,
                "available_memory_gb": 8.0,
                "used_memory_gb": 24.0,
                "memory_percent": 75.0,
                "free_memory_gb": 4.0,
                "total_swap_gb": 8.0,
                "used_swap_gb": 1.0,
                "swap_percent": 12.5,
                "process_memory_gb": 2.0,
                "children_memory_gb": 1.0,
                "total_process_memory_gb": 3.0,
            }
            for _ in range(5)
        ]

        with open(log_file, "w") as f:
            for data in test_data:
                f.write(json.dumps(data) + "\n")

        # Run analysis
        result = subprocess.run(
            [sys.executable, str(analysis_script), "--log-dir", str(log_dir)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Analysis failed: {result.stderr}"
        assert "MEMORY USAGE ANALYSIS REPORT" in result.stdout
        assert "Test Phase" in result.stdout
