#!/usr/bin/env python3
"""Tests for memory monitoring functionality."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psutil
except ImportError:
    print("WARNING: psutil not installed, skipping memory monitor tests")
    sys.exit(0)

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

    print("[PASS] Memory stats collection test passed")


def test_monitoring_loop():
    """Test that monitoring loop runs and collects samples."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        log_file = Path(f.name)

    try:
        monitor = MemoryMonitor(
            interval=0.5,  # Fast interval for testing
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

        print("[PASS] Monitoring loop test passed")

    finally:
        if log_file.exists():
            log_file.unlink()


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

        print("[PASS] Analysis script test passed")


def main():
    """Run all tests."""
    # Only run tests if ACTIONS_RUNNER_DEBUG is set to true
    if os.environ.get("ACTIONS_RUNNER_DEBUG", "").lower() != "true":
        print("Skipping memory monitor tests (ACTIONS_RUNNER_DEBUG not set to true)")
        return 0

    print("Running memory monitor tests...\n")

    tests = [
        test_memory_stats_collection,
        test_monitoring_loop,
        test_analysis_script,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__} failed: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"[FAIL] {test.__name__} error: {e}")
            failed.append(test.__name__)

    print(f"\n{'='*60}")
    if failed:
        print(f"[FAIL] {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1
    else:
        print(f"[PASS] All {len(tests)} tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
