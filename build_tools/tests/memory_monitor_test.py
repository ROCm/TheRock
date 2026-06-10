#!/usr/bin/env python3
"""Tests for memory_monitor.py"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_monitor import ResourceMonitor, get_storage_info, get_thread_info


def test_collect_stats():
    """Test that resource stats are collected correctly."""
    monitor = ResourceMonitor(phase="Test")
    stats = monitor._collect_stats()

    assert "timestamp" in stats
    assert stats["mem_total_gb"] > 0
    assert 0 <= stats["mem_percent"] <= 100
    assert 0 <= stats["swap_percent"] <= 100
    assert "cpu_count" in stats


def test_storage_info():
    """Test storage info collection."""
    info = get_storage_info(".")
    assert info["total_gb"] > 0
    assert info["free_gb"] >= 0
    assert 0 <= info["percent"] <= 100


def test_thread_info():
    """Test thread/CPU info collection."""
    info = get_thread_info()
    assert info["cpu_count"] > 0


def test_monitor_start_stop():
    """Test starting and stopping the monitor."""
    monitor = ResourceMonitor(interval=0.1, phase="Test")

    monitor.start()
    time.sleep(0.5)
    monitor.stop()

    assert len(monitor.samples) >= 1
    assert monitor.stop_event.is_set()


def test_responsive_shutdown():
    """Test that stop() returns quickly even with long interval."""
    monitor = ResourceMonitor(interval=60.0, phase="Test")

    monitor.start()
    time.sleep(0.2)

    start = time.time()
    monitor.stop()
    elapsed = time.time() - start

    assert elapsed < 3.0, f"Stop took {elapsed:.1f}s, should be < 3s"
