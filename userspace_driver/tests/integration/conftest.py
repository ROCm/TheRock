"""Integration test fixtures - skip if no GPU hardware available."""

import os
from pathlib import Path

import pytest


def _has_kfd_device() -> bool:
    """Check if /dev/kfd exists and is accessible."""
    kfd_path = Path("/dev/kfd")
    if not kfd_path.exists():
        return False
    try:
        fd = os.open(str(kfd_path), os.O_RDWR)
        os.close(fd)
        return True
    except OSError:
        return False


def _has_gpu_topology() -> bool:
    """Check if any GPU nodes exist in KFD topology."""
    topology = Path("/sys/devices/virtual/kfd/kfd/topology/nodes")
    if not topology.exists():
        return False
    for node_dir in topology.iterdir():
        if not node_dir.is_dir():
            continue
        props_file = node_dir / "properties"
        if not props_file.exists():
            continue
        text = props_file.read_text()
        if "simd_count" in text:
            for line in text.splitlines():
                if line.startswith("simd_count") and int(line.split()[1]) > 0:
                    return True
    return False


requires_gpu = pytest.mark.skipif(
    not (_has_kfd_device() and _has_gpu_topology()),
    reason="No AMD GPU with KFD support available",
)


@pytest.fixture
def amd_device():
    """Create and yield an AMDDevice, closing it after the test."""
    from amd_gpu_driver import AMDDevice

    dev = AMDDevice()
    yield dev
    dev.close()
