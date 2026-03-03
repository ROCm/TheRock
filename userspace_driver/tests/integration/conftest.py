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


def _gpu_count() -> int:
    """Count available AMD GPU nodes from topology."""
    try:
        from amd_gpu_driver.topology import discover_gpu_nodes

        return len(discover_gpu_nodes())
    except Exception:
        return 0


requires_gpu = pytest.mark.skipif(
    not (_has_kfd_device() and _has_gpu_topology()),
    reason="No AMD GPU with KFD support available",
)

requires_multi_gpu = pytest.mark.skipif(
    _gpu_count() < 2,
    reason="Requires at least 2 AMD GPUs",
)

requires_3_gpus = pytest.mark.skipif(
    _gpu_count() < 3,
    reason="Requires at least 3 AMD GPUs",
)


@pytest.fixture
def amd_device():
    """Create and yield an AMDDevice, closing it after the test."""
    from amd_gpu_driver import AMDDevice

    dev = AMDDevice()
    yield dev
    dev.close()


@pytest.fixture
def multi_gpu_context():
    """Create and yield a MultiGPUContext, closing it after the test."""
    from amd_gpu_driver.multi_gpu import MultiGPUContext

    ctx = MultiGPUContext()
    yield ctx
    ctx.close()
