"""Integration tests for compute dispatch.

These tests require a pre-compiled GPU kernel (.co file).
Skip if no test kernel is available.
"""

import pytest

from tests.integration.conftest import requires_gpu


@requires_gpu
class TestComputeDispatch:
    """Test compute kernel dispatch."""

    @pytest.mark.skip(reason="Requires pre-compiled .co kernel binary")
    def test_dispatch_fill_kernel(self, amd_device):
        """Dispatch a kernel that fills a buffer with a constant value."""
        # This test requires a compiled GPU kernel binary.
        # Example kernel (HIP/OpenCL):
        #   __global__ void fill(uint32_t* out, uint32_t val) {
        #       out[threadIdx.x] = val;
        #   }
        #
        # Compile with: hipcc --genco -o fill.co fill.hip --offload-arch=gfx942
        pass

    def test_submit_nop_packets(self, amd_device):
        """Submit NOP packets to verify queue submission works."""
        from amd_gpu_driver.commands.pm4 import PM4PacketBuilder

        backend = amd_device.backend
        queue = backend.create_compute_queue()

        # Build NOP packets
        pm4 = PM4PacketBuilder()
        pm4.nop(4)

        # Submit
        backend.submit_packets(queue, pm4.build())

        backend.destroy_queue(queue)

    def test_signal_event(self, amd_device):
        """Test GPU signal event creation and destroy."""
        backend = amd_device.backend
        signal = backend.create_signal()
        assert signal.event_id > 0
        backend.destroy_signal(signal)
