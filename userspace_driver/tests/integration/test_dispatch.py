"""Integration tests for compute dispatch.

These tests require a pre-compiled GPU kernel (.co file) and GPU hardware.
"""

from pathlib import Path

import pytest

from tests.integration.conftest import requires_gpu

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FILL_KERNEL = FIXTURES_DIR / "fill_kernel_gfx942.co"


@requires_gpu
class TestComputeDispatch:
    """Test compute kernel dispatch."""

    @pytest.mark.skipif(
        not FILL_KERNEL.exists(),
        reason=f"Requires pre-compiled kernel at {FILL_KERNEL}",
    )
    def test_dispatch_fill_kernel(self, amd_device):
        """Dispatch a kernel that fills a buffer with a constant value."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        program = amd_device.load_program(str(FILL_KERNEL))

        # Verify kernel was loaded
        assert program.name == "fill_kernel"
        assert program.kernarg_size > 0

        # Allocate output buffer (256 uint32s = 1024 bytes)
        num_elements = 256
        out_buf = amd_device.alloc(num_elements * 4, location="vram")
        out_buf.fill(0x00)

        # Create compute queue and timeline
        backend = amd_device.backend
        queue = backend.create_compute_queue()
        timeline = TimelineSemaphore(backend)

        # Dispatch: fill_kernel(out, val=0xDEADBEEF)
        fill_value = 0xDEADBEEF
        program.dispatch(
            queue,
            grid=(num_elements // 64, 1, 1),  # workgroups
            block=(64, 1, 1),  # threads per workgroup
            args=[out_buf, fill_value],
            timeline=timeline,
        )

        # Wait for completion
        timeline.cpu_wait(timeout_ms=5000)

        # Verify output
        import struct

        data = out_buf.read(num_elements * 4)
        values = struct.unpack(f"<{num_elements}I", data)
        assert values[0] == fill_value, f"values[0] = 0x{values[0]:08x}"
        assert values[63] == fill_value, f"values[63] = 0x{values[63]:08x}"
        assert values[255] == fill_value, f"values[255] = 0x{values[255]:08x}"
        assert all(v == fill_value for v in values), (
            f"Not all values match: first mismatch at index "
            f"{next(i for i, v in enumerate(values) if v != fill_value)}"
        )

        # Cleanup
        program.free()
        timeline.destroy()
        backend.destroy_queue(queue)
        out_buf.free()

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

    @pytest.mark.skipif(
        not FILL_KERNEL.exists(),
        reason=f"Requires pre-compiled kernel at {FILL_KERNEL}",
    )
    def test_dispatch_small(self, amd_device):
        """Dispatch with a single workgroup of 64 threads."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        program = amd_device.load_program(str(FILL_KERNEL))

        out_buf = amd_device.alloc(64 * 4, location="vram")
        out_buf.fill(0x00)

        backend = amd_device.backend
        queue = backend.create_compute_queue()
        timeline = TimelineSemaphore(backend)

        program.dispatch(
            queue,
            grid=(1, 1, 1),
            block=(64, 1, 1),
            args=[out_buf, 42],
            timeline=timeline,
        )

        timeline.cpu_wait(timeout_ms=5000)

        import struct

        data = out_buf.read(64 * 4)
        values = struct.unpack("<64I", data)
        assert values[0] == 42
        assert values[63] == 42

        program.free()
        timeline.destroy()
        backend.destroy_queue(queue)
        out_buf.free()
