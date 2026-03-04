"""Integration tests for compute-communication overlap.

Tests verify that compute (PM4 kernel dispatch) and communication
(SDMA / XGMI copy) can run concurrently on the same or different GPUs,
and that GPU-side synchronization primitives (WAIT_REG_MEM, poll_regmem)
correctly enforce ordering between engines.

Overlap dimensions tested:
  1. Same-GPU: compute queue + SDMA queue run in parallel
  2. Cross-GPU: compute on GPU_A while P2P XGMI copy to GPU_B
  3. GPU-side sync: SDMA waits for compute (poll_regmem) and vice versa
  4. Pipeline: double-buffering patterns with compute + send overlap
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from amd_gpu_driver.backends.base import MemoryLocation
from tests.integration.conftest import (
    requires_3_gpus,
    requires_gpu,
    requires_multi_gpu,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FILL_KERNEL = FIXTURES_DIR / "fill_kernel_gfx942.co"

requires_fill_kernel = pytest.mark.skipif(
    not FILL_KERNEL.exists(),
    reason=f"Requires pre-compiled kernel at {FILL_KERNEL}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch_fill(dev, queue, timeline, out_buf, value, num_elements=256):
    """Dispatch fill_kernel: fill out_buf with value, signal timeline."""
    program = dev.load_program(str(FILL_KERNEL))
    program.dispatch(
        queue,
        grid=(num_elements // 64, 1, 1),
        block=(64, 1, 1),
        args=[out_buf, value],
        timeline=timeline,
    )
    return program


def _verify_fill(buf, value, num_elements=256):
    """Read back and verify all uint32 values match expected."""
    data = buf.read(num_elements * 4)
    values = struct.unpack(f"<{num_elements}I", data)
    assert all(v == value for v in values), (
        f"Expected 0x{value:08X}, first mismatch at index "
        f"{next(i for i, v in enumerate(values) if v != value)}: "
        f"got 0x{values[next(i for i, v in enumerate(values) if v != value)]:08X}"
    )


def _get_xgmi_or_sdma_queue(backend):
    """Create an XGMI SDMA queue if available, else regular SDMA."""
    node = backend.node
    if node is not None and node.num_sdma_xgmi_engines > 0:
        return backend.create_xgmi_sdma_queue()
    return backend.create_sdma_queue()


# ---------------------------------------------------------------------------
# TestSameGPUOverlap — compute + SDMA on same GPU
# ---------------------------------------------------------------------------


@requires_gpu
class TestSameGPUOverlap:
    """Compute and SDMA engines on the same GPU run concurrently."""

    @requires_fill_kernel
    def test_compute_and_sdma_overlap(self, amd_device):
        """Dispatch fill_kernel while an SDMA copy runs on the same GPU."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend

        # Compute side
        num_elements = 256
        compute_out = amd_device.alloc(num_elements * 4, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        # SDMA side
        sdma_src = amd_device.alloc(4096, location="vram")
        sdma_dst = amd_device.alloc(4096, location="vram")
        sdma_src.fill(0xCC)
        sdma_dst.fill(0x00)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        # Submit both without waiting
        fill_value = 0xDEADBEEF
        program = _dispatch_fill(
            amd_device, compute_queue, compute_tl, compute_out, fill_value
        )

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(sdma_dst.gpu_addr, sdma_src.gpu_addr, 4096)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend.submit_packets(sdma_queue, sdma.build())

        # Wait for both
        compute_tl.cpu_wait(timeout_ms=5000)
        sdma_tl.cpu_wait(timeout_ms=5000)

        # Verify both
        _verify_fill(compute_out, fill_value)
        assert sdma_dst.read(16) == b"\xCC" * 16

        # Cleanup
        program.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    @requires_fill_kernel
    def test_compute_while_sdma_large_copy(self, amd_device):
        """Kernel dispatch completes even while SDMA is doing a 16 MB copy."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend

        # Large SDMA copy (submitted first so it occupies the engine)
        big_size = 16 * 1024 * 1024
        sdma_src = amd_device.alloc(big_size, location="vram")
        sdma_dst = amd_device.alloc(big_size, location="vram")
        sdma_src.fill(0xAA)
        sdma_dst.fill(0x00)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(sdma_dst.gpu_addr, sdma_src.gpu_addr, big_size)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend.submit_packets(sdma_queue, sdma.build())

        # Now dispatch kernel on compute queue (should run in parallel)
        num_elements = 256
        compute_out = amd_device.alloc(num_elements * 4, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        fill_value = 0xCAFEBABE
        program = _dispatch_fill(
            amd_device, compute_queue, compute_tl, compute_out, fill_value
        )

        # Wait for both
        compute_tl.cpu_wait(timeout_ms=10000)
        sdma_tl.cpu_wait(timeout_ms=10000)

        # Verify both
        _verify_fill(compute_out, fill_value)
        assert sdma_dst.read(64) == b"\xAA" * 64
        assert sdma_dst.read(64, offset=big_size - 64) == b"\xAA" * 64

        program.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    @requires_fill_kernel
    def test_sdma_while_compute_fills(self, amd_device):
        """SDMA copy of one buffer while kernel fills a different buffer."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 256

        # Compute side (submitted first)
        compute_out = amd_device.alloc(num_elements * 4, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        fill_value = 0xBAADF00D
        program = _dispatch_fill(
            amd_device, compute_queue, compute_tl, compute_out, fill_value
        )

        # SDMA side (submitted second, different buffers)
        sdma_src = amd_device.alloc(4096, location="vram")
        sdma_dst = amd_device.alloc(4096, location="vram")
        sdma_src.fill(0x55)
        sdma_dst.fill(0x00)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(sdma_dst.gpu_addr, sdma_src.gpu_addr, 4096)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend.submit_packets(sdma_queue, sdma.build())

        compute_tl.cpu_wait(timeout_ms=5000)
        sdma_tl.cpu_wait(timeout_ms=5000)

        _verify_fill(compute_out, fill_value)
        assert sdma_dst.read(16) == b"\x55" * 16

        program.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    @requires_fill_kernel
    def test_multiple_sdma_and_compute_waves(self, amd_device):
        """Interleave 4 kernel dispatches and 4 SDMA copies, wait once."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 256

        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        programs = []
        compute_bufs = []
        sdma_dsts = []

        for i in range(4):
            # Kernel dispatch
            out = amd_device.alloc(num_elements * 4, location="vram")
            out.fill(0x00)
            fill_val = 0x10000000 + i
            prog = _dispatch_fill(
                amd_device, compute_queue, compute_tl, out, fill_val
            )
            compute_bufs.append((out, fill_val))
            programs.append(prog)

            # SDMA copy
            src = amd_device.alloc(4096, location="vram")
            dst = amd_device.alloc(4096, location="vram")
            src.fill((i + 1) * 0x11)
            dst.fill(0x00)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(dst.gpu_addr, src.gpu_addr, 4096)
            sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
            backend.submit_packets(sdma_queue, sdma.build())
            sdma_dsts.append((dst, (i + 1) * 0x11))

        # Wait once for everything
        compute_tl.cpu_wait(timeout_ms=10000)
        sdma_tl.cpu_wait(timeout_ms=10000)

        # Verify all 4 kernel outputs
        for out, fill_val in compute_bufs:
            _verify_fill(out, fill_val)

        # Verify all 4 SDMA copies
        for dst, val in sdma_dsts:
            assert dst.read(4) == bytes([val]) * 4

        for prog in programs:
            prog.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)


# ---------------------------------------------------------------------------
# TestCrossGPUComputeCommOverlap — compute || P2P copy
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestCrossGPUComputeCommOverlap:
    """Compute on one GPU while P2P XGMI copy runs to another GPU."""

    @requires_fill_kernel
    def test_compute_on_src_while_p2p_copy(self, multi_gpu_context):
        """GPU0 runs fill_kernel while GPU0 XGMI sends a different buffer to GPU1."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0 = dev0.backend

        num_elements = 256
        size = num_elements * 4

        # Compute side on GPU0
        compute_out = dev0.alloc(size, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend0.create_compute_queue()
        compute_tl = TimelineSemaphore(backend0)

        fill_value = 0xDEADBEEF
        program = _dispatch_fill(
            dev0, compute_queue, compute_tl, compute_out, fill_value
        )

        # P2P copy: separate buffer from GPU0 to GPU1
        p2p_src = dev0.alloc(4096, location="vram")
        p2p_dst = dev1.alloc(4096, location="vram")
        p2p_src.fill(0xBB)
        p2p_dst.fill(0x00)
        ctx.enable_peer_access(p2p_src, dev1)
        ctx.enable_peer_access(p2p_dst, dev0)

        xgmi_queue = _get_xgmi_or_sdma_queue(backend0)
        xgmi_tl = TimelineSemaphore(backend0)

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(p2p_dst.gpu_addr, p2p_src.gpu_addr, 4096)
        sdma.fence(xgmi_tl.signal_addr, xgmi_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma.build())

        # Wait for both
        compute_tl.cpu_wait(timeout_ms=5000)
        xgmi_tl.cpu_wait(timeout_ms=5000)

        # Verify both
        _verify_fill(compute_out, fill_value)
        assert p2p_dst.read(16) == b"\xBB" * 16

        program.free()
        compute_tl.destroy()
        xgmi_tl.destroy()
        backend0.destroy_queue(compute_queue)
        backend0.destroy_queue(xgmi_queue)

    @requires_fill_kernel
    def test_compute_on_dst_while_receiving_p2p(self, multi_gpu_context):
        """GPU1 runs a kernel while receiving a P2P copy from GPU0."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0, backend1 = dev0.backend, dev1.backend

        num_elements = 256
        size = num_elements * 4

        # Compute on GPU1 (the receiver)
        compute_out = dev1.alloc(size, location="vram")
        compute_out.fill(0x00)
        compute_queue1 = backend1.create_compute_queue()
        compute_tl1 = TimelineSemaphore(backend1)

        fill_value = 0xFACEFEED
        program = _dispatch_fill(
            dev1, compute_queue1, compute_tl1, compute_out, fill_value
        )

        # P2P copy from GPU0 to a DIFFERENT buffer on GPU1
        p2p_src = dev0.alloc(4096, location="vram")
        p2p_recv = dev1.alloc(4096, location="vram")
        p2p_src.fill(0x77)
        p2p_recv.fill(0x00)
        ctx.enable_peer_access(p2p_src, dev1)
        ctx.enable_peer_access(p2p_recv, dev0)

        xgmi_queue = _get_xgmi_or_sdma_queue(backend0)
        xgmi_tl = TimelineSemaphore(backend0)

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(p2p_recv.gpu_addr, p2p_src.gpu_addr, 4096)
        sdma.fence(xgmi_tl.signal_addr, xgmi_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma.build())

        # Wait for both
        compute_tl1.cpu_wait(timeout_ms=5000)
        xgmi_tl.cpu_wait(timeout_ms=5000)

        _verify_fill(compute_out, fill_value)
        assert p2p_recv.read(16) == b"\x77" * 16

        program.free()
        compute_tl1.destroy()
        xgmi_tl.destroy()
        backend1.destroy_queue(compute_queue1)
        backend0.destroy_queue(xgmi_queue)

    @requires_fill_kernel
    def test_all_gpus_compute_while_ring_copy(self, multi_gpu_context):
        """Every GPU dispatches a kernel while a ring of P2P copies proceeds."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices

        # Set up ring copy buffers with peer access
        ring_srcs = []
        ring_dsts = []
        for i in range(n):
            j = (i + 1) % n
            src = ctx.devices[i].alloc(4096, location="vram")
            dst = ctx.devices[j].alloc(4096, location="vram")
            fill_val = ((i + 1) * 0x11) & 0xFF
            src.fill(fill_val)
            dst.fill(0x00)
            ctx.enable_peer_access(src, ctx.devices[j])
            ctx.enable_peer_access(dst, ctx.devices[i])
            ring_srcs.append((src, fill_val))
            ring_dsts.append(dst)

        # Dispatch kernel on every GPU
        compute_results = []
        programs = []
        compute_queues = []
        compute_tls = []
        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            num_elements = 256
            out = dev.alloc(num_elements * 4, location="vram")
            out.fill(0x00)
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            fill_val = 0xA0000000 + i
            prog = _dispatch_fill(dev, queue, tl, out, fill_val)
            compute_results.append((out, fill_val))
            programs.append(prog)
            compute_queues.append(queue)
            compute_tls.append(tl)

        # Start ring copies
        for i in range(n):
            j = (i + 1) % n
            ctx.copy_peer(
                ring_dsts[i], ctx.devices[j],
                ring_srcs[i][0], ctx.devices[i], 4096,
            )

        # Wait for everything
        for tl in compute_tls:
            tl.cpu_wait(timeout_ms=10000)
        ctx.synchronize_all()

        # Verify kernels
        for out, fill_val in compute_results:
            _verify_fill(out, fill_val)

        # Verify ring copies
        for i, dst in enumerate(ring_dsts):
            expected_val = ring_srcs[i][1]
            assert dst.read(4) == bytes([expected_val]) * 4, (
                f"Ring copy {i}->{(i+1)%n} failed"
            )

        for prog in programs:
            prog.free()
        for tl in compute_tls:
            tl.destroy()
        for i, q in enumerate(compute_queues):
            ctx.devices[i].backend.destroy_queue(q)


# ---------------------------------------------------------------------------
# TestGPUSideSync — GPU-side ordering between engines
# ---------------------------------------------------------------------------


@requires_gpu
class TestGPUSideSync:
    """GPU-side synchronization between compute and SDMA engines."""

    @requires_fill_kernel
    def test_sdma_waits_for_compute_via_poll_regmem(self, amd_device):
        """SDMA polls compute timeline, then copies the kernel's output.

        If poll_regmem didn't wait, SDMA would copy stale zeros.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 256
        size = num_elements * 4

        # Compute: fill buffer with known value
        compute_out = amd_device.alloc(size, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        fill_value = 0xDEADBEEF
        program = _dispatch_fill(
            amd_device, compute_queue, compute_tl, compute_out, fill_value
        )
        expected_tl_val = compute_tl.timeline_value  # = 1

        # SDMA: poll for compute completion, then copy result to verify_buf
        verify_buf = amd_device.alloc(size, location="vram")
        verify_buf.fill(0x00)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        sdma = SDMAPacketBuilder()
        sdma.poll_regmem(
            compute_tl.signal_addr,
            expected_tl_val,
            interval=160,
            retry_count=0xFFF,
        )
        sdma.copy_linear(verify_buf.gpu_addr, compute_out.gpu_addr, size)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend.submit_packets(sdma_queue, sdma.build())

        # CPU waits for SDMA (which internally waited for compute)
        sdma_tl.cpu_wait(timeout_ms=10000)

        # verify_buf should have the kernel's output, not zeros
        _verify_fill(verify_buf, fill_value)

        program.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    @requires_fill_kernel
    def test_compute_waits_for_sdma_via_wait_reg_mem(self, amd_device):
        """Compute queue uses WAIT_REG_MEM to poll SDMA fence before dispatch.

        WAIT_REG_MEM (PM4) polls indefinitely, so this is safe for
        arbitrarily long SDMA operations. Uses VRAM for sync memory
        since CDNA3 ME may not reliably poll GTT addresses.
        """
        from amd_gpu_driver.commands.pm4 import PM4PacketBuilder
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 256
        size = num_elements * 4

        # Sync memory in VRAM: SDMA writes fence here, compute polls it
        sync_buf = amd_device.alloc(4096, location="vram")
        sync_buf.fill(0x00)
        sync_val = 1

        # SDMA: copy data, then write fence to sync_buf
        sdma_src = amd_device.alloc(4096, location="vram")
        sdma_dst = amd_device.alloc(4096, location="vram")
        sdma_src.fill(0xEE)
        sdma_dst.fill(0x00)
        sdma_queue = backend.create_sdma_queue()

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(sdma_dst.gpu_addr, sdma_src.gpu_addr, 4096)
        sdma.fence(sync_buf.gpu_addr, sync_val)
        backend.submit_packets(sdma_queue, sdma.build())

        # Compute: WAIT_REG_MEM for SDMA fence, then dispatch kernel
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        # Submit WAIT_REG_MEM + kernel in one submission
        pm4_wait = PM4PacketBuilder()
        pm4_wait.wait_reg_mem(
            addr=sync_buf.gpu_addr,
            expected=sync_val,
        )
        backend.submit_packets(compute_queue, pm4_wait.build())

        # Then dispatch kernel (processed in ring order after the wait)
        compute_out = amd_device.alloc(size, location="vram")
        compute_out.fill(0x00)
        fill_value = 0xBAADF00D
        program = _dispatch_fill(
            amd_device, compute_queue, compute_tl, compute_out, fill_value
        )

        # CPU wait for compute timeline (implies SDMA also finished)
        compute_tl.cpu_wait(timeout_ms=10000)

        # Verify both
        _verify_fill(compute_out, fill_value)
        assert sdma_dst.read(16) == b"\xEE" * 16

        program.free()
        compute_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    def test_sdma_poll_regmem_basic(self, amd_device):
        """Basic poll_regmem: SDMA polls a CPU-written value, then copies."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        import ctypes

        # Signal memory: CPU will write here before SDMA reads it
        signal_mem = backend.alloc_memory(4096, location=MemoryLocation.GTT, uncached=True)
        ctypes.c_uint32.from_address(signal_mem.cpu_addr).value = 1

        # SDMA: poll until signal >= 1, then copy
        src = amd_device.alloc(4096, location="vram")
        dst = amd_device.alloc(4096, location="vram")
        src.fill(0xDD)
        dst.fill(0x00)

        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        sdma = SDMAPacketBuilder()
        sdma.poll_regmem(signal_mem.gpu_addr, 1, interval=10, retry_count=0xFFF)
        sdma.copy_linear(dst.gpu_addr, src.gpu_addr, 4096)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend.submit_packets(sdma_queue, sdma.build())

        sdma_tl.cpu_wait(timeout_ms=5000)
        assert dst.read(16) == b"\xDD" * 16

        sdma_tl.destroy()
        backend.destroy_queue(sdma_queue)
        backend.free_memory(signal_mem)

    def test_wait_reg_mem_basic(self, amd_device):
        """Basic WAIT_REG_MEM: compute polls SDMA-written fence in VRAM.

        Uses SDMA to write the sync value (instead of CPU+GTT) since
        CDNA3 ME reliably polls VRAM addresses.
        """
        from amd_gpu_driver.commands.pm4 import PM4PacketBuilder
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend

        # Sync memory in VRAM: SDMA writes fence, compute polls it
        sync_buf = amd_device.alloc(4096, location="vram")
        sync_buf.fill(0x00)

        # SDMA writes fence value to sync_buf
        sdma_queue = backend.create_sdma_queue()
        sdma = SDMAPacketBuilder()
        sdma.fence(sync_buf.gpu_addr, 1)
        backend.submit_packets(sdma_queue, sdma.build())

        # Compute: WAIT_REG_MEM for fence, then signal timeline
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        pm4 = PM4PacketBuilder()
        pm4.wait_reg_mem(addr=sync_buf.gpu_addr, expected=1)
        pm4.nop(1)
        packets = pm4.build() + compute_tl.signal_packets(compute_tl.next_value())
        backend.submit_packets(compute_queue, packets)

        compute_tl.cpu_wait(timeout_ms=5000)

        compute_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)


# ---------------------------------------------------------------------------
# TestCrossGPUSync — GPU-side sync across devices
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestCrossGPUSync:
    """Cross-GPU GPU-side synchronization without CPU round-trips."""

    @requires_fill_kernel
    def test_cross_gpu_compute_then_p2p_gpu_sync(self, multi_gpu_context):
        """GPU0 compute signals timeline, GPU0 SDMA polls it, then P2P copies
        the result to GPU1. Entire pipeline uses GPU-side sync only.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0 = dev0.backend

        num_elements = 256
        size = num_elements * 4

        # Compute: fill on GPU0
        compute_out = dev0.alloc(size, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend0.create_compute_queue()
        compute_tl = TimelineSemaphore(backend0)

        fill_value = 0xCAFEBABE
        program = _dispatch_fill(
            dev0, compute_queue, compute_tl, compute_out, fill_value
        )
        compute_signal_val = compute_tl.timeline_value

        # Receive buffer on GPU1
        recv_buf = dev1.alloc(size, location="vram")
        recv_buf.fill(0x00)
        ctx.enable_peer_access(compute_out, dev1)
        ctx.enable_peer_access(recv_buf, dev0)

        # SDMA on GPU0: poll compute timeline, then P2P copy to GPU1
        xgmi_queue = _get_xgmi_or_sdma_queue(backend0)
        sdma_tl = TimelineSemaphore(backend0)

        sdma = SDMAPacketBuilder()
        sdma.poll_regmem(
            compute_tl.signal_addr,
            compute_signal_val,
            interval=160,
            retry_count=0xFFF,
        )
        sdma.copy_linear(recv_buf.gpu_addr, compute_out.gpu_addr, size)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma.build())

        # CPU only waits for the final SDMA fence
        sdma_tl.cpu_wait(timeout_ms=10000)

        # GPU1's buffer should have the kernel output
        _verify_fill(recv_buf, fill_value)

        program.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend0.destroy_queue(compute_queue)
        backend0.destroy_queue(xgmi_queue)

    @requires_fill_kernel
    def test_cross_gpu_timeline_map_to_peers(self, multi_gpu_context):
        """Map a timeline's signal memory to a peer GPU and verify access."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0 = dev0.backend

        # Create timeline on GPU0 and map to GPU1
        tl = TimelineSemaphore(backend0)
        tl.map_to_peers([dev1.gpu_id])

        assert dev1.gpu_id in tl._signal_mem.mapped_gpu_ids

        # Dispatch kernel on GPU0 using this timeline
        num_elements = 256
        out = dev0.alloc(num_elements * 4, location="vram")
        out.fill(0x00)
        compute_queue = backend0.create_compute_queue()

        fill_value = 0x12345678
        program = _dispatch_fill(dev0, compute_queue, tl, out, fill_value)

        tl.cpu_wait(timeout_ms=5000)
        _verify_fill(out, fill_value)

        program.free()
        tl.destroy()
        backend0.destroy_queue(compute_queue)


# ---------------------------------------------------------------------------
# TestPipelinePatterns — double-buffering, compute-then-send
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestPipelinePatterns:
    """Double-buffering and pipeline overlap patterns."""

    @requires_fill_kernel
    def test_pipeline_compute_then_send(self, multi_gpu_context):
        """Compute on GPU0, GPU-side wait, then SDMA sends result to GPU1.

        Full pipeline without CPU in the critical path:
          GPU0 compute → RELEASE_MEM → poll_regmem → XGMI copy → fence
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0 = dev0.backend

        num_elements = 256
        size = num_elements * 4

        # Compute on GPU0
        compute_out = dev0.alloc(size, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend0.create_compute_queue()
        compute_tl = TimelineSemaphore(backend0)

        fill_value = 0xA5A5A5A5
        program = _dispatch_fill(
            dev0, compute_queue, compute_tl, compute_out, fill_value
        )

        # Receive on GPU1
        recv = dev1.alloc(size, location="vram")
        recv.fill(0x00)
        ctx.enable_peer_access(compute_out, dev1)
        ctx.enable_peer_access(recv, dev0)

        # GPU0 SDMA: wait for compute, then P2P copy
        xgmi_queue = _get_xgmi_or_sdma_queue(backend0)
        sdma_tl = TimelineSemaphore(backend0)

        sdma = SDMAPacketBuilder()
        sdma.poll_regmem(
            compute_tl.signal_addr,
            compute_tl.timeline_value,
            interval=160,
            retry_count=0xFFF,
        )
        sdma.copy_linear(recv.gpu_addr, compute_out.gpu_addr, size)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma.build())

        sdma_tl.cpu_wait(timeout_ms=10000)
        _verify_fill(recv, fill_value)

        program.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend0.destroy_queue(compute_queue)
        backend0.destroy_queue(xgmi_queue)

    @requires_fill_kernel
    def test_double_buffer_compute_send(self, multi_gpu_context):
        """Double-buffering: compute into buf_A while sending buf_B.

        Iteration 0: compute → buf_A  (no prior send)
        Iteration 1: compute → buf_B  ∥  SDMA send buf_A → GPU1
        Final:       SDMA send buf_B → GPU1
        Verify GPU1 received both buffers correctly.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0 = dev0.backend

        num_elements = 256
        size = num_elements * 4

        buf_a = dev0.alloc(size, location="vram")
        buf_b = dev0.alloc(size, location="vram")
        recv_a = dev1.alloc(size, location="vram")
        recv_b = dev1.alloc(size, location="vram")
        buf_a.fill(0x00)
        buf_b.fill(0x00)
        recv_a.fill(0x00)
        recv_b.fill(0x00)

        ctx.enable_peer_access(buf_a, dev1)
        ctx.enable_peer_access(buf_b, dev1)
        ctx.enable_peer_access(recv_a, dev0)
        ctx.enable_peer_access(recv_b, dev0)

        compute_queue = backend0.create_compute_queue()
        compute_tl = TimelineSemaphore(backend0)
        xgmi_queue = _get_xgmi_or_sdma_queue(backend0)
        sdma_tl = TimelineSemaphore(backend0)

        # Iteration 0: compute into buf_A
        val_a = 0x11111111
        prog_a = _dispatch_fill(dev0, compute_queue, compute_tl, buf_a, val_a)
        compute_tl.cpu_wait(timeout_ms=5000)

        # Iteration 1: send buf_A ∥ compute into buf_B
        sdma = SDMAPacketBuilder()
        sdma.copy_linear(recv_a.gpu_addr, buf_a.gpu_addr, size)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma.build())

        val_b = 0x22222222
        prog_b = _dispatch_fill(dev0, compute_queue, compute_tl, buf_b, val_b)

        # Wait for overlapped ops
        compute_tl.cpu_wait(timeout_ms=5000)
        sdma_tl.cpu_wait(timeout_ms=5000)

        # Final: send buf_B
        sdma2 = SDMAPacketBuilder()
        sdma2.copy_linear(recv_b.gpu_addr, buf_b.gpu_addr, size)
        sdma2.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma2.build())
        sdma_tl.cpu_wait(timeout_ms=5000)

        # Verify GPU1 received both
        _verify_fill(recv_a, val_a)
        _verify_fill(recv_b, val_b)

        prog_a.free()
        prog_b.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend0.destroy_queue(compute_queue)
        backend0.destroy_queue(xgmi_queue)

    @requires_fill_kernel
    def test_double_buffer_recv_compute(self, multi_gpu_context):
        """GPU1 computes on chunk 0 while receiving chunk 1 from GPU0."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        backend0, backend1 = dev0.backend, dev1.backend

        num_elements = 256
        size = num_elements * 4

        # GPU0 has two source buffers
        src_0 = dev0.alloc(size, location="vram")
        src_1 = dev0.alloc(size, location="vram")
        src_0.fill(0x10)
        src_1.fill(0x20)

        # GPU1 receive slots
        recv_0 = dev1.alloc(size, location="vram")
        recv_1 = dev1.alloc(size, location="vram")
        recv_0.fill(0x00)
        recv_1.fill(0x00)

        ctx.enable_peer_access(src_0, dev1)
        ctx.enable_peer_access(src_1, dev1)
        ctx.enable_peer_access(recv_0, dev0)
        ctx.enable_peer_access(recv_1, dev0)

        xgmi_queue = _get_xgmi_or_sdma_queue(backend0)
        xgmi_tl = TimelineSemaphore(backend0)

        # Send chunk 0 from GPU0 → GPU1
        sdma = SDMAPacketBuilder()
        sdma.copy_linear(recv_0.gpu_addr, src_0.gpu_addr, size)
        sdma.fence(xgmi_tl.signal_addr, xgmi_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma.build())
        xgmi_tl.cpu_wait(timeout_ms=5000)

        # Now overlap: GPU1 computes while GPU0 sends chunk 1
        compute_queue1 = backend1.create_compute_queue()
        compute_tl1 = TimelineSemaphore(backend1)
        compute_out = dev1.alloc(num_elements * 4, location="vram")
        compute_out.fill(0x00)

        fill_value = 0xFACE0001
        program = _dispatch_fill(
            dev1, compute_queue1, compute_tl1, compute_out, fill_value
        )

        sdma2 = SDMAPacketBuilder()
        sdma2.copy_linear(recv_1.gpu_addr, src_1.gpu_addr, size)
        sdma2.fence(xgmi_tl.signal_addr, xgmi_tl.next_value())
        backend0.submit_packets(xgmi_queue, sdma2.build())

        # Wait for both
        compute_tl1.cpu_wait(timeout_ms=5000)
        xgmi_tl.cpu_wait(timeout_ms=5000)

        # Verify
        _verify_fill(compute_out, fill_value)
        assert recv_0.read(4) == b"\x10" * 4
        assert recv_1.read(4) == b"\x20" * 4

        program.free()
        compute_tl1.destroy()
        xgmi_tl.destroy()
        backend1.destroy_queue(compute_queue1)
        backend0.destroy_queue(xgmi_queue)


# ---------------------------------------------------------------------------
# TestOverlapStress — heavier overlapped workloads
# ---------------------------------------------------------------------------


@requires_gpu
class TestOverlapStressSingleGPU:
    """Stress tests for overlapped operations on a single GPU."""

    @requires_fill_kernel
    def test_saturate_compute_and_sdma(self, amd_device):
        """10 kernel dispatches and 10 SDMA copies submitted back-to-back."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 256

        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        programs = []
        compute_bufs = []
        sdma_dsts = []

        for i in range(10):
            # Kernel dispatch
            out = amd_device.alloc(num_elements * 4, location="vram")
            out.fill(0x00)
            fill_val = 0x01000000 + i
            prog = _dispatch_fill(
                amd_device, compute_queue, compute_tl, out, fill_val
            )
            compute_bufs.append((out, fill_val))
            programs.append(prog)

            # SDMA copy (1 MB)
            sdma_size = 1024 * 1024
            src = amd_device.alloc(sdma_size, location="vram")
            dst = amd_device.alloc(sdma_size, location="vram")
            src.fill((i + 1) & 0xFF)
            dst.fill(0x00)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(dst.gpu_addr, src.gpu_addr, sdma_size)
            sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
            backend.submit_packets(sdma_queue, sdma.build())
            sdma_dsts.append((dst, (i + 1) & 0xFF))

        compute_tl.cpu_wait(timeout_ms=30000)
        sdma_tl.cpu_wait(timeout_ms=30000)

        for out, fill_val in compute_bufs:
            _verify_fill(out, fill_val)
        for dst, val in sdma_dsts:
            assert dst.read(4) == bytes([val]) * 4

        for prog in programs:
            prog.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)


@requires_multi_gpu
class TestOverlapStressMultiGPU:
    """Stress tests for overlapped operations across multiple GPUs."""

    @requires_fill_kernel
    def test_bidirectional_p2p_with_compute(self, multi_gpu_context):
        """Both GPUs compute while doing bidirectional P2P copies."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]
        b0, b1 = dev0.backend, dev1.backend

        num_elements = 256
        size = num_elements * 4

        # Compute on GPU0
        cq0 = b0.create_compute_queue()
        ct0 = TimelineSemaphore(b0)
        cout0 = dev0.alloc(size, location="vram")
        cout0.fill(0x00)
        p0 = _dispatch_fill(dev0, cq0, ct0, cout0, 0xAAAAAAAA)

        # Compute on GPU1
        cq1 = b1.create_compute_queue()
        ct1 = TimelineSemaphore(b1)
        cout1 = dev1.alloc(size, location="vram")
        cout1.fill(0x00)
        p1 = _dispatch_fill(dev1, cq1, ct1, cout1, 0xBBBBBBBB)

        # P2P: GPU0 → GPU1
        p2p_s0 = dev0.alloc(4096, location="vram")
        p2p_d1 = dev1.alloc(4096, location="vram")
        p2p_s0.fill(0x0A)
        p2p_d1.fill(0x00)
        ctx.enable_peer_access(p2p_s0, dev1)
        ctx.enable_peer_access(p2p_d1, dev0)

        # P2P: GPU1 → GPU0
        p2p_s1 = dev1.alloc(4096, location="vram")
        p2p_d0 = dev0.alloc(4096, location="vram")
        p2p_s1.fill(0x0B)
        p2p_d0.fill(0x00)
        ctx.enable_peer_access(p2p_s1, dev0)
        ctx.enable_peer_access(p2p_d0, dev1)

        # Submit copies
        ctx.copy_peer(p2p_d1, dev1, p2p_s0, dev0, 4096)
        ctx.copy_peer(p2p_d0, dev0, p2p_s1, dev1, 4096)

        # Wait for everything
        ct0.cpu_wait(timeout_ms=5000)
        ct1.cpu_wait(timeout_ms=5000)
        ctx.synchronize_all()

        # Verify all 4 operations
        _verify_fill(cout0, 0xAAAAAAAA)
        _verify_fill(cout1, 0xBBBBBBBB)
        assert p2p_d1.read(16) == b"\x0A" * 16
        assert p2p_d0.read(16) == b"\x0B" * 16

        p0.free()
        p1.free()
        ct0.destroy()
        ct1.destroy()
        b0.destroy_queue(cq0)
        b1.destroy_queue(cq1)

    @requires_fill_kernel
    def test_all_gpus_compute_and_sdma_independently(self, multi_gpu_context):
        """Each GPU does compute + SDMA concurrently, all at the same time."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        per_gpu = []

        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            num_elements = 256
            size = num_elements * 4

            # Compute
            cq = backend.create_compute_queue()
            ct = TimelineSemaphore(backend)
            cout = dev.alloc(size, location="vram")
            cout.fill(0x00)
            fill_val = 0xC0000000 + i
            prog = _dispatch_fill(dev, cq, ct, cout, fill_val)

            # SDMA
            sq = backend.create_sdma_queue()
            st = TimelineSemaphore(backend)
            ssrc = dev.alloc(4096, location="vram")
            sdst = dev.alloc(4096, location="vram")
            sdma_val = ((i + 1) * 0x11) & 0xFF
            ssrc.fill(sdma_val)
            sdst.fill(0x00)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(sdst.gpu_addr, ssrc.gpu_addr, 4096)
            sdma.fence(st.signal_addr, st.next_value())
            backend.submit_packets(sq, sdma.build())

            per_gpu.append({
                "backend": backend, "prog": prog,
                "cq": cq, "ct": ct, "cout": cout, "fill_val": fill_val,
                "sq": sq, "st": st, "sdst": sdst, "sdma_val": sdma_val,
            })

        # Wait for all
        for g in per_gpu:
            g["ct"].cpu_wait(timeout_ms=10000)
            g["st"].cpu_wait(timeout_ms=10000)

        # Verify all
        for g in per_gpu:
            _verify_fill(g["cout"], g["fill_val"])
            assert g["sdst"].read(4) == bytes([g["sdma_val"]]) * 4

        # Cleanup
        for g in per_gpu:
            g["prog"].free()
            g["ct"].destroy()
            g["st"].destroy()
            g["backend"].destroy_queue(g["cq"])
            g["backend"].destroy_queue(g["sq"])

    def test_all_gpus_nop_compute_with_sdma_copy(self, multi_gpu_context):
        """NOP compute + SDMA on each GPU (no kernel binary needed)."""
        from amd_gpu_driver.commands.pm4 import PM4PacketBuilder
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        per_gpu = []

        for dev in ctx.devices:
            backend = dev.backend

            # Compute: NOP + signal
            cq = backend.create_compute_queue()
            ct = TimelineSemaphore(backend)
            pm4 = PM4PacketBuilder()
            pm4.nop(4)
            packets = pm4.build() + ct.signal_packets(ct.next_value())
            backend.submit_packets(cq, packets)

            # SDMA: copy + fence
            sq = backend.create_sdma_queue()
            st = TimelineSemaphore(backend)
            src = dev.alloc(4096, location="vram")
            dst = dev.alloc(4096, location="vram")
            src.fill(0xEE)
            dst.fill(0x00)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(dst.gpu_addr, src.gpu_addr, 4096)
            sdma.fence(st.signal_addr, st.next_value())
            backend.submit_packets(sq, sdma.build())

            per_gpu.append({
                "backend": backend,
                "cq": cq, "ct": ct,
                "sq": sq, "st": st, "dst": dst,
            })

        for g in per_gpu:
            g["ct"].cpu_wait(timeout_ms=5000)
            g["st"].cpu_wait(timeout_ms=5000)

        for g in per_gpu:
            assert g["dst"].read(16) == b"\xEE" * 16

        for g in per_gpu:
            g["ct"].destroy()
            g["st"].destroy()
            g["backend"].destroy_queue(g["cq"])
            g["backend"].destroy_queue(g["sq"])
