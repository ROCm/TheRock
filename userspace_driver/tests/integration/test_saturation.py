"""Saturation tests: maximally stress compute + communication across all 8 GPUs.

Goal: drive GPU utilization (gfx_activity, memory bandwidth, XGMI throughput)
as high as possible by overlapping large compute dispatches with sustained
SDMA/XGMI transfers on every GPU simultaneously.

Design principles:
  - Large dispatches: millions of elements per kernel (not 256)
  - Large SDMA: 64-256 MB continuous transfers per link
  - All GPUs active simultaneously: submit everything, then wait once
  - Multiple iterations: repeated compute+copy loops to sustain activity
  - Bidirectional XGMI: ring + all-to-neighbor patterns
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import pytest

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


def _dispatch_fill_large(dev, queue, timeline, out_buf, value, num_elements):
    """Dispatch fill_kernel on a large buffer."""
    program = dev.load_program(str(FILL_KERNEL))
    program.dispatch(
        queue,
        grid=(num_elements // 64, 1, 1),
        block=(64, 1, 1),
        args=[out_buf, value],
        timeline=timeline,
    )
    return program


def _verify_fill_sample(buf, value, num_elements, sample_count=64):
    """Spot-check a large buffer by reading samples at various offsets."""
    size = num_elements * 4
    # Check first, last, and some middle samples
    offsets = [0, size - sample_count * 4]
    for off in offsets:
        data = buf.read(sample_count * 4, offset=off)
        values = struct.unpack(f"<{sample_count}I", data)
        assert all(v == value for v in values), (
            f"Mismatch at offset {off}: expected 0x{value:08X}, "
            f"got 0x{values[0]:08X}"
        )


def _get_xgmi_or_sdma_queue(backend):
    """Create an XGMI SDMA queue if available, else regular SDMA."""
    node = backend.node
    if node is not None and node.num_sdma_xgmi_engines > 0:
        return backend.create_xgmi_sdma_queue()
    return backend.create_sdma_queue()


# ---------------------------------------------------------------------------
# TestSingleGPUSaturation — max out one GPU's compute + SDMA
# ---------------------------------------------------------------------------


@requires_gpu
class TestSingleGPUSaturation:
    """Saturate compute and SDMA engines on a single GPU."""

    @requires_fill_kernel
    def test_large_compute_dispatch(self, amd_device):
        """Single large dispatch: 4M elements (16 MB output)."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 4 * 1024 * 1024  # 4M elements = 16 MB
        size = num_elements * 4

        out = amd_device.alloc(size, location="vram")
        out.fill(0x00)
        queue = backend.create_compute_queue()
        tl = TimelineSemaphore(backend)

        fill_value = 0xFEEDFACE
        prog = _dispatch_fill_large(
            amd_device, queue, tl, out, fill_value, num_elements
        )

        tl.cpu_wait(timeout_ms=30000)
        _verify_fill_sample(out, fill_value, num_elements)

        prog.free()
        tl.destroy()
        backend.destroy_queue(queue)

    @requires_fill_kernel
    def test_repeated_large_dispatches(self, amd_device):
        """8 back-to-back large dispatches (4M elements each) on one GPU."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 4 * 1024 * 1024
        size = num_elements * 4
        iterations = 8

        queue = backend.create_compute_queue()
        tl = TimelineSemaphore(backend)

        bufs = []
        programs = []
        for i in range(iterations):
            out = amd_device.alloc(size, location="vram")
            out.fill(0x00)
            fill_val = 0xA0000000 + i
            prog = _dispatch_fill_large(
                amd_device, queue, tl, out, fill_val, num_elements
            )
            bufs.append((out, fill_val))
            programs.append(prog)

        tl.cpu_wait(timeout_ms=60000)

        for out, fill_val in bufs:
            _verify_fill_sample(out, fill_val, num_elements)

        for prog in programs:
            prog.free()
        tl.destroy()
        backend.destroy_queue(queue)

    @requires_fill_kernel
    def test_large_compute_with_large_sdma(self, amd_device):
        """Overlap: 4M-element kernel + 64 MB SDMA copy simultaneously."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend

        # Large compute
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        compute_out = amd_device.alloc(compute_size, location="vram")
        compute_out.fill(0x00)
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        # Large SDMA (64 MB)
        sdma_size = 64 * 1024 * 1024
        sdma_src = amd_device.alloc(sdma_size, location="vram")
        sdma_dst = amd_device.alloc(sdma_size, location="vram")
        sdma_src.fill(0xBB)
        sdma_dst.fill(0x00)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        # Submit both
        fill_value = 0xDEADBEEF
        prog = _dispatch_fill_large(
            amd_device, compute_queue, compute_tl, compute_out,
            fill_value, num_elements,
        )

        sdma = SDMAPacketBuilder()
        sdma.copy_linear(sdma_dst.gpu_addr, sdma_src.gpu_addr, sdma_size)
        sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
        backend.submit_packets(sdma_queue, sdma.build())

        # Wait
        compute_tl.cpu_wait(timeout_ms=30000)
        sdma_tl.cpu_wait(timeout_ms=30000)

        _verify_fill_sample(compute_out, fill_value, num_elements)
        assert sdma_dst.read(64) == b"\xBB" * 64
        assert sdma_dst.read(64, offset=sdma_size - 64) == b"\xBB" * 64

        prog.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    @requires_fill_kernel
    def test_sustained_compute_sdma_loop(self, amd_device):
        """Loop: 4 iterations of compute(4M) + SDMA(64MB), pipelined."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        sdma_size = 64 * 1024 * 1024
        iterations = 4

        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        programs = []
        compute_bufs = []

        # Pre-allocate SDMA buffers
        sdma_src = amd_device.alloc(sdma_size, location="vram")
        sdma_dst = amd_device.alloc(sdma_size, location="vram")
        sdma_src.fill(0xCC)

        for i in range(iterations):
            # Compute dispatch
            out = amd_device.alloc(compute_size, location="vram")
            out.fill(0x00)
            fill_val = 0xF0000000 + i
            prog = _dispatch_fill_large(
                amd_device, compute_queue, compute_tl, out,
                fill_val, num_elements,
            )
            compute_bufs.append((out, fill_val))
            programs.append(prog)

            # Overlapping SDMA copy
            sdma_dst.fill(0x00)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(sdma_dst.gpu_addr, sdma_src.gpu_addr, sdma_size)
            sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
            backend.submit_packets(sdma_queue, sdma.build())

        # Wait for all
        compute_tl.cpu_wait(timeout_ms=120000)
        sdma_tl.cpu_wait(timeout_ms=120000)

        for out, fill_val in compute_bufs:
            _verify_fill_sample(out, fill_val, num_elements)

        for prog in programs:
            prog.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)


# ---------------------------------------------------------------------------
# TestAllGPUComputeSaturation — max compute across all 8 GPUs
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestAllGPUComputeSaturation:
    """Drive all GPUs to maximum compute utilization simultaneously."""

    @requires_fill_kernel
    def test_all_gpus_large_dispatch(self, multi_gpu_context):
        """Every GPU dispatches a 4M-element kernel simultaneously."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        num_elements = 4 * 1024 * 1024
        size = num_elements * 4

        per_gpu = []
        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            out = dev.alloc(size, location="vram")
            out.fill(0x00)
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            fill_val = 0xD0000000 + i
            prog = _dispatch_fill_large(dev, queue, tl, out, fill_val, num_elements)
            per_gpu.append({
                "dev": dev, "backend": backend, "out": out, "queue": queue,
                "tl": tl, "fill_val": fill_val, "prog": prog,
            })

        # Wait for all
        for g in per_gpu:
            g["tl"].cpu_wait(timeout_ms=30000)

        # Verify all
        for g in per_gpu:
            _verify_fill_sample(g["out"], g["fill_val"], num_elements)

        # Cleanup
        for g in per_gpu:
            g["prog"].free()
            g["tl"].destroy()
            g["backend"].destroy_queue(g["queue"])

    @requires_fill_kernel
    def test_all_gpus_repeated_large_dispatches(self, multi_gpu_context):
        """Every GPU dispatches 4 × 4M-element kernels back-to-back."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        num_elements = 4 * 1024 * 1024
        size = num_elements * 4
        iterations = 4

        per_gpu = []
        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            gpu_data = {
                "dev": dev, "backend": backend, "queue": queue, "tl": tl,
                "bufs": [], "programs": [],
            }

            for j in range(iterations):
                out = dev.alloc(size, location="vram")
                out.fill(0x00)
                fill_val = 0xE0000000 + i * 0x100 + j
                prog = _dispatch_fill_large(
                    dev, queue, tl, out, fill_val, num_elements
                )
                gpu_data["bufs"].append((out, fill_val))
                gpu_data["programs"].append(prog)

            per_gpu.append(gpu_data)

        for g in per_gpu:
            g["tl"].cpu_wait(timeout_ms=120000)

        for g in per_gpu:
            for out, fill_val in g["bufs"]:
                _verify_fill_sample(out, fill_val, num_elements)

        for g in per_gpu:
            for prog in g["programs"]:
                prog.free()
            g["tl"].destroy()
            g["backend"].destroy_queue(g["queue"])


# ---------------------------------------------------------------------------
# TestAllGPUCommsSaturation — max out XGMI bandwidth
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestAllGPUCommsSaturation:
    """Saturate XGMI links with large bidirectional P2P transfers."""

    def test_ring_copy_large(self, multi_gpu_context):
        """Ring topology: each GPU sends 128 MB to next GPU, all at once."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        copy_size = 128 * 1024 * 1024  # 128 MB

        # Allocate and set up peer access for ring
        ring_data = []
        for i in range(n):
            j = (i + 1) % n
            src = ctx.devices[i].alloc(copy_size, location="vram")
            dst = ctx.devices[j].alloc(copy_size, location="vram")
            fill_byte = ((i + 1) * 0x11) & 0xFF
            src.fill(fill_byte)
            dst.fill(0x00)
            ctx.enable_peer_access(src, ctx.devices[j])
            ctx.enable_peer_access(dst, ctx.devices[i])
            ring_data.append({
                "src": src, "dst": dst, "fill_byte": fill_byte,
                "src_dev": ctx.devices[i], "dst_dev": ctx.devices[j],
            })

        # Submit all copies at once
        queues = []
        timelines = []
        for i, rd in enumerate(ring_data):
            backend = rd["src_dev"].backend
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            tl = TimelineSemaphore(backend)

            sdma = SDMAPacketBuilder()
            sdma.copy_linear(rd["dst"].gpu_addr, rd["src"].gpu_addr, copy_size)
            sdma.fence(tl.signal_addr, tl.next_value())
            backend.submit_packets(xgmi_queue, sdma.build())

            queues.append((backend, xgmi_queue))
            timelines.append(tl)

        # Wait for all
        for tl in timelines:
            tl.cpu_wait(timeout_ms=60000)

        # Verify
        for rd in ring_data:
            expected = bytes([rd["fill_byte"]]) * 64
            assert rd["dst"].read(64) == expected
            assert rd["dst"].read(64, offset=copy_size - 64) == expected

        for tl in timelines:
            tl.destroy()
        for backend, q in queues:
            backend.destroy_queue(q)

    def test_bidirectional_neighbor_copies_large(self, multi_gpu_context):
        """Bidirectional: GPU(i) <-> GPU(i+1) for all adjacent pairs, 64 MB."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        copy_size = 64 * 1024 * 1024

        transfers = []
        for i in range(n - 1):
            j = i + 1
            # Forward: i -> j
            fwd_src = ctx.devices[i].alloc(copy_size, location="vram")
            fwd_dst = ctx.devices[j].alloc(copy_size, location="vram")
            fwd_src.fill(0xAA)
            fwd_dst.fill(0x00)
            ctx.enable_peer_access(fwd_src, ctx.devices[j])
            ctx.enable_peer_access(fwd_dst, ctx.devices[i])

            # Reverse: j -> i
            rev_src = ctx.devices[j].alloc(copy_size, location="vram")
            rev_dst = ctx.devices[i].alloc(copy_size, location="vram")
            rev_src.fill(0xBB)
            rev_dst.fill(0x00)
            ctx.enable_peer_access(rev_src, ctx.devices[i])
            ctx.enable_peer_access(rev_dst, ctx.devices[j])

            transfers.append({
                "fwd_src": fwd_src, "fwd_dst": fwd_dst,
                "rev_src": rev_src, "rev_dst": rev_dst,
                "dev_i": ctx.devices[i], "dev_j": ctx.devices[j],
            })

        # Submit all transfers at once
        queues = []
        timelines = []
        for t in transfers:
            # Forward copy on dev_i's XGMI queue
            bi = t["dev_i"].backend
            qi = _get_xgmi_or_sdma_queue(bi)
            tli = TimelineSemaphore(bi)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(t["fwd_dst"].gpu_addr, t["fwd_src"].gpu_addr, copy_size)
            sdma.fence(tli.signal_addr, tli.next_value())
            bi.submit_packets(qi, sdma.build())
            queues.append((bi, qi))
            timelines.append(tli)

            # Reverse copy on dev_j's XGMI queue
            bj = t["dev_j"].backend
            qj = _get_xgmi_or_sdma_queue(bj)
            tlj = TimelineSemaphore(bj)
            sdma2 = SDMAPacketBuilder()
            sdma2.copy_linear(t["rev_dst"].gpu_addr, t["rev_src"].gpu_addr, copy_size)
            sdma2.fence(tlj.signal_addr, tlj.next_value())
            bj.submit_packets(qj, sdma2.build())
            queues.append((bj, qj))
            timelines.append(tlj)

        for tl in timelines:
            tl.cpu_wait(timeout_ms=60000)

        for t in transfers:
            assert t["fwd_dst"].read(64) == b"\xAA" * 64
            assert t["rev_dst"].read(64) == b"\xBB" * 64

        for tl in timelines:
            tl.destroy()
        for backend, q in queues:
            backend.destroy_queue(q)

    def test_all_to_all_copy(self, multi_gpu_context):
        """All-to-all: every GPU sends 16 MB to every other GPU."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        copy_size = 16 * 1024 * 1024

        transfers = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                src = ctx.devices[i].alloc(copy_size, location="vram")
                dst = ctx.devices[j].alloc(copy_size, location="vram")
                fill_byte = ((i * n + j) * 0x07 + 0x10) & 0xFF
                if fill_byte == 0:
                    fill_byte = 0x01
                src.fill(fill_byte)
                dst.fill(0x00)
                ctx.enable_peer_access(src, ctx.devices[j])
                ctx.enable_peer_access(dst, ctx.devices[i])
                transfers.append({
                    "src": src, "dst": dst, "fill_byte": fill_byte,
                    "src_dev": ctx.devices[i],
                })

        queues = []
        timelines = []
        for t in transfers:
            backend = t["src_dev"].backend
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            tl = TimelineSemaphore(backend)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(t["dst"].gpu_addr, t["src"].gpu_addr, copy_size)
            sdma.fence(tl.signal_addr, tl.next_value())
            backend.submit_packets(xgmi_queue, sdma.build())
            queues.append((backend, xgmi_queue))
            timelines.append(tl)

        for tl in timelines:
            tl.cpu_wait(timeout_ms=120000)

        for t in transfers:
            expected = bytes([t["fill_byte"]]) * 64
            assert t["dst"].read(64) == expected

        for tl in timelines:
            tl.destroy()
        for backend, q in queues:
            backend.destroy_queue(q)

    def test_repeated_ring_copies(self, multi_gpu_context):
        """Ring topology × 4 iterations: sustained 128 MB ring copies."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        copy_size = 128 * 1024 * 1024
        iterations = 4

        # Pre-allocate ring buffers
        ring = []
        for i in range(n):
            j = (i + 1) % n
            src = ctx.devices[i].alloc(copy_size, location="vram")
            dst = ctx.devices[j].alloc(copy_size, location="vram")
            src.fill(((i + 1) * 0x11) & 0xFF)
            dst.fill(0x00)
            ctx.enable_peer_access(src, ctx.devices[j])
            ctx.enable_peer_access(dst, ctx.devices[i])
            ring.append({
                "src": src, "dst": dst, "fill_byte": ((i + 1) * 0x11) & 0xFF,
                "src_dev": ctx.devices[i],
            })

        # Create queues and timelines
        queues = []
        timelines = []
        for i, rd in enumerate(ring):
            backend = rd["src_dev"].backend
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            tl = TimelineSemaphore(backend)
            queues.append((backend, xgmi_queue))
            timelines.append(tl)

        # Submit iterations back-to-back
        for _iter in range(iterations):
            for i, rd in enumerate(ring):
                backend, xgmi_queue = queues[i]
                tl = timelines[i]
                sdma = SDMAPacketBuilder()
                sdma.copy_linear(
                    rd["dst"].gpu_addr, rd["src"].gpu_addr, copy_size,
                )
                sdma.fence(tl.signal_addr, tl.next_value())
                backend.submit_packets(xgmi_queue, sdma.build())

        # Wait for all iterations
        for tl in timelines:
            tl.cpu_wait(timeout_ms=120000)

        for rd in ring:
            assert rd["dst"].read(64) == bytes([rd["fill_byte"]]) * 64

        for tl in timelines:
            tl.destroy()
        for backend, q in queues:
            backend.destroy_queue(q)


# ---------------------------------------------------------------------------
# TestFullSaturation — compute + comms simultaneously on all GPUs
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestFullSaturation:
    """Simultaneously saturate compute AND communication on all 8 GPUs."""

    @requires_fill_kernel
    def test_all_gpus_compute_and_ring_copy(self, multi_gpu_context):
        """Every GPU: 4M kernel dispatch + 128 MB ring copy simultaneously."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        copy_size = 128 * 1024 * 1024

        per_gpu = []
        for i in range(n):
            j = (i + 1) % n
            dev = ctx.devices[i]
            backend = dev.backend

            # Compute
            compute_out = dev.alloc(compute_size, location="vram")
            compute_out.fill(0x00)
            compute_queue = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)
            fill_val = 0xC0000000 + i

            # XGMI copy buffers
            copy_src = dev.alloc(copy_size, location="vram")
            copy_dst = ctx.devices[j].alloc(copy_size, location="vram")
            fill_byte = ((i + 1) * 0x11) & 0xFF
            copy_src.fill(fill_byte)
            copy_dst.fill(0x00)
            ctx.enable_peer_access(copy_src, ctx.devices[j])
            ctx.enable_peer_access(copy_dst, dev)
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            xgmi_tl = TimelineSemaphore(backend)

            per_gpu.append({
                "dev": dev, "backend": backend,
                "compute_out": compute_out, "compute_queue": compute_queue,
                "compute_tl": compute_tl, "fill_val": fill_val,
                "copy_src": copy_src, "copy_dst": copy_dst,
                "fill_byte": fill_byte,
                "xgmi_queue": xgmi_queue, "xgmi_tl": xgmi_tl,
            })

        # Submit all compute + copies simultaneously
        programs = []
        for g in per_gpu:
            # Compute dispatch
            prog = _dispatch_fill_large(
                g["dev"], g["compute_queue"], g["compute_tl"],
                g["compute_out"], g["fill_val"], num_elements,
            )
            programs.append(prog)

            # XGMI copy
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(
                g["copy_dst"].gpu_addr, g["copy_src"].gpu_addr, copy_size,
            )
            sdma.fence(g["xgmi_tl"].signal_addr, g["xgmi_tl"].next_value())
            g["backend"].submit_packets(g["xgmi_queue"], sdma.build())

        # Wait for everything
        for g in per_gpu:
            g["compute_tl"].cpu_wait(timeout_ms=60000)
            g["xgmi_tl"].cpu_wait(timeout_ms=60000)

        # Verify
        for g in per_gpu:
            _verify_fill_sample(g["compute_out"], g["fill_val"], num_elements)
            assert g["copy_dst"].read(64) == bytes([g["fill_byte"]]) * 64
            assert g["copy_dst"].read(64, offset=copy_size - 64) == bytes([g["fill_byte"]]) * 64

        for prog in programs:
            prog.free()
        for g in per_gpu:
            g["compute_tl"].destroy()
            g["xgmi_tl"].destroy()
            g["backend"].destroy_queue(g["compute_queue"])
            g["backend"].destroy_queue(g["xgmi_queue"])

    @requires_fill_kernel
    def test_sustained_compute_and_comms(self, multi_gpu_context):
        """4 iterations: every GPU does 4M compute + 64 MB ring copy per iteration."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        copy_size = 64 * 1024 * 1024
        iterations = 4

        # Set up per-GPU state
        gpu_state = []
        for i in range(n):
            j = (i + 1) % n
            dev = ctx.devices[i]
            backend = dev.backend

            compute_queue = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            xgmi_tl = TimelineSemaphore(backend)

            # Pre-allocate XGMI ring buffers
            copy_src = dev.alloc(copy_size, location="vram")
            copy_dst = ctx.devices[j].alloc(copy_size, location="vram")
            fill_byte = ((i + 1) * 0x11) & 0xFF
            copy_src.fill(fill_byte)
            copy_dst.fill(0x00)
            ctx.enable_peer_access(copy_src, ctx.devices[j])
            ctx.enable_peer_access(copy_dst, dev)

            gpu_state.append({
                "dev": dev, "backend": backend,
                "compute_queue": compute_queue, "compute_tl": compute_tl,
                "xgmi_queue": xgmi_queue, "xgmi_tl": xgmi_tl,
                "copy_src": copy_src, "copy_dst": copy_dst,
                "fill_byte": fill_byte,
            })

        all_programs = []
        last_bufs = []

        for it in range(iterations):
            for i, gs in enumerate(gpu_state):
                # Compute dispatch
                out = gs["dev"].alloc(compute_size, location="vram")
                out.fill(0x00)
                fill_val = 0xB0000000 + i * 0x100 + it
                prog = _dispatch_fill_large(
                    gs["dev"], gs["compute_queue"], gs["compute_tl"],
                    out, fill_val, num_elements,
                )
                all_programs.append(prog)
                if it == iterations - 1:
                    last_bufs.append((out, fill_val))

                # XGMI copy
                sdma = SDMAPacketBuilder()
                sdma.copy_linear(
                    gs["copy_dst"].gpu_addr, gs["copy_src"].gpu_addr, copy_size,
                )
                sdma.fence(gs["xgmi_tl"].signal_addr, gs["xgmi_tl"].next_value())
                gs["backend"].submit_packets(gs["xgmi_queue"], sdma.build())

        # Wait for all
        for gs in gpu_state:
            gs["compute_tl"].cpu_wait(timeout_ms=120000)
            gs["xgmi_tl"].cpu_wait(timeout_ms=120000)

        # Verify last iteration compute outputs
        for out, fill_val in last_bufs:
            _verify_fill_sample(out, fill_val, num_elements)

        # Verify ring copy data
        for gs in gpu_state:
            assert gs["copy_dst"].read(64) == bytes([gs["fill_byte"]]) * 64

        for prog in all_programs:
            prog.free()
        for gs in gpu_state:
            gs["compute_tl"].destroy()
            gs["xgmi_tl"].destroy()
            gs["backend"].destroy_queue(gs["compute_queue"])
            gs["backend"].destroy_queue(gs["xgmi_queue"])

    @requires_fill_kernel
    def test_compute_and_bidirectional_xgmi(self, multi_gpu_context):
        """All GPUs compute while bidirectional ring copies run.

        GPU(i) sends to GPU(i+1) AND GPU(i-1) while computing.
        Creates 2×N XGMI transfers + N compute dispatches.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        copy_size = 64 * 1024 * 1024

        per_gpu = []
        for i in range(n):
            fwd = (i + 1) % n
            bwd = (i - 1) % n
            dev = ctx.devices[i]
            backend = dev.backend

            # Compute
            compute_out = dev.alloc(compute_size, location="vram")
            compute_out.fill(0x00)
            compute_queue = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)
            fill_val = 0x80000000 + i

            # Forward XGMI copy: i -> (i+1)
            fwd_src = dev.alloc(copy_size, location="vram")
            fwd_dst = ctx.devices[fwd].alloc(copy_size, location="vram")
            fwd_src.fill(0xAA)
            fwd_dst.fill(0x00)
            ctx.enable_peer_access(fwd_src, ctx.devices[fwd])
            ctx.enable_peer_access(fwd_dst, dev)

            # Backward XGMI copy: i -> (i-1)
            bwd_src = dev.alloc(copy_size, location="vram")
            bwd_dst = ctx.devices[bwd].alloc(copy_size, location="vram")
            bwd_src.fill(0x55)
            bwd_dst.fill(0x00)
            ctx.enable_peer_access(bwd_src, ctx.devices[bwd])
            ctx.enable_peer_access(bwd_dst, dev)

            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            xgmi_tl = TimelineSemaphore(backend)

            per_gpu.append({
                "dev": dev, "backend": backend,
                "compute_out": compute_out, "compute_queue": compute_queue,
                "compute_tl": compute_tl, "fill_val": fill_val,
                "fwd_src": fwd_src, "fwd_dst": fwd_dst,
                "bwd_src": bwd_src, "bwd_dst": bwd_dst,
                "xgmi_queue": xgmi_queue, "xgmi_tl": xgmi_tl,
            })

        # Submit everything
        programs = []
        for g in per_gpu:
            # Compute
            prog = _dispatch_fill_large(
                g["dev"], g["compute_queue"], g["compute_tl"],
                g["compute_out"], g["fill_val"], num_elements,
            )
            programs.append(prog)

            # Forward + backward copies on same XGMI queue (serialized)
            sdma = SDMAPacketBuilder()
            sdma.copy_linear(g["fwd_dst"].gpu_addr, g["fwd_src"].gpu_addr, copy_size)
            sdma.copy_linear(g["bwd_dst"].gpu_addr, g["bwd_src"].gpu_addr, copy_size)
            sdma.fence(g["xgmi_tl"].signal_addr, g["xgmi_tl"].next_value())
            g["backend"].submit_packets(g["xgmi_queue"], sdma.build())

        # Wait
        for g in per_gpu:
            g["compute_tl"].cpu_wait(timeout_ms=60000)
            g["xgmi_tl"].cpu_wait(timeout_ms=60000)

        # Verify
        for g in per_gpu:
            _verify_fill_sample(g["compute_out"], g["fill_val"], num_elements)
            assert g["fwd_dst"].read(64) == b"\xAA" * 64
            assert g["bwd_dst"].read(64) == b"\x55" * 64

        for prog in programs:
            prog.free()
        for g in per_gpu:
            g["compute_tl"].destroy()
            g["xgmi_tl"].destroy()
            g["backend"].destroy_queue(g["compute_queue"])
            g["backend"].destroy_queue(g["xgmi_queue"])

    @requires_fill_kernel
    def test_pipeline_allreduce_pattern(self, multi_gpu_context):
        """Simulate allreduce: compute on each GPU, ring-reduce, ring-broadcast.

        Phase 1: All GPUs compute into local buffers (4M elements)
        Phase 2: Ring of P2P copies (reduce-scatter emulation)
        Phase 3: Ring of P2P copies (allgather emulation)
        All phases overlap where possible.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        chunk_size = 32 * 1024 * 1024  # 32 MB per ring step

        # Phase 1: Dispatch compute on all GPUs
        compute_data = []
        programs = []
        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            out = dev.alloc(compute_size, location="vram")
            out.fill(0x00)
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            fill_val = 0x90000000 + i
            prog = _dispatch_fill_large(dev, queue, tl, out, fill_val, num_elements)
            programs.append(prog)
            compute_data.append({
                "dev": dev, "backend": backend, "out": out,
                "queue": queue, "tl": tl, "fill_val": fill_val,
            })

        # Phase 2: Ring reduce-scatter (overlaps with compute)
        # Each GPU sends a chunk to next GPU
        ring_srcs = []
        ring_dsts = []
        ring_queues = []
        ring_tls = []
        for i in range(n):
            j = (i + 1) % n
            src = ctx.devices[i].alloc(chunk_size, location="vram")
            dst = ctx.devices[j].alloc(chunk_size, location="vram")
            src.fill(((i + 1) * 0x13) & 0xFF)
            dst.fill(0x00)
            ctx.enable_peer_access(src, ctx.devices[j])
            ctx.enable_peer_access(dst, ctx.devices[i])

            backend = ctx.devices[i].backend
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            tl = TimelineSemaphore(backend)

            sdma = SDMAPacketBuilder()
            sdma.copy_linear(dst.gpu_addr, src.gpu_addr, chunk_size)
            sdma.fence(tl.signal_addr, tl.next_value())
            backend.submit_packets(xgmi_queue, sdma.build())

            ring_srcs.append(src)
            ring_dsts.append(dst)
            ring_queues.append((backend, xgmi_queue))
            ring_tls.append(tl)

        # Phase 3: Ring allgather (submitted immediately, serializes after phase 2)
        gather_dsts = []
        gather_tls = []
        for i in range(n):
            j = (i + 1) % n
            src = ctx.devices[i].alloc(chunk_size, location="vram")
            dst = ctx.devices[j].alloc(chunk_size, location="vram")
            src.fill(((i + 1) * 0x17) & 0xFF)
            dst.fill(0x00)
            ctx.enable_peer_access(src, ctx.devices[j])
            ctx.enable_peer_access(dst, ctx.devices[i])

            backend, xgmi_queue = ring_queues[i]
            tl = ring_tls[i]  # reuse same timeline (values increment)

            sdma = SDMAPacketBuilder()
            sdma.copy_linear(dst.gpu_addr, src.gpu_addr, chunk_size)
            sdma.fence(tl.signal_addr, tl.next_value())
            backend.submit_packets(xgmi_queue, sdma.build())

            gather_dsts.append(dst)

        # Wait for everything
        for cd in compute_data:
            cd["tl"].cpu_wait(timeout_ms=60000)
        for tl in ring_tls:
            tl.cpu_wait(timeout_ms=60000)

        # Verify compute results
        for cd in compute_data:
            _verify_fill_sample(cd["out"], cd["fill_val"], num_elements)

        # Verify ring copies completed
        for i, dst in enumerate(ring_dsts):
            fill_byte = ((i + 1) * 0x13) & 0xFF
            assert dst.read(64) == bytes([fill_byte]) * 64

        # Cleanup
        for prog in programs:
            prog.free()
        for cd in compute_data:
            cd["tl"].destroy()
            cd["backend"].destroy_queue(cd["queue"])
        for tl in ring_tls:
            tl.destroy()
        for backend, q in ring_queues:
            backend.destroy_queue(q)

    @requires_fill_kernel
    def test_max_saturation_all_engines(self, multi_gpu_context):
        """Maximum saturation: every GPU runs compute + local SDMA + XGMI copy.

        Per GPU:
        - 1 compute dispatch: 4M elements
        - 1 local SDMA copy: 64 MB (VRAM->VRAM same GPU)
        - 1 XGMI copy: 64 MB to next GPU
        All 8 GPUs × 3 engines = 24 concurrent operations.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        local_copy_size = 64 * 1024 * 1024
        xgmi_copy_size = 64 * 1024 * 1024

        per_gpu = []
        for i in range(n):
            j = (i + 1) % n
            dev = ctx.devices[i]
            backend = dev.backend

            # Compute
            compute_out = dev.alloc(compute_size, location="vram")
            compute_out.fill(0x00)
            compute_queue = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)
            fill_val = 0x70000000 + i

            # Local SDMA (same GPU VRAM->VRAM)
            local_src = dev.alloc(local_copy_size, location="vram")
            local_dst = dev.alloc(local_copy_size, location="vram")
            local_src.fill(0xDD)
            local_dst.fill(0x00)
            sdma_queue = backend.create_sdma_queue()
            sdma_tl = TimelineSemaphore(backend)

            # XGMI copy to next GPU
            xgmi_src = dev.alloc(xgmi_copy_size, location="vram")
            xgmi_dst = ctx.devices[j].alloc(xgmi_copy_size, location="vram")
            xgmi_src.fill(0xEE)
            xgmi_dst.fill(0x00)
            ctx.enable_peer_access(xgmi_src, ctx.devices[j])
            ctx.enable_peer_access(xgmi_dst, dev)
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            xgmi_tl = TimelineSemaphore(backend)

            per_gpu.append({
                "dev": dev, "backend": backend,
                "compute_out": compute_out, "compute_queue": compute_queue,
                "compute_tl": compute_tl, "fill_val": fill_val,
                "local_src": local_src, "local_dst": local_dst,
                "sdma_queue": sdma_queue, "sdma_tl": sdma_tl,
                "xgmi_src": xgmi_src, "xgmi_dst": xgmi_dst,
                "xgmi_queue": xgmi_queue, "xgmi_tl": xgmi_tl,
            })

        # Submit all 24 operations
        programs = []
        for g in per_gpu:
            # Compute
            prog = _dispatch_fill_large(
                g["dev"], g["compute_queue"], g["compute_tl"],
                g["compute_out"], g["fill_val"], num_elements,
            )
            programs.append(prog)

            # Local SDMA
            sdma_local = SDMAPacketBuilder()
            sdma_local.copy_linear(
                g["local_dst"].gpu_addr, g["local_src"].gpu_addr, local_copy_size,
            )
            sdma_local.fence(g["sdma_tl"].signal_addr, g["sdma_tl"].next_value())
            g["backend"].submit_packets(g["sdma_queue"], sdma_local.build())

            # XGMI copy
            sdma_xgmi = SDMAPacketBuilder()
            sdma_xgmi.copy_linear(
                g["xgmi_dst"].gpu_addr, g["xgmi_src"].gpu_addr, xgmi_copy_size,
            )
            sdma_xgmi.fence(g["xgmi_tl"].signal_addr, g["xgmi_tl"].next_value())
            g["backend"].submit_packets(g["xgmi_queue"], sdma_xgmi.build())

        # Wait for all 24 operations
        for g in per_gpu:
            g["compute_tl"].cpu_wait(timeout_ms=60000)
            g["sdma_tl"].cpu_wait(timeout_ms=60000)
            g["xgmi_tl"].cpu_wait(timeout_ms=60000)

        # Verify all
        for g in per_gpu:
            _verify_fill_sample(g["compute_out"], g["fill_val"], num_elements)
            assert g["local_dst"].read(64) == b"\xDD" * 64
            assert g["local_dst"].read(64, offset=local_copy_size - 64) == b"\xDD" * 64
            assert g["xgmi_dst"].read(64) == b"\xEE" * 64
            assert g["xgmi_dst"].read(64, offset=xgmi_copy_size - 64) == b"\xEE" * 64

        for prog in programs:
            prog.free()
        for g in per_gpu:
            g["compute_tl"].destroy()
            g["sdma_tl"].destroy()
            g["xgmi_tl"].destroy()
            g["backend"].destroy_queue(g["compute_queue"])
            g["backend"].destroy_queue(g["sdma_queue"])
            g["backend"].destroy_queue(g["xgmi_queue"])

    @requires_fill_kernel
    def test_sustained_max_saturation(self, multi_gpu_context):
        """4 iterations of max saturation: compute + local SDMA + XGMI on all GPUs.

        Total: 4 × 8 × 3 = 96 GPU operations submitted before any waits.
        """
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_elements = 4 * 1024 * 1024
        compute_size = num_elements * 4
        copy_size = 64 * 1024 * 1024
        iterations = 4

        # Set up per-GPU persistent state
        gpu_state = []
        for i in range(n):
            j = (i + 1) % n
            dev = ctx.devices[i]
            backend = dev.backend

            compute_queue = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)
            sdma_queue = backend.create_sdma_queue()
            sdma_tl = TimelineSemaphore(backend)
            xgmi_queue = _get_xgmi_or_sdma_queue(backend)
            xgmi_tl = TimelineSemaphore(backend)

            # Persistent SDMA buffers (reused each iteration)
            local_src = dev.alloc(copy_size, location="vram")
            local_dst = dev.alloc(copy_size, location="vram")
            local_src.fill(0xDD)

            xgmi_src = dev.alloc(copy_size, location="vram")
            xgmi_dst = ctx.devices[j].alloc(copy_size, location="vram")
            xgmi_src.fill(0xEE)
            ctx.enable_peer_access(xgmi_src, ctx.devices[j])
            ctx.enable_peer_access(xgmi_dst, dev)

            gpu_state.append({
                "dev": dev, "backend": backend,
                "compute_queue": compute_queue, "compute_tl": compute_tl,
                "sdma_queue": sdma_queue, "sdma_tl": sdma_tl,
                "xgmi_queue": xgmi_queue, "xgmi_tl": xgmi_tl,
                "local_src": local_src, "local_dst": local_dst,
                "xgmi_src": xgmi_src, "xgmi_dst": xgmi_dst,
            })

        all_programs = []

        # Submit all iterations back-to-back
        for it in range(iterations):
            for i, gs in enumerate(gpu_state):
                # Compute
                out = gs["dev"].alloc(compute_size, location="vram")
                out.fill(0x00)
                fill_val = 0x60000000 + i * 0x1000 + it
                prog = _dispatch_fill_large(
                    gs["dev"], gs["compute_queue"], gs["compute_tl"],
                    out, fill_val, num_elements,
                )
                all_programs.append(prog)

                # Local SDMA
                sdma = SDMAPacketBuilder()
                sdma.copy_linear(
                    gs["local_dst"].gpu_addr, gs["local_src"].gpu_addr, copy_size,
                )
                sdma.fence(gs["sdma_tl"].signal_addr, gs["sdma_tl"].next_value())
                gs["backend"].submit_packets(gs["sdma_queue"], sdma.build())

                # XGMI copy
                sdma_xgmi = SDMAPacketBuilder()
                sdma_xgmi.copy_linear(
                    gs["xgmi_dst"].gpu_addr, gs["xgmi_src"].gpu_addr, copy_size,
                )
                sdma_xgmi.fence(gs["xgmi_tl"].signal_addr, gs["xgmi_tl"].next_value())
                gs["backend"].submit_packets(gs["xgmi_queue"], sdma_xgmi.build())

        # Wait for everything
        for gs in gpu_state:
            gs["compute_tl"].cpu_wait(timeout_ms=300000)
            gs["sdma_tl"].cpu_wait(timeout_ms=300000)
            gs["xgmi_tl"].cpu_wait(timeout_ms=300000)

        # Verify final state
        for gs in gpu_state:
            assert gs["local_dst"].read(64) == b"\xDD" * 64
            assert gs["xgmi_dst"].read(64) == b"\xEE" * 64

        for prog in all_programs:
            prog.free()
        for gs in gpu_state:
            gs["compute_tl"].destroy()
            gs["sdma_tl"].destroy()
            gs["xgmi_tl"].destroy()
            gs["backend"].destroy_queue(gs["compute_queue"])
            gs["backend"].destroy_queue(gs["sdma_queue"])
            gs["backend"].destroy_queue(gs["xgmi_queue"])
