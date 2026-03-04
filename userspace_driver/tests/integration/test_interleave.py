"""Interleaved compute + communication tests using compute-bound GPU kernels.

Uses busy_kernel (Knuth hash loop) to keep CUs occupied for seconds while
simultaneous SDMA/XGMI transfers run.  This ensures GPU utilization (gfx_activity)
registers on ``amd-smi monitor`` at 1-second sampling.

Kernel signatures (from compute_kernels.hip):
    busy_kernel(uint32_t* out, uint32_t seed, uint32_t iters)
    reduce_kernel(const uint32_t* in, uint32_t* out, uint32_t num_elements,
                  uint32_t elements_per_thread)
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
COMPUTE_KERNEL = FIXTURES_DIR / "compute_kernels_gfx942.co"
FILL_KERNEL = FIXTURES_DIR / "fill_kernel_gfx942.co"

requires_compute_kernel = pytest.mark.skipif(
    not COMPUTE_KERNEL.exists(),
    reason=f"Requires pre-compiled kernel at {COMPUTE_KERNEL}",
)
requires_fill_kernel = pytest.mark.skipif(
    not FILL_KERNEL.exists(),
    reason=f"Requires pre-compiled kernel at {FILL_KERNEL}",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Iteration counts calibrated for MI300X (304 CUs, ~2.1 GHz).
# 65K threads fully saturate 304 CUs.  100K iters ≈ 120s, so we scale down
# to keep test runtimes reasonable while maintaining measurable workloads.
BUSY_ITERS_SHORT = 2_000      # ~2-3 seconds
BUSY_ITERS_MEDIUM = 5_000     # ~6-8 seconds
BUSY_ITERS_LONG = 10_000      # ~12-15 seconds

NUM_THREADS_DEFAULT = 65536   # 64K threads = 1024 wavefronts (saturates 304 CUs)
SDMA_COPY_SIZE = 64 * 1024 * 1024  # 64 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch_busy(dev, queue, timeline, out_buf, num_threads, seed, iters):
    """Dispatch busy_kernel: each thread does `iters` Knuth hash iterations."""
    program = dev.load_program(str(COMPUTE_KERNEL), kernel_name="busy_kernel")
    program.dispatch(
        queue,
        grid=(num_threads // 64, 1, 1),
        block=(64, 1, 1),
        args=[out_buf, (seed, 4), (iters, 4)],
        timeline=timeline,
    )
    return program


def _dispatch_reduce(dev, queue, timeline, in_buf, out_buf, num_elements,
                     elements_per_thread):
    """Dispatch reduce_kernel: each thread sums `elements_per_thread` values."""
    num_threads = num_elements // elements_per_thread
    program = dev.load_program(str(COMPUTE_KERNEL), kernel_name="reduce_kernel")
    program.dispatch(
        queue,
        grid=(num_threads // 64, 1, 1),
        block=(64, 1, 1),
        args=[in_buf, out_buf, (num_elements, 4), (elements_per_thread, 4)],
        timeline=timeline,
    )
    return program


def _sdma_copy(backend, sdma_queue, sdma_tl, dst_addr, src_addr, size):
    """Submit an SDMA linear copy with fence."""
    from amd_gpu_driver.commands.sdma import SDMAPacketBuilder

    sdma = SDMAPacketBuilder()
    sdma.copy_linear(dst_addr, src_addr, size)
    sdma.fence(sdma_tl.signal_addr, sdma_tl.next_value())
    backend.submit_packets(sdma_queue, sdma.build())


def _verify_busy_output(buf, seed, iters, num_threads, samples=8):
    """Verify busy_kernel output by recomputing the Knuth hash for a few threads.

    For large iteration counts (>10000), just verify output is non-trivial
    (not seed+idx, meaning the hash loop actually ran).
    """
    data = buf.read(samples * 4)
    if iters <= 10000:
        # Full verification for small iter counts
        for idx in range(samples):
            expected = (seed + idx) & 0xFFFFFFFF
            for _ in range(iters):
                expected = (expected * 2654435761 + 1) & 0xFFFFFFFF
            actual = int.from_bytes(data[idx * 4:(idx + 1) * 4], "little")
            assert actual == expected, (
                f"Thread {idx}: expected 0x{expected:08X}, got 0x{actual:08X}"
            )
    else:
        # For large iters, just check output differs from seed+idx (hash ran)
        for idx in range(samples):
            actual = int.from_bytes(data[idx * 4:(idx + 1) * 4], "little")
            trivial = (seed + idx) & 0xFFFFFFFF
            assert actual != trivial, (
                f"Thread {idx}: output == seed+idx ({trivial}), hash loop didn't run"
            )
            assert actual != 0, f"Thread {idx}: output is zero"


def _get_xgmi_or_sdma_queue(backend):
    """Create an XGMI SDMA queue if available, else regular SDMA."""
    node = backend.node
    if node is not None and node.num_sdma_xgmi_engines > 0:
        return backend.create_xgmi_sdma_queue()
    return backend.create_sdma_queue()


# ---------------------------------------------------------------------------
# TestSingleGPUInterleave — compute-bound kernel + SDMA on one GPU
# ---------------------------------------------------------------------------


@requires_gpu
class TestSingleGPUInterleave:
    """Overlap compute-bound kernel with SDMA on a single GPU."""

    @requires_compute_kernel
    def test_busy_kernel_basic(self, amd_device):
        """Basic busy_kernel dispatch: verify correct output."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_threads = NUM_THREADS_DEFAULT
        out = amd_device.alloc(num_threads * 4, location="vram")
        out.fill(0)
        queue = backend.create_compute_queue()
        tl = TimelineSemaphore(backend)

        seed, iters = 42, 10000  # Small iters for quick verification
        prog = _dispatch_busy(amd_device, queue, tl, out, num_threads, seed, iters)
        tl.cpu_wait(timeout_ms=30000)

        _verify_busy_output(out, seed, iters, num_threads)

        prog.free()
        tl.destroy()
        backend.destroy_queue(queue)

    @requires_compute_kernel
    def test_busy_kernel_sustained(self, amd_device):
        """Sustained compute: 500K iterations should take several seconds."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_threads = NUM_THREADS_DEFAULT
        out = amd_device.alloc(num_threads * 4, location="vram")
        out.fill(0)
        queue = backend.create_compute_queue()
        tl = TimelineSemaphore(backend)

        t0 = time.time()
        prog = _dispatch_busy(
            amd_device, queue, tl, out, num_threads, 99, BUSY_ITERS_MEDIUM,
        )
        tl.cpu_wait(timeout_ms=60000)
        elapsed = time.time() - t0

        # Should take at least 2 seconds (CUs actually busy)
        assert elapsed > 2.0, f"Kernel finished too fast: {elapsed:.3f}s"

        prog.free()
        tl.destroy()
        backend.destroy_queue(queue)

    @requires_compute_kernel
    def test_busy_compute_with_sdma(self, amd_device):
        """Overlap: busy_kernel on compute engine + 64 MB SDMA copy."""
        from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend

        # Compute: busy_kernel
        num_threads = NUM_THREADS_DEFAULT
        compute_out = amd_device.alloc(num_threads * 4, location="vram")
        compute_out.fill(0)
        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)

        # SDMA: 64 MB copy
        sdma_src = amd_device.alloc(SDMA_COPY_SIZE, location="vram")
        sdma_dst = amd_device.alloc(SDMA_COPY_SIZE, location="vram")
        sdma_src.fill(0xCC)
        sdma_dst.fill(0x00)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        # Submit both
        t0 = time.time()
        prog = _dispatch_busy(
            amd_device, compute_queue, compute_tl, compute_out,
            num_threads, 7, BUSY_ITERS_MEDIUM,
        )
        _sdma_copy(
            backend, sdma_queue, sdma_tl,
            sdma_dst.gpu_addr, sdma_src.gpu_addr, SDMA_COPY_SIZE,
        )

        # Wait for both
        compute_tl.cpu_wait(timeout_ms=60000)
        sdma_tl.cpu_wait(timeout_ms=30000)
        elapsed = time.time() - t0

        # Verify SDMA copy
        check = sdma_dst.read(64)
        assert check == b"\xCC" * 64, "SDMA copy verification failed"

        # Verify compute
        _verify_busy_output(compute_out, 7, BUSY_ITERS_MEDIUM, num_threads, samples=4)

        prog.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)

    @requires_compute_kernel
    def test_repeated_busy_with_sdma(self, amd_device):
        """4 iterations of compute + SDMA overlap on a single GPU."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        num_threads = NUM_THREADS_DEFAULT
        iterations = 4

        compute_queue = backend.create_compute_queue()
        compute_tl = TimelineSemaphore(backend)
        sdma_queue = backend.create_sdma_queue()
        sdma_tl = TimelineSemaphore(backend)

        compute_out = amd_device.alloc(num_threads * 4, location="vram")
        sdma_src = amd_device.alloc(SDMA_COPY_SIZE, location="vram")
        sdma_dst = amd_device.alloc(SDMA_COPY_SIZE, location="vram")
        sdma_src.fill(0xDD)

        programs = []
        for i in range(iterations):
            compute_out.fill(0)
            sdma_dst.fill(0)
            prog = _dispatch_busy(
                amd_device, compute_queue, compute_tl, compute_out,
                num_threads, i + 1, BUSY_ITERS_SHORT,
            )
            _sdma_copy(
                backend, sdma_queue, sdma_tl,
                sdma_dst.gpu_addr, sdma_src.gpu_addr, SDMA_COPY_SIZE,
            )
            programs.append(prog)

        compute_tl.cpu_wait(timeout_ms=120000)
        sdma_tl.cpu_wait(timeout_ms=60000)

        # Verify last iteration's SDMA copy
        check = sdma_dst.read(64)
        assert check == b"\xDD" * 64

        for prog in programs:
            prog.free()
        compute_tl.destroy()
        sdma_tl.destroy()
        backend.destroy_queue(compute_queue)
        backend.destroy_queue(sdma_queue)


# ---------------------------------------------------------------------------
# TestReduceKernel — verify reduce_kernel correctness
# ---------------------------------------------------------------------------


@requires_gpu
class TestReduceKernel:
    """Test reduce_kernel correctness."""

    @requires_compute_kernel
    @requires_fill_kernel
    def test_reduce_basic(self, amd_device):
        """Fill input with 1s, reduce 256 elements/thread, verify sums."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        backend = amd_device.backend
        elements_per_thread = 256
        num_threads = 1024
        num_elements = num_threads * elements_per_thread  # 256K elements

        # Fill input with 1s using fill_kernel
        in_buf = amd_device.alloc(num_elements * 4, location="vram")
        in_buf.fill(0)
        fill_prog = amd_device.load_program(str(FILL_KERNEL))
        fill_queue = backend.create_compute_queue()
        fill_tl = TimelineSemaphore(backend)
        fill_prog.dispatch(
            fill_queue,
            grid=(num_elements // 64, 1, 1),
            block=(64, 1, 1),
            args=[in_buf, 1],
            timeline=fill_tl,
        )
        fill_tl.cpu_wait(timeout_ms=10000)

        # Now reduce
        out_buf = amd_device.alloc(num_threads * 4, location="vram")
        out_buf.fill(0)
        reduce_tl = TimelineSemaphore(backend)

        reduce_prog = _dispatch_reduce(
            amd_device, fill_queue, reduce_tl,
            in_buf, out_buf, num_elements, elements_per_thread,
        )
        reduce_tl.cpu_wait(timeout_ms=10000)

        # Each thread should sum to elements_per_thread (256 × 1 = 256)
        data = out_buf.read(32)
        vals = struct.unpack("<8I", data)
        for i, v in enumerate(vals):
            assert v == elements_per_thread, (
                f"Thread {i}: expected {elements_per_thread}, got {v}"
            )

        fill_prog.free()
        reduce_prog.free()
        fill_tl.destroy()
        reduce_tl.destroy()
        backend.destroy_queue(fill_queue)


# ---------------------------------------------------------------------------
# TestAllGPUComputeInterleave — busy_kernel on all GPUs simultaneously
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestAllGPUComputeInterleave:
    """Dispatch compute-bound kernels on all GPUs simultaneously."""

    @requires_compute_kernel
    def test_all_gpus_busy_kernel(self, multi_gpu_context):
        """busy_kernel on all GPUs at once — should show GFX utilization."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        num_threads = NUM_THREADS_DEFAULT

        resources = []
        t0 = time.time()

        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            out = dev.alloc(num_threads * 4, location="vram")
            out.fill(0)
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            prog = _dispatch_busy(
                dev, queue, tl, out, num_threads, i + 1, BUSY_ITERS_MEDIUM,
            )
            resources.append((dev, backend, out, queue, tl, prog, i + 1))

        # Wait for all
        for dev, backend, out, queue, tl, prog, seed in resources:
            tl.cpu_wait(timeout_ms=60000)

        elapsed = time.time() - t0

        # Verify each GPU's output
        for dev, backend, out, queue, tl, prog, seed in resources:
            _verify_busy_output(out, seed, BUSY_ITERS_MEDIUM, num_threads, samples=4)

        # Cleanup
        for dev, backend, out, queue, tl, prog, seed in resources:
            prog.free()
            tl.destroy()
            backend.destroy_queue(queue)

        assert elapsed > 2.0, f"All GPUs finished too fast: {elapsed:.3f}s"

    @requires_compute_kernel
    def test_all_gpus_repeated_busy(self, multi_gpu_context):
        """3 iterations of busy_kernel on all GPUs — sustained compute load."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        num_threads = NUM_THREADS_DEFAULT
        iterations = 3

        resources = []
        for dev in ctx.devices:
            backend = dev.backend
            out = dev.alloc(num_threads * 4, location="vram")
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            resources.append((dev, backend, out, queue, tl))

        t0 = time.time()
        all_progs = []
        for iteration in range(iterations):
            for idx, (dev, backend, out, queue, tl) in enumerate(resources):
                out.fill(0)
                prog = _dispatch_busy(
                    dev, queue, tl, out, num_threads,
                    idx * 100 + iteration, BUSY_ITERS_SHORT,
                )
                all_progs.append(prog)

        for dev, backend, out, queue, tl in resources:
            tl.cpu_wait(timeout_ms=120000)
        elapsed = time.time() - t0

        for dev, backend, out, queue, tl in resources:
            tl.destroy()
            backend.destroy_queue(queue)
        for prog in all_progs:
            prog.free()

        assert elapsed > 3.0, f"Sustained compute finished too fast: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# TestComputeCommsInterleave — busy_kernel + XGMI/SDMA across all GPUs
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestComputeCommsInterleave:
    """Overlap compute-bound kernels with XGMI/SDMA copies across GPUs."""

    @requires_compute_kernel
    def test_compute_and_ring_copy(self, multi_gpu_context):
        """All GPUs run busy_kernel while a ring copy pattern runs on XGMI."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_threads = NUM_THREADS_DEFAULT

        # Setup per-GPU compute resources
        compute_res = []
        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            out = dev.alloc(num_threads * 4, location="vram")
            out.fill(0)
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            compute_res.append((dev, backend, out, queue, tl))

        # Setup ring copy: GPU[i] → GPU[(i+1) % n]
        copy_size = 32 * 1024 * 1024  # 32 MB per link
        ring_res = []
        for i in range(n):
            src_dev = ctx.devices[i]
            dst_dev = ctx.devices[(i + 1) % n]
            src_buf = src_dev.alloc(copy_size, location="vram")
            dst_buf = dst_dev.alloc(copy_size, location="vram")
            src_buf.fill(0xAA + i)
            dst_buf.fill(0)
            # P2P mapping
            ctx.enable_peer_access(src_buf, dst_dev)
            ctx.enable_peer_access(dst_buf, src_dev)
            xgmi_queue = _get_xgmi_or_sdma_queue(src_dev.backend)
            xgmi_tl = TimelineSemaphore(src_dev.backend)
            ring_res.append((src_dev, dst_dev, src_buf, dst_buf,
                             xgmi_queue, xgmi_tl))

        # Submit everything
        t0 = time.time()

        # Compute on all GPUs
        programs = []
        for i, (dev, backend, out, queue, tl) in enumerate(compute_res):
            prog = _dispatch_busy(
                dev, queue, tl, out, num_threads, i + 10, BUSY_ITERS_MEDIUM,
            )
            programs.append(prog)

        # Ring copies
        for src_dev, dst_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            _sdma_copy(
                src_dev.backend, xgmi_q, xgmi_tl,
                dst_buf.gpu_addr, src_buf.gpu_addr, copy_size,
            )

        # Wait for all
        for dev, backend, out, queue, tl in compute_res:
            tl.cpu_wait(timeout_ms=60000)
        for src_dev, dst_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.cpu_wait(timeout_ms=30000)
        elapsed = time.time() - t0

        # Verify compute
        for i, (dev, backend, out, queue, tl) in enumerate(compute_res):
            _verify_busy_output(out, i + 10, BUSY_ITERS_MEDIUM, num_threads, samples=4)

        # Verify ring copies
        for i, (src_dev, dst_dev, src_buf, dst_buf, xgmi_q, xgmi_tl) in enumerate(ring_res):
            check = dst_buf.read(64)
            fill_byte = (0xAA + i) & 0xFF
            assert check == bytes([fill_byte]) * 64, (
                f"Ring copy {i}→{(i+1) % n} failed"
            )

        # Cleanup
        for prog in programs:
            prog.free()
        for dev, backend, out, queue, tl in compute_res:
            tl.destroy()
            backend.destroy_queue(queue)
        for src_dev, dst_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.destroy()
            src_dev.backend.destroy_queue(xgmi_q)

    @requires_compute_kernel
    def test_compute_and_bidirectional_xgmi(self, multi_gpu_context):
        """Busy compute on all GPUs + bidirectional neighbor copies."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_threads = NUM_THREADS_DEFAULT
        copy_size = 32 * 1024 * 1024

        # Compute on all GPUs
        compute_res = []
        programs = []
        for i, dev in enumerate(ctx.devices):
            backend = dev.backend
            out = dev.alloc(num_threads * 4, location="vram")
            out.fill(0)
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            compute_res.append((dev, backend, out, queue, tl))

        # Bidirectional: GPU[i] ↔ GPU[i+1] for adjacent pairs
        xgmi_res = []
        for i in range(n - 1):
            dev_a = ctx.devices[i]
            dev_b = ctx.devices[i + 1]
            # A→B
            buf_ab_src = dev_a.alloc(copy_size, location="vram")
            buf_ab_dst = dev_b.alloc(copy_size, location="vram")
            buf_ab_src.fill(0x10 + i)
            buf_ab_dst.fill(0)
            ctx.enable_peer_access(buf_ab_src, dev_b)
            ctx.enable_peer_access(buf_ab_dst, dev_a)
            q_ab = _get_xgmi_or_sdma_queue(dev_a.backend)
            tl_ab = TimelineSemaphore(dev_a.backend)
            # B→A
            buf_ba_src = dev_b.alloc(copy_size, location="vram")
            buf_ba_dst = dev_a.alloc(copy_size, location="vram")
            buf_ba_src.fill(0x50 + i)
            buf_ba_dst.fill(0)
            ctx.enable_peer_access(buf_ba_src, dev_a)
            ctx.enable_peer_access(buf_ba_dst, dev_b)
            q_ba = _get_xgmi_or_sdma_queue(dev_b.backend)
            tl_ba = TimelineSemaphore(dev_b.backend)
            xgmi_res.append((
                dev_a, dev_b,
                buf_ab_src, buf_ab_dst, q_ab, tl_ab,
                buf_ba_src, buf_ba_dst, q_ba, tl_ba,
            ))

        t0 = time.time()

        # Submit compute
        for i, (dev, backend, out, queue, tl) in enumerate(compute_res):
            prog = _dispatch_busy(
                dev, queue, tl, out, num_threads, i + 20, BUSY_ITERS_MEDIUM,
            )
            programs.append(prog)

        # Submit bidirectional copies
        for entry in xgmi_res:
            dev_a, dev_b = entry[0], entry[1]
            buf_ab_src, buf_ab_dst, q_ab, tl_ab = entry[2:6]
            buf_ba_src, buf_ba_dst, q_ba, tl_ba = entry[6:10]
            _sdma_copy(dev_a.backend, q_ab, tl_ab,
                       buf_ab_dst.gpu_addr, buf_ab_src.gpu_addr, copy_size)
            _sdma_copy(dev_b.backend, q_ba, tl_ba,
                       buf_ba_dst.gpu_addr, buf_ba_src.gpu_addr, copy_size)

        # Wait
        for dev, backend, out, queue, tl in compute_res:
            tl.cpu_wait(timeout_ms=60000)
        for entry in xgmi_res:
            entry[5].cpu_wait(timeout_ms=30000)  # tl_ab
            entry[9].cpu_wait(timeout_ms=30000)  # tl_ba

        elapsed = time.time() - t0

        # Verify compute output
        for i, (dev, backend, out, queue, tl) in enumerate(compute_res):
            _verify_busy_output(out, i + 20, BUSY_ITERS_MEDIUM, num_threads, samples=4)

        # Cleanup
        for prog in programs:
            prog.free()
        for dev, backend, out, queue, tl in compute_res:
            tl.destroy()
            backend.destroy_queue(queue)
        for entry in xgmi_res:
            entry[5].destroy()   # tl_ab
            entry[9].destroy()   # tl_ba
            entry[0].backend.destroy_queue(entry[4])  # q_ab
            entry[1].backend.destroy_queue(entry[8])  # q_ba

    @requires_compute_kernel
    def test_sustained_compute_comms_loop(self, multi_gpu_context):
        """3 iterations of busy compute + ring XGMI on all GPUs."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_threads = NUM_THREADS_DEFAULT
        copy_size = 16 * 1024 * 1024  # 16 MB per link
        iterations = 3

        # Persistent resources
        compute_res = []
        for dev in ctx.devices:
            backend = dev.backend
            out = dev.alloc(num_threads * 4, location="vram")
            queue = backend.create_compute_queue()
            tl = TimelineSemaphore(backend)
            compute_res.append((dev, backend, out, queue, tl))

        ring_res = []
        for i in range(n):
            src_dev = ctx.devices[i]
            dst_dev = ctx.devices[(i + 1) % n]
            src_buf = src_dev.alloc(copy_size, location="vram")
            dst_buf = dst_dev.alloc(copy_size, location="vram")
            ctx.enable_peer_access(src_buf, dst_dev)
            ctx.enable_peer_access(dst_buf, src_dev)
            xgmi_q = _get_xgmi_or_sdma_queue(src_dev.backend)
            xgmi_tl = TimelineSemaphore(src_dev.backend)
            ring_res.append((src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl))

        t0 = time.time()
        all_progs = []

        for iteration in range(iterations):
            # Dispatch compute on all GPUs
            for idx, (dev, backend, out, queue, tl) in enumerate(compute_res):
                out.fill(0)
                prog = _dispatch_busy(
                    dev, queue, tl, out, num_threads,
                    idx * 10 + iteration, BUSY_ITERS_SHORT,
                )
                all_progs.append(prog)

            # Ring copies
            for i, (src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl) in enumerate(ring_res):
                src_buf.fill(0xE0 + iteration)
                dst_buf.fill(0)
                _sdma_copy(
                    src_dev.backend, xgmi_q, xgmi_tl,
                    dst_buf.gpu_addr, src_buf.gpu_addr, copy_size,
                )

        # Wait for all
        for dev, backend, out, queue, tl in compute_res:
            tl.cpu_wait(timeout_ms=180000)
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.cpu_wait(timeout_ms=60000)
        elapsed = time.time() - t0

        # Cleanup
        for prog in all_progs:
            prog.free()
        for dev, backend, out, queue, tl in compute_res:
            tl.destroy()
            backend.destroy_queue(queue)
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.destroy()
            src_dev.backend.destroy_queue(xgmi_q)

        assert elapsed > 5.0, f"Sustained loop too fast: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# TestMaxInterleave — all engines saturated with compute-bound kernels
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestMaxInterleave:
    """Maximum saturation: compute-bound kernels + all SDMA/XGMI engines."""

    @requires_compute_kernel
    def test_all_engines_busy(self, multi_gpu_context):
        """Each GPU: busy_kernel on compute + local SDMA + XGMI ring copy.

        8 GPUs × 3 engines = 24 concurrent operations, all with substantial
        workloads visible to amd-smi.
        """
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_threads = NUM_THREADS_DEFAULT
        local_copy_size = 32 * 1024 * 1024   # 32 MB local SDMA
        ring_copy_size = 32 * 1024 * 1024    # 32 MB XGMI ring

        # Per-GPU: compute + local SDMA
        gpu_res = []
        for dev in ctx.devices:
            backend = dev.backend
            compute_out = dev.alloc(num_threads * 4, location="vram")
            compute_out.fill(0)
            compute_q = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)

            sdma_src = dev.alloc(local_copy_size, location="vram")
            sdma_dst = dev.alloc(local_copy_size, location="vram")
            sdma_src.fill(0x55)
            sdma_dst.fill(0)
            sdma_q = backend.create_sdma_queue()
            sdma_tl = TimelineSemaphore(backend)

            gpu_res.append((
                dev, backend,
                compute_out, compute_q, compute_tl,
                sdma_src, sdma_dst, sdma_q, sdma_tl,
            ))

        # Ring XGMI
        ring_res = []
        for i in range(n):
            src_dev = ctx.devices[i]
            dst_dev = ctx.devices[(i + 1) % n]
            src_buf = src_dev.alloc(ring_copy_size, location="vram")
            dst_buf = dst_dev.alloc(ring_copy_size, location="vram")
            src_buf.fill(0x77)
            dst_buf.fill(0)
            ctx.enable_peer_access(src_buf, dst_dev)
            ctx.enable_peer_access(dst_buf, src_dev)
            xgmi_q = _get_xgmi_or_sdma_queue(src_dev.backend)
            xgmi_tl = TimelineSemaphore(src_dev.backend)
            ring_res.append((src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl))

        t0 = time.time()

        # Submit all compute dispatches
        programs = []
        for i, entry in enumerate(gpu_res):
            dev = entry[0]
            compute_out, compute_q, compute_tl = entry[2], entry[3], entry[4]
            prog = _dispatch_busy(
                dev, compute_q, compute_tl, compute_out,
                num_threads, i + 100, BUSY_ITERS_MEDIUM,
            )
            programs.append(prog)

        # Submit all local SDMA copies
        for entry in gpu_res:
            backend = entry[1]
            sdma_src, sdma_dst, sdma_q, sdma_tl = entry[5], entry[6], entry[7], entry[8]
            _sdma_copy(backend, sdma_q, sdma_tl,
                       sdma_dst.gpu_addr, sdma_src.gpu_addr, local_copy_size)

        # Submit all ring XGMI copies
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            _sdma_copy(src_dev.backend, xgmi_q, xgmi_tl,
                       dst_buf.gpu_addr, src_buf.gpu_addr, ring_copy_size)

        # Wait for everything
        for entry in gpu_res:
            entry[4].cpu_wait(timeout_ms=60000)   # compute_tl
            entry[8].cpu_wait(timeout_ms=30000)   # sdma_tl
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.cpu_wait(timeout_ms=30000)

        elapsed = time.time() - t0

        # Verify compute
        for i, entry in enumerate(gpu_res):
            _verify_busy_output(entry[2], i + 100, BUSY_ITERS_MEDIUM, num_threads, samples=4)

        # Verify local SDMA
        for entry in gpu_res:
            check = entry[6].read(64)  # sdma_dst
            assert check == b"\x55" * 64

        # Verify ring XGMI
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            check = dst_buf.read(64)
            assert check == b"\x77" * 64

        # Cleanup
        for prog in programs:
            prog.free()
        for entry in gpu_res:
            entry[4].destroy()
            entry[8].destroy()
            entry[1].destroy_queue(entry[3])
            entry[1].destroy_queue(entry[7])
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.destroy()
            src_dev.backend.destroy_queue(xgmi_q)

        assert elapsed > 3.0, f"Max saturation finished too fast: {elapsed:.3f}s"

    @requires_compute_kernel
    def test_sustained_max_interleave(self, multi_gpu_context):
        """3 iterations of max saturation: compute + local SDMA + XGMI ring.

        Total operations: 3 × 8 GPUs × 3 engines = 72 submissions.
        """
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        n = ctx.num_devices
        num_threads = NUM_THREADS_DEFAULT
        local_copy_size = 16 * 1024 * 1024
        ring_copy_size = 16 * 1024 * 1024
        iterations = 3

        # Persistent per-GPU resources
        gpu_res = []
        for dev in ctx.devices:
            backend = dev.backend
            compute_out = dev.alloc(num_threads * 4, location="vram")
            compute_q = backend.create_compute_queue()
            compute_tl = TimelineSemaphore(backend)
            sdma_src = dev.alloc(local_copy_size, location="vram")
            sdma_dst = dev.alloc(local_copy_size, location="vram")
            sdma_q = backend.create_sdma_queue()
            sdma_tl = TimelineSemaphore(backend)
            gpu_res.append((
                dev, backend,
                compute_out, compute_q, compute_tl,
                sdma_src, sdma_dst, sdma_q, sdma_tl,
            ))

        ring_res = []
        for i in range(n):
            src_dev = ctx.devices[i]
            dst_dev = ctx.devices[(i + 1) % n]
            src_buf = src_dev.alloc(ring_copy_size, location="vram")
            dst_buf = dst_dev.alloc(ring_copy_size, location="vram")
            ctx.enable_peer_access(src_buf, dst_dev)
            ctx.enable_peer_access(dst_buf, src_dev)
            xgmi_q = _get_xgmi_or_sdma_queue(src_dev.backend)
            xgmi_tl = TimelineSemaphore(src_dev.backend)
            ring_res.append((src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl))

        t0 = time.time()
        all_progs = []

        for iteration in range(iterations):
            # Compute
            for idx, entry in enumerate(gpu_res):
                dev = entry[0]
                entry[2].fill(0)  # compute_out
                prog = _dispatch_busy(
                    dev, entry[3], entry[4], entry[2],
                    num_threads, idx * 10 + iteration, BUSY_ITERS_SHORT,
                )
                all_progs.append(prog)

            # Local SDMA
            for entry in gpu_res:
                entry[5].fill(0x33 + iteration)  # sdma_src
                entry[6].fill(0)                  # sdma_dst
                _sdma_copy(entry[1], entry[7], entry[8],
                           entry[6].gpu_addr, entry[5].gpu_addr, local_copy_size)

            # Ring XGMI
            for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
                src_buf.fill(0x90 + iteration)
                dst_buf.fill(0)
                _sdma_copy(src_dev.backend, xgmi_q, xgmi_tl,
                           dst_buf.gpu_addr, src_buf.gpu_addr, ring_copy_size)

        # Wait for all
        for entry in gpu_res:
            entry[4].cpu_wait(timeout_ms=180000)
            entry[8].cpu_wait(timeout_ms=60000)
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.cpu_wait(timeout_ms=60000)

        elapsed = time.time() - t0

        # Cleanup
        for prog in all_progs:
            prog.free()
        for entry in gpu_res:
            entry[4].destroy()
            entry[8].destroy()
            entry[1].destroy_queue(entry[3])
            entry[1].destroy_queue(entry[7])
        for src_dev, src_buf, dst_buf, xgmi_q, xgmi_tl in ring_res:
            xgmi_tl.destroy()
            src_dev.backend.destroy_queue(xgmi_q)

        assert elapsed > 8.0, f"Sustained max too fast: {elapsed:.3f}s"
