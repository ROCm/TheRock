"""Integration tests for multi-GPU P2P operations.

Tests are organized by GPU count requirement:
- @requires_multi_gpu: needs 2+ GPUs
- @requires_3_gpus: needs 3+ GPUs

All tests are designed to work with up to 8 GPUs (MI300X configurations).
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tests.integration.conftest import requires_3_gpus, requires_multi_gpu

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
FILL_KERNEL = FIXTURES_DIR / "fill_kernel_gfx942.co"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_peer_pair(ctx, dev_a, dev_b, size=4096, location="vram"):
    """Allocate src on dev_a and dst on dev_b with bidirectional peer access."""
    src = dev_a.alloc(size, location=location)
    dst = dev_b.alloc(size, location=location)
    ctx.enable_peer_access(src, dev_b)
    ctx.enable_peer_access(dst, dev_a)
    return src, dst


# ---------------------------------------------------------------------------
# TestMultiGPUContext — constructor, lifecycle, edge cases
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestMultiGPUContext:
    """Test MultiGPUContext construction and lifecycle."""

    def test_open_multiple_devices(self, multi_gpu_context):
        ctx = multi_gpu_context
        assert ctx.num_devices >= 2

    def test_unique_gpu_ids(self, multi_gpu_context):
        ctx = multi_gpu_context
        gpu_ids = [dev.gpu_id for dev in ctx.devices]
        assert len(set(gpu_ids)) == len(gpu_ids), "GPU IDs must be unique"

    def test_all_have_vram(self, multi_gpu_context):
        ctx = multi_gpu_context
        for dev in ctx.devices:
            assert dev.vram_size > 0, f"GPU {dev.device_index} has no VRAM"

    def test_device_index_matches(self, multi_gpu_context):
        ctx = multi_gpu_context
        for i, dev in enumerate(ctx.devices):
            assert dev.device_index == i

    def test_device_accessor(self, multi_gpu_context):
        ctx = multi_gpu_context
        for i in range(ctx.num_devices):
            dev = ctx.device(i)
            assert dev is ctx.devices[i]

    def test_context_manager_protocol(self):
        """with-statement opens and closes cleanly."""
        from amd_gpu_driver.multi_gpu import MultiGPUContext

        with MultiGPUContext() as ctx:
            assert ctx.num_devices >= 2
            names = [dev.name for dev in ctx.devices]
            assert all(names)
        # After exit, devices list is empty (closed)
        assert ctx.num_devices == 0

    def test_open_specific_device_indices(self):
        """Explicitly requesting [0, 1] returns exactly those devices."""
        from amd_gpu_driver.multi_gpu import MultiGPUContext

        with MultiGPUContext(device_indices=[0, 1]) as ctx:
            assert ctx.num_devices == 2
            assert ctx.device(0).device_index == 0
            assert ctx.device(1).device_index == 1

    def test_open_single_device_via_context(self):
        """Degenerate case: context with one GPU still works."""
        from amd_gpu_driver.multi_gpu import MultiGPUContext

        with MultiGPUContext(device_indices=[0]) as ctx:
            assert ctx.num_devices == 1
            buf = ctx.device(0).alloc(4096, location="vram")
            buf.fill(0x42)
            assert buf.read(4) == b"\x42" * 4

    def test_max_devices_cap(self):
        """max_devices limits how many GPUs are opened."""
        from amd_gpu_driver.multi_gpu import MultiGPUContext

        with MultiGPUContext(max_devices=2) as ctx:
            assert ctx.num_devices <= 2

    def test_repr(self, multi_gpu_context):
        r = repr(multi_gpu_context)
        assert "MultiGPUContext" in r
        assert str(multi_gpu_context.num_devices) in r

    def test_devices_returns_copy(self, multi_gpu_context):
        """devices property returns a new list, not the internal list."""
        ctx = multi_gpu_context
        list_a = ctx.devices
        list_b = ctx.devices
        assert list_a is not list_b
        assert list_a == list_b

    def test_all_devices_have_gfx_target(self, multi_gpu_context):
        ctx = multi_gpu_context
        for dev in ctx.devices:
            assert dev.gfx_target.startswith("gfx"), (
                f"GPU {dev.device_index} has unexpected gfx_target: {dev.gfx_target}"
            )

    def test_all_devices_have_name(self, multi_gpu_context):
        ctx = multi_gpu_context
        for dev in ctx.devices:
            assert dev.name, f"GPU {dev.device_index} has empty name"


# ---------------------------------------------------------------------------
# TestPeerMemoryAccess — P2P mapping scenarios
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestPeerMemoryAccess:
    """Test P2P memory mapping between GPUs."""

    def test_enable_peer_access(self, multi_gpu_context):
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        buf = dev0.alloc(4096, location="vram")
        ctx.enable_peer_access(buf, dev1)
        assert dev0.gpu_id in buf.handle.mapped_gpu_ids
        assert dev1.gpu_id in buf.handle.mapped_gpu_ids

    def test_bidirectional_peer_access(self, multi_gpu_context):
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        buf0 = dev0.alloc(4096, location="vram")
        buf1 = dev1.alloc(4096, location="vram")

        ctx.enable_peer_access(buf0, dev1)
        ctx.enable_peer_access(buf1, dev0)

        assert dev1.gpu_id in buf0.handle.mapped_gpu_ids
        assert dev0.gpu_id in buf1.handle.mapped_gpu_ids

    def test_enable_peer_access_multiple_peers(self, multi_gpu_context):
        """Map one buffer to all other GPUs in a single call."""
        ctx = multi_gpu_context
        dev0 = ctx.devices[0]
        peers = ctx.devices[1:]

        buf = dev0.alloc(4096, location="vram")
        ctx.enable_peer_access(buf, *peers)

        for peer in peers:
            assert peer.gpu_id in buf.handle.mapped_gpu_ids
        assert dev0.gpu_id in buf.handle.mapped_gpu_ids

    def test_enable_peer_access_idempotent(self, multi_gpu_context):
        """Calling enable_peer_access twice on the same peer does not crash."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        buf = dev0.alloc(4096, location="vram")
        ctx.enable_peer_access(buf, dev1)
        # Second call should not raise
        ctx.enable_peer_access(buf, dev1)
        assert dev1.gpu_id in buf.handle.mapped_gpu_ids

    def test_peer_access_gtt_buffer(self, multi_gpu_context):
        """GTT (system memory) buffers can also be peer-mapped."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        buf = dev0.alloc(4096, location="gtt")
        ctx.enable_peer_access(buf, dev1)
        assert dev1.gpu_id in buf.handle.mapped_gpu_ids

    def test_owner_gpu_id_set(self, multi_gpu_context):
        """After alloc, owner_gpu_id reflects the allocating device."""
        ctx = multi_gpu_context
        for dev in ctx.devices:
            buf = dev.alloc(4096, location="vram")
            assert buf.handle.owner_gpu_id == dev.gpu_id

    def test_peer_access_all_devices_vram(self, multi_gpu_context):
        """Map a buffer from each device to every other device."""
        ctx = multi_gpu_context
        for owner in ctx.devices:
            peers = [d for d in ctx.devices if d is not owner]
            buf = owner.alloc(4096, location="vram")
            ctx.enable_peer_access(buf, *peers)
            for peer in peers:
                assert peer.gpu_id in buf.handle.mapped_gpu_ids


# ---------------------------------------------------------------------------
# TestPeerCopy — cross-GPU SDMA copies
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestPeerCopy:
    """Test cross-GPU SDMA copies."""

    def test_basic_copy_4kb(self, multi_gpu_context):
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)
        src.fill(0xAA)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0)
        ctx.synchronize_all()

        assert dst.read(16) == b"\xAA" * 16

    def test_bidirectional_copy(self, multi_gpu_context):
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        buf0 = dev0.alloc(4096, location="vram")
        buf1 = dev1.alloc(4096, location="vram")
        buf0.fill(0x11)
        buf1.fill(0x22)

        ctx.enable_peer_access(buf0, dev1)
        ctx.enable_peer_access(buf1, dev0)

        dst1 = dev1.alloc(4096, location="vram")
        ctx.enable_peer_access(dst1, dev0)
        ctx.copy_peer(dst1, dev1, buf0, dev0)

        dst0 = dev0.alloc(4096, location="vram")
        ctx.enable_peer_access(dst0, dev1)
        ctx.copy_peer(dst0, dev0, buf1, dev1)

        ctx.synchronize_all()

        assert dst1.read(16) == b"\x11" * 16
        assert dst0.read(16) == b"\x22" * 16

    def test_large_copy_16mb(self, multi_gpu_context):
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        size = 16 * 1024 * 1024
        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=size)
        src.fill(0xBB)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size)
        ctx.synchronize_all()

        assert dst.read(64) == b"\xBB" * 64
        assert dst.read(64, offset=size - 64) == b"\xBB" * 64

    def test_copy_partial_peer(self, multi_gpu_context):
        """P2P copy with explicit size smaller than buffer."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=8192)
        src.fill(0xDD)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size=4096)
        ctx.synchronize_all()

        assert dst.read(16) == b"\xDD" * 16

    def test_copy_preserves_uncopied_region_peer(self, multi_gpu_context):
        """Partial P2P copy does not overwrite beyond the copy size."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        buf_size = 8192
        copy_size = 4096
        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=buf_size)
        src.fill(0xAA)
        dst.fill(0xFF)

        ctx.copy_peer(dst, dev1, src, dev0, size=copy_size)
        ctx.synchronize_all()

        assert dst.read(16, offset=0) == b"\xAA" * 16
        assert dst.read(16, offset=copy_size) == b"\xFF" * 16

    def test_copy_non_power_of_two_peer(self, multi_gpu_context):
        """P2P copy of a non-power-of-two byte count."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)
        pattern = bytes(i & 0xFF for i in range(1000))
        src.write(pattern)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size=1000)
        ctx.synchronize_all()

        assert dst.read(1000) == pattern

    def test_copy_default_size_peer(self, multi_gpu_context):
        """copy_peer with size=None copies min(dst.size, src.size)."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src = dev0.alloc(4096, location="vram")
        dst = dev1.alloc(8192, location="vram")
        ctx.enable_peer_access(src, dev1)
        ctx.enable_peer_access(dst, dev0)

        src.fill(0xEE)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0)  # size=None -> min(8192,4096)=4096
        ctx.synchronize_all()

        assert dst.read(16) == b"\xEE" * 16

    def test_sequential_peer_copies(self, multi_gpu_context):
        """Multiple P2P copies with different patterns, each verified."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)

        for value in [0x11, 0x22, 0x33, 0x44, 0xFF]:
            src.fill(value)
            dst.fill(0x00)
            ctx.copy_peer(dst, dev1, src, dev0)
            ctx.synchronize_all()
            data = dst.read(16)
            assert data == bytes([value]) * 16, f"Failed for 0x{value:02x}"

    def test_copy_data_integrity_sequential_u32(self, multi_gpu_context):
        """Copy sequential u32 values across GPUs and verify each one."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        count = 1024
        size = count * 4
        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=size)

        pattern = struct.pack(f"<{count}I", *range(count))
        src.write(pattern)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size)
        ctx.synchronize_all()

        data = dst.read(size)
        values = struct.unpack(f"<{count}I", data)
        for i in range(count):
            assert values[i] == i, f"Mismatch at index {i}: got {values[i]}"

    def test_copy_alternating_pattern_peer(self, multi_gpu_context):
        """Copy alternating bit pattern to detect bit-level errors."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        size = 4096
        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=size)
        pattern = b"\x55\xAA" * (size // 2)
        src.write(pattern)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size)
        ctx.synchronize_all()

        assert dst.read(size) == pattern

    def test_copy_all_byte_values_peer(self, multi_gpu_context):
        """All 256 byte values survive a cross-GPU copy."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        size = 4096
        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=size)
        pattern = bytes(range(256)) * (size // 256)
        src.write(pattern)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size)
        ctx.synchronize_all()

        assert dst.read(size) == pattern

    def test_copy_8mb_chunked_peer(self, multi_gpu_context):
        """8 MB P2P copy — exceeds SDMA_MAX_COPY_SIZE, requires chunking."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        size = 8 * 1024 * 1024
        src, dst = _setup_peer_pair(ctx, dev0, dev1, size=size)
        src.fill(0xAB)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0, size)
        ctx.synchronize_all()

        assert dst.read(64) == b"\xAB" * 64
        assert dst.read(64, offset=size - 64) == b"\xAB" * 64


# ---------------------------------------------------------------------------
# TestPeerCopySynchronization — sync correctness
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestPeerCopySynchronization:
    """Test synchronization semantics for cross-GPU operations."""

    def test_synchronize_single_device(self, multi_gpu_context):
        """synchronize(dev) waits only for that device's timeline."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)
        src.fill(0xCC)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0)
        ctx.synchronize(dev0)  # dev0 is the source, its timeline advances

        assert dst.read(16) == b"\xCC" * 16

    def test_synchronize_without_pending_ops(self, multi_gpu_context):
        """synchronize on a device with no pending ops is a no-op."""
        ctx = multi_gpu_context
        for dev in ctx.devices:
            ctx.synchronize(dev)  # Should not raise

    def test_synchronize_all_without_pending_ops(self, multi_gpu_context):
        """synchronize_all with no pending ops is a no-op."""
        ctx = multi_gpu_context
        ctx.synchronize_all()  # Should not raise

    def test_multiple_peer_copies_then_sync(self, multi_gpu_context):
        """Submit copies from multiple sources, then wait once."""
        ctx = multi_gpu_context
        n = ctx.num_devices

        results = []
        for i in range(n):
            j = (i + 1) % n
            dev_src = ctx.devices[i]
            dev_dst = ctx.devices[j]
            src, dst = _setup_peer_pair(ctx, dev_src, dev_dst)
            fill_val = (i + 1) * 0x10
            src.fill(fill_val)
            dst.fill(0x00)
            ctx.copy_peer(dst, dev_dst, src, dev_src)
            results.append((dst, fill_val))

        ctx.synchronize_all()

        for dst, fill_val in results:
            data = dst.read(16)
            assert data == bytes([fill_val]) * 16, (
                f"Failed for fill 0x{fill_val:02x}"
            )

    def test_back_to_back_sync(self, multi_gpu_context):
        """Two synchronize_all calls back-to-back do not crash."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)
        src.fill(0x77)
        dst.fill(0x00)

        ctx.copy_peer(dst, dev1, src, dev0)
        ctx.synchronize_all()
        ctx.synchronize_all()  # Second call is a no-op

        assert dst.read(16) == b"\x77" * 16


# ---------------------------------------------------------------------------
# TestPeerCopyAllPairs — every direction between all GPUs
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestPeerCopyAllPairs:
    """Test P2P copy between every ordered pair of GPUs."""

    def test_all_pairs_copy(self, multi_gpu_context):
        """Copy between every (i, j) pair where i != j."""
        ctx = multi_gpu_context
        n = ctx.num_devices

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dev_src = ctx.devices[i]
                dev_dst = ctx.devices[j]

                src, dst = _setup_peer_pair(ctx, dev_src, dev_dst)
                fill_val = ((i * n + j) & 0xFF) | 0x01  # nonzero
                src.fill(fill_val)
                dst.fill(0x00)

                ctx.copy_peer(dst, dev_dst, src, dev_src)
                ctx.synchronize_all()

                data = dst.read(16)
                assert data == bytes([fill_val]) * 16, (
                    f"P2P copy GPU{i}->GPU{j} failed for 0x{fill_val:02x}"
                )


# ---------------------------------------------------------------------------
# TestXGMIQueue — XGMI-specific queue tests
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestXGMIQueue:
    """Test XGMI SDMA queue creation and fallback."""

    def test_create_xgmi_sdma_queue(self, multi_gpu_context):
        """Create an XGMI queue on a device that supports it."""
        from amd_gpu_driver.backends.base import QueueType
        from amd_gpu_driver.backends.kfd import KFDDevice

        ctx = multi_gpu_context
        for dev in ctx.devices:
            backend = dev.backend
            assert isinstance(backend, KFDDevice)
            node = backend.node
            if node is not None and node.num_sdma_xgmi_engines > 0:
                queue = backend.create_xgmi_sdma_queue()
                assert queue.queue_type == QueueType.SDMA_XGMI
                assert queue.queue_id >= 0
                backend.destroy_queue(queue)
                return  # Found at least one device with XGMI

        pytest.skip("No device with XGMI SDMA engines found")

    def test_copy_peer_uses_sdma_fallback(self, multi_gpu_context):
        """copy_peer works even if XGMI engines are absent (uses SDMA)."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)
        src.fill(0xFE)
        dst.fill(0x00)

        # copy_peer internally decides XGMI vs SDMA based on topology
        ctx.copy_peer(dst, dev1, src, dev0)
        ctx.synchronize_all()

        assert dst.read(16) == b"\xFE" * 16


# ---------------------------------------------------------------------------
# TestTimelineSemaphorePeers — cross-GPU timeline mapping
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestTimelineSemaphorePeers:
    """Test TimelineSemaphore peer mapping."""

    def test_timeline_map_to_peers(self, multi_gpu_context):
        """Map a timeline's signal memory to peer GPUs."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        dev0 = ctx.devices[0]
        peer_ids = [d.gpu_id for d in ctx.devices[1:]]

        timeline = TimelineSemaphore(dev0.backend)
        timeline.map_to_peers(peer_ids)

        # Verify signal memory is mapped to all peers
        for gid in peer_ids:
            assert gid in timeline._signal_mem.mapped_gpu_ids

        timeline.destroy()


# ---------------------------------------------------------------------------
# TestMultiGPUAllocation — memory allocation across devices
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestMultiGPUAllocation:
    """Test memory allocation on each device in a multi-GPU context."""

    def test_alloc_vram_on_each_device(self, multi_gpu_context):
        ctx = multi_gpu_context
        for dev in ctx.devices:
            buf = dev.alloc(4096, location="vram")
            assert buf.size >= 4096
            assert buf.gpu_addr != 0
            assert buf.handle.owner_gpu_id == dev.gpu_id

    def test_alloc_gtt_on_each_device(self, multi_gpu_context):
        ctx = multi_gpu_context
        for dev in ctx.devices:
            buf = dev.alloc(4096, location="gtt")
            assert buf.size >= 4096
            assert buf.gpu_addr != 0
            assert buf.handle.owner_gpu_id == dev.gpu_id

    def test_independent_buffer_readwrite(self, multi_gpu_context):
        """Write different data to each GPU, verify no cross-contamination."""
        ctx = multi_gpu_context
        buffers = []
        for i, dev in enumerate(ctx.devices):
            buf = dev.alloc(4096, location="vram")
            fill_val = (i + 1) * 0x11
            buf.fill(fill_val)
            buffers.append((buf, fill_val))

        for buf, fill_val in buffers:
            data = buf.read(16)
            assert data == bytes([fill_val]) * 16, (
                f"Cross-contamination for 0x{fill_val:02x}"
            )

    def test_large_alloc_on_each_device(self, multi_gpu_context):
        """16 MB VRAM allocation on each device."""
        ctx = multi_gpu_context
        size = 16 * 1024 * 1024
        for dev in ctx.devices:
            buf = dev.alloc(size, location="vram")
            assert buf.size >= size


# ---------------------------------------------------------------------------
# TestIndependentOps — each GPU works independently
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestIndependentOps:
    """Test independent operations on each GPU."""

    def test_independent_sdma_copies(self, multi_gpu_context):
        """Each GPU performs its own SDMA copy independently."""
        ctx = multi_gpu_context
        results = []

        for i, dev in enumerate(ctx.devices):
            src = dev.alloc(4096, location="vram")
            dst = dev.alloc(4096, location="vram")
            fill_val = (i + 1) * 0x11
            src.fill(fill_val)
            dst.fill(0x00)
            dev.copy(dst, src)
            results.append((dev, dst, fill_val))

        for dev, dst, fill_val in results:
            dev.synchronize()
            data = dst.read(16)
            assert data == bytes([fill_val]) * 16, (
                f"GPU {dev.device_index} SDMA failed for 0x{fill_val:02x}"
            )

    def test_independent_sdma_copy_large(self, multi_gpu_context):
        """Each GPU copies 1 MB independently."""
        ctx = multi_gpu_context
        size = 1024 * 1024

        for dev in ctx.devices:
            src = dev.alloc(size, location="vram")
            dst = dev.alloc(size, location="vram")
            src.fill(0xCD)
            dst.fill(0x00)
            dev.copy(dst, src, size)

        for dev in ctx.devices:
            dev.synchronize()

    def test_independent_gtt_to_vram_copies(self, multi_gpu_context):
        """Each GPU copies GTT->VRAM independently."""
        ctx = multi_gpu_context
        results = []

        for i, dev in enumerate(ctx.devices):
            src = dev.alloc(4096, location="gtt")
            dst = dev.alloc(4096, location="vram")
            pattern = bytes([(i * 37 + j) & 0xFF for j in range(4096)])
            src.write(pattern)
            dst.fill(0x00)
            dev.copy(dst, src, 4096)
            results.append((dev, dst, pattern))

        for dev, dst, pattern in results:
            dev.synchronize()
            data = dst.read(4096)
            assert data == pattern, f"GPU {dev.device_index} GTT->VRAM failed"


# ---------------------------------------------------------------------------
# TestMultiGPUDispatch — kernel dispatch on each GPU
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestMultiGPUDispatch:
    """Test compute operations on each GPU independently."""

    def test_independent_fills(self, multi_gpu_context):
        """CPU fill on each device with unique values."""
        ctx = multi_gpu_context

        buffers = []
        for i, dev in enumerate(ctx.devices):
            buf = dev.alloc(4096, location="vram")
            buf.fill(i & 0xFF)
            buffers.append((dev, buf, i & 0xFF))

        for dev, buf, val in buffers:
            data = buf.read(16)
            assert data == bytes([val]) * 16

    def test_submit_nop_each_gpu(self, multi_gpu_context):
        """Submit NOP PM4 packets to a compute queue on each GPU."""
        from amd_gpu_driver.commands.pm4 import PM4PacketBuilder

        ctx = multi_gpu_context
        for dev in ctx.devices:
            backend = dev.backend
            queue = backend.create_compute_queue()
            pm4 = PM4PacketBuilder()
            pm4.nop(4)
            backend.submit_packets(queue, pm4.build())
            backend.destroy_queue(queue)

    def test_signal_event_each_gpu(self, multi_gpu_context):
        """Create and destroy a signal event on each GPU."""
        ctx = multi_gpu_context
        for dev in ctx.devices:
            backend = dev.backend
            signal = backend.create_signal()
            assert signal.event_id > 0
            backend.destroy_signal(signal)

    @pytest.mark.skipif(
        not FILL_KERNEL.exists(),
        reason=f"Requires pre-compiled kernel at {FILL_KERNEL}",
    )
    def test_dispatch_fill_kernel_each_gpu(self, multi_gpu_context):
        """Dispatch fill_kernel on every GPU and verify output."""
        from amd_gpu_driver.sync.timeline import TimelineSemaphore

        ctx = multi_gpu_context
        for dev in ctx.devices:
            # Only dispatch if device matches the kernel's ISA
            if dev.gfx_target != "gfx942":
                continue

            program = dev.load_program(str(FILL_KERNEL))
            num_elements = 256
            out_buf = dev.alloc(num_elements * 4, location="vram")
            out_buf.fill(0x00)

            backend = dev.backend
            queue = backend.create_compute_queue()
            timeline = TimelineSemaphore(backend)

            fill_value = 0xDEADBEEF
            program.dispatch(
                queue,
                grid=(num_elements // 64, 1, 1),
                block=(64, 1, 1),
                args=[out_buf, fill_value],
                timeline=timeline,
            )
            timeline.cpu_wait(timeout_ms=5000)

            data = out_buf.read(num_elements * 4)
            values = struct.unpack(f"<{num_elements}I", data)
            assert all(v == fill_value for v in values), (
                f"GPU {dev.device_index}: fill_kernel mismatch"
            )

            program.free()
            timeline.destroy()
            backend.destroy_queue(queue)
            out_buf.free()

    def test_multiple_compute_queues_each_gpu(self, multi_gpu_context):
        """Create 3 compute queues on each GPU, verify unique IDs."""
        ctx = multi_gpu_context
        for dev in ctx.devices:
            backend = dev.backend
            queues = [backend.create_compute_queue() for _ in range(3)]
            ids = [q.queue_id for q in queues]
            assert len(set(ids)) == 3, (
                f"GPU {dev.device_index}: non-unique queue IDs"
            )
            for q in queues:
                backend.destroy_queue(q)


# ---------------------------------------------------------------------------
# TestMultiGPUCopyChain — chained copies across devices
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestMultiGPUCopyChain:
    """Test chained P2P copies through multiple GPUs."""

    def test_chain_copy_two_hops(self, multi_gpu_context):
        """Copy A(GPU0) -> B(GPU1) -> C(GPU0), verify C == A."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        size = 4096
        a = dev0.alloc(size, location="vram")
        b = dev1.alloc(size, location="vram")
        c = dev0.alloc(size, location="vram")

        pattern = struct.pack("<I", 0xCAFEBABE) * (size // 4)
        a.write(pattern)
        b.fill(0x00)
        c.fill(0x00)

        # A -> B
        ctx.enable_peer_access(a, dev1)
        ctx.enable_peer_access(b, dev0)
        ctx.copy_peer(b, dev1, a, dev0, size)
        ctx.synchronize_all()

        # B -> C
        ctx.enable_peer_access(c, dev1)
        ctx.copy_peer(c, dev0, b, dev1, size)
        ctx.synchronize_all()

        assert c.read(size) == pattern


@requires_3_gpus
class TestMultiGPUCopyChain3:
    """Test P2P copy chains requiring 3+ GPUs."""

    def test_chain_copy_three_gpus(self, multi_gpu_context):
        """Copy A(GPU0) -> B(GPU1) -> C(GPU2), verify C == A."""
        ctx = multi_gpu_context
        dev0, dev1, dev2 = ctx.devices[0], ctx.devices[1], ctx.devices[2]

        size = 4096
        a = dev0.alloc(size, location="vram")
        b = dev1.alloc(size, location="vram")
        c = dev2.alloc(size, location="vram")

        pattern = struct.pack("<I", 0xDEADC0DE) * (size // 4)
        a.write(pattern)
        b.fill(0x00)
        c.fill(0x00)

        ctx.enable_peer_access(a, dev1)
        ctx.enable_peer_access(b, dev0, dev2)
        ctx.enable_peer_access(c, dev1)

        ctx.copy_peer(b, dev1, a, dev0, size)
        ctx.synchronize_all()

        ctx.copy_peer(c, dev2, b, dev1, size)
        ctx.synchronize_all()

        assert c.read(size) == pattern

    def test_ring_copy(self, multi_gpu_context):
        """Copy around a ring: GPU0 -> GPU1 -> GPU2 -> ... -> GPU0."""
        ctx = multi_gpu_context
        n = ctx.num_devices
        size = 4096

        # Allocate a buffer on each device
        bufs = []
        for dev in ctx.devices:
            buf = dev.alloc(size, location="vram")
            buf.fill(0x00)
            bufs.append(buf)

        # Write the original pattern on GPU 0
        pattern = struct.pack("<I", 0xFACEFEED) * (size // 4)
        bufs[0].write(pattern)

        # Enable peer access for the ring
        for i in range(n):
            j = (i + 1) % n
            ctx.enable_peer_access(bufs[i], ctx.devices[j])
            ctx.enable_peer_access(bufs[j], ctx.devices[i])

        # Copy around the ring
        for i in range(n):
            j = (i + 1) % n
            ctx.copy_peer(bufs[j], ctx.devices[j], bufs[i], ctx.devices[i], size)
            ctx.synchronize_all()

        # After full ring, the last copy lands back on GPU 0's neighbor
        # which is bufs[0] itself (when n hops are completed)
        # Actually the final destination is bufs[0] after n hops
        assert bufs[0].read(size) == pattern

    def test_broadcast_from_one_gpu(self, multi_gpu_context):
        """Copy from GPU 0 to every other GPU (broadcast pattern)."""
        ctx = multi_gpu_context
        dev0 = ctx.devices[0]
        size = 4096

        src = dev0.alloc(size, location="vram")
        pattern = bytes([(i * 7) & 0xFF for i in range(size)])
        src.write(pattern)

        dsts = []
        for dev in ctx.devices[1:]:
            dst = dev.alloc(size, location="vram")
            dst.fill(0x00)
            ctx.enable_peer_access(src, dev)
            ctx.enable_peer_access(dst, dev0)
            ctx.copy_peer(dst, dev, src, dev0, size)
            dsts.append(dst)

        ctx.synchronize_all()

        for i, dst in enumerate(dsts):
            data = dst.read(size)
            assert data == pattern, f"Broadcast to GPU{i+1} failed"

    def test_gather_to_one_gpu(self, multi_gpu_context):
        """Copy from every GPU to GPU 0 (gather pattern)."""
        ctx = multi_gpu_context
        dev0 = ctx.devices[0]
        size = 4096

        results = []
        for i, dev in enumerate(ctx.devices[1:], start=1):
            src = dev.alloc(size, location="vram")
            dst = dev0.alloc(size, location="vram")
            fill_val = i * 0x11
            src.fill(fill_val)
            dst.fill(0x00)
            ctx.enable_peer_access(src, dev0)
            ctx.enable_peer_access(dst, dev)
            ctx.copy_peer(dst, dev0, src, dev)
            results.append((dst, fill_val))

        ctx.synchronize_all()

        for dst, fill_val in results:
            data = dst.read(16)
            assert data == bytes([fill_val]) * 16, (
                f"Gather failed for 0x{fill_val:02x}"
            )


# ---------------------------------------------------------------------------
# TestMultiGPUStress — heavier workloads across all GPUs
# ---------------------------------------------------------------------------


@requires_multi_gpu
class TestMultiGPUStress:
    """Heavier multi-GPU workloads."""

    def test_repeated_peer_copies(self, multi_gpu_context):
        """Repeat P2P copy 10 times on the same buffers."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        src, dst = _setup_peer_pair(ctx, dev0, dev1)

        for i in range(10):
            fill_val = (i * 23 + 1) & 0xFF
            src.fill(fill_val)
            dst.fill(0x00)
            ctx.copy_peer(dst, dev1, src, dev0)
            ctx.synchronize_all()
            assert dst.read(4) == bytes([fill_val]) * 4, f"Iteration {i} failed"

    def test_many_small_buffers_per_device(self, multi_gpu_context):
        """Allocate 10 small buffers on each GPU, write/read each."""
        ctx = multi_gpu_context
        all_buffers = []

        for dev in ctx.devices:
            for j in range(10):
                buf = dev.alloc(4096, location="vram")
                val = (dev.device_index * 10 + j) & 0xFF
                buf.fill(val)
                all_buffers.append((buf, val))

        for buf, val in all_buffers:
            assert buf.read(1) == bytes([val])

    def test_concurrent_bidirectional_copies(self, multi_gpu_context):
        """Submit copies in both directions before synchronizing."""
        ctx = multi_gpu_context
        dev0, dev1 = ctx.devices[0], ctx.devices[1]

        size = 4096
        a0 = dev0.alloc(size, location="vram")
        a1 = dev1.alloc(size, location="vram")
        b0 = dev0.alloc(size, location="vram")
        b1 = dev1.alloc(size, location="vram")

        a0.fill(0xAA)
        a1.fill(0x00)
        b1.fill(0xBB)
        b0.fill(0x00)

        ctx.enable_peer_access(a0, dev1)
        ctx.enable_peer_access(a1, dev0)
        ctx.enable_peer_access(b0, dev1)
        ctx.enable_peer_access(b1, dev0)

        # Submit both directions
        ctx.copy_peer(a1, dev1, a0, dev0, size)
        ctx.copy_peer(b0, dev0, b1, dev1, size)

        ctx.synchronize_all()

        assert a1.read(16) == b"\xAA" * 16
        assert b0.read(16) == b"\xBB" * 16
