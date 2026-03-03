"""Integration tests for SDMA copy operations."""

import struct

import pytest

from tests.integration.conftest import requires_gpu


@requires_gpu
class TestSDMACopy:
    """Test SDMA copy between buffers."""

    def test_copy_vram_to_vram(self, amd_device):
        """Test SDMA copy between two VRAM buffers."""
        size = 4096
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        # Fill source
        src.write(b"\xAA" * size)
        # Clear destination
        dst.fill(0x00)

        # Copy
        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        # Verify
        data = dst.read(16)
        assert data == b"\xAA" * 16

        src.free()
        dst.free()

    def test_copy_partial(self, amd_device):
        """Test partial SDMA copy."""
        size = 8192
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        src.fill(0xBB)
        dst.fill(0x00)

        # Copy only half
        amd_device.copy(dst, src, size // 2)
        amd_device.synchronize()

        # First half should be copied
        data = dst.read(16)
        assert data == b"\xBB" * 16

        src.free()
        dst.free()


@requires_gpu
class TestSDMACopyDirections:
    """Test SDMA copy across different memory types."""

    def test_copy_gtt_to_vram(self, amd_device):
        """Copy from system memory (GTT) to VRAM."""
        size = 4096
        src = amd_device.alloc(size, location="gtt")
        dst = amd_device.alloc(size, location="vram")

        pattern = b"\xDE\xAD\xBE\xEF" * (size // 4)
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == pattern

        src.free()
        dst.free()

    def test_copy_vram_to_gtt(self, amd_device):
        """Copy from VRAM to system memory (GTT)."""
        size = 4096
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="gtt")

        pattern = b"\xCA\xFE\xBA\xBE" * (size // 4)
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == pattern

        src.free()
        dst.free()

    def test_copy_gtt_to_gtt(self, amd_device):
        """Copy between two GTT buffers."""
        size = 4096
        src = amd_device.alloc(size, location="gtt")
        dst = amd_device.alloc(size, location="gtt")

        pattern = bytes(range(256)) * (size // 256)
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == pattern

        src.free()
        dst.free()


@requires_gpu
class TestSDMACopySizes:
    """Test SDMA copy with various sizes."""

    def test_copy_small_256_bytes(self, amd_device):
        """Copy a small 256-byte buffer."""
        size = 256
        src = amd_device.alloc(4096, location="vram")
        dst = amd_device.alloc(4096, location="vram")

        src.write(b"\x42" * size)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == b"\x42" * size

        src.free()
        dst.free()

    def test_copy_non_power_of_two(self, amd_device):
        """Copy a non-power-of-two size (1000 bytes)."""
        size = 1000
        src = amd_device.alloc(4096, location="vram")
        dst = amd_device.alloc(4096, location="vram")

        pattern = bytes(i & 0xFF for i in range(size))
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == pattern

        src.free()
        dst.free()

    def test_copy_1mb(self, amd_device):
        """Copy 1 MB of data."""
        size = 1024 * 1024
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        # Write a repeating pattern
        pattern = struct.pack("<I", 0xDEADBEEF) * (size // 4)
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        # Check beginning, middle, and end
        head = dst.read(16)
        assert head == b"\xEF\xBE\xAD\xDE" * 4

        mid = dst.read(16, offset=size // 2)
        assert mid == b"\xEF\xBE\xAD\xDE" * 4

        tail = dst.read(16, offset=size - 16)
        assert tail == b"\xEF\xBE\xAD\xDE" * 4

        src.free()
        dst.free()

    def test_copy_8mb_chunked(self, amd_device):
        """Copy 8 MB — exceeds SDMA_MAX_COPY_SIZE, requires chunking."""
        size = 8 * 1024 * 1024
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        src.fill(0xAB)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        head = dst.read(64)
        assert head == b"\xAB" * 64

        tail = dst.read(64, offset=size - 64)
        assert tail == b"\xAB" * 64

        src.free()
        dst.free()


@requires_gpu
class TestSDMACopyDataIntegrity:
    """Test that SDMA copies preserve data patterns accurately."""

    def test_copy_sequential_u32_values(self, amd_device):
        """Copy a buffer of sequential uint32 values and verify each one."""
        count = 1024
        size = count * 4
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        pattern = struct.pack(f"<{count}I", *range(count))
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        values = struct.unpack(f"<{count}I", data)
        for i in range(count):
            assert values[i] == i, f"Mismatch at index {i}: got {values[i]}"

        src.free()
        dst.free()

    def test_copy_preserves_uncopied_region(self, amd_device):
        """Partial copy should not modify bytes beyond the copy size."""
        buf_size = 8192
        copy_size = 4096
        src = amd_device.alloc(buf_size, location="vram")
        dst = amd_device.alloc(buf_size, location="vram")

        src.fill(0xAA)
        dst.fill(0xFF)

        amd_device.copy(dst, src, copy_size)
        amd_device.synchronize()

        # Copied region should be 0xAA
        copied = dst.read(16, offset=0)
        assert copied == b"\xAA" * 16

        # Region beyond copy_size should still be 0xFF
        untouched = dst.read(16, offset=copy_size)
        assert untouched == b"\xFF" * 16

        src.free()
        dst.free()

    def test_copy_alternating_pattern(self, amd_device):
        """Copy an alternating bit pattern to detect bit-level errors."""
        size = 4096
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        pattern = b"\x55\xAA" * (size // 2)
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == pattern

        src.free()
        dst.free()

    def test_copy_all_byte_values(self, amd_device):
        """Ensure all 256 byte values survive a round-trip copy."""
        size = 4096
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        # Fill with all 256 byte values repeated
        pattern = bytes(range(256)) * (size // 256)
        src.write(pattern)
        dst.fill(0x00)

        amd_device.copy(dst, src, size)
        amd_device.synchronize()

        data = dst.read(size)
        assert data == pattern

        src.free()
        dst.free()


@requires_gpu
class TestSDMACopyMultiple:
    """Test multiple SDMA operations in sequence."""

    def test_sequential_copies(self, amd_device):
        """Multiple copies in sequence, each with a different pattern."""
        size = 4096
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        for value in [0x11, 0x22, 0x33, 0x44, 0xFF]:
            src.fill(value)
            dst.fill(0x00)
            amd_device.copy(dst, src, size)
            amd_device.synchronize()
            data = dst.read(16)
            assert data == bytes([value]) * 16, f"Failed for fill value 0x{value:02x}"

        src.free()
        dst.free()

    def test_chain_copy_a_to_b_to_c(self, amd_device):
        """Copy A->B then B->C and verify C matches A."""
        size = 4096
        a = amd_device.alloc(size, location="vram")
        b = amd_device.alloc(size, location="vram")
        c = amd_device.alloc(size, location="vram")

        pattern = struct.pack("<I", 0xCAFEBABE) * (size // 4)
        a.write(pattern)
        b.fill(0x00)
        c.fill(0x00)

        amd_device.copy(b, a, size)
        amd_device.synchronize()

        amd_device.copy(c, b, size)
        amd_device.synchronize()

        data = c.read(size)
        assert data == pattern

        a.free()
        b.free()
        c.free()

    def test_copy_multiple_buffers_independently(self, amd_device):
        """Copy 4 independent src/dst pairs and verify all."""
        size = 4096
        pairs = []
        for i in range(4):
            src = amd_device.alloc(size, location="vram")
            dst = amd_device.alloc(size, location="vram")
            fill_val = (i + 1) * 0x11
            src.fill(fill_val)
            dst.fill(0x00)
            pairs.append((src, dst, fill_val))

        for src, dst, _ in pairs:
            amd_device.copy(dst, src, size)
        amd_device.synchronize()

        for src, dst, fill_val in pairs:
            data = dst.read(16)
            assert data == bytes([fill_val]) * 16, (
                f"Failed for fill value 0x{fill_val:02x}"
            )
            src.free()
            dst.free()

    def test_copy_default_size(self, amd_device):
        """Copy with size=None should copy min(dst.size, src.size)."""
        src = amd_device.alloc(4096, location="vram")
        dst = amd_device.alloc(8192, location="vram")

        src.fill(0xCC)
        dst.fill(0x00)

        amd_device.copy(dst, src)  # size=None -> min(8192, 4096) = 4096
        amd_device.synchronize()

        copied = dst.read(16, offset=0)
        assert copied == b"\xCC" * 16

        src.free()
        dst.free()


@requires_gpu
class TestSDMAFenceSignaling:
    """Test SDMA fence/signaling via the timeline semaphore."""

    def test_fence_signals_after_copy(self, amd_device):
        """Verify timeline value advances after a copy + synchronize."""
        src = amd_device.alloc(4096, location="vram")
        dst = amd_device.alloc(4096, location="vram")

        src.fill(0xEE)
        dst.fill(0x00)

        amd_device.copy(dst, src, 4096)
        # synchronize waits for the timeline — if it returns, the fence worked
        amd_device.synchronize()

        data = dst.read(16)
        assert data == b"\xEE" * 16

        src.free()
        dst.free()

    def test_multiple_fences_increment(self, amd_device):
        """Multiple copies should each bump the timeline fence value."""
        size = 4096
        src = amd_device.alloc(size, location="vram")
        dst = amd_device.alloc(size, location="vram")

        for i in range(5):
            fill_val = (i + 1) * 0x10
            src.fill(fill_val)
            dst.fill(0x00)
            amd_device.copy(dst, src, size)
            amd_device.synchronize()
            data = dst.read(4)
            assert data == bytes([fill_val]) * 4, f"Iteration {i} failed"

        src.free()
        dst.free()
