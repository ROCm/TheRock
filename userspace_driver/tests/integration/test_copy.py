"""Integration tests for SDMA copy operations."""

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
