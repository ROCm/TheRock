"""Integration tests for GPU memory allocation."""

import pytest

from tests.integration.conftest import requires_gpu


@requires_gpu
class TestMemoryAllocation:
    """Test VRAM and GTT memory allocation."""

    def test_alloc_vram(self, amd_device):
        buf = amd_device.alloc(4096, location="vram")
        assert buf.size >= 4096
        assert buf.gpu_addr != 0
        buf.free()

    def test_alloc_gtt(self, amd_device):
        buf = amd_device.alloc(4096, location="gtt")
        assert buf.size >= 4096
        assert buf.gpu_addr != 0
        buf.free()

    def test_write_read_vram(self, amd_device):
        buf = amd_device.alloc(4096, location="vram")
        test_data = b"\x42" * 4096
        buf.write(test_data)
        read_back = buf.read(4)
        assert read_back == b"\x42\x42\x42\x42"
        buf.free()

    def test_write_read_gtt(self, amd_device):
        buf = amd_device.alloc(4096, location="gtt")
        test_data = b"\xAB\xCD" * 2048
        buf.write(test_data)
        read_back = buf.read(4)
        assert read_back == b"\xAB\xCD\xAB\xCD"
        buf.free()

    def test_fill_buffer(self, amd_device):
        buf = amd_device.alloc(4096, location="vram")
        buf.fill(0xFF)
        data = buf.read(16)
        assert data == b"\xFF" * 16
        buf.free()

    def test_partial_read(self, amd_device):
        buf = amd_device.alloc(4096, location="vram")
        buf.fill(0x00)
        buf.write(b"\x01\x02\x03\x04", offset=100)
        data = buf.read(4, offset=100)
        assert data == b"\x01\x02\x03\x04"
        buf.free()

    def test_large_allocation(self, amd_device):
        size = 16 * 1024 * 1024  # 16MB
        buf = amd_device.alloc(size, location="vram")
        assert buf.size >= size
        buf.free()

    def test_multiple_allocations(self, amd_device):
        buffers = []
        for i in range(10):
            buf = amd_device.alloc(4096, location="vram")
            buf.fill(i & 0xFF)
            buffers.append(buf)

        for i, buf in enumerate(buffers):
            data = buf.read(1)
            assert data == bytes([i & 0xFF])
            buf.free()
