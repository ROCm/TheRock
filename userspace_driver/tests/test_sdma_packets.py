"""Tests for SDMA packet builder."""

import struct

from amd_gpu_driver.commands.sdma import (
    SDMAPacketBuilder,
    SDMA_OP_COPY,
    SDMA_OP_FENCE,
    SDMA_OP_POLL_REGMEM,
    SDMA_OP_TRAP,
    SDMA_SUBOP_COPY_LINEAR,
    SDMA_MAX_COPY_SIZE,
)


class TestSDMACopyLinear:
    """Test SDMA linear copy packet."""

    def test_small_copy(self):
        sdma = SDMAPacketBuilder()
        sdma.copy_linear(dst=0x2000, src=0x1000, size=4096)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        # Header: OP_COPY | (SUBOP_COPY_LINEAR << 8)
        assert dwords[0] == SDMA_OP_COPY | (SDMA_SUBOP_COPY_LINEAR << 8)
        # Size (1-based: byte_count - 1)
        assert dwords[1] == 4095
        # Reserved
        assert dwords[2] == 0
        # Source addr (lo, hi)
        assert dwords[3] == 0x1000
        assert dwords[4] == 0
        # Dest addr (lo, hi)
        assert dwords[5] == 0x2000
        assert dwords[6] == 0

    def test_64bit_addresses(self):
        sdma = SDMAPacketBuilder()
        src = 0x0000_7FFF_0000_1000
        dst = 0x0000_7FFF_0000_2000
        sdma.copy_linear(dst=dst, src=src, size=256)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert dwords[3] == (src & 0xFFFFFFFF)
        assert dwords[4] == ((src >> 32) & 0xFFFFFFFF)
        assert dwords[5] == (dst & 0xFFFFFFFF)
        assert dwords[6] == ((dst >> 32) & 0xFFFFFFFF)

    def test_chunking(self):
        sdma = SDMAPacketBuilder()
        large_size = SDMA_MAX_COPY_SIZE * 2 + 1024
        sdma.copy_linear(dst=0x2000, src=0x1000, size=large_size)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        # Should produce 3 copy packets
        # Each packet is 7 dwords
        copy_headers = [
            i for i, d in enumerate(dwords)
            if d == (SDMA_OP_COPY | (SDMA_SUBOP_COPY_LINEAR << 8))
        ]
        assert len(copy_headers) == 3

    def test_zero_size_produces_no_packets(self):
        sdma = SDMAPacketBuilder()
        sdma.copy_linear(dst=0x2000, src=0x1000, size=0)
        assert sdma.build() == b""


class TestSDMAFence:
    """Test SDMA fence packet."""

    def test_basic_fence(self):
        sdma = SDMAPacketBuilder()
        addr = 0x0000_7FFF_DEAD_0000
        sdma.fence(addr=addr, value=42)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert dwords[0] == SDMA_OP_FENCE
        assert dwords[1] == (addr & 0xFFFFFFFF)
        assert dwords[2] == ((addr >> 32) & 0xFFFFFFFF)
        assert dwords[3] == 42


class TestSDMAPollRegmem:
    """Test SDMA poll regmem packet."""

    def test_poll(self):
        sdma = SDMAPacketBuilder()
        addr = 0x0000_7FFF_0000_0100
        sdma.poll_regmem(addr=addr, expected=1, mask=0xFFFFFFFF)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        # Header should have memory space flag and function
        assert (dwords[0] & 0xFF) == SDMA_OP_POLL_REGMEM
        # Address
        assert dwords[1] == (addr & 0xFFFFFFFF)
        assert dwords[2] == ((addr >> 32) & 0xFFFFFFFF)
        # Expected value
        assert dwords[3] == 1
        # Mask
        assert dwords[4] == 0xFFFFFFFF


class TestSDMATrap:
    """Test SDMA trap packet."""

    def test_trap(self):
        sdma = SDMAPacketBuilder()
        sdma.trap(event_id=7)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert dwords[0] == SDMA_OP_TRAP
        assert dwords[1] == 7


class TestSDMABuilder:
    """Test builder operations."""

    def test_empty(self):
        sdma = SDMAPacketBuilder()
        assert sdma.build() == b""
        assert sdma.size_bytes == 0

    def test_chaining(self):
        sdma = SDMAPacketBuilder()
        result = sdma.copy_linear(0x2000, 0x1000, 256).fence(0x3000, 1)
        assert result is sdma
        assert sdma.size_dwords > 0

    def test_clear(self):
        sdma = SDMAPacketBuilder()
        sdma.fence(0x1000, 1)
        assert sdma.size_dwords > 0
        sdma.clear()
        assert sdma.size_dwords == 0

    def test_nop(self):
        sdma = SDMAPacketBuilder()
        sdma.nop(3)
        data = sdma.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)
        assert len(dwords) == 3
        assert all(d == 0 for d in dwords)
