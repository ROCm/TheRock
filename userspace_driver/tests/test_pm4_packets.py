"""Tests for PM4 packet builder."""

import struct

from amd_gpu_driver.commands.pm4 import (
    PM4PacketBuilder,
    PACKET3_NOP,
    PACKET3_SET_SH_REG,
    PACKET3_SET_UCONFIG_REG,
    PACKET3_DISPATCH_DIRECT,
    PACKET3_ACQUIRE_MEM,
    PACKET3_RELEASE_MEM,
    PACKET3_WAIT_REG_MEM,
    SH_REG_BASE,
)


class TestPM4Header:
    """Test PM4 Type-3 header encoding."""

    def test_nop_header(self):
        pm4 = PM4PacketBuilder()
        pm4.nop()
        data = pm4.build()
        header = struct.unpack_from("<I", data, 0)[0]
        # Type 3 = (3 << 30), opcode in bits 15:8, N-1 in bits 29:16
        assert (header >> 30) == 3
        assert ((header >> 8) & 0xFF) == PACKET3_NOP
        assert ((header >> 16) & 0x3FFF) == 0  # N-1 = 0 (1 payload dword)

    def test_header_count(self):
        pm4 = PM4PacketBuilder()
        pm4.dispatch_direct(1, 1, 1, 1)
        data = pm4.build()
        header = struct.unpack_from("<I", data, 0)[0]
        # dispatch_direct has 4 payload dwords, so N-1 = 3
        assert ((header >> 16) & 0x3FFF) == 3


class TestPM4SetSHReg:
    """Test SET_SH_REG packet encoding."""

    def test_single_value(self):
        pm4 = PM4PacketBuilder()
        pm4.set_sh_reg(0x100, 0xDEADBEEF)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        # Header
        assert (dwords[0] >> 30) == 3
        assert ((dwords[0] >> 8) & 0xFF) == PACKET3_SET_SH_REG
        # Offset (relative to SH_REG_BASE)
        assert dwords[1] == 0x100
        # Value
        assert dwords[2] == 0xDEADBEEF

    def test_multiple_values(self):
        pm4 = PM4PacketBuilder()
        pm4.set_sh_reg(0x200, 0x11111111, 0x22222222)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        # 1 header + 1 offset + 2 values = 4 dwords
        assert len(dwords) == 4
        # N-1 = 2 (3 payload dwords: offset + 2 values)
        assert ((dwords[0] >> 16) & 0x3FFF) == 2
        assert dwords[2] == 0x11111111
        assert dwords[3] == 0x22222222

    def test_absolute_offset_conversion(self):
        pm4 = PM4PacketBuilder()
        # Pass absolute register address (>= SH_REG_BASE)
        pm4.set_sh_reg(SH_REG_BASE + 0x100, 0x42)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)
        # Should convert to relative offset
        assert dwords[1] == 0x100


class TestPM4DispatchDirect:
    """Test DISPATCH_DIRECT packet."""

    def test_basic_dispatch(self):
        pm4 = PM4PacketBuilder()
        pm4.dispatch_direct(64, 1, 1, 1)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert ((dwords[0] >> 8) & 0xFF) == PACKET3_DISPATCH_DIRECT
        assert dwords[1] == 64  # dim_x
        assert dwords[2] == 1  # dim_y
        assert dwords[3] == 1  # dim_z
        assert dwords[4] == 1  # initiator

    def test_3d_dispatch(self):
        pm4 = PM4PacketBuilder()
        pm4.dispatch_direct(8, 8, 8, 1)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert dwords[1] == 8
        assert dwords[2] == 8
        assert dwords[3] == 8


class TestPM4ReleaseMem:
    """Test RELEASE_MEM packet."""

    def test_signal_address(self):
        pm4 = PM4PacketBuilder()
        addr = 0x0000_7FFF_DEAD_0000
        value = 42
        pm4.release_mem(addr=addr, value=value)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert ((dwords[0] >> 8) & 0xFF) == PACKET3_RELEASE_MEM
        # addr_lo is dwords[3], addr_hi is dwords[4]
        assert dwords[3] == (addr & 0xFFFFFFFF)
        assert dwords[4] == ((addr >> 32) & 0xFFFFFFFF)
        # data_lo is dwords[5], data_hi is dwords[6]
        assert dwords[5] == 42
        assert dwords[6] == 0


class TestPM4WaitRegMem:
    """Test WAIT_REG_MEM packet."""

    def test_memory_poll(self):
        pm4 = PM4PacketBuilder()
        addr = 0x1234_5678_9ABC_DEF0
        pm4.wait_reg_mem(addr=addr, expected=100, mask=0xFFFFFFFF)
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert ((dwords[0] >> 8) & 0xFF) == PACKET3_WAIT_REG_MEM
        # addr split
        assert dwords[2] == (addr & 0xFFFFFFFF)
        assert dwords[3] == ((addr >> 32) & 0xFFFFFFFF)
        # expected value
        assert dwords[4] == 100
        # mask
        assert dwords[5] == 0xFFFFFFFF


class TestPM4AcquireMem:
    """Test ACQUIRE_MEM packet."""

    def test_default_acquire(self):
        pm4 = PM4PacketBuilder()
        pm4.acquire_mem()
        data = pm4.build()
        dwords = struct.unpack(f"<{len(data)//4}I", data)

        assert ((dwords[0] >> 8) & 0xFF) == PACKET3_ACQUIRE_MEM
        # Should have 6 payload dwords, N-1 = 5
        assert ((dwords[0] >> 16) & 0x3FFF) == 5


class TestPM4Builder:
    """Test builder chaining and serialization."""

    def test_empty_builder(self):
        pm4 = PM4PacketBuilder()
        assert pm4.build() == b""
        assert pm4.size_bytes == 0
        assert pm4.size_dwords == 0

    def test_chaining(self):
        pm4 = PM4PacketBuilder()
        result = pm4.nop().set_sh_reg(0x100, 0x42).dispatch_direct(1, 1, 1, 1)
        assert result is pm4

    def test_clear(self):
        pm4 = PM4PacketBuilder()
        pm4.nop()
        assert pm4.size_dwords > 0
        pm4.clear()
        assert pm4.size_dwords == 0

    def test_multiple_packets(self):
        pm4 = PM4PacketBuilder()
        pm4.set_sh_reg(0x100, 0x42)
        pm4.dispatch_direct(1, 1, 1, 1)
        data = pm4.build()
        # First packet: 3 dwords (header + offset + value)
        # Second packet: 5 dwords (header + 4 payload)
        assert len(data) == (3 + 5) * 4

    def test_little_endian(self):
        pm4 = PM4PacketBuilder()
        pm4.set_sh_reg(0x0, 0x01020304)
        data = pm4.build()
        # Value should be at offset 8 (after header + offset)
        assert data[8:12] == b"\x04\x03\x02\x01"  # little-endian
