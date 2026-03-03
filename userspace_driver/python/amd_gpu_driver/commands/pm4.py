"""PM4 Type-3 packet builder for AMD GPU compute dispatch."""

from __future__ import annotations

import struct

# PM4 opcodes
PACKET3_NOP = 0x10
PACKET3_WAIT_REG_MEM = 0x0C
PACKET3_DISPATCH_DIRECT = 0x15
PACKET3_EVENT_WRITE = 0x46
PACKET3_RELEASE_MEM = 0x49
PACKET3_ACQUIRE_MEM = 0x58
PACKET3_SET_SH_REG = 0x76
PACKET3_SET_UCONFIG_REG = 0x79

# RELEASE_MEM event types
EVENT_TYPE_CACHE_FLUSH_AND_INV_TS_EVENT = 0x14
EVENT_TYPE_CACHE_FLUSH = 0x06

# RELEASE_MEM data select
DATA_SEL_NONE = 0
DATA_SEL_SEND_32BIT = 1
DATA_SEL_SEND_64BIT = 2
DATA_SEL_GPU_CLOCK = 3

# RELEASE_MEM int select
INT_SEL_NONE = 0
INT_SEL_SEND_INT = 1
INT_SEL_SEND_INT_ON_CONFIRM = 2

# WAIT_REG_MEM function
WAIT_REG_MEM_FUNC_ALWAYS = 0
WAIT_REG_MEM_FUNC_LT = 1
WAIT_REG_MEM_FUNC_LE = 2
WAIT_REG_MEM_FUNC_EQ = 3
WAIT_REG_MEM_FUNC_NE = 4
WAIT_REG_MEM_FUNC_GE = 5
WAIT_REG_MEM_FUNC_GT = 6

# WAIT_REG_MEM memory space
WAIT_REG_MEM_MEM_SPACE_REG = 0
WAIT_REG_MEM_MEM_SPACE_MEM = 1

# ACQUIRE_MEM engine
ACQUIRE_MEM_ENGINE_ME = 0

# Cache operations for ACQUIRE_MEM
CP_COHER_CNTL_TC_ACTION = 1 << 23
CP_COHER_CNTL_TCL1_ACTION = 1 << 22
CP_COHER_CNTL_TC_WB_ACTION = 1 << 18
CP_COHER_CNTL_SH_KCACHE_ACTION = 1 << 27
CP_COHER_CNTL_SH_ICACHE_ACTION = 1 << 29

# RELEASE_MEM cache flush flags (GFX9)
EOP_TC_WB_ACTION_EN = 1 << 15
EOP_TC_NC_ACTION_EN = 1 << 19

# EVENT_WRITE event types
CS_PARTIAL_FLUSH = 7
EVENT_INDEX_EOP = 5
EVENT_INDEX_CS_PARTIAL_FLUSH = 4

# Register bases for SET_SH_REG/SET_UCONFIG_REG
SH_REG_BASE = 0x2C00
UCONFIG_REG_BASE = 0xC000


class PM4PacketBuilder:
    """Builds PM4 Type-3 command packets for compute dispatch."""

    def __init__(self) -> None:
        self._dwords: list[int] = []

    def _pkt3(self, opcode: int, *payload: int) -> None:
        """Encode a PM4 Type-3 packet: header + payload dwords.

        Header format: (3 << 30) | ((count - 2) << 16) | (opcode << 8)
        where count = number of payload dwords + 1 (header itself).
        Bits 31:30 = type (3), bits 29:16 = N-2, bits 15:8 = opcode.
        """
        n = len(payload)  # number of dwords following the header
        header = (3 << 30) | (((n - 1) & 0x3FFF) << 16) | (opcode << 8)
        self._dwords.append(header)
        self._dwords.extend(payload)

    def nop(self, count: int = 1) -> PM4PacketBuilder:
        """Insert NOP padding."""
        for _ in range(count):
            self._pkt3(PACKET3_NOP, 0)
        return self

    def set_sh_reg(self, offset: int, *values: int) -> PM4PacketBuilder:
        """SET_SH_REG: write values to shader registers.

        offset is relative to SH_REG_BASE (0x2C00).
        """
        reg_offset = offset - SH_REG_BASE if offset >= SH_REG_BASE else offset
        self._pkt3(PACKET3_SET_SH_REG, reg_offset, *values)
        return self

    def set_uconfig_reg(self, offset: int, *values: int) -> PM4PacketBuilder:
        """SET_UCONFIG_REG: write values to uconfig registers.

        offset is relative to UCONFIG_REG_BASE (0xC000).
        """
        reg_offset = offset - UCONFIG_REG_BASE if offset >= UCONFIG_REG_BASE else offset
        self._pkt3(PACKET3_SET_UCONFIG_REG, reg_offset, *values)
        return self

    def acquire_mem(
        self,
        coher_cntl: int = (
            CP_COHER_CNTL_TC_ACTION
            | CP_COHER_CNTL_TC_WB_ACTION
            | CP_COHER_CNTL_SH_KCACHE_ACTION
            | CP_COHER_CNTL_SH_ICACHE_ACTION
        ),
        coher_size: int = 0xFFFFFFFF,
        coher_base: int = 0,
        poll_interval: int = 10,
    ) -> PM4PacketBuilder:
        """ACQUIRE_MEM: invalidate/flush caches before dispatch."""
        self._pkt3(
            PACKET3_ACQUIRE_MEM,
            coher_cntl,
            coher_size,
            coher_size >> 32 if coher_size > 0xFFFFFFFF else 0,
            coher_base,
            coher_base >> 32 if coher_base > 0xFFFFFFFF else 0,
            poll_interval,
        )
        return self

    def release_mem(
        self,
        addr: int,
        value: int,
        *,
        event_type: int = EVENT_TYPE_CACHE_FLUSH_AND_INV_TS_EVENT,
        data_sel: int = DATA_SEL_SEND_64BIT,
        int_sel: int = INT_SEL_SEND_INT_ON_CONFIRM,
        event_index: int = 5,  # EOP event
        cache_flush: bool = False,
    ) -> PM4PacketBuilder:
        """RELEASE_MEM: write a value to memory and optionally raise interrupt.

        Used for signaling completion of dispatch.
        GFX9 format: 7 payload dwords (header + 7).
        """
        # dword 0: event_type | event_index | optional cache flags
        dw0 = (event_type & 0x3F) | ((event_index & 0xF) << 8)
        if cache_flush:
            dw0 |= EOP_TC_WB_ACTION_EN | EOP_TC_NC_ACTION_EN
        # dword 1: data_sel | int_sel
        dw1 = ((data_sel & 0x7) << 29) | ((int_sel & 0x3) << 24)
        # dword 2-3: address (low, high)
        addr_lo = addr & 0xFFFFFFFF
        addr_hi = (addr >> 32) & 0xFFFFFFFF
        # dword 4-5: data (low, high)
        data_lo = value & 0xFFFFFFFF
        data_hi = (value >> 32) & 0xFFFFFFFF
        # dword 6: ctxid (always 0 for GFX9)
        ctxid = 0

        self._pkt3(PACKET3_RELEASE_MEM, dw0, dw1, addr_lo, addr_hi, data_lo, data_hi, ctxid)
        return self

    def wait_reg_mem(
        self,
        addr: int,
        expected: int,
        mask: int = 0xFFFFFFFF,
        *,
        func: int = WAIT_REG_MEM_FUNC_GE,
        mem_space: int = WAIT_REG_MEM_MEM_SPACE_MEM,
        poll_interval: int = 10,
    ) -> PM4PacketBuilder:
        """WAIT_REG_MEM: poll memory/register until condition met."""
        # dword 0: function | mem_space
        dw0 = (func & 0x7) | ((mem_space & 0x1) << 4)
        # dword 1-2: address
        addr_lo = addr & 0xFFFFFFFF
        addr_hi = (addr >> 32) & 0xFFFFFFFF
        # dword 3: reference value
        # dword 4: mask
        # dword 5: poll interval

        self._pkt3(
            PACKET3_WAIT_REG_MEM,
            dw0, addr_lo, addr_hi, expected, mask, poll_interval,
        )
        return self

    def dispatch_direct(
        self,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        initiator: int = 0x5,
    ) -> PM4PacketBuilder:
        """DISPATCH_DIRECT: launch compute shader.

        dim_x/y/z are the global dispatch dimensions in workgroups.
        initiator: bit 0 = compute_shader_en, bit 2 = force_start_at_000.
        Default 0x5 = both enabled.
        """
        self._pkt3(
            PACKET3_DISPATCH_DIRECT,
            dim_x, dim_y, dim_z, initiator,
        )
        return self

    def event_write(
        self,
        event_type: int,
        event_index: int,
    ) -> PM4PacketBuilder:
        """EVENT_WRITE: emit a GPU event (e.g. CS_PARTIAL_FLUSH)."""
        dw0 = (event_type & 0x3F) | ((event_index & 0xF) << 8)
        self._pkt3(PACKET3_EVENT_WRITE, dw0)
        return self

    def build(self) -> bytes:
        """Serialize all packets to little-endian bytes."""
        return struct.pack(f"<{len(self._dwords)}I", *self._dwords)

    def clear(self) -> PM4PacketBuilder:
        """Clear the packet buffer."""
        self._dwords.clear()
        return self

    @property
    def size_bytes(self) -> int:
        """Current packet buffer size in bytes."""
        return len(self._dwords) * 4

    @property
    def size_dwords(self) -> int:
        """Current packet buffer size in dwords."""
        return len(self._dwords)
