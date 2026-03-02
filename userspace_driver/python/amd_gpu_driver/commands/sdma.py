"""SDMA packet builder for AMD GPU DMA copy operations."""

from __future__ import annotations

import struct

# SDMA opcodes
SDMA_OP_NOP = 0
SDMA_OP_COPY = 1
SDMA_OP_FENCE = 5
SDMA_OP_POLL_REGMEM = 8
SDMA_OP_TIMESTAMP = 13
SDMA_OP_ATOMIC = 10
SDMA_OP_TRAP = 6

# SDMA copy sub-opcodes
SDMA_SUBOP_COPY_LINEAR = 0

# SDMA poll function
SDMA_POLL_FUNC_EQ = 3
SDMA_POLL_FUNC_GE = 5

# Max bytes per single SDMA linear copy
SDMA_MAX_COPY_SIZE = 0x3FFFE0  # ~4MB per packet, 32-byte aligned


class SDMAPacketBuilder:
    """Builds SDMA command packets for DMA operations."""

    def __init__(self) -> None:
        self._dwords: list[int] = []

    def _append(self, *dwords: int) -> None:
        self._dwords.extend(dwords)

    def nop(self, count: int = 1) -> SDMAPacketBuilder:
        """Insert NOP padding."""
        for _ in range(count):
            self._append(SDMA_OP_NOP)
        return self

    def copy_linear(
        self,
        dst: int,
        src: int,
        size: int,
    ) -> SDMAPacketBuilder:
        """Linear copy from src to dst, with automatic chunking.

        Handles copies larger than SDMA_MAX_COPY_SIZE by splitting
        into multiple packets.
        """
        offset = 0
        while offset < size:
            chunk = min(size - offset, SDMA_MAX_COPY_SIZE)
            self._copy_linear_single(dst + offset, src + offset, chunk)
            offset += chunk
        return self

    def _copy_linear_single(self, dst: int, src: int, size: int) -> None:
        """Emit a single SDMA linear copy packet.

        DWORD 0: opcode | sub_op
        DWORD 1: byte count - 1 (1-based: 0 means 1 byte)
        DWORD 2: reserved/parameter (0)
        DWORD 3-4: src addr (lo, hi)
        DWORD 5-6: dst addr (lo, hi)
        """
        header = SDMA_OP_COPY | (SDMA_SUBOP_COPY_LINEAR << 8)
        self._append(
            header,
            size - 1,  # 1-based count per SDMA spec
            0,  # parameter
            src & 0xFFFFFFFF,
            (src >> 32) & 0xFFFFFFFF,
            dst & 0xFFFFFFFF,
            (dst >> 32) & 0xFFFFFFFF,
        )

    def fence(
        self,
        addr: int,
        value: int,
    ) -> SDMAPacketBuilder:
        """Write a fence value to memory.

        Used to signal completion of SDMA operations.
        """
        self._append(
            SDMA_OP_FENCE,
            addr & 0xFFFFFFFF,
            (addr >> 32) & 0xFFFFFFFF,
            value & 0xFFFFFFFF,
        )
        return self

    def poll_regmem(
        self,
        addr: int,
        expected: int,
        mask: int = 0xFFFFFFFF,
        *,
        func: int = SDMA_POLL_FUNC_GE,
        interval: int = 10,
        retry_count: int = 0xFFF,
    ) -> SDMAPacketBuilder:
        """Poll memory until condition is met."""
        # Header with mem flag (bit 31)
        header = SDMA_OP_POLL_REGMEM | (1 << 31)  # memory space
        header |= (func & 0x7) << 28
        self._append(
            header,
            addr & 0xFFFFFFFF,
            (addr >> 32) & 0xFFFFFFFF,
            expected & 0xFFFFFFFF,
            mask & 0xFFFFFFFF,
            (interval & 0xFFFF) | ((retry_count & 0xFFF) << 16),
        )
        return self

    def trap(self, event_id: int = 0) -> SDMAPacketBuilder:
        """Insert a trap packet to trigger an interrupt."""
        self._append(SDMA_OP_TRAP, event_id)
        return self

    def build(self) -> bytes:
        """Serialize all packets to little-endian bytes."""
        return struct.pack(f"<{len(self._dwords)}I", *self._dwords)

    def clear(self) -> SDMAPacketBuilder:
        """Clear the packet buffer."""
        self._dwords.clear()
        return self

    @property
    def size_bytes(self) -> int:
        return len(self._dwords) * 4

    @property
    def size_dwords(self) -> int:
        return len(self._dwords)
