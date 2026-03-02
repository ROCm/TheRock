"""Ring buffer management for command submission."""

from __future__ import annotations

import ctypes


class RingBuffer:
    """Manages a power-of-2 ring buffer with wrap-around write logic.

    The ring buffer is used for submitting PM4/SDMA packets to the GPU.
    Write and read pointers track positions in dword units.
    """

    def __init__(
        self,
        base_addr: int,
        size: int,
        write_ptr_addr: int,
        read_ptr_addr: int,
    ) -> None:
        if size & (size - 1) != 0:
            raise ValueError(f"Ring size must be a power of 2, got {size}")

        self._base_addr = base_addr
        self._size = size
        self._mask = size - 1
        self._write_ptr_addr = write_ptr_addr
        self._read_ptr_addr = read_ptr_addr

    @property
    def size(self) -> int:
        return self._size

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def write_ptr(self) -> int:
        """Read current write pointer (in dwords)."""
        return ctypes.c_uint64.from_address(self._write_ptr_addr).value

    @write_ptr.setter
    def write_ptr(self, value: int) -> None:
        """Set write pointer (in dwords)."""
        ctypes.c_uint64.from_address(self._write_ptr_addr).value = value

    @property
    def read_ptr(self) -> int:
        """Read current read pointer (in dwords)."""
        return ctypes.c_uint64.from_address(self._read_ptr_addr).value

    def available_space(self) -> int:
        """Available space in bytes before ring wraps into unread data."""
        wp = self.write_ptr
        rp = self.read_ptr
        used_dwords = wp - rp
        total_dwords = self._size // 4
        free_dwords = total_dwords - used_dwords
        return free_dwords * 4

    def write(self, data: bytes) -> int:
        """Write data to ring buffer, returning new write pointer.

        Handles wrap-around automatically. Does NOT check for
        available space (caller must ensure there's room).
        """
        wp = self.write_ptr
        byte_offset = (wp * 4) & self._mask

        space_to_end = self._size - byte_offset
        if len(data) <= space_to_end:
            ctypes.memmove(self._base_addr + byte_offset, data, len(data))
        else:
            # Wrap around
            ctypes.memmove(
                self._base_addr + byte_offset, data[:space_to_end], space_to_end
            )
            remainder = len(data) - space_to_end
            ctypes.memmove(self._base_addr, data[space_to_end:], remainder)

        new_wp = wp + (len(data) // 4)
        self.write_ptr = new_wp
        return new_wp
