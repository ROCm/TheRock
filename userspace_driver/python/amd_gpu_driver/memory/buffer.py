"""Buffer: public API for GPU memory read/write operations."""

from __future__ import annotations

import ctypes

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation
from amd_gpu_driver.ioctl.helpers import libc


class Buffer:
    """High-level wrapper around a GPU memory allocation.

    Provides read/write/fill operations via the CPU-mapped address.
    """

    def __init__(self, handle: MemoryHandle, backend: object) -> None:
        self._handle = handle
        self._backend = backend

    @property
    def gpu_addr(self) -> int:
        """GPU virtual address."""
        return self._handle.gpu_addr

    @property
    def cpu_addr(self) -> int:
        """CPU-mapped address for direct access."""
        return self._handle.cpu_addr

    @property
    def size(self) -> int:
        """Size in bytes."""
        return self._handle.size

    @property
    def location(self) -> MemoryLocation:
        """Memory location (VRAM, GTT, etc.)."""
        return self._handle.location

    @property
    def handle(self) -> MemoryHandle:
        """Underlying memory handle."""
        return self._handle

    def read(self, size: int = 0, offset: int = 0) -> bytes:
        """Read bytes from the buffer via CPU mapping.

        Args:
            size: Number of bytes to read. 0 means read entire buffer.
            offset: Byte offset into the buffer.

        Returns:
            Raw bytes read from the buffer.
        """
        if size == 0:
            size = self._handle.size - offset
        if offset + size > self._handle.size:
            raise ValueError(
                f"Read of {size} bytes at offset {offset} exceeds "
                f"buffer size {self._handle.size}"
            )
        addr = self._handle.cpu_addr + offset
        return (ctypes.c_char * size).from_address(addr).raw

    def write(self, data: bytes, offset: int = 0) -> None:
        """Write bytes to the buffer via CPU mapping.

        Args:
            data: Bytes to write.
            offset: Byte offset into the buffer.
        """
        if offset + len(data) > self._handle.size:
            raise ValueError(
                f"Write of {len(data)} bytes at offset {offset} exceeds "
                f"buffer size {self._handle.size}"
            )
        addr = self._handle.cpu_addr + offset
        ctypes.memmove(addr, data, len(data))

    def fill(self, value: int, size: int = 0, offset: int = 0) -> None:
        """Fill buffer with a byte value.

        Args:
            value: Byte value (0-255) to fill with.
            size: Number of bytes to fill. 0 means fill entire buffer.
            offset: Byte offset into the buffer.
        """
        if size == 0:
            size = self._handle.size - offset
        if offset + size > self._handle.size:
            raise ValueError(
                f"Fill of {size} bytes at offset {offset} exceeds "
                f"buffer size {self._handle.size}"
            )
        addr = self._handle.cpu_addr + offset
        libc.memset(ctypes.c_void_p(addr), value, size)

    def free(self) -> None:
        """Free this buffer's GPU memory."""
        from amd_gpu_driver.backends.base import DeviceBackend
        if isinstance(self._backend, DeviceBackend):
            self._backend.free_memory(self._handle)

    def __repr__(self) -> str:
        return (
            f"Buffer(gpu_addr=0x{self.gpu_addr:x}, size={self.size}, "
            f"location={self.location.value})"
        )
