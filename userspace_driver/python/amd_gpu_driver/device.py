"""AMDDevice: high-level facade for the AMD GPU driver."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from amd_gpu_driver.backends.base import DeviceBackend, MemoryLocation
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
from amd_gpu_driver.gpu.family import GPUFamilyConfig
from amd_gpu_driver.memory.buffer import Buffer
from amd_gpu_driver.program import Program, load_program
from amd_gpu_driver.sync.timeline import TimelineSemaphore


class AMDDevice:
    """High-level interface to an AMD GPU.

    Usage:
        with AMDDevice() as dev:
            buf = dev.alloc(4096, location="vram")
            buf.write(b"\\x42" * 4096)
            assert buf.read(4) == b"\\x42\\x42\\x42\\x42"

    Args:
        device_index: Index of the GPU to open (0 = first).
        backend: Backend type to use ("kfd" for now).
    """

    def __init__(self, device_index: int = 0, *, backend: str | None = None) -> None:
        self._device_index = device_index
        self._backend: DeviceBackend

        # Auto-detect backend based on platform
        if backend is None:
            backend = "windows" if sys.platform == "win32" else "kfd"

        if backend == "kfd":
            if sys.platform == "win32":
                raise RuntimeError(
                    "KFD backend is not available on Windows. "
                    "Use the 'windows' backend instead."
                )
            from amd_gpu_driver.backends.kfd import KFDDevice
            self._backend = KFDDevice()
        elif backend == "windows":
            if sys.platform != "win32":
                raise RuntimeError(
                    "Windows backend is only available on Windows."
                )
            from amd_gpu_driver.backends.windows import WindowsDevice
            self._backend = WindowsDevice()
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        self._backend.open(device_index)
        self._compute_queue = None
        self._sdma_queue = None
        self._timeline: TimelineSemaphore | None = None

    def alloc(
        self,
        size: int,
        *,
        location: str = "vram",
        executable: bool = False,
    ) -> Buffer:
        """Allocate GPU memory.

        Args:
            size: Size in bytes.
            location: "vram", "gtt", or "userptr".
            executable: Whether the memory should be executable.

        Returns:
            A Buffer object for reading/writing data.
        """
        loc = MemoryLocation(location)
        public = location == "vram"  # CPU-visible VRAM
        handle = self._backend.alloc_memory(
            size, loc, executable=executable, public=public, uncached=(loc == MemoryLocation.GTT)
        )
        return Buffer(handle, self._backend)

    def load_program(
        self, path: str | Path, kernel_name: str | None = None
    ) -> Program:
        """Load a GPU program from an ELF code object file.

        Args:
            path: Path to .co or .hsaco file.
            kernel_name: Specific kernel to load (None = first kernel found).

        Returns:
            A Program object ready for dispatch.
        """
        from amd_gpu_driver.backends.kfd import KFDDevice
        if not isinstance(self._backend, KFDDevice):
            raise RuntimeError("load_program requires KFD backend")
        family = self._backend.family
        if family is None:
            raise RuntimeError("GPU family not identified")
        return load_program(self._backend, path, family, kernel_name=kernel_name)

    def copy(self, dst: Buffer, src: Buffer, size: int | None = None) -> None:
        """Copy data between GPU buffers using SDMA.

        Args:
            dst: Destination buffer.
            src: Source buffer.
            size: Number of bytes to copy (None = min of both buffers).
        """
        if size is None:
            size = min(dst.size, src.size)

        # Ensure timeline semaphore exists before creating queue so signal
        # memory is mapped to GPU page tables before SDMA engine starts.
        if self._timeline is None:
            self._timeline = TimelineSemaphore(self._backend)

        # Get or create SDMA queue
        if self._sdma_queue is None:
            self._sdma_queue = self._backend.create_sdma_queue()

        # Build SDMA copy packets
        sdma = SDMAPacketBuilder()
        sdma.copy_linear(dst.gpu_addr, src.gpu_addr, size)

        fence_value = self._timeline.next_value()
        sdma.fence(self._timeline.signal_addr, fence_value)

        self._backend.submit_packets(self._sdma_queue, sdma.build())

    def synchronize(self) -> None:
        """Wait for all pending GPU operations to complete."""
        if self._timeline is not None:
            self._timeline.cpu_wait()

    @property
    def name(self) -> str:
        """Human-readable device name."""
        return self._backend.name

    @property
    def gfx_target(self) -> str:
        """GFX target string (e.g. 'gfx942')."""
        from amd_gpu_driver.backends.kfd import KFDDevice
        if isinstance(self._backend, KFDDevice) and self._backend.node:
            return self._backend.node.gfx_name
        return "unknown"

    @property
    def device_index(self) -> int:
        """Index of this GPU (as passed to constructor)."""
        return self._device_index

    @property
    def gpu_id(self) -> int:
        """KFD GPU ID for this device."""
        return self._backend.gpu_id

    @property
    def vram_size(self) -> int:
        """Total VRAM in bytes."""
        return self._backend.vram_size

    @property
    def backend(self) -> DeviceBackend:
        """The underlying device backend."""
        return self._backend

    def close(self) -> None:
        """Release all resources."""
        if self._timeline is not None:
            self._timeline.destroy()
            self._timeline = None
        if self._sdma_queue is not None:
            self._backend.destroy_queue(self._sdma_queue)
            self._sdma_queue = None
        if self._compute_queue is not None:
            self._backend.destroy_queue(self._compute_queue)
            self._compute_queue = None
        self._backend.close()

    def __enter__(self) -> AMDDevice:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"AMDDevice({self.name}, {self.gfx_target}, VRAM={self.vram_size // (1024**3)}GB)"
