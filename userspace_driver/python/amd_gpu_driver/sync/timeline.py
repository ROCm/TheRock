"""Timeline semaphore for GPU-CPU synchronization.

Uses a 16-byte GPU memory allocation:
  - bytes 0-7: signal value (uint64)
  - bytes 8-15: timestamp (uint64, reserved)

The GPU writes to the signal value via RELEASE_MEM packets.
The CPU waits by either spin-polling the memory or using KFD events.
"""

from __future__ import annotations

import ctypes
import time

from amd_gpu_driver.backends.base import DeviceBackend, MemoryLocation, SignalHandle
from amd_gpu_driver.commands.pm4 import (
    PM4PacketBuilder,
    WAIT_REG_MEM_FUNC_GE,
    DATA_SEL_SEND_64BIT,
    INT_SEL_SEND_INT_ON_CONFIRM,
)
from amd_gpu_driver.errors import TimeoutError


class TimelineSemaphore:
    """Timeline semaphore using GPU-visible memory + KFD events."""

    def __init__(self, backend: DeviceBackend) -> None:
        self._backend = backend
        # Allocate 16-byte signal memory (GTT for CPU+GPU access)
        self._signal_mem = backend.alloc_memory(
            4096,  # Minimum page size
            MemoryLocation.GTT,
            uncached=True,
        )
        self._signal_addr = self._signal_mem.gpu_addr
        self._timeline_value: int = 0
        self._kfd_event: SignalHandle | None = None

        # Zero the signal memory
        if self._signal_mem.cpu_addr:
            ctypes.memset(self._signal_mem.cpu_addr, 0, 16)

    @property
    def signal_addr(self) -> int:
        """GPU address of the signal value."""
        return self._signal_addr

    @property
    def timeline_value(self) -> int:
        """Current timeline value (monotonically increasing)."""
        return self._timeline_value

    @property
    def gpu_value(self) -> int:
        """Read the current signal value from GPU memory."""
        if self._signal_mem.cpu_addr:
            return ctypes.c_uint64.from_address(self._signal_mem.cpu_addr).value
        return 0

    def next_value(self) -> int:
        """Increment and return the next timeline value."""
        self._timeline_value += 1
        return self._timeline_value

    def signal_packets(self, value: int | None = None) -> bytes:
        """Build PM4 RELEASE_MEM packets to signal from GPU.

        Returns packet bytes to append to a command stream.
        """
        if value is None:
            value = self.next_value()

        # Optionally create a KFD event for interrupt-based waiting
        event_id = 0
        if self._kfd_event is None:
            self._kfd_event = self._backend.create_signal()
        event_id = self._kfd_event.event_id

        builder = PM4PacketBuilder()
        builder.release_mem(
            addr=self._signal_addr,
            value=value,
            data_sel=DATA_SEL_SEND_64BIT,
            int_sel=INT_SEL_SEND_INT_ON_CONFIRM,
            cache_flush=True,
        )
        return builder.build()

    def wait_packets(self, value: int | None = None) -> bytes:
        """Build PM4 WAIT_REG_MEM packets to wait on GPU.

        Returns packet bytes to prepend to a command stream that needs
        to wait for a prior dispatch to complete.
        """
        if value is None:
            value = self._timeline_value

        builder = PM4PacketBuilder()
        builder.wait_reg_mem(
            addr=self._signal_addr,
            expected=value,
            func=WAIT_REG_MEM_FUNC_GE,
        )
        return builder.build()

    def cpu_wait(self, value: int | None = None, timeout_ms: int = 5000) -> None:
        """Wait on CPU for GPU to reach the specified timeline value.

        Uses KFD event waiting if available, falls back to spin-polling.
        """
        if value is None:
            value = self._timeline_value

        if self._kfd_event is not None:
            # Use KFD event-based waiting
            try:
                self._backend.wait_signal(self._kfd_event, timeout_ms)
                return
            except TimeoutError:
                pass

        # Spin-poll fallback
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            current = self.gpu_value
            if current >= value:
                return
            time.sleep(0.0001)  # 100us sleep between polls

        raise TimeoutError("timeline_semaphore.cpu_wait", timeout_ms)

    def reset(self) -> None:
        """Reset timeline to 0."""
        self._timeline_value = 0
        if self._signal_mem.cpu_addr:
            ctypes.c_uint64.from_address(self._signal_mem.cpu_addr).value = 0

    def map_to_peers(self, peer_gpu_ids: list[int]) -> None:
        """Map signal memory to peer GPUs for cross-GPU waiting.

        This enables a peer GPU to poll this timeline's signal memory
        via WAIT_REG_MEM or SDMA poll_regmem.

        Args:
            peer_gpu_ids: KFD GPU IDs of peer devices to map to.
        """
        from amd_gpu_driver.backends.kfd.device import KFDDevice

        if not isinstance(self._backend, KFDDevice):
            raise RuntimeError("map_to_peers requires KFD backend")
        self._backend.map_memory_to_peers(self._signal_mem, peer_gpu_ids)

    def destroy(self) -> None:
        """Free resources."""
        if self._kfd_event is not None:
            self._backend.destroy_signal(self._kfd_event)
            self._kfd_event = None
        self._backend.free_memory(self._signal_mem)
