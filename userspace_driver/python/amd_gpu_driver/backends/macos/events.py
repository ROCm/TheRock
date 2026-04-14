"""macOS event/signal manager — GPU-CPU synchronization.

Provides signal events for synchronizing GPU command completion with
CPU waits. Two modes:

  1. Polling (initial implementation):
     - Allocate a signal slot in GTT memory
     - GPU writes completion value via RELEASE_MEM PM4 packet
     - CPU busy-waits on the signal address
     - Simple but burns CPU cycles

  2. Interrupt-driven (future):
     - Configure IH (Interrupt Handler) ring
     - GPU RELEASE_MEM triggers MSI-X interrupt
     - DEXT forwards interrupt to userspace
     - CPU blocks in WaitInterrupt() call
"""

from __future__ import annotations

import ctypes
import time

from amd_gpu_driver.backends.base import MemoryHandle, MemoryLocation, SignalHandle
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.memory import MacOSMemoryManager

# Signal page layout: array of 64-bit signal slots
SIGNAL_SLOT_SIZE = 8         # 8 bytes per signal (uint64)
SIGNAL_PAGE_SIZE = 4096      # One page of signals
MAX_SIGNALS = SIGNAL_PAGE_SIZE // SIGNAL_SLOT_SIZE  # 512 signals per page

# Signal states
SIGNAL_UNSIGNALED = 0
SIGNAL_SIGNALED = 1


class MacOSEventManager:
    """Manages GPU-CPU synchronization signals on macOS.

    Each signal is an 8-byte slot in a shared GTT page. The GPU writes
    to this slot via RELEASE_MEM packets, and the CPU polls or waits
    for the value to change.

    Usage:
        signal = event_mgr.create_signal()
        # ... submit GPU work with RELEASE_MEM targeting signal.signal_addr ...
        event_mgr.wait_signal(signal, timeout_ms=5000)
        event_mgr.destroy_signal(signal)
    """

    def __init__(
        self,
        client: IOKitClient,
        memory: MacOSMemoryManager,
    ) -> None:
        self._client = client
        self._memory = memory
        self._signal_page: MemoryHandle | None = None
        self._signal_page_addr: int = 0
        self._slot_in_use: list[bool] = [False] * MAX_SIGNALS
        self._signals: dict[int, SignalHandle] = {}

    def _ensure_signal_page(self) -> None:
        """Lazily allocate the signal page on first use."""
        if self._signal_page is not None:
            return

        self._signal_page = self._memory.alloc(
            SIGNAL_PAGE_SIZE,
            MemoryLocation.GTT,
            uncached=True,
        )
        self._signal_page_addr = self._signal_page.cpu_addr

        # Initialize all slots to unsignaled
        ctypes.memset(self._signal_page_addr, 0, SIGNAL_PAGE_SIZE)

    def create_signal(self) -> SignalHandle:
        """Create a new signal event.

        Returns a SignalHandle with:
          - signal_addr: GPU-accessible address for RELEASE_MEM target
          - event_slot_index: index into the signal page
          - event_id: unique identifier
        """
        self._ensure_signal_page()

        # Find a free slot
        slot_index = -1
        for i, in_use in enumerate(self._slot_in_use):
            if not in_use:
                slot_index = i
                break

        if slot_index < 0:
            raise RuntimeError(f"Signal slots exhausted (max {MAX_SIGNALS})")

        self._slot_in_use[slot_index] = True

        # Calculate addresses
        cpu_addr = self._signal_page_addr + (slot_index * SIGNAL_SLOT_SIZE)

        # GPU address = physical address of the signal slot
        # (for GPU page table mapping, use the DMA segment address)
        phys_addr = self._memory.get_phys_addr(self._signal_page)
        gpu_addr = phys_addr + (slot_index * SIGNAL_SLOT_SIZE)

        # Initialize to unsignaled
        ctypes.c_uint64.from_address(cpu_addr).value = SIGNAL_UNSIGNALED

        handle = SignalHandle(
            event_id=slot_index,
            signal_addr=gpu_addr,
            event_slot_index=slot_index,
            event_page_offset=slot_index * SIGNAL_SLOT_SIZE,
        )

        self._signals[slot_index] = handle
        return handle

    def destroy_signal(self, handle: SignalHandle) -> None:
        """Release a signal slot."""
        idx = handle.event_slot_index
        if 0 <= idx < MAX_SIGNALS:
            self._slot_in_use[idx] = False
            self._signals.pop(idx, None)

    def wait_signal(
        self,
        handle: SignalHandle,
        timeout_ms: int = 5000,
        expected_value: int = SIGNAL_SIGNALED,
    ) -> None:
        """Wait for a signal to reach the expected value.

        Polling implementation: spins on the CPU address until the GPU
        writes the expected value or timeout is reached.

        For production use, this should be replaced with interrupt-driven
        waiting via the DEXT's EnableMSI + WaitInterrupt commands.
        """
        if self._signal_page is None:
            raise RuntimeError("No signal page allocated")

        cpu_addr = (
            self._signal_page_addr
            + handle.event_slot_index * SIGNAL_SLOT_SIZE
        )

        deadline = time.monotonic() + (timeout_ms / 1000.0)
        poll_interval = 0.000001  # Start at 1us

        while time.monotonic() < deadline:
            value = ctypes.c_uint64.from_address(cpu_addr).value
            if value >= expected_value:
                return

            # Exponential backoff: 1us -> 2us -> 4us -> ... -> 1ms max
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, 0.001)

        # Timeout — read final value for diagnostics
        final_value = ctypes.c_uint64.from_address(cpu_addr).value
        raise TimeoutError(
            f"Signal wait timed out after {timeout_ms}ms "
            f"(slot={handle.event_slot_index}, "
            f"expected={expected_value}, got={final_value})"
        )

    def reset_signal(self, handle: SignalHandle) -> None:
        """Reset a signal to unsignaled state for reuse."""
        if self._signal_page is None:
            return

        cpu_addr = (
            self._signal_page_addr
            + handle.event_slot_index * SIGNAL_SLOT_SIZE
        )
        ctypes.c_uint64.from_address(cpu_addr).value = SIGNAL_UNSIGNALED

    def get_signal_value(self, handle: SignalHandle) -> int:
        """Read the current signal value without waiting."""
        if self._signal_page is None:
            return 0

        cpu_addr = (
            self._signal_page_addr
            + handle.event_slot_index * SIGNAL_SLOT_SIZE
        )
        return ctypes.c_uint64.from_address(cpu_addr).value

    def cleanup(self) -> None:
        """Release all signal resources."""
        if self._signal_page is not None:
            self._memory.free(self._signal_page)
            self._signal_page = None
            self._signal_page_addr = 0
            self._slot_in_use = [False] * MAX_SIGNALS
            self._signals.clear()
