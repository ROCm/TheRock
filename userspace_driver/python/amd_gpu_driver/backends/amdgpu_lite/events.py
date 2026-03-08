"""Event/signal management via amdgpu_lite eventfd-based interrupts.

Unlike KFD which has a dedicated event mechanism with shared event pages,
amdgpu_lite uses Linux eventfd for interrupt forwarding. The kernel ISR
signals all registered eventfds when an MSI-X interrupt fires.
"""

from __future__ import annotations

import ctypes
import os
import select

from amd_gpu_driver.backends.base import SignalHandle
from amd_gpu_driver.errors import TimeoutError
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.amdgpu_lite import (
    AMDGPU_LITE_IOC_SETUP_IRQ,
    amdgpu_lite_setup_irq,
)


class LiteEventManager:
    """Manages signal events via amdgpu_lite eventfd-based interrupts."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._events: list[SignalHandle] = []
        self._eventfds: dict[int, int] = {}  # event_id -> eventfd

    def create_signal(self, irq_source: int = 0) -> SignalHandle:
        """Create a signal event backed by an eventfd.

        The kernel module's ISR will signal this eventfd when the specified
        interrupt source fires.
        """
        # Create a Linux eventfd (semaphore mode = 0)
        efd = os.eventfd(0)  # type: ignore[attr-defined]

        # Register with kernel module
        args = amdgpu_lite_setup_irq()
        args.eventfd = efd
        args.irq_source = irq_source
        args.registration_id = 0  # 0 = new registration

        helpers.ioctl(self._fd, AMDGPU_LITE_IOC_SETUP_IRQ, args, "SETUP_IRQ")

        reg_id = args.out_registration_id

        handle = SignalHandle(
            event_id=reg_id,
            signal_addr=0,  # No shared memory page — eventfd-based
            event_slot_index=irq_source,
        )
        self._events.append(handle)
        self._eventfds[reg_id] = efd
        return handle

    def wait(self, handle: SignalHandle, timeout_ms: int = 5000) -> None:
        """Wait for a signal event (eventfd becomes readable)."""
        efd = self._eventfds.get(handle.event_id)
        if efd is None:
            raise ValueError(f"Unknown event ID {handle.event_id}")

        timeout_s = timeout_ms / 1000.0
        ready, _, _ = select.select([efd], [], [], timeout_s)
        if not ready:
            raise TimeoutError("wait_signal", timeout_ms)

        # Consume the eventfd counter
        os.eventfd_read(efd)  # type: ignore[attr-defined]

    def destroy(self, handle: SignalHandle) -> None:
        """Tear down a signal event registration."""
        efd = self._eventfds.pop(handle.event_id, None)

        # Tell kernel to release this registration
        args = amdgpu_lite_setup_irq()
        args.eventfd = -1
        args.irq_source = 0
        args.registration_id = handle.event_id  # nonzero = teardown
        try:
            helpers.ioctl(self._fd, AMDGPU_LITE_IOC_SETUP_IRQ, args, "SETUP_IRQ")
        except Exception:
            pass

        # Close the eventfd
        if efd is not None:
            try:
                os.close(efd)
            except Exception:
                pass

        if handle in self._events:
            self._events.remove(handle)

    def destroy_all(self) -> None:
        """Destroy all tracked events."""
        for evt in list(self._events):
            self.destroy(evt)
