"""KFD event creation and waiting."""

from __future__ import annotations

import ctypes

from amd_gpu_driver.backends.base import SignalHandle
from amd_gpu_driver.errors import TimeoutError
from amd_gpu_driver.ioctl import helpers
from amd_gpu_driver.ioctl.kfd import (
    AMDKFD_IOC_CREATE_EVENT,
    AMDKFD_IOC_DESTROY_EVENT,
    AMDKFD_IOC_WAIT_EVENTS,
    KFD_IOC_EVENT_SIGNAL,
    KFD_IOC_WAIT_RESULT_COMPLETE,
    KFD_IOC_WAIT_RESULT_TIMEOUT,
    KFD_SIGNAL_EVENT_LIMIT,
    kfd_event_data,
    kfd_ioctl_create_event_args,
    kfd_ioctl_destroy_event_args,
    kfd_ioctl_wait_events_args,
)


class KFDEventManager:
    """Manages KFD signal events."""

    def __init__(self, kfd_fd: int, gpu_id: int) -> None:
        self._kfd_fd = kfd_fd
        self._gpu_id = gpu_id
        self._events: list[SignalHandle] = []
        self._event_page_addr: int = 0

    def create_signal(self, auto_reset: bool = True) -> SignalHandle:
        """Create a signal event."""
        args = kfd_ioctl_create_event_args()
        args.event_type = KFD_IOC_EVENT_SIGNAL
        args.auto_reset = 1 if auto_reset else 0
        args.node_id = 0

        helpers.ioctl(
            self._kfd_fd, AMDKFD_IOC_CREATE_EVENT, args, "CREATE_EVENT"
        )

        # mmap the event page if this is the first event
        if self._event_page_addr == 0 and args.event_page_offset:
            page_size = 4096
            self._event_page_addr = helpers.libc_mmap(
                None,
                page_size,
                helpers.PROT_READ | helpers.PROT_WRITE,
                helpers.MAP_SHARED,
                self._kfd_fd,
                args.event_page_offset,
            )

        signal_addr = 0
        if self._event_page_addr and args.event_slot_index < KFD_SIGNAL_EVENT_LIMIT:
            # Each event slot is 8 bytes (uint64)
            signal_addr = self._event_page_addr + args.event_slot_index * 8

        handle = SignalHandle(
            event_id=args.event_id,
            signal_addr=signal_addr,
            event_slot_index=args.event_slot_index,
            event_page_offset=args.event_page_offset,
        )
        self._events.append(handle)
        return handle

    def wait(self, handle: SignalHandle, timeout_ms: int = 5000) -> None:
        """Wait for a signal event."""
        event_data = kfd_event_data()
        event_data.event_id = handle.event_id

        wait_args = kfd_ioctl_wait_events_args()
        wait_args.events_ptr = ctypes.addressof(event_data)
        wait_args.num_events = 1
        wait_args.wait_for_all = 1
        wait_args.timeout = timeout_ms

        helpers.ioctl(
            self._kfd_fd, AMDKFD_IOC_WAIT_EVENTS, wait_args, "WAIT_EVENTS"
        )

        if wait_args.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT:
            raise TimeoutError("wait_signal", timeout_ms)

    def destroy(self, handle: SignalHandle) -> None:
        """Destroy a signal event."""
        args = kfd_ioctl_destroy_event_args()
        args.event_id = handle.event_id
        try:
            helpers.ioctl(
                self._kfd_fd, AMDKFD_IOC_DESTROY_EVENT, args, "DESTROY_EVENT"
            )
        except Exception:
            pass
        if handle in self._events:
            self._events.remove(handle)

    def destroy_all(self) -> None:
        """Destroy all tracked events."""
        for evt in list(self._events):
            self.destroy(evt)
