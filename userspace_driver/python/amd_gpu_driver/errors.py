"""Exception hierarchy for the AMD GPU userspace driver."""

from __future__ import annotations


class AMDGPUDriverError(Exception):
    """Base exception for all AMD GPU driver errors."""


class DeviceNotFoundError(AMDGPUDriverError):
    """No AMD GPU device found at the specified index."""

    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        super().__init__(f"No AMD GPU device found at index {device_index}")


class IoctlError(AMDGPUDriverError):
    """An ioctl call failed."""

    def __init__(self, ioctl_name: str, errno: int) -> None:
        self.ioctl_name = ioctl_name
        self.errno = errno
        import os

        self.strerror = os.strerror(errno)
        super().__init__(f"ioctl {ioctl_name} failed: [{errno}] {self.strerror}")


class MemoryAllocationError(AMDGPUDriverError):
    """Failed to allocate GPU memory."""

    def __init__(self, size: int, location: str, reason: str = "") -> None:
        self.size = size
        self.location = location
        msg = f"Failed to allocate {size} bytes of {location} memory"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class QueueError(AMDGPUDriverError):
    """Failed to create or operate on a hardware queue."""


class KernelLoadError(AMDGPUDriverError):
    """Failed to load a GPU kernel from an ELF code object."""


class TimeoutError(AMDGPUDriverError):
    """A wait operation timed out."""

    def __init__(self, operation: str, timeout_ms: int) -> None:
        self.operation = operation
        self.timeout_ms = timeout_ms
        super().__init__(f"{operation} timed out after {timeout_ms}ms")


class UnsupportedGPUError(AMDGPUDriverError):
    """The GPU is not supported by this driver."""

    def __init__(self, gfx_target_version: int) -> None:
        self.gfx_target_version = gfx_target_version
        major = gfx_target_version // 10000
        minor = (gfx_target_version // 100) % 100
        stepping = gfx_target_version % 100
        super().__init__(
            f"Unsupported GPU: gfx{major}{minor:x}{stepping:x} "
            f"(gfx_target_version={gfx_target_version})"
        )
