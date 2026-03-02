"""KFD backend - talks to /dev/kfd and /dev/dri/renderD*."""

from amd_gpu_driver.backends.kfd.device import KFDDevice

__all__ = ["KFDDevice"]
