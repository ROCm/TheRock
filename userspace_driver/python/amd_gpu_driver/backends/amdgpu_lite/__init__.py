"""amdgpu_lite backend: talks to /dev/amdgpu_lite0 via our lightweight kernel module."""

from amd_gpu_driver.backends.amdgpu_lite.device import AmdgpuLiteDevice

__all__ = ["AmdgpuLiteDevice"]
