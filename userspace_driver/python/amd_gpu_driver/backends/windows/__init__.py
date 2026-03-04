"""Windows MCDM backend - talks to amdgpu_mcdm.sys via D3DKMTEscape."""

from amd_gpu_driver.backends.windows.device import WindowsDevice
from amd_gpu_driver.backends.windows.compute_dispatch import (
    GPUContext,
    full_gpu_bringup,
    run_demo,
    shutdown,
)

__all__ = [
    "WindowsDevice",
    "GPUContext",
    "full_gpu_bringup",
    "run_demo",
    "shutdown",
]
