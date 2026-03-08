"""Windows MCDM backend - talks to amdgpu_mcdm.sys via D3DKMTEscape.

The device and dispatch modules require Windows (ctypes.windll). On Linux,
importing this package still works — individual submodules like ip_discovery,
nbio_init, gmc_init, etc. are platform-independent and can be imported
directly. Only WindowsDevice and compute_dispatch require Windows.
"""

import sys

if sys.platform == "win32":
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
else:
    __all__: list[str] = []
