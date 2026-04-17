"""Windows MCDM backend - talks to amdgpu_mcdm.sys via D3DKMTEscape.

The device/compute_dispatch modules require the Windows D3DKMT API and
can't be imported on other platforms. Register-programming init modules
(nbio_init, gmc_init, psp_init, ih_init, ring_init, ip_discovery) are
OS-agnostic and can be imported directly (e.g. by the macOS backend).
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
    __all__ = []
