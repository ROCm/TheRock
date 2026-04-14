"""macOS DriverKit backend for AMD eGPU access.

Communicates with the ROCmGPU.dext DriverKit extension via IOKit
(IOConnectCallScalarMethod / IOConnectMapMemory64) to provide PCIe
BAR mapping, MMIO register access, DMA allocation, and interrupt
handling for Thunderbolt-attached AMD GPUs.
"""

from amd_gpu_driver.backends.macos.device import MacOSDevice

__all__ = ["MacOSDevice"]
