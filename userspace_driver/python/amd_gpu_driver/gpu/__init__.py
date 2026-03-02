"""GPU family configurations.

Importing this module registers all known GPU family configs.
"""

from amd_gpu_driver.gpu.family import GPUFamilyConfig, get_gpu_family, register_gpu_family

# Import all family modules to trigger registration
from amd_gpu_driver.gpu import cdna2, cdna3, rdna2, rdna3, rdna4

__all__ = ["GPUFamilyConfig", "get_gpu_family", "register_gpu_family"]
