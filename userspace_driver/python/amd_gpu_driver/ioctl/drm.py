"""Minimal DRM/AMDGPU ioctl bindings for device queries and mmap offsets."""

from __future__ import annotations

import ctypes

from amd_gpu_driver.ioctl.helpers import _IOR, _IOC_READ, _IOC_WRITE, _IOWR

# DRM ioctl base
DRM_IOCTL_BASE = ord("d")

# DRM command numbers for AMDGPU
DRM_COMMAND_BASE = 0x40

# AMDGPU-specific DRM command offsets
DRM_AMDGPU_GEM_MMAP = 0x05
DRM_AMDGPU_INFO = 0x07

# --- AMDGPU_INFO query types ---
AMDGPU_INFO_DEV_INFO = 0x16
AMDGPU_INFO_MEMORY = 0x19
AMDGPU_INFO_VRAM_GTT = 0x14


# --- Structs ---


class drm_amdgpu_info_request(ctypes.Structure):
    """Simplified AMDGPU info request - query field + return_pointer + return_size."""

    _fields_ = [
        ("return_pointer", ctypes.c_uint64),
        ("return_size", ctypes.c_uint32),
        ("query", ctypes.c_uint32),
        # Padding for the union fields in the full struct
        ("_pad", ctypes.c_uint8 * 24),
    ]


class drm_amdgpu_gem_mmap_in(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
    ]


class drm_amdgpu_gem_mmap_out(ctypes.Structure):
    _fields_ = [
        ("addr_ptr", ctypes.c_uint64),
    ]


class drm_amdgpu_gem_mmap(ctypes.Union):
    _fields_ = [
        ("in_", drm_amdgpu_gem_mmap_in),
        ("out", drm_amdgpu_gem_mmap_out),
    ]


# --- Ioctl numbers ---

# DRM_IOCTL_AMDGPU_INFO = _IOW('d', DRM_COMMAND_BASE + DRM_AMDGPU_INFO, sizeof(info_request))
DRM_IOCTL_AMDGPU_INFO = _IOW(
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_AMDGPU_INFO,
    ctypes.sizeof(drm_amdgpu_info_request),
)

# DRM_IOCTL_AMDGPU_GEM_MMAP = _IOWR('d', DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP, sizeof(gem_mmap))
DRM_IOCTL_AMDGPU_GEM_MMAP = _IOWR(
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + DRM_AMDGPU_GEM_MMAP,
    ctypes.sizeof(drm_amdgpu_gem_mmap),
)
