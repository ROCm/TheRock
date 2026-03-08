"""Ioctl definitions for the amdgpu_lite kernel module.

Mirrors the structs and ioctl numbers from amdgpu_lite.h.
"""

from __future__ import annotations

import ctypes

from amd_gpu_driver.ioctl.helpers import _IOR, _IOW, _IOWR

AMDGPU_LITE_IOC_MAGIC = ord("L")
AMDGPU_LITE_MAX_BARS = 6

# Mmap offset encoding
AMDGPU_LITE_MMAP_TYPE_SHIFT = 60
AMDGPU_LITE_MMAP_TYPE_BAR = 0
AMDGPU_LITE_MMAP_TYPE_GTT = 1
AMDGPU_LITE_MMAP_TYPE_VRAM = 2

# GART constants (must match kernel module)
AMDGPU_LITE_GART_TABLE_SIZE = 1024 * 1024  # 1MB
AMDGPU_LITE_GART_NUM_ENTRIES = AMDGPU_LITE_GART_TABLE_SIZE // 8  # 128K
AMDGPU_LITE_GART_VA_START = 0x100000000  # 4GB


# --- Structures ---


class amdgpu_lite_bar_info(ctypes.Structure):
    _fields_ = [
        ("phys_addr", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("is_memory", ctypes.c_uint32),
        ("is_64bit", ctypes.c_uint32),
        ("is_prefetchable", ctypes.c_uint32),
        ("bar_index", ctypes.c_uint32),
    ]


class amdgpu_lite_get_info(ctypes.Structure):
    _fields_ = [
        ("vendor_id", ctypes.c_uint16),
        ("device_id", ctypes.c_uint16),
        ("subsystem_vendor_id", ctypes.c_uint16),
        ("subsystem_id", ctypes.c_uint16),
        ("revision_id", ctypes.c_uint8),
        ("reserved1", ctypes.c_uint8 * 3),
        ("num_bars", ctypes.c_uint32),
        ("bars", amdgpu_lite_bar_info * AMDGPU_LITE_MAX_BARS),
        ("vram_size", ctypes.c_uint64),
        ("visible_vram_size", ctypes.c_uint64),
        ("mmio_bar_index", ctypes.c_uint32),
        ("vram_bar_index", ctypes.c_uint32),
        ("doorbell_bar_index", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("gart_table_bus_addr", ctypes.c_uint64),
        ("gart_table_size", ctypes.c_uint64),
        ("gart_gpu_va_start", ctypes.c_uint64),
    ]


class amdgpu_lite_map_bar(ctypes.Structure):
    _fields_ = [
        ("bar_index", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("mmap_offset", ctypes.c_uint64),
        ("reserved2", ctypes.c_uint32 * 4),
    ]


class amdgpu_lite_alloc_gtt(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("reserved1", ctypes.c_uint32 * 2),
        ("handle", ctypes.c_uint64),
        ("bus_addr", ctypes.c_uint64),
        ("mmap_offset", ctypes.c_uint64),
        ("reserved2", ctypes.c_uint32 * 4),
    ]


class amdgpu_lite_free_gtt(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class amdgpu_lite_alloc_vram(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("handle", ctypes.c_uint64),
        ("gpu_addr", ctypes.c_uint64),
        ("mmap_offset", ctypes.c_uint64),
        ("reserved2", ctypes.c_uint32 * 4),
    ]


class amdgpu_lite_free_vram(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class amdgpu_lite_map_gpu(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),
        ("gpu_va", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
        ("mapped_gpu_va", ctypes.c_uint64),
    ]


class amdgpu_lite_unmap_gpu(ctypes.Structure):
    _fields_ = [
        ("gpu_va", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class amdgpu_lite_setup_irq(ctypes.Structure):
    _fields_ = [
        ("eventfd", ctypes.c_int32),
        ("irq_source", ctypes.c_uint32),
        ("registration_id", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
        ("out_registration_id", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
    ]


# --- Ioctl numbers ---


def _lite_ior(nr: int, struct_type: type) -> int:
    return _IOR(AMDGPU_LITE_IOC_MAGIC, nr, ctypes.sizeof(struct_type))


def _lite_iow(nr: int, struct_type: type) -> int:
    return _IOW(AMDGPU_LITE_IOC_MAGIC, nr, ctypes.sizeof(struct_type))


def _lite_iowr(nr: int, struct_type: type) -> int:
    return _IOWR(AMDGPU_LITE_IOC_MAGIC, nr, ctypes.sizeof(struct_type))


AMDGPU_LITE_IOC_GET_INFO = _lite_ior(0x01, amdgpu_lite_get_info)
AMDGPU_LITE_IOC_MAP_BAR = _lite_iowr(0x02, amdgpu_lite_map_bar)
AMDGPU_LITE_IOC_ALLOC_GTT = _lite_iowr(0x10, amdgpu_lite_alloc_gtt)
AMDGPU_LITE_IOC_FREE_GTT = _lite_iow(0x11, amdgpu_lite_free_gtt)
AMDGPU_LITE_IOC_ALLOC_VRAM = _lite_iowr(0x20, amdgpu_lite_alloc_vram)
AMDGPU_LITE_IOC_FREE_VRAM = _lite_iow(0x21, amdgpu_lite_free_vram)
AMDGPU_LITE_IOC_MAP_GPU = _lite_iowr(0x30, amdgpu_lite_map_gpu)
AMDGPU_LITE_IOC_UNMAP_GPU = _lite_iow(0x31, amdgpu_lite_unmap_gpu)
AMDGPU_LITE_IOC_SETUP_IRQ = _lite_iowr(0x40, amdgpu_lite_setup_irq)
