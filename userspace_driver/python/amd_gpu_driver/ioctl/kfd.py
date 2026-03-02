"""KFD ioctl structs and numbers translated from kfd_ioctl.h."""

from __future__ import annotations

import ctypes

from amd_gpu_driver.ioctl.helpers import _IOC_READ, _IOC_WRITE, _IOR, _IOW, _IOWR

# AMDKFD_IOCTL_BASE = 'K' = 0x4B
AMDKFD_IOCTL_BASE = ord("K")

# --- KFD version ---
KFD_IOCTL_MAJOR_VERSION = 1
KFD_IOCTL_MINOR_VERSION = 18

# --- Queue types ---
KFD_IOC_QUEUE_TYPE_COMPUTE = 0x0
KFD_IOC_QUEUE_TYPE_SDMA = 0x1
KFD_IOC_QUEUE_TYPE_COMPUTE_AQL = 0x2
KFD_IOC_QUEUE_TYPE_SDMA_XGMI = 0x3
KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID = 0x4

KFD_MAX_QUEUE_PERCENTAGE = 100
KFD_MAX_QUEUE_PRIORITY = 15

# --- Event types ---
KFD_IOC_EVENT_SIGNAL = 0
KFD_IOC_EVENT_NODECHANGE = 1
KFD_IOC_EVENT_DEVICESTATECHANGE = 2
KFD_IOC_EVENT_HW_EXCEPTION = 3
KFD_IOC_EVENT_SYSTEM_EVENT = 4
KFD_IOC_EVENT_DEBUG_EVENT = 5
KFD_IOC_EVENT_PROFILE_EVENT = 6
KFD_IOC_EVENT_QUEUE_EVENT = 7
KFD_IOC_EVENT_MEMORY = 8

# --- Wait results ---
KFD_IOC_WAIT_RESULT_COMPLETE = 0
KFD_IOC_WAIT_RESULT_TIMEOUT = 1
KFD_IOC_WAIT_RESULT_FAIL = 2

KFD_SIGNAL_EVENT_LIMIT = 4096

# --- Memory allocation flags ---
KFD_IOC_ALLOC_MEM_FLAGS_VRAM = 1 << 0
KFD_IOC_ALLOC_MEM_FLAGS_GTT = 1 << 1
KFD_IOC_ALLOC_MEM_FLAGS_USERPTR = 1 << 2
KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL = 1 << 3
KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP = 1 << 4
KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE = 1 << 31
KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE = 1 << 30
KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC = 1 << 29
KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE = 1 << 28
KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM = 1 << 27
KFD_IOC_ALLOC_MEM_FLAGS_COHERENT = 1 << 26
KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED = 1 << 25
KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT = 1 << 24
KFD_IOC_ALLOC_MEM_FLAGS_CONTIGUOUS_BEST_EFFORT = 1 << 23

# --- MMIO remap offsets ---
KFD_MMIO_REMAP_HDP_MEM_FLUSH_CNTL = 0
KFD_MMIO_REMAP_HDP_REG_FLUSH_CNTL = 4

# --- Runtime enable masks ---
KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK = 1
KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK = 2


# ============================================================================
# ctypes struct definitions - matching kfd_ioctl.h exactly
# ============================================================================


class kfd_ioctl_get_version_args(ctypes.Structure):
    _fields_ = [
        ("major_version", ctypes.c_uint32),  # from KFD
        ("minor_version", ctypes.c_uint32),  # from KFD
    ]


class kfd_ioctl_create_queue_args(ctypes.Structure):
    _fields_ = [
        ("ring_base_address", ctypes.c_uint64),  # to KFD
        ("write_pointer_address", ctypes.c_uint64),  # to KFD
        ("read_pointer_address", ctypes.c_uint64),  # to KFD
        ("doorbell_offset", ctypes.c_uint64),  # from KFD
        ("ring_size", ctypes.c_uint32),  # to KFD
        ("gpu_id", ctypes.c_uint32),  # to KFD
        ("queue_type", ctypes.c_uint32),  # to KFD
        ("queue_percentage", ctypes.c_uint32),  # to KFD
        ("queue_priority", ctypes.c_uint32),  # to KFD
        ("queue_id", ctypes.c_uint32),  # from KFD
        ("eop_buffer_address", ctypes.c_uint64),  # to KFD
        ("eop_buffer_size", ctypes.c_uint64),  # to KFD
        ("ctx_save_restore_address", ctypes.c_uint64),  # to KFD
        ("ctx_save_restore_size", ctypes.c_uint32),  # to KFD
        ("ctl_stack_size", ctypes.c_uint32),  # to KFD
        ("sdma_engine_id", ctypes.c_uint32),  # to KFD
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_destroy_queue_args(ctypes.Structure):
    _fields_ = [
        ("queue_id", ctypes.c_uint32),  # to KFD
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_update_queue_args(ctypes.Structure):
    _fields_ = [
        ("ring_base_address", ctypes.c_uint64),  # to KFD
        ("queue_id", ctypes.c_uint32),  # to KFD
        ("ring_size", ctypes.c_uint32),  # to KFD
        ("queue_percentage", ctypes.c_uint32),  # to KFD
        ("queue_priority", ctypes.c_uint32),  # to KFD
    ]


class kfd_ioctl_set_memory_policy_args(ctypes.Structure):
    _fields_ = [
        ("alternate_aperture_base", ctypes.c_uint64),
        ("alternate_aperture_size", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("default_policy", ctypes.c_uint32),
        ("alternate_policy", ctypes.c_uint32),
        ("misc_process_flag", ctypes.c_uint32),
    ]


class kfd_ioctl_get_clock_counters_args(ctypes.Structure):
    _fields_ = [
        ("gpu_clock_counter", ctypes.c_uint64),
        ("cpu_clock_counter", ctypes.c_uint64),
        ("system_clock_counter", ctypes.c_uint64),
        ("system_clock_freq", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_process_device_apertures(ctypes.Structure):
    _fields_ = [
        ("lds_base", ctypes.c_uint64),
        ("lds_limit", ctypes.c_uint64),
        ("scratch_base", ctypes.c_uint64),
        ("scratch_limit", ctypes.c_uint64),
        ("gpuvm_base", ctypes.c_uint64),
        ("gpuvm_limit", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_get_process_apertures_new_args(ctypes.Structure):
    _fields_ = [
        ("kfd_process_device_apertures_ptr", ctypes.c_uint64),
        ("num_of_nodes", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_acquire_vm_args(ctypes.Structure):
    _fields_ = [
        ("drm_fd", ctypes.c_uint32),  # to KFD
        ("gpu_id", ctypes.c_uint32),  # to KFD
    ]


class kfd_ioctl_alloc_memory_of_gpu_args(ctypes.Structure):
    _fields_ = [
        ("va_addr", ctypes.c_uint64),  # to KFD
        ("size", ctypes.c_uint64),  # to KFD
        ("handle", ctypes.c_uint64),  # from KFD
        ("mmap_offset", ctypes.c_uint64),  # to/from KFD
        ("gpu_id", ctypes.c_uint32),  # to KFD
        ("flags", ctypes.c_uint32),
    ]


class kfd_ioctl_free_memory_of_gpu_args(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),  # to KFD
    ]


class kfd_ioctl_get_available_memory_args(ctypes.Structure):
    _fields_ = [
        ("available", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_map_memory_to_gpu_args(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),  # to KFD
        ("device_ids_array_ptr", ctypes.c_uint64),  # to KFD
        ("n_devices", ctypes.c_uint32),  # to KFD
        ("n_success", ctypes.c_uint32),  # to/from KFD
    ]


class kfd_ioctl_unmap_memory_from_gpu_args(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),  # to KFD
        ("device_ids_array_ptr", ctypes.c_uint64),  # to KFD
        ("n_devices", ctypes.c_uint32),  # to KFD
        ("n_success", ctypes.c_uint32),  # to/from KFD
    ]


class kfd_ioctl_create_event_args(ctypes.Structure):
    _fields_ = [
        ("event_page_offset", ctypes.c_uint64),  # from KFD
        ("event_trigger_data", ctypes.c_uint32),  # from KFD
        ("event_type", ctypes.c_uint32),  # to KFD
        ("auto_reset", ctypes.c_uint32),  # to KFD
        ("node_id", ctypes.c_uint32),  # to KFD
        ("event_id", ctypes.c_uint32),  # from KFD
        ("event_slot_index", ctypes.c_uint32),  # from KFD
    ]


class kfd_ioctl_destroy_event_args(ctypes.Structure):
    _fields_ = [
        ("event_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_set_event_args(ctypes.Structure):
    _fields_ = [
        ("event_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_reset_event_args(ctypes.Structure):
    _fields_ = [
        ("event_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_memory_exception_failure(ctypes.Structure):
    _fields_ = [
        ("NotPresent", ctypes.c_uint32),
        ("ReadOnly", ctypes.c_uint32),
        ("NoExecute", ctypes.c_uint32),
        ("imprecise", ctypes.c_uint32),
    ]


class kfd_hsa_memory_exception_data(ctypes.Structure):
    _fields_ = [
        ("failure", kfd_memory_exception_failure),
        ("va", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("ErrorType", ctypes.c_uint32),
    ]


class kfd_hsa_hw_exception_data(ctypes.Structure):
    _fields_ = [
        ("reset_type", ctypes.c_uint32),
        ("reset_cause", ctypes.c_uint32),
        ("memory_lost", ctypes.c_uint32),
        ("gpu_id", ctypes.c_uint32),
    ]


class kfd_hsa_signal_event_data(ctypes.Structure):
    _fields_ = [
        ("last_event_age", ctypes.c_uint64),
    ]


class kfd_event_data(ctypes.Structure):
    """Event data union - sized to the largest member."""

    class _event_union(ctypes.Union):
        _fields_ = [
            ("memory_exception_data", kfd_hsa_memory_exception_data),
            ("hw_exception_data", kfd_hsa_hw_exception_data),
            ("signal_event_data", kfd_hsa_signal_event_data),
        ]

    _fields_ = [
        ("event_data", _event_union),
        ("kfd_event_data_ext", ctypes.c_uint64),
        ("event_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_wait_events_args(ctypes.Structure):
    _fields_ = [
        ("events_ptr", ctypes.c_uint64),  # to KFD
        ("num_events", ctypes.c_uint32),  # to KFD
        ("wait_for_all", ctypes.c_uint32),  # to KFD
        ("timeout", ctypes.c_uint32),  # to KFD
        ("wait_result", ctypes.c_uint32),  # from KFD
    ]


class kfd_ioctl_set_scratch_backing_va_args(ctypes.Structure):
    _fields_ = [
        ("va_addr", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_set_trap_handler_args(ctypes.Structure):
    _fields_ = [
        ("tba_addr", ctypes.c_uint64),
        ("tma_addr", ctypes.c_uint64),
        ("gpu_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


class kfd_ioctl_export_dmabuf_args(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("dmabuf_fd", ctypes.c_uint32),
    ]


class kfd_ioctl_runtime_enable_args(ctypes.Structure):
    _fields_ = [
        ("r_debug", ctypes.c_uint64),
        ("mode_mask", ctypes.c_uint32),
        ("capabilities_mask", ctypes.c_uint32),
    ]


# ============================================================================
# Ioctl number definitions
# ============================================================================

def _kfd_ior(nr: int, struct_type: type) -> int:
    return _IOR(AMDKFD_IOCTL_BASE, nr, ctypes.sizeof(struct_type))


def _kfd_iow(nr: int, struct_type: type) -> int:
    return _IOW(AMDKFD_IOCTL_BASE, nr, ctypes.sizeof(struct_type))


def _kfd_iowr(nr: int, struct_type: type) -> int:
    return _IOWR(AMDKFD_IOCTL_BASE, nr, ctypes.sizeof(struct_type))


AMDKFD_IOC_GET_VERSION = _kfd_ior(0x01, kfd_ioctl_get_version_args)
AMDKFD_IOC_CREATE_QUEUE = _kfd_iowr(0x02, kfd_ioctl_create_queue_args)
AMDKFD_IOC_DESTROY_QUEUE = _kfd_iowr(0x03, kfd_ioctl_destroy_queue_args)
AMDKFD_IOC_SET_MEMORY_POLICY = _kfd_iow(0x04, kfd_ioctl_set_memory_policy_args)
AMDKFD_IOC_GET_CLOCK_COUNTERS = _kfd_iowr(0x05, kfd_ioctl_get_clock_counters_args)
AMDKFD_IOC_UPDATE_QUEUE = _kfd_iow(0x07, kfd_ioctl_update_queue_args)
AMDKFD_IOC_CREATE_EVENT = _kfd_iowr(0x08, kfd_ioctl_create_event_args)
AMDKFD_IOC_DESTROY_EVENT = _kfd_iow(0x09, kfd_ioctl_destroy_event_args)
AMDKFD_IOC_SET_EVENT = _kfd_iow(0x0A, kfd_ioctl_set_event_args)
AMDKFD_IOC_RESET_EVENT = _kfd_iow(0x0B, kfd_ioctl_reset_event_args)
AMDKFD_IOC_WAIT_EVENTS = _kfd_iowr(0x0C, kfd_ioctl_wait_events_args)
AMDKFD_IOC_SET_SCRATCH_BACKING_VA = _kfd_iowr(0x11, kfd_ioctl_set_scratch_backing_va_args)
AMDKFD_IOC_SET_TRAP_HANDLER = _kfd_iow(0x13, kfd_ioctl_set_trap_handler_args)
AMDKFD_IOC_GET_PROCESS_APERTURES_NEW = _kfd_iowr(0x14, kfd_ioctl_get_process_apertures_new_args)
AMDKFD_IOC_ACQUIRE_VM = _kfd_iow(0x15, kfd_ioctl_acquire_vm_args)
AMDKFD_IOC_ALLOC_MEMORY_OF_GPU = _kfd_iowr(0x16, kfd_ioctl_alloc_memory_of_gpu_args)
AMDKFD_IOC_FREE_MEMORY_OF_GPU = _kfd_iow(0x17, kfd_ioctl_free_memory_of_gpu_args)
AMDKFD_IOC_MAP_MEMORY_TO_GPU = _kfd_iowr(0x18, kfd_ioctl_map_memory_to_gpu_args)
AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU = _kfd_iowr(0x19, kfd_ioctl_unmap_memory_from_gpu_args)
AMDKFD_IOC_AVAILABLE_MEMORY = _kfd_iowr(0x23, kfd_ioctl_get_available_memory_args)
AMDKFD_IOC_EXPORT_DMABUF = _kfd_iowr(0x24, kfd_ioctl_export_dmabuf_args)
AMDKFD_IOC_RUNTIME_ENABLE = _kfd_iowr(0x25, kfd_ioctl_runtime_enable_args)
