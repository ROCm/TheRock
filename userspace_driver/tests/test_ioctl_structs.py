"""Verify ctypes struct sizes match expected C ABI sizes."""

import ctypes

from amd_gpu_driver.ioctl.kfd import (
    kfd_ioctl_get_version_args,
    kfd_ioctl_create_queue_args,
    kfd_ioctl_destroy_queue_args,
    kfd_ioctl_update_queue_args,
    kfd_ioctl_set_memory_policy_args,
    kfd_ioctl_get_clock_counters_args,
    kfd_process_device_apertures,
    kfd_ioctl_get_process_apertures_new_args,
    kfd_ioctl_acquire_vm_args,
    kfd_ioctl_alloc_memory_of_gpu_args,
    kfd_ioctl_free_memory_of_gpu_args,
    kfd_ioctl_get_available_memory_args,
    kfd_ioctl_map_memory_to_gpu_args,
    kfd_ioctl_unmap_memory_from_gpu_args,
    kfd_ioctl_create_event_args,
    kfd_ioctl_destroy_event_args,
    kfd_ioctl_set_event_args,
    kfd_ioctl_reset_event_args,
    kfd_ioctl_wait_events_args,
    kfd_ioctl_set_scratch_backing_va_args,
    kfd_ioctl_set_trap_handler_args,
    kfd_ioctl_export_dmabuf_args,
    kfd_ioctl_runtime_enable_args,
    kfd_event_data,
    kfd_memory_exception_failure,
    kfd_hsa_memory_exception_data,
    kfd_hsa_hw_exception_data,
    kfd_hsa_signal_event_data,
)


class TestKFDStructSizes:
    """Verify struct sizes match the C header definitions.

    These sizes are based on the Linux kernel ABI for KFD ioctls.
    Fields are __u32 (4 bytes) and __u64 (8 bytes) with natural alignment.
    """

    def test_get_version_args(self):
        # __u32 major + __u32 minor = 8
        assert ctypes.sizeof(kfd_ioctl_get_version_args) == 8

    def test_create_queue_args(self):
        # 3x __u64 (ring,wr,rd) + 1x __u64 (doorbell) = 32
        # 6x __u32 (ring_size..queue_id) = 24
        # 2x __u64 (eop_addr,eop_size) = 16
        # 1x __u64 (ctx_save_addr) = 8
        # 4x __u32 (ctx_save_size,ctl_stack,sdma_eng,pad) = 16
        # Total = 32 + 24 + 16 + 8 + 16 = 96
        assert ctypes.sizeof(kfd_ioctl_create_queue_args) == 96

    def test_destroy_queue_args(self):
        # __u32 queue_id + __u32 pad = 8
        assert ctypes.sizeof(kfd_ioctl_destroy_queue_args) == 8

    def test_update_queue_args(self):
        # __u64 ring_base + 4x __u32 = 8 + 16 = 24
        assert ctypes.sizeof(kfd_ioctl_update_queue_args) == 24

    def test_set_memory_policy_args(self):
        # 2x __u64 + 4x __u32 = 16 + 16 = 32
        assert ctypes.sizeof(kfd_ioctl_set_memory_policy_args) == 32

    def test_get_clock_counters_args(self):
        # 4x __u64 + 2x __u32 = 32 + 8 = 40
        assert ctypes.sizeof(kfd_ioctl_get_clock_counters_args) == 40

    def test_process_device_apertures(self):
        # 6x __u64 + 2x __u32 = 48 + 8 = 56
        assert ctypes.sizeof(kfd_process_device_apertures) == 56

    def test_get_process_apertures_new_args(self):
        # __u64 ptr + __u32 num + __u32 pad = 16
        assert ctypes.sizeof(kfd_ioctl_get_process_apertures_new_args) == 16

    def test_acquire_vm_args(self):
        # 2x __u32 = 8
        assert ctypes.sizeof(kfd_ioctl_acquire_vm_args) == 8

    def test_alloc_memory_of_gpu_args(self):
        # 4x __u64 + 2x __u32 = 32 + 8 = 40
        assert ctypes.sizeof(kfd_ioctl_alloc_memory_of_gpu_args) == 40

    def test_free_memory_of_gpu_args(self):
        # __u64 handle = 8
        assert ctypes.sizeof(kfd_ioctl_free_memory_of_gpu_args) == 8

    def test_get_available_memory_args(self):
        # __u64 + 2x __u32 = 16
        assert ctypes.sizeof(kfd_ioctl_get_available_memory_args) == 16

    def test_map_memory_to_gpu_args(self):
        # 2x __u64 + 2x __u32 = 24
        assert ctypes.sizeof(kfd_ioctl_map_memory_to_gpu_args) == 24

    def test_unmap_memory_from_gpu_args(self):
        # 2x __u64 + 2x __u32 = 24
        assert ctypes.sizeof(kfd_ioctl_unmap_memory_from_gpu_args) == 24

    def test_create_event_args(self):
        # __u64 + 6x __u32 = 8 + 24 = 32
        assert ctypes.sizeof(kfd_ioctl_create_event_args) == 32

    def test_destroy_event_args(self):
        # 2x __u32 = 8
        assert ctypes.sizeof(kfd_ioctl_destroy_event_args) == 8

    def test_set_event_args(self):
        assert ctypes.sizeof(kfd_ioctl_set_event_args) == 8

    def test_reset_event_args(self):
        assert ctypes.sizeof(kfd_ioctl_reset_event_args) == 8

    def test_wait_events_args(self):
        # __u64 + 4x __u32 = 8 + 16 = 24
        assert ctypes.sizeof(kfd_ioctl_wait_events_args) == 24

    def test_scratch_backing_va_args(self):
        # __u64 + 2x __u32 = 16
        assert ctypes.sizeof(kfd_ioctl_set_scratch_backing_va_args) == 16

    def test_set_trap_handler_args(self):
        # 2x __u64 + 2x __u32 = 24
        assert ctypes.sizeof(kfd_ioctl_set_trap_handler_args) == 24

    def test_export_dmabuf_args(self):
        # __u64 + 2x __u32 = 16
        assert ctypes.sizeof(kfd_ioctl_export_dmabuf_args) == 16

    def test_runtime_enable_args(self):
        # __u64 + 2x __u32 = 16
        assert ctypes.sizeof(kfd_ioctl_runtime_enable_args) == 16

    def test_memory_exception_failure(self):
        # 4x __u32 = 16
        assert ctypes.sizeof(kfd_memory_exception_failure) == 16

    def test_signal_event_data(self):
        # __u64 = 8
        assert ctypes.sizeof(kfd_hsa_signal_event_data) == 8

    def test_hw_exception_data(self):
        # 4x __u32 = 16
        assert ctypes.sizeof(kfd_hsa_hw_exception_data) == 16


class TestKFDIoctlNumbers:
    """Verify ioctl numbers are computed correctly."""

    def test_ioctl_base(self):
        from amd_gpu_driver.ioctl.kfd import AMDKFD_IOCTL_BASE
        assert AMDKFD_IOCTL_BASE == ord("K")

    def test_get_version_ioctl(self):
        from amd_gpu_driver.ioctl.kfd import AMDKFD_IOC_GET_VERSION
        # IOR with type='K'=0x4B, nr=0x01, size=8
        # direction = _IOC_READ = 2
        # (2 << 30) | (8 << 16) | (0x4B << 8) | 0x01
        expected = (2 << 30) | (8 << 16) | (0x4B << 8) | 0x01
        assert AMDKFD_IOC_GET_VERSION == expected

    def test_acquire_vm_ioctl(self):
        from amd_gpu_driver.ioctl.kfd import AMDKFD_IOC_ACQUIRE_VM
        # IOW with type='K'=0x4B, nr=0x15, size=8
        # direction = _IOC_WRITE = 1
        expected = (1 << 30) | (8 << 16) | (0x4B << 8) | 0x15
        assert AMDKFD_IOC_ACQUIRE_VM == expected

    def test_alloc_memory_ioctl(self):
        from amd_gpu_driver.ioctl.kfd import AMDKFD_IOC_ALLOC_MEMORY_OF_GPU
        # IOWR with type='K'=0x4B, nr=0x16, size=40
        expected = (3 << 30) | (40 << 16) | (0x4B << 8) | 0x16
        assert AMDKFD_IOC_ALLOC_MEMORY_OF_GPU == expected

    def test_create_queue_ioctl(self):
        from amd_gpu_driver.ioctl.kfd import AMDKFD_IOC_CREATE_QUEUE
        expected = (3 << 30) | (96 << 16) | (0x4B << 8) | 0x02
        assert AMDKFD_IOC_CREATE_QUEUE == expected


class TestKFDMemoryFlags:
    """Verify memory flag constants."""

    def test_vram_flag(self):
        from amd_gpu_driver.ioctl.kfd import KFD_IOC_ALLOC_MEM_FLAGS_VRAM
        assert KFD_IOC_ALLOC_MEM_FLAGS_VRAM == 1

    def test_gtt_flag(self):
        from amd_gpu_driver.ioctl.kfd import KFD_IOC_ALLOC_MEM_FLAGS_GTT
        assert KFD_IOC_ALLOC_MEM_FLAGS_GTT == 2

    def test_writable_flag(self):
        from amd_gpu_driver.ioctl.kfd import KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
        assert KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE == (1 << 31)

    def test_executable_flag(self):
        from amd_gpu_driver.ioctl.kfd import KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
        assert KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE == (1 << 30)

    def test_no_substitute_flag(self):
        from amd_gpu_driver.ioctl.kfd import KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
        assert KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE == (1 << 28)
