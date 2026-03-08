#!/usr/bin/env python3
"""Test the amdgpu_lite Python backend against a real device.

Run on a machine with amdgpu_lite.ko loaded and /dev/amdgpu_lite0 present.
"""

import sys
import os
import ctypes

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(__file__))

from amd_gpu_driver.backends.amdgpu_lite import AmdgpuLiteDevice
from amd_gpu_driver.backends.base import MemoryLocation


def test_open_and_info():
    """Test device open and GET_INFO."""
    dev = AmdgpuLiteDevice()
    dev.open(0)

    info = dev.info
    assert info is not None
    print(f"Device: {dev.name}")
    print(f"  Vendor: {info.vendor_id:#06x}  Device: {info.device_id:#06x}")
    print(f"  VRAM: {dev.vram_size / (1024*1024):.0f} MB")
    print(f"  GFX target version: {dev.gfx_target_version}")
    print(f"  Num BARs: {info.num_bars}")
    for i in range(info.num_bars):
        bar = info.bars[i]
        if bar.size > 0:
            label = ""
            if i == info.mmio_bar_index:
                label = " [MMIO]"
            elif i == info.vram_bar_index:
                label = " [VRAM]"
            elif i == info.doorbell_bar_index:
                label = " [DOORBELL]"
            print(f"    BAR{bar.bar_index}: phys={bar.phys_addr:#018x} "
                  f"size={bar.size:#010x}{label}")
    print(f"  GART: bus_addr={info.gart_table_bus_addr:#018x} "
          f"size={info.gart_table_size} "
          f"va_start={info.gart_gpu_va_start:#018x}")
    print(f"  MMIO mapped at: {dev.mmio_addr:#018x}")
    print(f"  Doorbell mapped at: {dev.doorbell_addr:#018x}")

    dev.close()
    print("PASS: open_and_info\n")


def test_mmio_register_read():
    """Test reading a register via the MMIO BAR mapping."""
    dev = AmdgpuLiteDevice()
    dev.open(0)

    mmio = dev.mmio_addr
    assert mmio != 0, "MMIO BAR not mapped"

    # Read RCC_CONFIG_MEMSIZE at offset 0xDE3 * 4 = 0x378C
    val = ctypes.c_uint32.from_address(mmio + 0x378C).value
    print(f"Register RCC_CONFIG_MEMSIZE (0x378C): {val:#010x} = {val / (1024*1024):.0f} MB")
    assert val > 0, "Expected non-zero VRAM size register"

    dev.close()
    print("PASS: mmio_register_read\n")


def test_gtt_alloc_readback():
    """Test GTT memory allocation and CPU readback."""
    dev = AmdgpuLiteDevice()
    dev.open(0)

    size = 4096
    mem = dev.alloc_memory(size, MemoryLocation.GTT)
    print(f"GTT alloc: handle={mem.kfd_handle} cpu_addr={mem.cpu_addr:#018x} "
          f"gpu_addr={mem.gpu_addr:#018x}")

    assert mem.cpu_addr != 0, "GTT CPU address is 0"
    assert mem.gpu_addr != 0, "GTT GPU address is 0 (MAP_GPU failed?)"

    # Write a pattern and read it back
    pattern = 0xDEADBEEF
    ctypes.c_uint32.from_address(mem.cpu_addr).value = pattern
    readback = ctypes.c_uint32.from_address(mem.cpu_addr).value
    assert readback == pattern, f"GTT readback mismatch: {readback:#x} != {pattern:#x}"
    print(f"  Write {pattern:#010x}, read back {readback:#010x} - OK")

    dev.free_memory(mem)
    dev.close()
    print("PASS: gtt_alloc_readback\n")


def test_vram_alloc_readback():
    """Test VRAM allocation and CPU readback via BAR aperture."""
    dev = AmdgpuLiteDevice()
    dev.open(0)

    size = 4096
    mem = dev.alloc_memory(size, MemoryLocation.VRAM)
    print(f"VRAM alloc: handle={mem.kfd_handle} cpu_addr={mem.cpu_addr:#018x} "
          f"gpu_addr={mem.gpu_addr:#018x}")

    if mem.cpu_addr == 0:
        print("  VRAM CPU mapping not available (BAR too small?) - SKIP")
    else:
        pattern = 0xCAFEBABE
        ctypes.c_uint32.from_address(mem.cpu_addr).value = pattern
        readback = ctypes.c_uint32.from_address(mem.cpu_addr).value
        assert readback == pattern, f"VRAM readback mismatch: {readback:#x} != {pattern:#x}"
        print(f"  Write {pattern:#010x}, read back {readback:#010x} - OK")

    dev.free_memory(mem)
    dev.close()
    print("PASS: vram_alloc_readback\n")


def test_multiple_allocs():
    """Test multiple GTT allocations and verify GPU VA auto-assignment."""
    dev = AmdgpuLiteDevice()
    dev.open(0)

    allocs = []
    for i in range(4):
        mem = dev.alloc_memory(4096, MemoryLocation.GTT)
        print(f"  Alloc {i}: gpu_va={mem.gpu_addr:#018x}")
        allocs.append(mem)

    # Verify GPU VAs are distinct and ascending
    gpu_vas = [m.gpu_addr for m in allocs]
    assert len(set(gpu_vas)) == 4, "GPU VAs not unique"
    assert gpu_vas == sorted(gpu_vas), "GPU VAs not ascending"
    print("  GPU VAs are unique and ascending - OK")

    for mem in allocs:
        dev.free_memory(mem)

    dev.close()
    print("PASS: multiple_allocs\n")


def test_queue_creation():
    """Test compute queue creation (ring buffer + doorbell)."""
    dev = AmdgpuLiteDevice()
    dev.open(0)

    queue = dev.create_compute_queue()
    print(f"Compute queue created:")
    print(f"  queue_id={queue.queue_id}")
    print(f"  ring_size={queue.ring_size}")
    print(f"  doorbell_addr={queue.doorbell_addr:#018x}")
    print(f"  write_ptr_addr={queue.write_ptr_addr:#018x}")

    assert queue.ring_buffer is not None
    assert queue.ring_buffer.cpu_addr != 0
    assert queue.ring_buffer.gpu_addr != 0
    print(f"  ring gpu_va={queue.ring_buffer.gpu_addr:#018x}")

    dev.destroy_queue(queue)
    dev.close()
    print("PASS: queue_creation\n")


def main():
    print("=" * 60)
    print("amdgpu_lite Python backend tests")
    print("=" * 60)
    print()

    tests = [
        test_open_and_info,
        test_mmio_register_read,
        test_gtt_alloc_readback,
        test_vram_alloc_readback,
        test_multiple_allocs,
        test_queue_creation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
