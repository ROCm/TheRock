"""Debug SDMA fence with timeline."""
import ctypes
import struct
import time

from amd_gpu_driver import AMDDevice
from amd_gpu_driver.backends.base import MemoryLocation
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder
from amd_gpu_driver.sync.timeline import TimelineSemaphore


def main():
    dev = AMDDevice()
    src = dev.alloc(4096, location="vram")
    dst = dev.alloc(4096, location="vram")
    src.write(b"\xAA" * 4096)
    dst.fill(0x00)

    if dev._sdma_queue is None:
        dev._sdma_queue = dev._backend.create_sdma_queue()

    if dev._timeline is None:
        dev._timeline = TimelineSemaphore(dev._backend)
    tl = dev._timeline

    print(f"signal_addr = 0x{tl.signal_addr:x}")
    print(f"signal_mem.gpu_addr = 0x{tl._signal_mem.gpu_addr:x}")
    print(f"signal_mem.cpu_addr = 0x{tl._signal_mem.cpu_addr:x}")
    print(f"gpu_addr == cpu_addr: {tl._signal_mem.gpu_addr == tl._signal_mem.cpu_addr}")

    # signal_addr comes from gpu_addr
    # The SDMA fence writes to the GPU address space
    # But we read from the CPU address
    # If gpu_addr == cpu_addr, they're the same mapping (GTT)

    sdma = SDMAPacketBuilder()
    sdma.copy_linear(dst.gpu_addr, src.gpu_addr, 4096)
    fence_value = tl.next_value()
    sdma.fence(tl.signal_addr, fence_value)
    packets = sdma.build()
    dwords = struct.unpack(f"<{len(packets) // 4}I", packets)
    print(f"\nPacket ({len(dwords)} dwords):")
    for i, d in enumerate(dwords):
        print(f"  [{i}] 0x{d:08x}")

    # Fence address from packet
    fence_lo = dwords[8]
    fence_hi = dwords[9]
    fence_addr = fence_lo | (fence_hi << 32)
    print(f"\nFence addr in packet: 0x{fence_addr:x}")
    print(f"Signal addr: 0x{tl.signal_addr:x}")
    print(f"Match: {fence_addr == tl.signal_addr}")

    # Submit
    dev._backend.submit_packets(dev._sdma_queue, packets)
    time.sleep(0.5)

    # Check values
    gpu_val = ctypes.c_uint64.from_address(tl._signal_mem.cpu_addr).value
    print(f"\nGPU value (u64) at cpu_addr: {gpu_val}")
    gpu_val32 = ctypes.c_uint32.from_address(tl._signal_mem.cpu_addr).value
    print(f"GPU value (u32) at cpu_addr: {gpu_val32}")

    data = dst.read(16)
    print(f"Dst data: {data.hex()}")

    # Direct test: alloc a separate GTT buffer and fence to it
    print("\n--- Direct fence test with separate buffer ---")
    sig2 = dev._backend.alloc_memory(4096, MemoryLocation.GTT, uncached=True)
    ctypes.memset(sig2.cpu_addr, 0, 16)
    print(f"sig2.gpu_addr = 0x{sig2.gpu_addr:x}")
    print(f"sig2.cpu_addr = 0x{sig2.cpu_addr:x}")

    sdma2 = SDMAPacketBuilder()
    sdma2.copy_linear(dst.gpu_addr, src.gpu_addr, 4096)
    sdma2.fence(sig2.gpu_addr, 42)
    dev._backend.submit_packets(dev._sdma_queue, sdma2.build())
    time.sleep(0.5)
    val2 = ctypes.c_uint32.from_address(sig2.cpu_addr).value
    print(f"Fence value: {val2}")

    src.free()
    dst.free()
    dev._backend.free_memory(sig2)
    dev.close()


if __name__ == "__main__":
    main()
