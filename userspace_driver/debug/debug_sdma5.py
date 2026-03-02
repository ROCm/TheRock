"""Debug: compare working vs non-working SDMA submission."""
import ctypes
import struct
import time

from amd_gpu_driver import AMDDevice
from amd_gpu_driver.backends.base import MemoryLocation
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder


def main():
    dev = AMDDevice()

    # Allocate VRAM via AMDDevice.alloc (same as test)
    src = dev.alloc(4096, location="vram")
    dst = dev.alloc(4096, location="vram")
    src.write(b"\xAA" * 4096)
    dst.fill(0x00)

    # Create SDMA queue directly (as tests do via dev.copy)
    q = dev._backend.create_sdma_queue()

    # Allocate signal via same path as TimelineSemaphore
    sig = dev._backend.alloc_memory(4096, MemoryLocation.GTT, uncached=True)
    ctypes.memset(sig.cpu_addr, 0, 16)

    print(f"src.gpu_addr = 0x{src.gpu_addr:x}")
    print(f"dst.gpu_addr = 0x{dst.gpu_addr:x}")
    print(f"sig.gpu_addr = 0x{sig.gpu_addr:x}")

    # Build copy + fence
    sdma = SDMAPacketBuilder()
    sdma.copy_linear(dst.gpu_addr, src.gpu_addr, 4096)
    sdma.fence(sig.gpu_addr, 1)
    packets = sdma.build()

    dwords = struct.unpack(f"<{len(packets) // 4}I", packets)
    print(f"\nPacket ({len(dwords)} dwords):")
    for i, d in enumerate(dwords):
        print(f"  [{i}] 0x{d:08x}")

    # Submit
    wp_before = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    dev._backend.submit_packets(q, packets)
    wp_after = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    print(f"\nWP: {wp_before} -> {wp_after}")

    # Read ring buffer
    ring_base = q.ring_buffer.cpu_addr
    ring_offset = int(wp_before) & (q.ring_size - 1)
    ring_data = bytes((ctypes.c_uint8 * len(packets)).from_address(ring_base + ring_offset))
    ring_dwords = struct.unpack(f"<{len(ring_data) // 4}I", ring_data)
    print(f"Ring at offset {ring_offset}:")
    for i, d in enumerate(ring_dwords):
        print(f"  [{i}] 0x{d:08x}")

    # Check doorbell
    print(f"\nDoorbell addr: 0x{q.doorbell_addr:x}")
    db_val = ctypes.c_uint64.from_address(q.doorbell_addr).value
    print(f"Doorbell value: {db_val}")

    # Poll
    for i in range(10):
        time.sleep(0.1)
        val = ctypes.c_uint32.from_address(sig.cpu_addr).value
        rp = ctypes.c_uint64.from_address(q.read_ptr_addr).value
        print(f"  [{(i+1)*100}ms] fence={val} RP={rp}")
        if val > 0:
            break

    data = dst.read(16)
    print(f"\nDst: {data.hex()}")

    # Now try: submit another fence to the same queue
    ctypes.memset(sig.cpu_addr, 0, 16)
    sdma2 = SDMAPacketBuilder()
    sdma2.fence(sig.gpu_addr, 99)
    dev._backend.submit_packets(q, sdma2.build())
    time.sleep(0.2)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    print(f"\nSecond fence-only: {val}")

    dev._backend.destroy_queue(q)
    src.free()
    dst.free()
    dev._backend.free_memory(sig)
    dev.close()


if __name__ == "__main__":
    main()
