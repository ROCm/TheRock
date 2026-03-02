"""Debug SDMA copy + fence issue."""
import ctypes
import struct
import time

from amd_gpu_driver import AMDDevice
from amd_gpu_driver.backends.base import MemoryLocation
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder


def main():
    dev = AMDDevice()
    if dev._sdma_queue is None:
        dev._sdma_queue = dev._backend.create_sdma_queue()

    q = dev._sdma_queue
    print(f"Queue ring_size={q.ring_size}")

    sig = dev._backend.alloc_memory(4096, MemoryLocation.GTT, uncached=True)
    ctypes.memset(sig.cpu_addr, 0, 16)

    # Test 1: standalone fence
    sdma = SDMAPacketBuilder()
    sdma.fence(sig.gpu_addr, 1)
    p1 = sdma.build()
    print(f"\nTest 1: fence only ({len(p1)} bytes)")

    wp_before = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    dev._backend.submit_packets(q, p1)
    wp_after = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    print(f"  WP: {wp_before} -> {wp_after} (delta={wp_after - wp_before})")
    time.sleep(0.1)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    print(f"  Fence value: {val}")

    # Test 2: copy + fence
    ctypes.memset(sig.cpu_addr, 0, 16)
    src = dev.alloc(256, location="vram")
    dst = dev.alloc(256, location="vram")
    src.write(b"\xDD" * 256)
    dst.fill(0x00)

    sdma2 = SDMAPacketBuilder()
    sdma2.copy_linear(dst.gpu_addr, src.gpu_addr, 256)
    sdma2.fence(sig.gpu_addr, 2)
    p2 = sdma2.build()
    dwords = struct.unpack(f"<{len(p2) // 4}I", p2)
    print(f"\nTest 2: copy+fence ({len(p2)} bytes, {len(dwords)} dwords)")
    print(f"  Dwords: {[hex(d) for d in dwords]}")

    wp_before = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    dev._backend.submit_packets(q, p2)
    wp_after = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    print(f"  WP: {wp_before} -> {wp_after} (delta={wp_after - wp_before})")

    # Read ring buffer contents
    ring_base = q.ring_buffer.cpu_addr
    ring_offset = int(wp_before) & (q.ring_size - 1)
    ring_data = (ctypes.c_uint8 * len(p2)).from_address(ring_base + ring_offset)
    ring_bytes = bytes(ring_data)
    ring_dwords = struct.unpack(f"<{len(ring_bytes) // 4}I", ring_bytes)
    print(f"  Ring at offset {ring_offset}: {[hex(d) for d in ring_dwords]}")

    time.sleep(0.5)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    data = dst.read(16)
    print(f"  Fence value: {val}")
    print(f"  Dst: {data.hex()}")

    # Test 3: two separate submissions
    ctypes.memset(sig.cpu_addr, 0, 16)
    dst.fill(0x00)

    sdma3 = SDMAPacketBuilder()
    sdma3.copy_linear(dst.gpu_addr, src.gpu_addr, 256)
    dev._backend.submit_packets(q, sdma3.build())

    sdma4 = SDMAPacketBuilder()
    sdma4.fence(sig.gpu_addr, 3)
    dev._backend.submit_packets(q, sdma4.build())

    time.sleep(0.5)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    data = dst.read(16)
    print(f"\nTest 3: separate submissions")
    print(f"  Fence value: {val}")
    print(f"  Dst: {data.hex()}")

    src.free()
    dst.free()
    dev._backend.free_memory(sig)
    dev.close()


if __name__ == "__main__":
    main()
