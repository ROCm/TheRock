"""Debug: verify SDMA doorbell is working."""
import ctypes
import struct
import time

from amd_gpu_driver import AMDDevice
from amd_gpu_driver.backends.base import MemoryLocation
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder


def main():
    dev = AMDDevice()

    # Verify VRAM allocation and CPU access work
    buf = dev.alloc(4096, location="vram")
    print(f"VRAM buf: gpu=0x{buf.gpu_addr:x} cpu=0x{buf._handle.cpu_addr:x}")

    # Write pattern
    buf.fill(0x42)
    data = buf.read(4)
    print(f"After fill(0x42): {data.hex()}")

    buf.fill(0x00)
    data = buf.read(4)
    print(f"After fill(0x00): {data.hex()}")
    buf.free()

    # Now test the SDMA queue doorbell
    q = dev._backend.create_sdma_queue()
    print(f"\nSDMA queue doorbell_offset=0x{q.doorbell_offset:x}")
    print(f"SDMA queue doorbell_addr=0x{q.doorbell_addr:x}")

    # Try reading the doorbell
    try:
        db_val = ctypes.c_uint64.from_address(q.doorbell_addr).value
        print(f"Doorbell read: 0x{db_val:x}")
    except Exception as e:
        print(f"Doorbell read failed: {e}")

    # Try writing to doorbell
    try:
        ctypes.c_uint64.from_address(q.doorbell_addr).value = 0
        print("Doorbell write(0): OK")
        db_val = ctypes.c_uint64.from_address(q.doorbell_addr).value
        print(f"Doorbell read after write: 0x{db_val:x}")
    except Exception as e:
        print(f"Doorbell write failed: {e}")

    # Check the queue manager's doorbell page
    qm = dev._backend._queues
    print(f"\nDoorbell page addr: 0x{qm._doorbell_page_addr:x}")
    print(f"Doorbell page size: {qm._doorbell_page_size}")

    # Check SDMA doorbell: for SDMA queues, the doorbell might be 4 bytes, not 8
    # Let's try writing as u32
    sig = dev._backend.alloc_memory(4096, MemoryLocation.GTT, uncached=True)
    ctypes.memset(sig.cpu_addr, 0, 16)

    sdma = SDMAPacketBuilder()
    sdma.fence(sig.gpu_addr, 77)
    packets = sdma.build()

    # Manual submit: write to ring, update WP, ring doorbell as u32
    ring_base = q.ring_buffer.cpu_addr
    ctypes.memmove(ring_base, packets, len(packets))

    # Write pointer as bytes (SDMA)
    wp = len(packets)
    ctypes.c_uint64.from_address(q.write_ptr_addr).value = wp

    # Try doorbell as u32 instead of u64
    print(f"\nManual submit: WP={wp}")
    ctypes.c_uint32.from_address(q.doorbell_addr).value = wp
    time.sleep(0.2)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    print(f"After u32 doorbell: fence={val}")

    if val == 0:
        # Try u64 doorbell
        ctypes.c_uint64.from_address(q.doorbell_addr).value = wp
        time.sleep(0.2)
        val = ctypes.c_uint32.from_address(sig.cpu_addr).value
        print(f"After u64 doorbell: fence={val}")

    dev._backend.destroy_queue(q)
    dev._backend.free_memory(sig)
    dev.close()


if __name__ == "__main__":
    main()
