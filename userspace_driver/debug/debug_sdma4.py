"""Replicate exact test_copy_vram_to_vram flow."""
import ctypes
import time

from amd_gpu_driver import AMDDevice


def main():
    dev = AMDDevice()

    size = 4096
    src = dev.alloc(size, location="vram")
    dst = dev.alloc(size, location="vram")

    src.write(b"\xAA" * size)
    dst.fill(0x00)

    dev.copy(dst, src, size)

    # Instead of synchronize, let's poll manually
    tl = dev._timeline
    print(f"Timeline value: {tl.timeline_value}")
    print(f"Signal addr: 0x{tl.signal_addr:x}")
    print(f"Signal mem cpu_addr: 0x{tl._signal_mem.cpu_addr:x}")

    for i in range(20):
        gpu_val_u32 = ctypes.c_uint32.from_address(tl._signal_mem.cpu_addr).value
        gpu_val_u64 = ctypes.c_uint64.from_address(tl._signal_mem.cpu_addr).value
        print(f"  [{i*100}ms] u32={gpu_val_u32} u64={gpu_val_u64}")
        if gpu_val_u32 > 0 or gpu_val_u64 > 0:
            break
        time.sleep(0.1)

    # Check SDMA queue state
    q = dev._sdma_queue
    wp = ctypes.c_uint64.from_address(q.write_ptr_addr).value
    rp = ctypes.c_uint64.from_address(q.read_ptr_addr).value
    print(f"SDMA queue WP={wp} RP={rp}")

    # Read dst
    data = dst.read(16)
    print(f"Dst: {data.hex()}")

    src.free()
    dst.free()
    dev.close()


if __name__ == "__main__":
    main()
