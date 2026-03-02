"""Debug: does the first SDMA submission fail?"""
import ctypes
import time

from amd_gpu_driver import AMDDevice
from amd_gpu_driver.backends.base import MemoryLocation
from amd_gpu_driver.commands.sdma import SDMAPacketBuilder


def main():
    dev = AMDDevice()
    q = dev._backend.create_sdma_queue()

    sig = dev._backend.alloc_memory(4096, MemoryLocation.GTT, uncached=True)
    ctypes.memset(sig.cpu_addr, 0, 16)

    # Test 1: first submission = standalone fence
    sdma = SDMAPacketBuilder()
    sdma.fence(sig.gpu_addr, 1)
    dev._backend.submit_packets(q, sdma.build())
    time.sleep(0.1)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    print(f"Test 1 (first fence): {val}")

    dev._backend.destroy_queue(q)

    # Test 2: new queue, first submission = copy + fence
    q2 = dev._backend.create_sdma_queue()
    ctypes.memset(sig.cpu_addr, 0, 16)
    src = dev.alloc(256, location="vram")
    dst = dev.alloc(256, location="vram")
    src.write(b"\xDD" * 256)
    dst.fill(0x00)

    sdma2 = SDMAPacketBuilder()
    sdma2.copy_linear(dst.gpu_addr, src.gpu_addr, 256)
    sdma2.fence(sig.gpu_addr, 2)
    dev._backend.submit_packets(q2, sdma2.build())
    time.sleep(0.5)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    data = dst.read(16)
    print(f"Test 2 (first copy+fence): fence={val}, dst={data.hex()}")

    dev._backend.destroy_queue(q2)

    # Test 3: new queue, first submission = just fence, second = copy+fence
    q3 = dev._backend.create_sdma_queue()
    ctypes.memset(sig.cpu_addr, 0, 16)
    dst.fill(0x00)

    sdma3 = SDMAPacketBuilder()
    sdma3.fence(sig.gpu_addr, 10)
    dev._backend.submit_packets(q3, sdma3.build())
    time.sleep(0.1)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    print(f"Test 3a (first fence): {val}")

    ctypes.memset(sig.cpu_addr, 0, 16)
    sdma4 = SDMAPacketBuilder()
    sdma4.copy_linear(dst.gpu_addr, src.gpu_addr, 256)
    sdma4.fence(sig.gpu_addr, 20)
    dev._backend.submit_packets(q3, sdma4.build())
    time.sleep(0.5)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    data = dst.read(16)
    print(f"Test 3b (second copy+fence): fence={val}, dst={data.hex()}")

    dev._backend.destroy_queue(q3)

    # Test 4: new queue, first submission = copy + fence, using VRAM source already written
    q4 = dev._backend.create_sdma_queue()
    ctypes.memset(sig.cpu_addr, 0, 16)
    dst.fill(0x00)

    # Use a NOP before the copy
    sdma5 = SDMAPacketBuilder()
    sdma5.nop(1)
    sdma5.copy_linear(dst.gpu_addr, src.gpu_addr, 256)
    sdma5.fence(sig.gpu_addr, 30)
    dev._backend.submit_packets(q4, sdma5.build())
    time.sleep(0.5)
    val = ctypes.c_uint32.from_address(sig.cpu_addr).value
    data = dst.read(16)
    print(f"Test 4 (NOP+copy+fence first): fence={val}, dst={data.hex()}")

    dev._backend.destroy_queue(q4)

    src.free()
    dst.free()
    dev._backend.free_memory(sig)
    dev.close()


if __name__ == "__main__":
    main()
