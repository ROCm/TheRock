"""Phase 11: smoke direct compute through the public macOS backend API.

This assumes the GPU is already brought up by the phase-9 flow. It uses
MacOSDevice.create_compute_queue() and submit_packets(), then verifies a PM4
WRITE_DATA packet by polling a VRAM-backed queue buffer from the CPU.
"""
from __future__ import annotations

import ctypes
import struct
import time

from amd_gpu_driver.backends.macos.device import MacOSDevice

PM4_NOP = struct.pack("<II", 0xC0001000, 0)
PACKET3_WRITE_DATA = 0x37
WRITE_DATA_DST_SEL_MEM_ASYNC = 5
WRITE_DATA_WR_CONFIRM = 1 << 20


def build_write_data(addr: int, values: list[int]) -> bytes:
    n_body = 3 + len(values)
    header = (3 << 30) | (((n_body - 1) & 0x3FFF) << 16) | (PACKET3_WRITE_DATA << 8)
    control = (WRITE_DATA_DST_SEL_MEM_ASYNC << 8) | WRITE_DATA_WR_CONFIRM
    dwords = [
        header,
        control,
        addr & 0xFFFFFFFF,
        (addr >> 32) & 0xFFFFFFFF,
        *values,
    ]
    return struct.pack(f"<{len(dwords)}I", *dwords)


def main() -> None:
    dev = MacOSDevice()
    dev.open(0)

    queue = dev.create_compute_queue()
    if queue.ring_buffer is None:
        raise RuntimeError("compute queue has no ring buffer")

    # Use a scratch slot inside the VRAM-backed ring. The direct queue uses a
    # 4 KiB ring and the packet stream starts at DW 0, so 0x800 is safely away
    # from this smoke's command packets.
    target_cpu = queue.ring_buffer.cpu_addr + 0x800
    target_gpu = queue.ring_buffer.gpu_addr + 0x800
    ctypes.c_uint64.from_address(target_cpu).value = 0

    dev.submit_packets(queue, PM4_NOP)
    time.sleep(0.05)

    values = [0xFACE0001, 0xFACE0002]
    packets = build_write_data(target_gpu, values)
    dev.submit_packets(queue, packets)

    expected = (values[1] << 32) | values[0]
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        written = ctypes.c_uint64.from_address(target_cpu).value
        wptr = ctypes.c_uint64.from_address(queue.write_ptr_addr).value
        if (written, wptr) != last:
            print(f"written=0x{written:016x} wptr=0x{wptr:x}")
            last = (written, wptr)
        if written == expected:
            print("MacOSDevice compute queue WRITE_DATA ✓")
            return
        time.sleep(0.05)

    raise TimeoutError("MacOSDevice compute queue WRITE_DATA did not complete")


if __name__ == "__main__":
    main()
