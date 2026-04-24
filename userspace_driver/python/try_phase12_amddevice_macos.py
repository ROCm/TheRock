"""Phase 12: smoke the high-level AMDDevice facade on macOS.

This verifies that AMDDevice(backend="macos") can allocate a VRAM Buffer,
create the direct compute queue, submit PM4 through the public backend, and
address the Buffer by its MemoryHandle.gpu_addr.
"""
from __future__ import annotations

import ctypes
import struct
import time

from amd_gpu_driver.device import AMDDevice

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
    dev = AMDDevice(backend="macos")
    print(f"device={dev.name} gfx={dev.gfx_target}")

    buf = dev.alloc(4096, location="vram")
    buf.fill(0)
    print(f"buffer gpu=0x{buf.gpu_addr:x} cpu=0x{buf.cpu_addr:x}")

    queue = dev.backend.create_compute_queue()
    values = [0xABCD0001, 0xABCD0002]
    dev.backend.submit_packets(queue, build_write_data(buf.gpu_addr, values))

    expected = (values[1] << 32) | values[0]
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        written = ctypes.c_uint64.from_address(buf.cpu_addr).value
        if written != last:
            print(f"written=0x{written:016x}")
            last = written
        if written == expected:
            print("AMDDevice macOS Buffer WRITE_DATA ✓")
            return
        time.sleep(0.05)

    raise TimeoutError("AMDDevice macOS Buffer WRITE_DATA did not complete")


if __name__ == "__main__":
    main()
