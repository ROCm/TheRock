"""Try HQD writes via direct BAR5 CPU mapping (bypass DEXT ioctl).

All HQD canary writes via mmio_write32 (ioctl path) drop silently,
regardless of MEC/MES state. Hypothesis: the DEXT may filter writes
to the CP_HQD_* range. Or there's something fundamental about these
registers on gfx12.

`map_bar(5)` gives us a CPU-visible mapping of the MMIO BAR. If we
write through that pointer directly (bypassing ioctl), we can
distinguish:
  - DEXT filter: direct writes work, ioctl doesn't.
  - HW truly rejects: neither works.

If direct writes work, we switch to that path for HQD programming.
"""
from __future__ import annotations

import ctypes
import logging
import os
import sys

from amd_gpu_driver.backends.macos.gfx_bringup import gfx_bring_up
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
_MMIO_BAR = 5
GC_B1 = 0xA000

regGRBM_GFX_CNTL    = 0x0900
regCP_HQD_PQ_BASE   = 0x1fb1
regCP_HQD_ACTIVE    = 0x1fab
regCP_MEC_RS64_CNTL = 0x2904


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # Map BAR5 directly.
    try:
        bar5_cpu, bar5_size = c.map_bar(5)
        print(f"\nBAR5 mapped at CPU=0x{bar5_cpu:x} size=0x{bar5_size:x}")
    except Exception as e:
        print(f"map_bar(5) failed: {e}")
        sys.exit(1)

    def bar5_rd(dw_off):
        return (ctypes.c_uint32 * 1).from_address(bar5_cpu + dw_off * 4)[0]

    def bar5_wr(dw_off, val):
        (ctypes.c_uint32 * 1).from_address(bar5_cpu + dw_off * 4)[0] = val & 0xFFFFFFFF

    # Sanity: read CP_MEC_RS64_CNTL via both paths.
    a_ioctl = c.mmio_read32(_MMIO_BAR, (GC_B1 + regCP_MEC_RS64_CNTL) * 4)
    a_direct = bar5_rd(GC_B1 + regCP_MEC_RS64_CNTL)
    print(f"\nCP_MEC_RS64_CNTL: ioctl=0x{a_ioctl:08x}  direct=0x{a_direct:08x}")

    # Select me=1, pipe=0, queue=0 via ioctl (for GRBM_GFX_CNTL).
    c.mmio_write32(_MMIO_BAR, (GC_B1 + regGRBM_GFX_CNTL) * 4, (1 << 2))

    # Canary via IOCTL.
    c.mmio_write32(_MMIO_BAR, (GC_B1 + regCP_HQD_PQ_BASE) * 4, 0xdeadbeef)
    rb_ioctl = c.mmio_read32(_MMIO_BAR, (GC_B1 + regCP_HQD_PQ_BASE) * 4)
    c.mmio_write32(_MMIO_BAR, (GC_B1 + regCP_HQD_PQ_BASE) * 4, 0)
    print(f"\nCP_HQD_PQ_BASE via ioctl: wrote 0xdeadbeef, read 0x{rb_ioctl:08x}")

    # Canary via direct BAR5 mapping.
    bar5_wr(GC_B1 + regCP_HQD_PQ_BASE, 0xCAFEBABE)
    rb_direct = bar5_rd(GC_B1 + regCP_HQD_PQ_BASE)
    bar5_wr(GC_B1 + regCP_HQD_PQ_BASE, 0)
    print(f"CP_HQD_PQ_BASE via direct: wrote 0xcafebabe, read 0x{rb_direct:08x}")

    # Also try writing a known-working register via direct and verify.
    bar5_wr(GC_B1 + regGRBM_GFX_CNTL, 0x8)  # me=2, pipe=0
    rb_grbm = bar5_rd(GC_B1 + regGRBM_GFX_CNTL)
    bar5_wr(GC_B1 + regGRBM_GFX_CNTL, 0)
    print(f"GRBM_GFX_CNTL via direct: wrote 0x8, read 0x{rb_grbm:08x}")

    c.mmio_write32(_MMIO_BAR, (GC_B1 + regGRBM_GFX_CNTL) * 4, 0)


if __name__ == "__main__":
    main()
