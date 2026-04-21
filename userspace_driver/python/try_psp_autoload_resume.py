"""Resume PSP LOAD_IP_FW probe after smu_bring_up already completed.

The previous run left SMU at PWR_SOC and the PSP ring live. This
script picks up from there, reconstructs the PSP ring state, and
exercises every fw_type LOAD_IP_FW → prints which PSP accepts and
which it rejects (with the returned status code).
"""
from __future__ import annotations

import logging
import os
import sys

from amd_gpu_driver.backends.macos.gfx_psp_autoload import psp_load_gfx_and_autoload
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_cmd import alloc_cmd_ctx
from amd_gpu_driver.backends.macos.psp_ring import ring_create
from amd_gpu_driver.backends.macos.smu import (
    MP0_BASE_DW,
    PPSMC_MSG_GetSmuVersion,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")


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
    # Sanity: SOS must be alive (otherwise PSP ring reuse won't work).
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) == 0:
        print("SOS not alive. Run try_psp_autoload.py first (fresh card).")
        sys.exit(1)

    drv = _DriverShim(c)

    # Ensure SMU still responds.
    try:
        resp, ver = smu_send(c, PPSMC_MSG_GetSmuVersion, 0, timeout_ms=500)
        print(f"SMU alive: version=0x{ver:x} (resp=0x{resp:x})")
    except TimeoutError:
        print("SMU not responding; replug and run try_psp_autoload.py.")
        sys.exit(2)

    # Rebuild a fresh PSP ring + cmd ctx.
    print("\n== Recreating PSP KM ring ==")
    ring = ring_create(c, drv, MP0_BASE_DW, destroy_first=True, verbose=False)
    print(f"  ring_bus = 0x{ring.ring_bus:x}")

    print("\n== PSP LOAD_IP_FW probe ==")
    try:
        psp_load_gfx_and_autoload(c, drv, MP0_BASE_DW, ring, FIRMWARE_DIR)
    except RuntimeError as e:
        print(f"  stopped early: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
