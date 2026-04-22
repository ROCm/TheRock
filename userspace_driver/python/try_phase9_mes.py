"""Phase 9c (attempt 4): load MES via gc_12_0_1_uni_mes.bin.

Prior experiments: we tried PSP LOAD_IP_FW for RS64_MES (type 76) +
RS64_MES_STACK (77) using gc_12_0_1_mes.bin. PSP rejected both with
0xFFFF0006. Without MES running, MEC appears to park in a wait loop
and direct HQD register writes are dropped.

Root cause per amdgpu_mes.c::amdgpu_mes_init_microcode:
  - When `enable_uni_mes=1` (default on gfx12), driver loads
    `gc_12_0_1_uni_mes.bin` (NOT gc_12_0_1_mes.bin — that's the
    legacy split path for gfx11 and older).
  - The SAME uni_mes file is submitted to PSP with TWO different
    fw_type pairs:
      SCHED pipe: GFX_FW_TYPE_CP_MES (33)      + MES_STACK (34)
      KIQ   pipe: GFX_FW_TYPE_CP_MES_KIQ (81) + MES_KIQ_STACK (82)

This script:
  1. Runs our full bring-up.
  2. Submits LOAD_IP_FW for the four MES fw_types from uni_mes.bin.
  3. Reports which succeed.

If all 4 accept, we have MES loaded in the TMR. Next step would be
_config_mes (write CP_MES_CNTL, CP_MES_PRGRM_CNTR_START) and enable.
"""
from __future__ import annotations

import logging
import os
import struct
import sys
import time

from amd_gpu_driver.backends.macos.gfx_bringup import gfx_bring_up
from amd_gpu_driver.backends.macos.gfx_psp_autoload import _load_one
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_cmd import alloc_cmd_ctx
from amd_gpu_driver.backends.macos.smu import MP0_BASE_DW

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")

# Not in our psp_cmd.py yet:
GFX_FW_TYPE_CP_MES           = 33
GFX_FW_TYPE_MES_STACK        = 34
GFX_FW_TYPE_CP_MES_KIQ       = 81
GFX_FW_TYPE_MES_KIQ_STACK    = 82


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_mes_header(blob: bytes) -> dict:
    """mes_firmware_header_v1_0 (32 + 44 bytes):
      +32 mes_ucode_version
      +36 mes_ucode_size_bytes
      +40 mes_ucode_offset_bytes
      +44 mes_ucode_data_version
      +48 mes_ucode_data_size_bytes
      +52 mes_ucode_data_offset_bytes
      +56 mes_uc_start_addr_lo
      +60 mes_uc_start_addr_hi
      +64 mes_data_start_addr_lo
      +68 mes_data_start_addr_hi
    """
    (u_ver, u_sz, u_off, d_ver, d_sz, d_off,
     uc_lo, uc_hi, dd_lo, dd_hi) = struct.unpack_from("<IIIIIIIIII", blob, 32)
    return {
        "ucode_version": u_ver,
        "ucode_size": u_sz,
        "ucode_offset": u_off,
        "data_size": d_sz,
        "data_offset": d_off,
        "uc_start_addr": (uc_hi << 32) | uc_lo,
        "data_start_addr": (dd_hi << 32) | dd_lo,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    # Bring GFX up.
    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # Parse uni_mes.bin.
    uni_path = os.path.join(FIRMWARE_DIR, "gc_12_0_1_uni_mes.bin")
    if not os.path.exists(uni_path):
        print(f"Missing {uni_path}")
        sys.exit(1)
    uni_blob = open(uni_path, "rb").read()
    hdr = _parse_mes_header(uni_blob)
    print(f"\n== gc_12_0_1_uni_mes.bin ==")
    print(f"  ucode_size        = {hdr['ucode_size']}")
    print(f"  ucode_offset      = {hdr['ucode_offset']}")
    print(f"  data_size         = {hdr['data_size']}")
    print(f"  data_offset       = {hdr['data_offset']}")
    print(f"  uc_start_addr     = 0x{hdr['uc_start_addr']:x}")
    print(f"  data_start_addr   = 0x{hdr['data_start_addr']:x}")

    ucode = uni_blob[hdr["ucode_offset"]:hdr["ucode_offset"] + hdr["ucode_size"]]
    data  = uni_blob[hdr["data_offset"]:hdr["data_offset"] + hdr["data_size"]]
    print(f"\n  ucode bytes = {len(ucode)}  data bytes = {len(data)}")

    # We'll reuse the PSP ring created inside gfx_bring_up. alloc a new ctx
    # because we're done with the previous batch.
    ctx = alloc_cmd_ctx(drv)

    # Try LOAD_IP_FW for all 4 MES types.
    print(f"\n== LOAD_IP_FW (MES) ==")
    probes = [
        ("CP_MES",         GFX_FW_TYPE_CP_MES,        ucode),
        ("MES_STACK",      GFX_FW_TYPE_MES_STACK,     data),
        ("CP_MES_KIQ",     GFX_FW_TYPE_CP_MES_KIQ,    ucode),
        ("MES_KIQ_STACK",  GFX_FW_TYPE_MES_KIQ_STACK, data),
    ]
    for label, fw_type, payload in probes:
        status = _load_one(c, drv, MP0_BASE_DW, r.ring, ctx,
                           payload, fw_type, label, strict=False)
        mark = "OK" if status == 0 else f"status=0x{status:08x}"
        print(f"  {label:16s} type={fw_type:3d} size={len(payload):8d}  {mark}")


if __name__ == "__main__":
    main()
