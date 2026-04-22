"""Phase 9c (attempt 5): MES enable + HQD canary.

Prior test: PSP accepts MES firmware from gc_12_0_1_uni_mes.bin under
types CP_MES (33), MES_STACK (34), CP_MES_KIQ (81), MES_KIQ_STACK (82).
So we now have both MES pipes' firmware in PSP's TMR.

This script:
  1. gfx_bring_up() — reaches BOOTLOAD_COMPLETE with MEC firmware in TMR.
  2. Loads MES firmware (all 4 types) via PSP LOAD_IP_FW.
  3. For each MES pipe (0=SCHED, 1=KIQ):
       _grbm_select(me=3, pipe=P, queue=0, vmid=0)
       CP_MES_PRGRM_CNTR_START    = (uc_start >> 2) & 0xFFFFFFFF
       CP_MES_PRGRM_CNTR_START_HI = (uc_start >> 34) & 0xFFFFFFFF
  4. _grbm_select(0,0,0,0) deselect.
  5. CP_MES_CNTL = PIPE0_ACTIVE | PIPE1_ACTIVE (halt=0, resets=0).
  6. Poll CP_MES_CNTL + CP_MES instr regs for MES liveness.
  7. Retry HQD canary after MES is running.

If HQD writes stick after MES enable, the missing link was MES.
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
_MMIO_BAR = 5
GC_B1 = 0xA000

GFX_FW_TYPE_CP_MES           = 33
GFX_FW_TYPE_MES_STACK        = 34
GFX_FW_TYPE_CP_MES_KIQ       = 81
GFX_FW_TYPE_MES_KIQ_STACK    = 82

# BASE_IDX=1 regs (GC_B1=0xA000):
regGRBM_GFX_CNTL                 = 0x0900
regCP_MES_PRGRM_CNTR_START       = 0x2800
regCP_MES_PRGRM_CNTR_START_HI    = 0x289d
regCP_MES_CNTL                   = 0x2807
# CP_MES_CNTL bits: PIPE0_RESET=16 .. PIPE3_RESET=19, PIPE0_ACTIVE=26 ..
#   PIPE3_ACTIVE=29, HALT=30, INVALIDATE_ICACHE=4.
regCP_HQD_ACTIVE                 = 0x1fab
regCP_HQD_PQ_BASE                = 0x1fb1


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _parse_mes(blob: bytes):
    (u_ver, u_sz, u_off, d_ver, d_sz, d_off,
     uc_lo, uc_hi, dd_lo, dd_hi) = struct.unpack_from("<IIIIIIIIII", blob, 32)
    return {
        "ucode_size": u_sz, "ucode_offset": u_off,
        "data_size": d_sz, "data_offset": d_off,
        "uc_start_addr": (uc_hi << 32) | uc_lo,
        "data_start_addr": (dd_hi << 32) | dd_lo,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    def gc1_rd(o): return c.mmio_read32(_MMIO_BAR, (GC_B1 + o) * 4)
    def gc1_wr(o, v): c.mmio_write32(_MMIO_BAR, (GC_B1 + o) * 4, v & 0xFFFFFFFF)

    # Bring GFX up.
    r = gfx_bring_up(c, drv, firmware_dir=FIRMWARE_DIR)
    if not (r.bootload_status & 0x80000000):
        print("BOOTLOAD_COMPLETE not set — aborting.")
        sys.exit(1)

    # Parse + load MES firmware via PSP.
    uni_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_uni_mes.bin"), "rb").read()
    h = _parse_mes(uni_blob)
    print(f"\n== MES uni_mes.bin ==")
    print(f"  uc_start_addr    = 0x{h['uc_start_addr']:x}")
    print(f"  data_start_addr  = 0x{h['data_start_addr']:x}")
    print(f"  ucode_size       = {h['ucode_size']}")
    print(f"  data_size        = {h['data_size']}")

    ucode = uni_blob[h["ucode_offset"]:h["ucode_offset"] + h["ucode_size"]]
    data  = uni_blob[h["data_offset"]:h["data_offset"] + h["data_size"]]

    ctx = alloc_cmd_ctx(drv)
    print(f"\n== LOAD_IP_FW (MES) ==")
    for label, fw_type, payload in [
        ("CP_MES",         GFX_FW_TYPE_CP_MES,        ucode),
        ("MES_STACK",      GFX_FW_TYPE_MES_STACK,     data),
        ("CP_MES_KIQ",     GFX_FW_TYPE_CP_MES_KIQ,    ucode),
        ("MES_KIQ_STACK",  GFX_FW_TYPE_MES_KIQ_STACK, data),
    ]:
        st = _load_one(c, drv, MP0_BASE_DW, r.ring, ctx, payload, fw_type, label, strict=True)
        print(f"  {label:16s} type={fw_type:3d}  status=0x{st:08x}")

    uc_addr_shifted = h["uc_start_addr"] >> 2
    lo = uc_addr_shifted & 0xFFFFFFFF
    hi = (uc_addr_shifted >> 32) & 0xFFFFFFFF

    # Step 1 (mes_v12_0_enable false): clean-disable first to known state.
    print(f"\n== MES disable (clean state) ==")
    pre_cntl = gc1_rd(regCP_MES_CNTL)
    # halt=1, invalidate_icache=1, pipe0/1_reset=1, pipe0/1_active=0
    disable_cntl = (pre_cntl & ~(0x0C000000)) | 0x40000000 | 0x00000010 | 0x00030000
    gc1_wr(regCP_MES_CNTL, disable_cntl)
    time.sleep(0.01)
    print(f"  CP_MES_CNTL = 0x{gc1_rd(regCP_MES_CNTL):08x} (wrote 0x{disable_cntl:08x})")

    # Step 2 (mes_v12_0_enable true): per-pipe setup.
    # Linux order per pipe:
    #   a. grbm_select(me=3, pipe=P)
    #   b. RMW CP_MES_CNTL: set PIPE<P>_RESET=1 (pipe-specific)
    #   c. Write PRGRM_CNTR_START_LO/HI
    #   d. Write CP_MES_CNTL from 0, with PIPE<P>_ACTIVE=1
    #      (pipe 1 iter: also PIPE0_ACTIVE=1 so both pipes are active)
    print(f"\n== MES enable per pipe (Linux pattern) ==")
    for pipe in range(2):
        grbm = (3 << 2) | pipe
        gc1_wr(regGRBM_GFX_CNTL, grbm)

        cntl = gc1_rd(regCP_MES_CNTL)
        reset_bit = 1 << (16 + pipe)
        gc1_wr(regCP_MES_CNTL, cntl | reset_bit)

        gc1_wr(regCP_MES_PRGRM_CNTR_START,    lo)
        gc1_wr(regCP_MES_PRGRM_CNTR_START_HI, hi)

        # Build CP_MES_CNTL from 0 per Linux. For pipe 0: PIPE0_ACTIVE only.
        # For pipe 1: both PIPE0 and PIPE1 active.
        if pipe == 0:
            new_cntl = 0x04000000       # PIPE0_ACTIVE bit 26
        else:
            new_cntl = 0x0C000000       # PIPE0+PIPE1 active
        gc1_wr(regCP_MES_CNTL, new_cntl)
        time.sleep(0.001)

        rb = gc1_rd(regCP_MES_CNTL)
        print(f"  pipe {pipe}: PRGRM_CNTR_START lo=0x{gc1_rd(regCP_MES_PRGRM_CNTR_START):08x} "
              f"hi=0x{gc1_rd(regCP_MES_PRGRM_CNTR_START_HI):08x}  "
              f"CP_MES_CNTL=0x{rb:08x}")

    gc1_wr(regGRBM_GFX_CNTL, 0)
    time.sleep(0.5)

    post_cntl = gc1_rd(regCP_MES_CNTL)
    print(f"\n  final CP_MES_CNTL = 0x{post_cntl:08x}")

    # Now try HQD canary on MEC.
    print(f"\n== HQD canary post-MES-enable ==")
    for me in [1]:
        for pipe in [0]:
            for queue in [0]:
                grbm = (queue << 8) | (me << 2) | pipe
                gc1_wr(regGRBM_GFX_CNTL, grbm)
                gc1_wr(regCP_HQD_PQ_BASE, 0xdeadbeef)
                rb = gc1_rd(regCP_HQD_PQ_BASE)
                stuck = (rb == 0xdeadbeef)
                print(f"  me={me} pipe={pipe} queue={queue}: read 0x{rb:08x} "
                      f"{'STUCK ✓' if stuck else 'dropped'}")
                gc1_wr(regCP_HQD_PQ_BASE, 0)
    gc1_wr(regGRBM_GFX_CNTL, 0)


if __name__ == "__main__":
    main()
