"""Match tinygrad's gfx12 bringup order exactly.

Key findings from reading tinygrad (ip.py) and amdgpu (gfx_v12_0.c)
on 2026-04-21:

  1. `EnableAllSmuFeatures(0)` — ALL features, not PWR_SOC(=3).
     tinygrad does this and it works on gfx11/gfx12. Our previous
     assumption that arg=0 hangs may have been from a bare GPU
     without the full autoload path in place.

  2. Register `GFX_IMU_RLC_BOOTLOADER_ADDR_*` reading 0xFFFFFFFF
     means GFX clocks are gated. `EnableAllSmuFeatures(0)` ungates
     them; `PWR_SOC(=3)` does not.

  3. For MP0 14.0.3 (ours), `boot_time_tmr=True` and
     `autoload_tmr=True`, so PSP uses its boot-time TMR and no
     SETUP_TMR command is sent. LOAD_TOC is still sent.

  4. tinygrad order is:
       LOAD_TOC -> LOAD_IP_FW(SMU) -> LOAD_IP_FW(everything incl
       IMU_I/D) -> AUTOLOAD_RLC -> SetDriverDramAddr{Hi,Lo} ->
       EnableAllSmuFeatures(0) -> wait_for_rlc_complete

Our sequence to test:
  1. `load_sos` + `ring_create` (via smu_bring_up's first steps).
  2. `LOAD_TOC` (PSP TOC from SOS).
  3. `LOAD_IP_FW(SMU)` (also inside smu_bring_up).
  4. `LOAD_IP_FW(IMU_I, IMU_D)`.
  5. Fill autoload buffer in VRAM (RLC, SDMA, RS64 CP, MES,
     patched TOC).
  6. `AUTOLOAD_RLC`.
  7. `SetDriverDramAddr{Hi,Lo}` (already done inside smu_bring_up).
  8. `EnableAllSmuFeatures(0)`    -- tinygrad's arg value.
  9. Wait 10 s for BOOTLOAD_COMPLETE + IMU_GFX_RESET_CTRL = 0x7F.

If BOOTLOADER_ADDR registers start returning non-FF after step 8,
GFX clocks are finally ungated — and we've found the missing
piece.
"""
from __future__ import annotations

import logging
import os
import sys
import time

from amd_gpu_driver.backends.macos.gfx_autoload import (
    build_autoload_buffer,
    plan_autoload,
)
from amd_gpu_driver.backends.macos.gfx_psp_autoload import (
    _extract_imu,
    _load_one,
    submit_autoload_rlc,
    submit_load_toc,
)
from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_bootloader import parse_psp_firmware
from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_FW_TYPE_IMU_D,
    GFX_FW_TYPE_IMU_I,
    alloc_cmd_ctx,
)
from amd_gpu_driver.backends.macos.smu import (
    MP0_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
GC_B1 = 0xA000

# Register offsets (BASE_IDX=1 — all via GC_B1):
regGFX_IMU_CORE_CTRL              = 0x40b6
regGFX_IMU_GFX_RESET_CTRL         = 0x40bc
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5f81
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5f82
regGFX_IMU_RLC_BOOTLOADER_SIZE    = 0x5f83
regRLC_RLCS_BOOTLOAD_STATUS       = 0x4e7c
regRLC_CNTL                       = 0x4b00


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _try(c, msg, arg, name, *, timeout=5000):
    try:
        r, a = smu_send(c, msg, arg, timeout_ms=timeout)
        print(f"  {name:52s} resp=0x{r:x} arg_out=0x{a:x}")
        return r, a
    except TimeoutError:
        print(f"  {name:52s} TIMEOUT")
        return None, None


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)
    drv = _DriverShim(c)

    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    def snapshot(label):
        core = gc_rd(regGFX_IMU_CORE_CTRL)
        rst  = gc_rd(regGFX_IMU_GFX_RESET_CTRL)
        bl   = gc_rd(regRLC_RLCS_BOOTLOAD_STATUS)
        cntl = gc_rd(regRLC_CNTL)
        blo  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO)
        bhi  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_ADDR_HI)
        bsz  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_SIZE)
        print(f"  [{label}]")
        print(f"    CORE=0x{core:x} RESET=0x{rst:08x} BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x}")
        print(f"    BOOTLOADER[hi=0x{bhi:08x} lo=0x{blo:08x} sz=0x{bsz:08x}]")

    # Step 1-3: smu_bring_up (includes load_sos, ring_create, LOAD_IP_FW(SMU),
    # SetDriverDramAddr{Hi,Lo}). enable_domain=None → skip EnableAll, we'll do
    # it ourselves AFTER AUTOLOAD_RLC.
    print("\n== 1-3: smu_bring_up(enable_domain=None) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    snapshot("after smu_bring_up")

    ctx = alloc_cmd_ctx(drv)

    # Step 4a: LOAD_TOC
    print("\n== 4: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)

    # Step 4b: LOAD_IP_FW(IMU_I, IMU_D)
    print("\n== 5: LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    # Step 5: Fill autoload buffer
    print("\n== 6: build autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)
    snapshot("after build_autoload_buffer")

    # Step 6: AUTOLOAD_RLC
    print("\n== 7: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")
    snapshot("just after AUTOLOAD_RLC")

    # Step 7-8: EnableAllSmuFeatures(0). Tinygrad's approach.
    # This is what should ungate GFX clocks and unlock BOOTLOADER_ADDR.
    print("\n== 8: EnableAllSmuFeatures(0) [tinygrad arg] ==")
    _try(c, PPSMC_MSG_EnableAllSmuFeatures, 0,
         "EnableAllSmuFeatures(PWR_ALL=0)", timeout=10000)
    snapshot("after EnableAllSmuFeatures(0)")

    # Step 9: Poll BOOTLOAD_COMPLETE for 15 s.
    print("\n== 9: poll BOOTLOAD_COMPLETE (15s) ==")
    deadline = time.time() + 15
    last = None
    start = time.time()
    while time.time() < deadline:
        core = gc_rd(regGFX_IMU_CORE_CTRL)
        rst  = gc_rd(regGFX_IMU_GFX_RESET_CTRL)
        bl   = gc_rd(regRLC_RLCS_BOOTLOAD_STATUS)
        cntl = gc_rd(regRLC_CNTL)
        blo  = gc_rd(regGFX_IMU_RLC_BOOTLOADER_ADDR_LO)
        snap = (core, rst, bl, cntl, blo)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x} "
                  f"BOOT_LO=0x{blo:08x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    print("\n== Final feature state ==")
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, "GetRunningFeaturesLo", timeout=1500)
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, "GetRunningFeaturesHi", timeout=1500)


if __name__ == "__main__":
    main()
