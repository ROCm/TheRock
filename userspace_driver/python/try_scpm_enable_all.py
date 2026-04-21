"""SCPM-aware bring-up path — skip DramToSmu pptable + SetAllowedMask.

Finding from the previous run: on gfx1201, both
`TransferTableDram2Smu(PPTABLE)` and `SetAllowedFeaturesMask` are
rejected (0xFF and 0xFD respectively). That matches Linux's
`smu_smc_hw_setup` which skips both when `adev->scpm_enabled`:

    if (!adev->scpm_enabled) {
        smu_write_pptable(smu);            // TransferTableDram2Smu
    }
    ...
    if (!adev->scpm_enabled) {
        smu_feature_set_allowed_mask(smu); // SetAllowedMask
    }

On SCPM cards PSP is responsible for the pptable and allowed-mask;
the driver just configures, runs BTC, updates PCIe params, and
kicks EnableAll. Our card is SCPM — confirmed by the rejections.

SCPM bring-up order (matching Linux):
  1. smu_bring_up (enable_domain=None).
  2. Full PSP autoload (LOAD_TOC + LOAD_IP_FW(IMU) + VRAM + AUTOLOAD_RLC).
  3. SetToolsDramAddr + SetSystemVirtualDramAddr.
  4. UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE=1).
  5. RunDcBtc.
  6. OverridePcieParameters (PPSMC_MSG 0x20) — new step for us.
  7. EnableAll(PWR_ALL=0).
"""
from __future__ import annotations

import ctypes
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
    FEATURE_PWR_ALL,
    MP0_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_GetRunningSmuFeaturesHi,
    PPSMC_MSG_GetRunningSmuFeaturesLo,
    PPSMC_MSG_SetSystemVirtualDramAddrHigh,
    PPSMC_MSG_SetSystemVirtualDramAddrLow,
    PPSMC_MSG_SetToolsDramAddrHigh,
    PPSMC_MSG_SetToolsDramAddrLow,
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
PPSMC_MSG_RunDcBtc               = 0x36
PPSMC_MSG_OverridePcieParameters = 0x20
TABLE_COMBO_PPTABLE = 1
TOOL_VRAM_OFF = 0x1810000
POOL_VRAM_OFF = 0x1900000


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _try(c, msg, arg, name, *, timeout=3000):
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

    print("\n== 1: smu_bring_up ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    fb_base = (c.mmio_read32(5, (0x1A000 + 0x0554) * 4) & 0xFFFFFF) << 24
    tbl_mc  = result.driver_table_mc
    tool_mc = fb_base + TOOL_VRAM_OFF
    pool_mc = fb_base + POOL_VRAM_OFF

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: full PSP autoload ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)
    print(f"  AUTOLOAD_RLC status = 0x{submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)['status']:08x}")

    print("\n== 3: SetToolsDramAddr + SetSystemVirtualDramAddr ==")
    _try(c, PPSMC_MSG_SetToolsDramAddrHigh, (tool_mc >> 32) & 0xFFFFFFFF, "SetToolsDramAddrHigh")
    _try(c, PPSMC_MSG_SetToolsDramAddrLow, tool_mc & 0xFFFFFFFF, "SetToolsDramAddrLow")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrHigh, (pool_mc >> 32) & 0xFFFFFFFF, "SetSystemVirtualDramAddrHigh")
    _try(c, PPSMC_MSG_SetSystemVirtualDramAddrLow, pool_mc & 0xFFFFFFFF, "SetSystemVirtualDramAddrLow")

    print("\n== 4: UseDefaultPPTable + TransferTableSmu2Dram(COMBO_PPTABLE) ==")
    _try(c, PPSMC_MSG_UseDefaultPPTable, 0, "UseDefaultPPTable")
    _try(c, PPSMC_MSG_TransferTableSmu2Dram, TABLE_COMBO_PPTABLE,
         "TransferTableSmu2Dram(COMBO_PPTABLE=1)", timeout=5000)

    print("\n== 5: RunDcBtc ==")
    _try(c, PPSMC_MSG_RunDcBtc, 0, "RunDcBtc", timeout=5000)

    # OverridePcieParameters param layout (from Linux
    # smu_v14_0_2_update_pcie_parameters):
    #   smu_pcie_arg = (link_level << 16) | (pcie_gen << 8) | pcie_width
    #
    # PcieGenSpeed enum:  0=Gen1  1=Gen2  2=Gen3  3=Gen4  4=Gen5
    # PcieLaneCount enum: 1=x1  2=x2  3=x4  4=x8  5=x12  6=x16
    # NUM_LINK_LEVELS = 3 — Linux sends one message per level.
    # Thunderbolt eGPU is effectively PCIe 3.0 x4 → gen=2, width=3.
    print("\n== 6: OverridePcieParameters × 3 levels (gen=2 Gen3, width=3 x4) ==")
    for level in range(3):
        arg = (level << 16) | (2 << 8) | 3
        _try(c, PPSMC_MSG_OverridePcieParameters, arg,
             f"OverridePcieParameters(level={level}, gen=2, width=3)", timeout=5000)

    # Send EnableAll(PWR_ALL=0) and while it pends, poll RLC state
    # every 100 ms for 60 s. If RLC is doing slow work we'll see
    # bits 6..30 or 31 of BOOTLOAD_STATUS progress over time. If
    # nothing changes in 60 s, the problem isn't a slow-completion.
    import time as _time
    GC_B1 = 0xA000
    def _rlc(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    MP1 = 0x16200
    c90 = (MP1 + 0x40 + 90) * 4
    c82 = (MP1 + 0x40 + 82) * 4
    c66 = (MP1 + 0x40 + 66) * 4

    # Parse FeaturesToRun from the pptable SMU wrote into our driver
    # table. We'll try enabling specific feature bits via
    # EnableSmuFeaturesLow/High instead of the big hammer EnableAll.
    bar0_cpu, _ = c.map_bar(0)
    base = bar0_cpu + (tbl_mc - fb_base)
    dw1 = (ctypes.c_uint32 * 1).from_address(base + 4)[0]
    pp_off = (dw1 >> 16) & 0xFFFF
    feat_low  = (ctypes.c_uint32 * 1).from_address(base + pp_off + 4)[0]
    feat_high = (ctypes.c_uint32 * 1).from_address(base + pp_off + 8)[0]
    print(f"\n  Pptable FeaturesToRun: low=0x{feat_low:08x} high=0x{feat_high:08x}")

    # First warm up SOC (known to work) so SMU's feature engine is live.
    print("\n== 7a: EnableAll(PWR_SOC=3) ==")
    _try(c, PPSMC_MSG_EnableAllSmuFeatures, 3, "EnableAll(PWR_SOC=3)", timeout=5000)

    # Previous run showed EnableSmuFeaturesLow(0x6) returns arg_out=0x2
    # — bit 1 (DPM_GFXCLK) stuck, bit 2 (DPM_GFX_POWER_OPTIMIZER)
    # didn't. So individual feature enables via 0x08 work even though
    # PWR_GFX/PWR_ALL hang. Try enabling every bit in
    # FeaturesToRun[0] one at a time, logging which ones stick and
    # which hang SMU. Each call has a short timeout to avoid a
    # per-bit hang taking out the whole probe.
    print("\n== 7b: per-bit EnableSmuFeaturesLow probe ==")
    running_lo = 0
    for bit in range(32):
        bit_mask = 1 << bit
        if not (feat_low & bit_mask):
            continue  # pptable doesn't want this feature
        if running_lo & bit_mask:
            continue  # already enabled via PWR_SOC warm-up
        try:
            r, arg_out = smu_send(c, 0x08, bit_mask, timeout_ms=2000)
            tag = "enabled" if (arg_out & bit_mask) else "refused"
            print(f"  bit {bit:2d} (0x{bit_mask:08x}): resp=0x{r:x} arg_out=0x{arg_out:08x}  {tag}")
            if arg_out & bit_mask:
                running_lo |= bit_mask
        except TimeoutError:
            print(f"  bit {bit:2d} (0x{bit_mask:08x}): HUNG — this bit triggered a hang")
            break
    print(f"\n  Final low-mask enabled via EnableSmuFeaturesLow: 0x{running_lo:08x}")

    # After enabling DPM_GFXCLK (bit 1) and friends, SMU gates the
    # GFX block → GRBM/EA/UTCL2/SDMA re-enter reset and RLC loses its
    # bootload state. DisallowGfxOff tells SMU to wake and keep GFX
    # up so bootload can resume.
    print("\n== 7c: DisallowGfxOff (wake GFX after DPM enable) ==")
    from amd_gpu_driver.backends.macos.smu import PPSMC_MSG_DisallowGfxOff
    _try(c, PPSMC_MSG_DisallowGfxOff, 0, "DisallowGfxOff", timeout=3000)

    # Also enable the high-mask features that the pptable wants.
    print("\n== 7d: EnableSmuFeaturesHigh(FeaturesToRun[1]) ==")
    _try(c, 0x09, feat_high,
         f"EnableSmuFeaturesHigh(0x{feat_high:x})", timeout=5000)

    print("\n== 8: poll BOOTLOAD_COMPLETE (5s) ==")
    GC_B1 = 0xA000
    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        bl = gc_rd(0x4e7c); rst = gc_rd(0x40bc); core = gc_rd(0x40b6)
        if (bl, rst, core) != last:
            t = time.time() - deadline + 5
            print(f"  t={t:5.2f}s CORE=0x{core:x} RESET=0x{rst:08x} BOOTLOAD=0x{bl:08x}")
            last = (bl, rst, core)
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)

    print("\n== Final feature state ==")
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0, "GetRunningFeaturesLo", timeout=1500)
    _try(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0, "GetRunningFeaturesHi", timeout=1500)


if __name__ == "__main__":
    main()
