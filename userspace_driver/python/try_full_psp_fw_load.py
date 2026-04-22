"""Full PSP LOAD_IP_FW probe — try all firmware types tinygrad loads.

Prior memory claimed PSP rejects everything except SMU + IMU_I/D on
gfx1201. That conclusion was from a run in a specific SMU/PSP state
and might not be accurate in the tinygrad-order flow we've now
validated.

Per comment in `psp_gfx_if.h` line 103:
  GFX_CMD_ID_AUTOLOAD_RLC: "Indicates all graphics fw loaded"

This strongly implies PSP expects ALL graphics firmware to be sent
via LOAD_IP_FW BEFORE AUTOLOAD_RLC. Our previous runs only sent
SMU + IMU_I/D, so PSP's AUTOLOAD_RLC had incomplete firmware in
TMR and RLC stalled at BOOTLOAD=0x3F.

This script:
  1. smu_bring_up(enable_domain=None)
  2. LOAD_TOC
  3. Probe LOAD_IP_FW for every firmware type tinygrad loads
     (matches `fw.descs` order). Continue on rejection, don't halt.
  4. Print accept/reject summary.
  5. Call AUTOLOAD_RLC.
  6. Call EnableAllSmuFeatures(0).
  7. Poll BOOTLOAD_COMPLETE.
"""
from __future__ import annotations

import logging
import os
import sys
import time

from amd_gpu_driver.backends.macos.gfx_autoload import (
    extract_mes,
    extract_rlc_subfw,
    extract_rs64_gfx,
    extract_sdma,
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
    GFX_FW_TYPE_RLC_G,
    GFX_FW_TYPE_RLC_IRAM,
    GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM,
    GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM,
    GFX_FW_TYPE_RS64_ME,
    GFX_FW_TYPE_RS64_ME_P0_STACK,
    GFX_FW_TYPE_RS64_ME_P1_STACK,
    GFX_FW_TYPE_RS64_MEC,
    GFX_FW_TYPE_RS64_MEC_P0_STACK,
    GFX_FW_TYPE_RS64_MEC_P1_STACK,
    GFX_FW_TYPE_RS64_MEC_P2_STACK,
    GFX_FW_TYPE_RS64_MEC_P3_STACK,
    GFX_FW_TYPE_RS64_MES,
    GFX_FW_TYPE_RS64_MES_STACK,
    GFX_FW_TYPE_RS64_PFP,
    GFX_FW_TYPE_RS64_PFP_P0_STACK,
    GFX_FW_TYPE_RS64_PFP_P1_STACK,
    GFX_FW_TYPE_SDMA_UCODE_TH0,
    alloc_cmd_ctx,
)
from amd_gpu_driver.backends.macos.smu import (
    MP0_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
GC_B1 = 0xA000

GFX_FW_TYPE_RLC_DRAM_BOOT = 48  # not exported but valid


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
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)
    drv = _DriverShim(c)

    def gc_rd(o): return c.mmio_read32(5, (GC_B1 + o) * 4)

    print("\n== 1: smu_bring_up(enable_domain=None) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)

    # Parse all firmware files.
    def _read(name): return open(os.path.join(FIRMWARE_DIR, name), "rb").read()
    imu_blob = _read("gc_12_0_1_imu.bin")
    rlc_blob = _read("gc_12_0_1_rlc.bin")
    pfp_blob = _read("gc_12_0_1_pfp.bin")
    me_blob  = _read("gc_12_0_1_me.bin")
    mec_blob = _read("gc_12_0_1_mec.bin")
    mes_blob = _read("gc_12_0_1_mes.bin")
    sdma_blob = _read("sdma_7_0_1.bin")

    imu_iram, imu_dram = _extract_imu(imu_blob)
    rlc_subs = extract_rlc_subfw(rlc_blob)
    pfp_u, pfp_d = extract_rs64_gfx(pfp_blob)
    me_u,  me_d  = extract_rs64_gfx(me_blob)
    mec_u, mec_d = extract_rs64_gfx(mec_blob)
    mes_u, mes_d = extract_mes(mes_blob)
    sdma_ucode   = extract_sdma(sdma_blob)

    # Build probe list. Order matches tinygrad (SDMA, RLC subs, PFP+stacks,
    # ME+stacks, MEC+stacks, MES, IMU_I, IMU_D, RLC_G).
    probes = []

    def append(label, fw_type, payload):
        probes.append((label, fw_type, payload))

    append("SDMA_TH0",          GFX_FW_TYPE_SDMA_UCODE_TH0,          sdma_ucode)
    append("RLC_IRAM",          GFX_FW_TYPE_RLC_IRAM,                rlc_subs.get(7, b""))   # SOC24_FW_ID_RLX6_UCODE=7
    append("RLC_DRAM_BOOT",     GFX_FW_TYPE_RLC_DRAM_BOOT,           rlc_subs.get(9, b""))   # SOC24_FW_ID_RLX6_DRAM_BOOT=9
    append("RLC_SRLG",          GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM, rlc_subs.get(3, b""))  # SOC24_FW_ID_RLCG_SCRATCH=3
    append("RLC_SRLS",          GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM, rlc_subs.get(4, b""))  # SOC24_FW_ID_RLC_SRM_ARAM=4
    append("RS64_PFP",          GFX_FW_TYPE_RS64_PFP,                pfp_u)
    append("RS64_PFP_P0_STACK", GFX_FW_TYPE_RS64_PFP_P0_STACK,       pfp_d)
    append("RS64_PFP_P1_STACK", GFX_FW_TYPE_RS64_PFP_P1_STACK,       pfp_d)
    append("RS64_ME",           GFX_FW_TYPE_RS64_ME,                 me_u)
    append("RS64_ME_P0_STACK",  GFX_FW_TYPE_RS64_ME_P0_STACK,        me_d)
    append("RS64_ME_P1_STACK",  GFX_FW_TYPE_RS64_ME_P1_STACK,        me_d)
    append("RS64_MEC",          GFX_FW_TYPE_RS64_MEC,                mec_u)
    append("RS64_MEC_P0_STACK", GFX_FW_TYPE_RS64_MEC_P0_STACK,       mec_d)
    append("RS64_MEC_P1_STACK", GFX_FW_TYPE_RS64_MEC_P1_STACK,       mec_d)
    append("RS64_MEC_P2_STACK", GFX_FW_TYPE_RS64_MEC_P2_STACK,       mec_d)
    append("RS64_MEC_P3_STACK", GFX_FW_TYPE_RS64_MEC_P3_STACK,       mec_d)
    append("RS64_MES",          GFX_FW_TYPE_RS64_MES,                mes_u)
    append("RS64_MES_STACK",    GFX_FW_TYPE_RS64_MES_STACK,          mes_d)
    append("IMU_I",             GFX_FW_TYPE_IMU_I,                   imu_iram)
    append("IMU_D",             GFX_FW_TYPE_IMU_D,                   imu_dram)
    # RLC_G is last (triggers AUTOLOAD_RLC on Linux PSP path):
    append("RLC_G",             GFX_FW_TYPE_RLC_G,                   rlc_subs.get(1, b""))   # SOC24_FW_ID_RLC_G_UCODE=1

    print(f"\n== 3: LOAD_IP_FW probe ({len(probes)} types) ==")
    results = []
    for label, fw_type, payload in probes:
        if not payload:
            print(f"  {label:22s} type={fw_type:3d} (empty payload, skipped)")
            results.append((label, fw_type, "skipped", 0))
            continue
        status = _load_one(c, drv, MP0_BASE_DW, result.ring, ctx,
                           payload, fw_type, label, strict=False)
        results.append((label, fw_type, "ok" if status == 0 else f"0x{status:08x}", len(payload)))

    print("\n== LOAD_IP_FW summary ==")
    print(f"  {'label':22s} {'type':>4} {'status':>12} {'size':>8}")
    oks = 0
    fails = 0
    for label, fw_type, status, size in results:
        print(f"  {label:22s} {fw_type:>4d} {status:>12} {size:>8d}")
        if status == "ok":
            oks += 1
        elif status != "skipped":
            fails += 1
    print(f"  TOTAL: {oks} OK, {fails} failed, {len(results) - oks - fails} skipped")

    print("\n== 4: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    print("\n== 5: EnableAllSmuFeatures(0) ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures, 0, timeout_ms=8000)
        print(f"  EnableAllSmuFeatures(0) resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError:
        print("  EnableAllSmuFeatures(0) TIMEOUT (expected)")

    print("\n== 6: poll BOOTLOAD_COMPLETE (30s) ==")
    deadline = time.time() + 30
    last = None
    start = time.time()
    while time.time() < deadline:
        bl = gc_rd(0x4e7c)
        rst = gc_rd(0x40bc)
        core = gc_rd(0x40b6)
        cntl = gc_rd(0x4c00)
        stat = gc_rd(0x4c04)
        snap = (bl, rst, core, cntl, stat)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x} RLC_STAT=0x{stat:x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)


if __name__ == "__main__":
    main()
