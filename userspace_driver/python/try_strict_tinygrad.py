"""Strict tinygrad order — all PSP commands, THEN all SMU mailbox messages.

Tinygrad's AM_PSP.init_hw + AM_SMU.init_hw ordering, preserved exactly:

  PSP side (AM_PSP.init_hw, ip.py:580-601):
    1. SOS bootstrap (if needed)
    2. ring_create
    3. LOAD_TOC
    4. LOAD_IP_FW(SMU)
    5. LOAD_IP_FW for every desc in fw.descs — SDMA, RLC sub-FW
       (IRAM, DRAM_BOOT, SRLG, SRLS), RS64 PFP/ME/MEC + stacks,
       MES + stack, IMU_I, IMU_D, and finally RLC_G
    6. AUTOLOAD_RLC

  SMU side (AM_SMU.init_hw, ip.py:179-182) — runs AFTER all PSP:
    7. SetDriverDramAddrHigh
    8. SetDriverDramAddrLow
    9. EnableAllSmuFeatures(0)

Our smu_bring_up() interleaves the PSP and SMU steps (it does SMU
GetSmuVersion + SetDriverDramAddr immediately after LOAD_IP_FW(SMU),
before LOAD_TOC). This script uses the lower-level primitives to
match tinygrad's strict PSP-then-SMU ordering.

Sanity: at MP0 14.0.3, boot_time_tmr=True and autoload_tmr=True in
tinygrad, so SETUP_TMR is skipped. LOAD_TOC is the only TMR-side
setup — matches what we do.
"""
from __future__ import annotations

import ctypes
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
from amd_gpu_driver.backends.macos.psp_bootloader import (
    c2pmsg_dw,
    is_sos_alive,
    load_sos,
)
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
    GFX_FW_TYPE_SMU,
    alloc_cmd_ctx,
)
from amd_gpu_driver.backends.macos.psp_ring import ring_create
from amd_gpu_driver.backends.macos.smu import (
    MP0_BASE_DW,
    MMHUB_BASE_DW,
    PPSMC_MSG_EnableAllSmuFeatures,
    PPSMC_MSG_SetDriverDramAddrHigh,
    PPSMC_MSG_SetDriverDramAddrLow,
    regMMMC_VM_FB_LOCATION_BASE,
    regMMMC_VM_FB_LOCATION_TOP,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
GC_B1 = 0xA000
_MMIO_BAR = 5

GFX_FW_TYPE_RLC_DRAM_BOOT = 48  # not exported but valid gfx12 type


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def _mmhub_rd(c, dw): return c.mmio_read32(_MMIO_BAR, (MMHUB_BASE_DW + dw) * 4)


def _vram_tbl_mc(c) -> int:
    fb = _mmhub_rd(c, regMMMC_VM_FB_LOCATION_BASE) & 0xFFFFFF
    top = _mmhub_rd(c, regMMMC_VM_FB_LOCATION_TOP) & 0xFFFFFF
    return (fb << 24) + (((top + 1) - fb) << 24) - 0x20000


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    drv = _DriverShim(c)

    def gc_rd(o): return c.mmio_read32(_MMIO_BAR, (GC_B1 + o) * 4)

    # ===== 1. SOS bootstrap =====
    print("\n== 1: load SOS (or assert already alive) ==")
    if not is_sos_alive(c, MP0_BASE_DW):
        load_sos(c, drv, MP0_BASE_DW, os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), verbose=False)
    c81 = c.mmio_read32(_MMIO_BAR, c2pmsg_dw(MP0_BASE_DW, 81) * 4)
    print(f"  SOS alive: C2PMSG_81=0x{c81:08x}")

    # ===== 2. PSP KM ring =====
    print("\n== 2: create PSP KM ring ==")
    ring = ring_create(c, drv, MP0_BASE_DW, destroy_first=True, verbose=False)

    ctx = alloc_cmd_ctx(drv)

    # ===== 3. LOAD_TOC =====
    print("\n== 3: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    from amd_gpu_driver.backends.macos.psp_bootloader import parse_psp_firmware
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, ring, ctx, toc_comp.data)

    # ===== 4. LOAD_IP_FW(SMU) — must be first before TMR on gfx12 =====
    print("\n== 4: LOAD_IP_FW(SMU) ==")

    def _read(name): return open(os.path.join(FIRMWARE_DIR, name), "rb").read()

    smu_blob = _read("smu_14_0_3.bin")
    smu_uoff, smu_usz, _ = _parse_common_hdr(smu_blob)
    smu_payload = smu_blob[smu_uoff:smu_uoff + smu_usz]
    status = _load_one(c, drv, MP0_BASE_DW, ring, ctx, smu_payload,
                       GFX_FW_TYPE_SMU, "SMU", strict=True)

    # ===== 5. LOAD_IP_FW for every other firmware type — mirror tinygrad fw.descs =====
    print("\n== 5: LOAD_IP_FW for all non-SMU firmware (tinygrad fw.descs order) ==")

    imu_blob  = _read("gc_12_0_1_imu.bin")
    rlc_blob  = _read("gc_12_0_1_rlc.bin")
    pfp_blob  = _read("gc_12_0_1_pfp.bin")
    me_blob   = _read("gc_12_0_1_me.bin")
    mec_blob  = _read("gc_12_0_1_mec.bin")
    mes_blob  = _read("gc_12_0_1_mes.bin")
    sdma_blob = _read("sdma_7_0_1.bin")

    imu_iram, imu_dram = _extract_imu(imu_blob)
    rlc_subs = extract_rlc_subfw(rlc_blob)
    pfp_u, pfp_d = extract_rs64_gfx(pfp_blob)
    me_u,  me_d  = extract_rs64_gfx(me_blob)
    mec_u, mec_d = extract_rs64_gfx(mec_blob)
    mes_u, mes_d = extract_mes(mes_blob)
    sdma_ucode   = extract_sdma(sdma_blob)

    probes = [
        ("SDMA_TH0",          GFX_FW_TYPE_SDMA_UCODE_TH0,           sdma_ucode),
        ("RLC_IRAM",          GFX_FW_TYPE_RLC_IRAM,                 rlc_subs.get(7, b"")),
        ("RLC_DRAM_BOOT",     GFX_FW_TYPE_RLC_DRAM_BOOT,            rlc_subs.get(9, b"")),
        ("RLC_SRLG",          GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM, rlc_subs.get(3, b"")),
        ("RLC_SRLS",          GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM, rlc_subs.get(4, b"")),
        ("RS64_PFP",          GFX_FW_TYPE_RS64_PFP,                 pfp_u),
        ("RS64_PFP_P0_STACK", GFX_FW_TYPE_RS64_PFP_P0_STACK,        pfp_d),
        ("RS64_PFP_P1_STACK", GFX_FW_TYPE_RS64_PFP_P1_STACK,        pfp_d),
        ("RS64_ME",           GFX_FW_TYPE_RS64_ME,                  me_u),
        ("RS64_ME_P0_STACK",  GFX_FW_TYPE_RS64_ME_P0_STACK,         me_d),
        ("RS64_ME_P1_STACK",  GFX_FW_TYPE_RS64_ME_P1_STACK,         me_d),
        ("RS64_MEC",          GFX_FW_TYPE_RS64_MEC,                 mec_u),
        ("RS64_MEC_P0_STACK", GFX_FW_TYPE_RS64_MEC_P0_STACK,        mec_d),
        ("RS64_MEC_P1_STACK", GFX_FW_TYPE_RS64_MEC_P1_STACK,        mec_d),
        ("RS64_MEC_P2_STACK", GFX_FW_TYPE_RS64_MEC_P2_STACK,        mec_d),
        ("RS64_MEC_P3_STACK", GFX_FW_TYPE_RS64_MEC_P3_STACK,        mec_d),
        ("RS64_MES",          GFX_FW_TYPE_RS64_MES,                 mes_u),
        ("RS64_MES_STACK",    GFX_FW_TYPE_RS64_MES_STACK,           mes_d),
        ("IMU_I",             GFX_FW_TYPE_IMU_I,                    imu_iram),
        ("IMU_D",             GFX_FW_TYPE_IMU_D,                    imu_dram),
        ("RLC_G",             GFX_FW_TYPE_RLC_G,                    rlc_subs.get(1, b"")),  # last
    ]

    oks = []
    fails = []
    for label, fw_type, payload in probes:
        if not payload:
            print(f"  {label:22s} type={fw_type:3d} (empty, skipped)")
            continue
        status = _load_one(c, drv, MP0_BASE_DW, ring, ctx, payload, fw_type, label, strict=False)
        if status == 0:
            oks.append(label)
        else:
            fails.append((label, fw_type, status))

    print(f"\n  OK: {len(oks)}   FAIL: {len(fails)}")
    for label, fw_type, status in fails:
        print(f"    FAIL type={fw_type:3d} {label:22s} status=0x{status:08x}")

    # ===== 6. AUTOLOAD_RLC =====
    print("\n== 6: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, ring, ctx)
    print(f"  AUTOLOAD_RLC status = 0x{resp['status']:08x}")

    # Snapshot immediately
    print(f"  post-AUTOLOAD: CORE=0x{gc_rd(0x40b6):x} RESET=0x{gc_rd(0x40bc):08x} "
          f"BOOTLOAD=0x{gc_rd(0x4e7c):08x} RLC_CNTL=0x{gc_rd(0x4c00):x}")

    # ===== 7-8. SMU side: SetDriverDramAddr{Hi,Lo} =====
    print("\n== 7: SetDriverDramAddrHigh/Low ==")
    tbl_mc = _vram_tbl_mc(c)
    print(f"  driver_table MC = 0x{tbl_mc:x}")
    try:
        r, _ = smu_send(c, PPSMC_MSG_SetDriverDramAddrHigh, (tbl_mc >> 32) & 0xFFFFFFFF, timeout_ms=3000)
        print(f"  SetDriverDramAddrHigh resp=0x{r:x}")
    except TimeoutError:
        print("  SetDriverDramAddrHigh TIMEOUT")
    try:
        r, _ = smu_send(c, PPSMC_MSG_SetDriverDramAddrLow, tbl_mc & 0xFFFFFFFF, timeout_ms=3000)
        print(f"  SetDriverDramAddrLow resp=0x{r:x}")
    except TimeoutError:
        print("  SetDriverDramAddrLow TIMEOUT")

    # ===== 9. EnableAllSmuFeatures(0) =====
    print("\n== 8: EnableAllSmuFeatures(0) ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures, 0, timeout_ms=10000)
        print(f"  EnableAllSmuFeatures(0) resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError:
        print("  EnableAllSmuFeatures(0) TIMEOUT")

    # ===== 10. Poll BOOTLOAD_COMPLETE =====
    print("\n== 9: poll BOOTLOAD_COMPLETE (30s) ==")
    deadline = time.time() + 30
    last = None
    start = time.time()
    while time.time() < deadline:
        bl   = gc_rd(0x4e7c)
        rst  = gc_rd(0x40bc)
        core = gc_rd(0x40b6)
        cntl = gc_rd(0x4c00)
        stat = gc_rd(0x4c04)
        gpm  = gc_rd(0x4e6c)
        snap = (bl, rst, core, cntl, stat, gpm)
        if snap != last:
            t = time.time() - start
            print(f"  t={t:6.3f}s CORE=0x{core:x} RESET=0x{rst:08x} "
                  f"BOOTLOAD=0x{bl:08x} RLC_CNTL=0x{cntl:x} RLC_STAT=0x{stat:x} GPM_STAT=0x{gpm:08x}")
            last = snap
        if bl & 0x80000000:
            print("  BOOTLOAD_COMPLETE ✓")
            break
        time.sleep(0.05)


def _parse_common_hdr(blob):
    """common_firmware_header: (ucode_offset, ucode_size_bytes, ucode_version)."""
    import struct
    _size, _hdr_sz, _maj, _min, _ipmaj, _ipmin, uver, usz, uoff, _crc = \
        struct.unpack_from("<IIHHHHIIII", blob, 0)
    return uoff, usz, uver


if __name__ == "__main__":
    main()
