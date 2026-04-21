"""Ask SMU to write its built-in combo pptable to our driver table.

Based on Linux `smu_cmn_get_combo_pptable`:
  smu_cmn_update_table(smu, SMU_TABLE_COMBO_PPTABLE, 0, ..., drv2smu=False)
    → smu_cmn_send_smc_msg_with_param(SMU_MSG_TransferTableSmu2Dram,
                                       table_id | ((0 & 0xFFFF) << 16))

On smu_v14_0_2 the per-ASIC table_id for COMBO_PPTABLE is 1
(TABLE_COMBO_PPTABLE from smu14_driver_if_v14_0.h). Size is 0x4000
(MP0_MP1_DATA_REGION_SIZE_COMBOPPTABLE). That matches the driver
table size we set in smu_bring_up.

Sequence (no enable_domain mid-flow):
  1. smu_bring_up(enable_domain=None).
  2. Full PSP LOAD_TOC + LOAD_IP_FW(IMU) + autoload + AUTOLOAD_RLC.
  3. PPSMC_MSG_TransferTableSmu2Dram arg=1.
  4. Read 512 bytes of driver table back via MM_INDEX/MM_DATA and
     dump so we can parse the allowed-features mask offline.
  5. Attempt EnableAll(PWR_ALL=0).
"""
from __future__ import annotations

import logging
import os
import struct
import sys

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
    PPSMC_MSG_SetAllowedFeaturesMaskHigh,
    PPSMC_MSG_SetAllowedFeaturesMaskLow,
    PPSMC_MSG_TransferTableSmu2Dram,
    PPSMC_MSG_UseDefaultPPTable,
    smu_bring_up,
    smu_send,
)

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")
TABLE_COMBO_PPTABLE = 1


class _DriverShim:
    def __init__(self, client): self.client = client
    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)
    def free_dma(self, h): self.client.free_dma(h)


def mmindex_read_bytes(client, mc_addr: int, size: int) -> bytes:
    """Read VRAM via MM_INDEX/MM_DATA at byte offsets 0x00/0x04/0x18."""
    out = bytearray()
    for off in range(0, size, 4):
        a = mc_addr + off
        client.mmio_write32(5, 0x00, 0x80000000 | (a & 0xFFFFFFFF))
        client.mmio_write32(5, 0x18, (a >> 32) & 0xFFFFFFFF)
        v = client.mmio_read32(5, 0x04)
        out += struct.pack("<I", v)
    return bytes(out)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    c = IOKitClient(); c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x}")
    if c.mmio_read32(5, (0x16000 + 0x40 + 81) * 4) != 0:
        print("SOS already alive — replug first.")
        sys.exit(0)
    drv = _DriverShim(c)

    print("\n== 1: smu_bring_up (no EnableAll) ==")
    result = smu_bring_up(c, drv, firmware_dir=FIRMWARE_DIR, enable_domain=None)
    tbl_mc = result.driver_table_mc
    print(f"  driver_table MC = 0x{tbl_mc:x}")

    ctx = alloc_cmd_ctx(drv)

    print("\n== 2a: LOAD_TOC ==")
    sos_blob = open(os.path.join(FIRMWARE_DIR, "psp_14_0_3_sos.bin"), "rb").read()
    toc_comp = next(x for x in parse_psp_firmware(sos_blob) if x.name == "TOC")
    submit_load_toc(c, drv, MP0_BASE_DW, result.ring, ctx, toc_comp.data)

    print("\n== 2b: LOAD_IP_FW(IMU_I, IMU_D) ==")
    imu_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_imu.bin"), "rb").read()
    iram, dram = _extract_imu(imu_blob)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, iram, GFX_FW_TYPE_IMU_I, "IMU_I", strict=True)
    _load_one(c, drv, MP0_BASE_DW, result.ring, ctx, dram, GFX_FW_TYPE_IMU_D, "IMU_D", strict=True)

    print("\n== 2c: VRAM autoload buffer ==")
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc_blob = f.read()
    layout = plan_autoload(toc_blob)
    build_autoload_buffer(c, FIRMWARE_DIR, layout, toc_blob)

    print("\n== 2d: AUTOLOAD_RLC ==")
    resp = submit_autoload_rlc(c, drv, MP0_BASE_DW, result.ring, ctx)
    print(f"  status = 0x{resp['status']:08x}")

    # Zero the driver table first via MM_INDEX/MM_DATA so we can tell
    # whether SMU wrote anything.
    print("\n== 3: zero 512 bytes of driver table (MM_INDEX) ==")
    for off in range(0, 512, 4):
        a = tbl_mc + off
        c.mmio_write32(5, 0x00, 0x80000000 | (a & 0xFFFFFFFF))
        c.mmio_write32(5, 0x18, (a >> 32) & 0xFFFFFFFF)
        c.mmio_write32(5, 0x04, 0)

    print("\n== 4: UseDefaultPPTable (might be a prereq) ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_UseDefaultPPTable, 0, timeout_ms=2000)
        print(f"  UseDefaultPPTable -> resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")

    print("\n== 5: TransferTableSmu2Dram(COMBO_PPTABLE=1) ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_TransferTableSmu2Dram, TABLE_COMBO_PPTABLE,
                        timeout_ms=5000)
        print(f"  TransferTableSmu2Dram(1) -> resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")

    # Read back the driver table
    print("\n== 6: read first 256 bytes of driver table ==")
    data = mmindex_read_bytes(c, tbl_mc, 256)
    for i in range(0, 256, 16):
        hex_s = " ".join(f"{b:02x}" for b in data[i:i+16])
        print(f"  0x{i:03x}: {hex_s}")

    # Heuristic: find where the table isn't just zeros
    nonzero = [i for i, b in enumerate(data) if b]
    print(f"\n  {len(nonzero)} non-zero bytes out of 256 in the head")

    print("\n== 7: EnableAll(PWR_ALL=0) attempt ==")
    try:
        r, a = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures, FEATURE_PWR_ALL, timeout_ms=5000)
        print(f"  EnableAll(0) -> resp=0x{r:x} arg_out=0x{a:x}")
    except TimeoutError as e:
        print(f"  TIMEOUT: {e}")


if __name__ == "__main__":
    main()
