"""Dump all TOC entries from gc_12_0_1_toc.bin to see what RLC expects.

Plan: if ids 5 (RLC_P_UCODE) or 6 (RLC_V_UCODE) or others are in the
TOC but we don't populate them, RLC will read zeros and stall past
its IRAM phase (which we're observing).
"""
from __future__ import annotations

import os
import struct

from amd_gpu_driver.backends.macos.gfx_autoload import parse_rlc_toc

FIRMWARE_DIR = os.path.expanduser("~/firmware/linux-firmware/amdgpu")

SOC24_ID_NAMES = {
    0:  "INVALID",
    1:  "RLC_G_UCODE",
    2:  "RLC_TOC",
    3:  "RLCG_SCRATCH",
    4:  "RLC_SRM_ARAM",
    5:  "RLC_P_UCODE",
    6:  "RLC_V_UCODE",
    7:  "RLX6_UCODE",
    8:  "RLX6_UCODE_CORE1",
    9:  "RLX6_DRAM_BOOT",
    10: "RLX6_DRAM_BOOT_CORE1",
    11: "SDMA_UCODE_TH0",
    12: "SDMA_UCODE_TH1",
    13: "CP_PFP",
    14: "CP_ME",
    15: "CP_MEC",
    16: "RS64_MES_P0",
    17: "RS64_MES_P1",
    18: "RS64_PFP",
    19: "RS64_ME",
    20: "RS64_MEC",
    21: "RS64_MES_P0_STACK",
    22: "RS64_MES_P1_STACK",
    23: "RS64_PFP_P0_STACK",
    24: "RS64_PFP_P1_STACK",
    25: "RS64_ME_P0_STACK",
    26: "RS64_ME_P1_STACK",
    27: "RS64_MEC_P0_STACK",
    28: "RS64_MEC_P1_STACK",
    29: "RS64_MEC_P2_STACK",
    30: "RS64_MEC_P3_STACK",
}


def main():
    with open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_toc.bin"), "rb") as f:
        toc = f.read()

    print(f"gc_12_0_1_toc.bin: {len(toc)} bytes")

    # Dump common header
    size, hdr_sz, maj, minr, ipmaj, ipmin, uver, usz, uoff, crc = \
        struct.unpack_from("<IIHHHHIIII", toc, 0)
    print(f"  common_header: size={size} hdr_sz={hdr_sz} "
          f"ver={maj}.{minr} ip={ipmaj}.{ipmin} ucode_ver=0x{uver:x} "
          f"ucode_size={usz} ucode_off={uoff} crc=0x{crc:x}")

    entries = parse_rlc_toc(toc)
    print(f"\n{len(entries)} entries:")
    print(f"  {'id':>3} {'name':24s} {'offset':>10} {'size':>10}")
    for e in sorted(entries, key=lambda e: e.id):
        name = SOC24_ID_NAMES.get(e.id, "?")
        print(f"  {e.id:3d} {name:24s} 0x{e.offset:08x} 0x{e.size:08x}")

    # Check RLC firmware header version
    rlc_blob = open(os.path.join(FIRMWARE_DIR, "gc_12_0_1_rlc.bin"), "rb").read()
    rlc_size, rlc_hdr_sz, rlc_maj, rlc_min = struct.unpack_from("<IIHH", rlc_blob, 0)
    print(f"\ngc_12_0_1_rlc.bin: size={rlc_size} ver={rlc_maj}.{rlc_min}")


if __name__ == "__main__":
    main()
