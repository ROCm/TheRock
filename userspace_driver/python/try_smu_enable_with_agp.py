"""Try EnableAllSmuFeatures with MMHUB AGP programmed as MC↔bus identity window.

Post-reboot flow:
  1. Open DEXT client
  2. Load SOS (psp_14_0_3_sos.bin)
  3. Create PSP KM ring
  4. Load SMU firmware (smu_14_0_3.bin)
  5. SMU mailbox sanity (GetSmuVersion)
  6. Allocate driver table (0x4000 bytes, DMA-mapped)
  7. Dump current MMHUB aperture state
  8. Program MMHUB MC_VM_AGP_{BASE,BOT,TOP} to cover [0, 0xFF000000) as
     an identity MC→bus window (BASE=0, BOT=0, TOP=0xFF in units of 16 MB)
  9. SetDriverDramAddrHigh/Low with driver-table bus address
 10. EnableAllSmuFeatures with SHORT 3 s timeout — if it hangs, dump MP1
     C2PMSG_* state rather than letting the SMU sit in a permanent hang.

Reference: previous attempt without AGP programming hung on EnableAll.
See memory/smu-feature-enable-attempt.md.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

from amd_gpu_driver.backends.macos.iokit_client import IOKitClient
from amd_gpu_driver.backends.macos.psp_bootloader import c2pmsg_dw, load_sos
from amd_gpu_driver.backends.macos.psp_cmd import (
    GFX_FW_TYPE_SMU,
    alloc_cmd_ctx,
    submit_ip_fw_load,
)
from amd_gpu_driver.backends.macos.psp_ring import ring_create

# ---- gfx1201 IP bases (DWORD units, from IP discovery) ----
MP0_BASE_DW = 0x16000   # PSP
MP1_BASE_DW = 0x16200   # SMU mailbox (BASE_IDX=1 for MP1)
MMHUB_BASE_DW = 0x1A000 # MMHUB system-domain

# ---- MMHUB MC_VM register DWORD offsets (mmhub_4_1_0_offset.h) ----
regMMMC_VM_FB_LOCATION_BASE = 0x0554
regMMMC_VM_FB_LOCATION_TOP  = 0x0555
regMMMC_VM_AGP_TOP          = 0x0556
regMMMC_VM_AGP_BOT          = 0x0557
regMMMC_VM_AGP_BASE         = 0x0558
regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR  = 0x0559
regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x055a

# ---- MP1 SMU mailbox C2PMSG slots ----
C2PMSG_BASE_DW = 0x40

# ---- PPSMC message IDs (smu_v14_0_2_ppsmc.h) ----
PPSMC_MSG_TestMessage             = 0x01
PPSMC_MSG_GetSmuVersion           = 0x02
PPSMC_MSG_EnableAllSmuFeatures    = 0x06
PPSMC_MSG_GetRunningSmuFeaturesLo = 0x0C
PPSMC_MSG_GetRunningSmuFeaturesHi = 0x0D
PPSMC_MSG_SetDriverDramAddrHigh   = 0x0E   # NOT 0x04 (that's SetAllowedFeaturesMaskLow)
PPSMC_MSG_SetDriverDramAddrLow    = 0x0F   # NOT 0x05 (that's SetAllowedFeaturesMaskHigh)

FIRMWARE_DIR = "/Users/anush/firmware/linux-firmware/amdgpu"
SOS_FW = f"{FIRMWARE_DIR}/psp_14_0_3_sos.bin"
SMU_FW = f"{FIRMWARE_DIR}/smu_14_0_3.bin"


class DriverShim:
    def __init__(self, client):
        self.client = client

    def alloc_dma(self, size):
        dma = self.client.alloc_dma(size)
        bus = dma.segments[0][0] if dma.segments else 0
        return (dma.cpu_addr, bus, dma.buffer_id)

    def free_dma(self, h):
        self.client.free_dma(h)


def mmhub_rd(client, dw_off):
    return client.mmio_read32(5, (MMHUB_BASE_DW + dw_off) * 4)


def mmhub_wr(client, dw_off, value):
    client.mmio_write32(5, (MMHUB_BASE_DW + dw_off) * 4, value & 0xFFFFFFFF)


def mp1_c2pmsg(client, n):
    return client.mmio_read32(5, (MP1_BASE_DW + C2PMSG_BASE_DW + n) * 4)


def smu_send(client, msg_id, arg, *, timeout_ms=2000):
    """Send SMU message via MP1 mailbox. Returns (resp, arg_out) or None on timeout."""
    c66 = (MP1_BASE_DW + C2PMSG_BASE_DW + 66) * 4
    c82 = (MP1_BASE_DW + C2PMSG_BASE_DW + 82) * 4
    c90 = (MP1_BASE_DW + C2PMSG_BASE_DW + 90) * 4
    client.mmio_write32(5, c90, 0)
    client.mmio_write32(5, c82, arg & 0xFFFFFFFF)
    client.mmio_write32(5, c66, msg_id)
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        v = client.mmio_read32(5, c90)
        if v != 0:
            return (v, client.mmio_read32(5, c82))
        time.sleep(0.002)
    return None


def dump_mmhub(client, label):
    print(f"  [{label}]")
    for name, off in [
        ("FB_LOCATION_BASE",  regMMMC_VM_FB_LOCATION_BASE),
        ("FB_LOCATION_TOP",   regMMMC_VM_FB_LOCATION_TOP),
        ("AGP_BASE",          regMMMC_VM_AGP_BASE),
        ("AGP_BOT",           regMMMC_VM_AGP_BOT),
        ("AGP_TOP",           regMMMC_VM_AGP_TOP),
        ("SYS_APERTURE_LOW",  regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR),
        ("SYS_APERTURE_HIGH", regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR),
    ]:
        print(f"    {name:20s} = 0x{mmhub_rd(client, off):08x}")


def dump_mp1_c2pmsg(client, label):
    print(f"  [{label}] MP1 C2PMSG snapshot:")
    for n in [33, 53, 54, 66, 75, 81, 82, 83, 90, 91, 92, 93]:
        try:
            v = mp1_c2pmsg(client, n)
            print(f"    C2PMSG_{n:<3d} = 0x{v:08x}")
        except Exception as e:
            print(f"    C2PMSG_{n:<3d} ERR: {e}")


def main():
    c = IOKitClient()
    c.open()
    info = c.get_info()
    print(f"device=0x{info.device_id:04x} rev=0x{info.revision_id:02x} vram={info.vram_size // (1024*1024)}MB")
    drv = DriverShim(c)

    # Step 0: Dump initial MMHUB state (but do NOT touch AGP — the
    # previous run showed AGP programming causes SMU to reject
    # SetDriverDramAddr with PPSMC_Result_CmdRejectedPrereq = 0xFD).
    print("\n== Step 0: MMHUB initial state (read-only) ==")
    dump_mmhub(c, "initial")

    # Step 2: Load SOS (skip if already alive from a prior run).
    print("\n== Step 2: Load SOS ==")
    c81 = c.mmio_read32(5, c2pmsg_dw(MP0_BASE_DW, 81) * 4)
    if c81 != 0:
        print(f"  SOS already alive: C2PMSG_81 = 0x{c81:08x} — skipping load")
    else:
        load_sos(c, drv, MP0_BASE_DW, SOS_FW, verbose=False)
        c81 = c.mmio_read32(5, c2pmsg_dw(MP0_BASE_DW, 81) * 4)
        print(f"  SOS alive: C2PMSG_81 = 0x{c81:08x}")

    # Step 3: PSP KM ring
    print("\n== Step 3: Create PSP KM ring ==")
    ring = ring_create(c, drv, MP0_BASE_DW, destroy_first=True, verbose=False)
    print(f"  ring_bus = 0x{ring.ring_bus:x} (size=0x{ring.ring_size:x})")

    # Step 4: Load SMU firmware.
    # Parse common_firmware_header and pass only the ucode payload
    # (not the whole container) to PSP — LOAD_IP_FW expects raw ucode
    # bytes, not the header-prefixed file.
    print("\n== Step 4: Load SMU firmware ==")
    import struct
    with open(SMU_FW, "rb") as f:
        smu_blob = f.read()
    (size_bytes, header_size_bytes, hver_maj, hver_min,
     ipver_maj, ipver_min, ucode_version, ucode_size_bytes,
     ucode_array_offset_bytes, crc32) = struct.unpack_from("<IIHHHHIIII", smu_blob, 0)
    print(f"  header: file={len(smu_blob)}B  size_bytes={size_bytes}  hver={hver_maj}.{hver_min}")
    print(f"          ucode_version=0x{ucode_version:08x}  ucode_size={ucode_size_bytes}  ucode_off={ucode_array_offset_bytes}")
    ucode_aligned = (ucode_size_bytes + 0xFFF) & ~0xFFF
    smu_cpu, smu_bus, smu_handle = drv.alloc_dma(ucode_aligned)
    ctypes.memset(smu_cpu, 0, ucode_aligned)
    (ctypes.c_ubyte * ucode_size_bytes).from_address(smu_cpu)[:] = \
        smu_blob[ucode_array_offset_bytes:ucode_array_offset_bytes + ucode_size_bytes]
    head = bytes((ctypes.c_ubyte * 16).from_address(smu_cpu))
    print(f"  ucode head:   {head.hex()}")
    print(f"  ucode bus:    0x{smu_bus:x}  size={ucode_size_bytes}")
    ctx = alloc_cmd_ctx(drv)
    r = submit_ip_fw_load(c, drv, MP0_BASE_DW, ring, ctx,
                          smu_bus, ucode_size_bytes, GFX_FW_TYPE_SMU,
                          verbose=True)
    print(f"  SMU FW load status = 0x{r['status']:08x}")
    # Dump first 32 bytes of raw_resp to see if PSP wrote fw_addr_lo/hi
    print(f"  raw_resp[0..32]: {r['raw_resp'][:32].hex()}")
    if r['status'] != 0:
        print("  PSP refused SMU firmware load — aborting.")
        sys.exit(1)

    # Step 5: SMU sanity
    print("\n== Step 5: SMU mailbox sanity ==")
    v = smu_send(c, PPSMC_MSG_GetSmuVersion, 0)
    print(f"  GetSmuVersion       -> {v}  (expected (0x1, 0x00684c00))")

    v = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    print(f"  RunningFeaturesLow  -> {v}")
    v = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  RunningFeaturesHigh -> {v}")

    # Step 6: Pick a driver_table location in VRAM (FB aperture).
    # Rather than allocating in system memory (which requires MMHUB
    # GART/AGP to be reachable from SMU), put the driver table in VRAM.
    # VRAM is unconditionally reachable to SMU via the FB aperture.
    # We pick a VRAM offset far from the IP discovery region at
    # VRAM_SIZE - 64KB.
    print("\n== Step 6: Pick driver_table location in VRAM ==")
    fb_base_val = mmhub_rd(c, regMMMC_VM_FB_LOCATION_BASE) & 0xFFFFFF
    fb_top_val = mmhub_rd(c, regMMMC_VM_FB_LOCATION_TOP) & 0xFFFFFF
    fb_start_mc = fb_base_val << 24
    fb_end_mc   = (fb_top_val + 1) << 24     # inclusive top + 1 byte
    vram_size   = fb_end_mc - fb_start_mc
    # Put driver_table at VRAM_SIZE - 128KB (leaving IP discovery
    # region at VRAM_SIZE - 64KB untouched).
    tbl_off     = vram_size - 0x20000
    tbl_mc      = fb_start_mc + tbl_off
    print(f"  fb_start=0x{fb_start_mc:x}  fb_end=0x{fb_end_mc:x}  vram_size=0x{vram_size:x}")
    print(f"  driver_table MC = 0x{tbl_mc:x}  (offset 0x{tbl_off:x} from fb_start)")

    # Step 7: SetDriverDramAddr with the VRAM MC address.
    print("\n== Step 7: SetDriverDramAddr (VRAM MC addr) ==")
    v = smu_send(c, PPSMC_MSG_SetDriverDramAddrHigh,
                 (tbl_mc >> 32) & 0xFFFFFFFF)
    print(f"  SetDriverDramAddrHigh(0x{(tbl_mc >> 32) & 0xFFFFFFFF:x}) -> {v}")
    v = smu_send(c, PPSMC_MSG_SetDriverDramAddrLow,
                 tbl_mc & 0xFFFFFFFF)
    print(f"  SetDriverDramAddrLow (0x{tbl_mc & 0xFFFFFFFF:x}) -> {v}")

    # Step 8: EnableAllSmuFeatures — SHORT timeout to avoid long hang.
    # The arg is a FEATURE_PWR_DOMAIN_e selector, not a bitmask:
    #   FEATURE_PWR_ALL  = 0   (everything — may require GMC + GFXHUB)
    #   FEATURE_PWR_S5   = 1
    #   FEATURE_PWR_BACO = 2
    #   FEATURE_PWR_SOC  = 3   (SOC-only: DPM for mem, fclk, soc, link, dcn)
    #   FEATURE_PWR_GFX  = 4
    # Try SOC-only first — if that works we can layer on GFX later.
    import os
    feature_arg = int(os.environ.get("SMU_FEATURE_PWR", "3"))
    feature_names = {0:"ALL", 1:"S5", 2:"BACO", 3:"SOC", 4:"GFX"}
    print(f"\n== Step 8: EnableAllSmuFeatures(FEATURE_PWR_{feature_names.get(feature_arg, feature_arg)}) — 3 s timeout ==")
    t0 = time.time()
    v = smu_send(c, PPSMC_MSG_EnableAllSmuFeatures, feature_arg, timeout_ms=3000)
    dt = time.time() - t0
    if v is None:
        print(f"  TIMEOUT after {dt:.2f}s — SMU may have hung. Dumping state...")
        dump_mp1_c2pmsg(c, "post-timeout")
        print("\n  Suggestion: trigger DEXT restart via `sudo kill $(pgrep -f ai.rocm.gpu.driver)`")
        sys.exit(2)
    else:
        print(f"  SUCCESS in {dt*1000:.0f} ms: resp=0x{v[0]:x} arg_out=0x{v[1]:x}")

    # Step 9: Query features
    print("\n== Step 9: Post-enable feature query ==")
    v = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesLo, 0)
    print(f"  RunningFeaturesLow  -> {v}")
    v = smu_send(c, PPSMC_MSG_GetRunningSmuFeaturesHi, 0)
    print(f"  RunningFeaturesHigh -> {v}")

    print("\n== Done ==")


if __name__ == "__main__":
    main()
