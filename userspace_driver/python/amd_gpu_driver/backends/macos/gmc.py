"""MMHUB (system-domain GMC) initialization for gfx1201.

Mirrors tinygrad's `AM_GMC.init_hub("MM")` (runtime/support/am/ip.py)
line-by-line — with two deliberate omissions:

  1. **No VM_CONTEXT0 page table.** Tinygrad builds a multi-level GART
     page table in VRAM and points VM_CONTEXT0 at it. We skip that for
     now: the SMU operates on FB-aperture addresses that never hit a
     VMID page walk, so VMID 0 can stay disabled for the bring-up path
     we care about (SMU's FEATURE_PWR_GFX / FEATURE_PWR_ALL).
  2. **No scratch-page allocator.** Tinygrad's `palloc` carves VRAM
     pages for the default-addr / dummy-page. We just pick VRAM
     offsets near the top of VRAM (below the IP discovery region and
     the SMU driver table).

Status: experimental. Programming MMHUB wrong on gfx1201 tends to
hang the SMU — if this module ever sets a register the ASIC doesn't
like, recovery is unplug/replug the eGPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---- MMHUB base (DWORD) ----
MMHUB_BASE_DW = 0x1A000
_MMIO_BAR = 5


# ---- Register DWORD offsets (mmhub_4_1_0_offset.h) ----

# FB + system aperture
regMMMC_VM_FB_LOCATION_BASE               = 0x0554
regMMMC_VM_FB_LOCATION_TOP                = 0x0555
regMMMC_VM_AGP_TOP                        = 0x0556
regMMMC_VM_AGP_BOT                        = 0x0557
regMMMC_VM_AGP_BASE                       = 0x0558
regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR       = 0x0559
regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR      = 0x055a
regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x04c8
regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x04c9

# TLB + L2
regMMMC_VM_MX_L1_TLB_CNTL                 = 0x055b
regMMVM_L2_CNTL                           = 0x04e4
regMMVM_L2_CNTL2                          = 0x04e5
regMMVM_L2_CNTL3                          = 0x04e6
regMMVM_L2_CNTL4                          = 0x04fd
regMMVM_L2_CNTL5                          = 0x0503
regMMVM_L2_PROTECTION_FAULT_CNTL          = 0x04ec
regMMVM_L2_PROTECTION_FAULT_CNTL2         = 0x04ed
regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x04f2
regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x04f3

# Identity aperture disable (tinygrad sets LOW > HIGH to disable)
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32  = 0x04f7
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32  = 0x04f8
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32 = 0x04f9
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32 = 0x04fa
regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32     = 0x04fb
regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32     = 0x04fc

# Invalidate engine 0 address-range regs; engine N is 2*N DWORDs later.
regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x05ab
regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x05ac


# ---- Precomputed register values (bit masks from mmhub_4_1_0_sh_mask.h) ----

# MC_VM_MX_L1_TLB_CNTL:
#   ENABLE_L1_TLB                 = bit 0       -> 0x00000001
#   SYSTEM_ACCESS_MODE            = bits [4:3]  -> 0x00000018 (= 3)
#   SYSTEM_APERTURE_UNMAPPED_ACCESS = bit 5     -> 0 (fault, per tinygrad)
#   ENABLE_ADVANCED_DRIVER_MODEL  = bit 6       -> 0x00000040
#   MTYPE                         = bits [12:11]-> 0 (MTYPE_UC)
_TLB_CNTL_VAL = 0x00000001 | 0x00000018 | 0x00000040

# MMVM_L2_CNTL (gfx12 → `enable_l2_fragment_processing=0`):
#   ENABLE_L2_CACHE                    = bit 0  -> 0x00000001
#   ENABLE_DEFAULT_PAGE_OUT_TO_SYSTEM  = bit 11 -> 0x00000800
#   L2_PDE0_CACHE_TAG_GENERATION_MODE  = bit 8  -> 0 (default)
#   PDE_FAULT_CLASSIFICATION           = bit 18 -> 0
#   CONTEXT1_IDENTITY_ACCESS_MODE      = bits [20:19] -> 1 << 19 = 0x00080000
#   IDENTITY_MODE_FRAGMENT_SIZE        = bits [25:21] -> 0
_L2_CNTL_VAL = 0x00000001 | 0x00000800 | 0x00080000

# MMVM_L2_CNTL2:
#   INVALIDATE_ALL_L1_TLBS = bit 0 -> 0x00000001
#   INVALIDATE_L2_CACHE    = bit 1 -> 0x00000002
_L2_CNTL2_VAL = 0x00000001 | 0x00000002

# MMVM_L2_CNTL3 (gfx12 / `trans_futher=False`):
#   BANK_SELECT              = bits [5:0]   -> 9
#   L2_CACHE_BIGK_FRAGMENT_SIZE= bits [19:15] -> 6  (shift 15)
#   L2_CACHE_BIGK_ASSOCIATIVITY = bit 20     -> 0x00100000
#   L2_CACHE_4K_ASSOCIATIVITY   = bit 31     -> 0x80000000
_L2_CNTL3_VAL = 9 | (6 << 15) | 0x00100000 | 0x80000000

# MMVM_L2_CNTL4:
#   L2_CACHE_4K_PARTITION_COUNT = bits [5:0] -> 1
_L2_CNTL4_VAL = 1

# MMVM_L2_CNTL5 (GC >= 10.0.0):
#   WALKER_PRIORITY_CLIENT_ID = bits [13:5] -> 0x1FF  (shift 5 -> 0x3FE0)
_L2_CNTL5_VAL = 0x1FF << 5

# MMVM_L2_PROTECTION_FAULT_CNTL2 (we OR this on top of the current value):
#   ACTIVE_PAGE_MIGRATION_PTE_READ_RETRY = bit 18 -> 0x00040000
_L2_PROT_FAULT_CNTL2_RETRY_BIT = 0x00040000


@dataclass
class GmcConfig:
    fb_start_mc:      int   # MC addr where VRAM starts
    fb_end_mc:        int   # MC addr (exclusive) where VRAM ends
    vram_size:        int   # bytes
    memscratch_paddr: int   # VRAM offset (bytes) used for SYSTEM_APERTURE_DEFAULT_ADDR
    dummy_page_paddr: int   # VRAM offset (bytes) used for L2_PROTECTION_FAULT_DEFAULT_ADDR


def _rd(client, dw_off: int) -> int:
    return client.mmio_read32(_MMIO_BAR, (MMHUB_BASE_DW + dw_off) * 4)


def _wr(client, dw_off: int, value: int) -> None:
    client.mmio_write32(_MMIO_BAR, (MMHUB_BASE_DW + dw_off) * 4, value & 0xFFFFFFFF)


def _probe_vram(client) -> tuple[int, int, int]:
    """Return (fb_start_mc, fb_end_mc, vram_size) from MMHUB FB_LOCATION."""
    fb_base = _rd(client, regMMMC_VM_FB_LOCATION_BASE) & 0xFFFFFF
    fb_top  = _rd(client, regMMMC_VM_FB_LOCATION_TOP)  & 0xFFFFFF
    fb_start = fb_base << 24
    fb_end   = (fb_top + 1) << 24
    return fb_start, fb_end, fb_end - fb_start


def init_mmhub(client) -> GmcConfig:
    """Configure MMHUB so SMU features beyond PWR_SOC can be enabled.

    Intended ordering (called after SMU FW load + SetDriverDramAddr +
    EnableAll(PWR_SOC) succeeded). Does NOT program VM_CONTEXT0 page
    tables — VMID 0 stays disabled. That suffices for SMU DMA to
    VRAM-resident buffers; it would not suffice for GFX clients that
    issue page walks.
    """
    fb_start, fb_end, vram_size = _probe_vram(client)

    # Pick scratch-page VRAM offsets (just shy of the SMU driver table
    # and IP discovery region, both parked at the top of VRAM).
    memscratch_paddr = vram_size - 0x30000
    dummy_page_paddr = vram_size - 0x40000

    logger.info("MMHUB init: fb=[0x%x, 0x%x) vram_size=0x%x", fb_start, fb_end, vram_size)
    logger.info("  memscratch_paddr=0x%x  dummy_page_paddr=0x%x",
                memscratch_paddr, dummy_page_paddr)

    # --- AGP: disabled (tinygrad-style — BOT > TOP). ---
    _wr(client, regMMMC_VM_AGP_BASE, 0)
    _wr(client, regMMMC_VM_AGP_BOT,  0xffffffffffff >> 24)  # = 0xFFFFFF
    _wr(client, regMMMC_VM_AGP_TOP,  0)

    # --- SYSTEM_APERTURE covers the FB range (same as vBIOS POST). ---
    _wr(client, regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR,  fb_start >> 18)
    _wr(client, regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR, fb_end   >> 18)

    # --- DEFAULT_ADDR / DEFAULT_FAULT_ADDR (VRAM page numbers). ---
    _wr(client, regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
        (memscratch_paddr >> 12) & 0xFFFFFFFF)
    _wr(client, regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
        (memscratch_paddr >> 12) >> 32)
    _wr(client, regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
        (dummy_page_paddr >> 12) & 0xFFFFFFFF)
    _wr(client, regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
        (dummy_page_paddr >> 12) >> 32)

    # --- L2_PROTECTION_FAULT_CNTL2 |= ACTIVE_PAGE_MIGRATION_PTE_READ_RETRY ---
    v = _rd(client, regMMVM_L2_PROTECTION_FAULT_CNTL2) | _L2_PROT_FAULT_CNTL2_RETRY_BIT
    _wr(client, regMMVM_L2_PROTECTION_FAULT_CNTL2, v)

    # --- TLB + L2 enable ---
    _wr(client, regMMMC_VM_MX_L1_TLB_CNTL, _TLB_CNTL_VAL)
    _wr(client, regMMVM_L2_CNTL,  _L2_CNTL_VAL)
    _wr(client, regMMVM_L2_CNTL2, _L2_CNTL2_VAL)
    _wr(client, regMMVM_L2_CNTL3, _L2_CNTL3_VAL)
    _wr(client, regMMVM_L2_CNTL4, _L2_CNTL4_VAL)
    _wr(client, regMMVM_L2_CNTL5, _L2_CNTL5_VAL)

    # --- Disable identity aperture (tinygrad-style LOW > HIGH). ---
    _wr(client, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32,  0xFFFFFFFF)
    _wr(client, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32,  0xF)
    _wr(client, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0)
    _wr(client, regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0)
    _wr(client, regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32,     0)
    _wr(client, regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32,     0)

    # --- Invalidate-engine address-range regs. tinygrad writes 0x1fffffffff
    # to all 18 engines' LO32+HI32 pair; 0x1fffffffff = 37-bit mask. ---
    for i in range(18):
        lo = regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + 2 * i
        hi = regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 + 2 * i
        _wr(client, lo, 0xFFFFFFFF)
        _wr(client, hi, 0x1F)

    return GmcConfig(
        fb_start_mc=fb_start,
        fb_end_mc=fb_end,
        vram_size=vram_size,
        memscratch_paddr=memscratch_paddr,
        dummy_page_paddr=dummy_page_paddr,
    )
