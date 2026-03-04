"""GMC v12.0 initialization for RDNA4 (Navi 48 / GFX1201).

Programs GPU memory controller registers needed for GART and VM setup:
- MMHUB: Memory-mapped HUB for VRAM/system memory access paths
- GFXHUB: Graphics/Compute HUB for GFX/compute VM translation
- Page table base, system aperture, L2 cache, TLB

Register offsets are SOC15-style (base_index + offset) resolved using
IP discovery base addresses.

Reference: Linux amdgpu gmc_v12_0.c, mmhub_v4_1_0.c, gfxhub_v12_0.c
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult
    from amd_gpu_driver.backends.windows.nbio_init import NBIOConfig


# ============================================================================
# MMHUB v4.1.0 register offsets (base_index 0 in MMHUB IP space)
# These are DWORD offsets — multiply by 4 for byte offset from MMHUB base.
# ============================================================================

# FB location
regMMMC_VM_FB_OFFSET = 0x04C7
regMMMC_VM_FB_LOCATION_BASE = 0x0554
regMMMC_VM_FB_LOCATION_TOP = 0x0555

# AGP aperture
regMMMC_VM_AGP_TOP = 0x0556
regMMMC_VM_AGP_BOT = 0x0557
regMMMC_VM_AGP_BASE = 0x0558

# System aperture
regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x0559
regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x055A
regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x04C8
regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x04C9

# L1 TLB
regMMMC_VM_MX_L1_TLB_CNTL = 0x055B

# L2 cache
regMMVM_L2_CNTL = 0x04E4
regMMVM_L2_CNTL2 = 0x04E5
regMMVM_L2_CNTL3 = 0x04E6
regMMVM_L2_CNTL4 = 0x04FD
regMMVM_L2_CNTL5 = 0x0503

# Protection fault
regMMVM_L2_PROTECTION_FAULT_CNTL = 0x04EC
regMMVM_L2_PROTECTION_FAULT_CNTL2 = 0x04ED
regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x04F4
regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x04F5
regMMVM_L2_PROTECTION_FAULT_STATUS_LO32 = 0x04F0

# Identity aperture
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32 = 0x04F7
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32 = 0x04F8
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32 = 0x04F9
regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32 = 0x04FA
regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32 = 0x04FB
regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32 = 0x04FC

# Private invalidation
regMMVM_L2_BANK_SELECT_RESERVED_CID2 = 0x0500

# Context control
regMMVM_CONTEXT0_CNTL = 0x0564
regMMVM_CONTEXT1_CNTL = 0x0565  # VMIDs 1-15

# Page table base
regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x05CF
regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x05D0
regMMVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32 = 0x05D1

# Page table start/end
regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x05EF
regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x05F0
regMMVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x05F1
regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x060F
regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x0610
regMMVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x0611

# Invalidation engines (18 engines, stride=1 per engine)
regMMVM_INVALIDATE_ENG0_SEM = 0x0575
regMMVM_INVALIDATE_ENG0_REQ = 0x0587
regMMVM_INVALIDATE_ENG0_ACK = 0x0599
regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x05AB
regMMVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x05AC

# Spacing
MMHUB_CTX_DISTANCE = 1       # Between CONTEXT0_CNTL and CONTEXT1_CNTL
MMHUB_CTX_ADDR_DISTANCE = 2  # Between CONTEXT0/CONTEXT1 page table base
MMHUB_ENG_DISTANCE = 1       # Between ENG0_REQ and ENG1_REQ
MMHUB_ENG_ADDR_DISTANCE = 2  # Between ENG0/ENG1 addr range


# ============================================================================
# GFXHUB v12.0 register offsets (GC base_index 0)
# ============================================================================

regGCMC_VM_FB_OFFSET = 0x15A7
regGCMC_VM_FB_LOCATION_BASE = 0x1614
regGCMC_VM_FB_LOCATION_TOP = 0x1615

regGCMC_VM_AGP_TOP = 0x1616
regGCMC_VM_AGP_BOT = 0x1617
regGCMC_VM_AGP_BASE = 0x1618

regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR = 0x1619
regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR = 0x161A
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB = 0x15A8
regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB = 0x15A9

regGCMC_VM_MX_L1_TLB_CNTL = 0x161B

regGCVM_L2_CNTL = 0x15C4
regGCVM_L2_CNTL2 = 0x15C5
regGCVM_L2_CNTL3 = 0x15C6
regGCVM_L2_CNTL4 = 0x15DD
regGCVM_L2_CNTL5 = 0x15E3

regGCVM_L2_PROTECTION_FAULT_CNTL = 0x15CC
regGCVM_L2_PROTECTION_FAULT_CNTL2 = 0x15CD
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32 = 0x15D4
regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32 = 0x15D5

regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32 = 0x15D7
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32 = 0x15D8
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32 = 0x15D9
regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32 = 0x15DA
regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32 = 0x15DB
regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32 = 0x15DC

regGCVM_CONTEXT0_CNTL = 0x1624
regGCVM_CONTEXT1_CNTL = 0x1625

regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32 = 0x168F
regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32 = 0x1690
regGCVM_CONTEXT1_PAGE_TABLE_BASE_ADDR_LO32 = 0x1691

regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32 = 0x16AF
regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32 = 0x16B0
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32 = 0x16CF
regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32 = 0x16D0
regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 = 0x16B1
regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 = 0x16D1

regGCVM_INVALIDATE_ENG0_SEM = 0x1635
regGCVM_INVALIDATE_ENG0_REQ = 0x1647
regGCVM_INVALIDATE_ENG0_ACK = 0x1659
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 = 0x166B
regGCVM_INVALIDATE_ENG0_ADDR_RANGE_HI32 = 0x166C

GFXHUB_CTX_DISTANCE = 1
GFXHUB_CTX_ADDR_DISTANCE = 2
GFXHUB_ENG_DISTANCE = 1
GFXHUB_ENG_ADDR_DISTANCE = 2

# CP debug (disable UTCL1 error halt for GFXHUB)
regCP_DEBUG = 0x1E1F


# ============================================================================
# GFX12 PTE/PDE flag definitions
# ============================================================================

AMDGPU_PTE_VALID = 1 << 0
AMDGPU_PTE_SYSTEM = 1 << 1
AMDGPU_PTE_SNOOPED = 1 << 2
AMDGPU_PTE_TMZ = 1 << 3
AMDGPU_PTE_EXECUTABLE = 1 << 4
AMDGPU_PTE_READABLE = 1 << 5
AMDGPU_PTE_WRITEABLE = 1 << 6
AMDGPU_PTE_IS_PTE = 1 << 63  # GFX12: marks entry as PTE (not PDE)

# MTYPE encoding for GFX12 (bits 55:54)
MTYPE_NC = 0  # Non-Coherent / Normal Cacheable
MTYPE_WC = 1  # Write Combine
MTYPE_CC = 2  # Coherently Cacheable
MTYPE_UC = 3  # Uncacheable


def gfx12_pte_mtype(mtype: int) -> int:
    """Encode MTYPE for GFX12 PTE (bits 55:54)."""
    return (mtype & 0x3) << 54


def gfx12_pte_fragment(frag: int) -> int:
    """Encode fragment size for GFX12 PTE (bits 11:7)."""
    return (frag & 0x1F) << 7


# L1 TLB control bits
L1_TLB_ENABLE = 1 << 0
L1_TLB_SYSTEM_ACCESS_MODE_MASK = 0x3 << 3  # bits 4:3
L1_TLB_ENABLE_ADV_DRIVER_MODEL = 1 << 6
L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS = 1 << 5

# VM context control bits
VM_CONTEXT_ENABLE_CONTEXT = 1 << 0
VM_CONTEXT_PAGE_TABLE_DEPTH_MASK = 0x3 << 1  # bits 2:1
VM_CONTEXT_RETRY_FAULT_ENABLE = 1 << 25


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class GMCConfig:
    """Resolved GMC configuration from IP discovery and hardware reads."""
    # Base addresses (DWORD offsets from IP discovery)
    mmhub_base: list[int]  # MMHUB base_index 0..N
    gfxhub_base: list[int]  # GC base_index 0..N

    # VRAM layout
    vram_start: int = 0    # GPU MC address of VRAM start
    vram_end: int = 0      # GPU MC address of VRAM end
    vram_size: int = 0     # VRAM size in bytes

    # GART (Graphics Address Remapping Table)
    gart_start: int = 0    # GPU MC address of GART start
    gart_end: int = 0      # GPU MC address of GART end
    gart_size: int = 512 * 1024 * 1024  # Default 512MB

    # GART page table physical address (in DMA memory)
    gart_table_bus_addr: int = 0

    # FB offset in MC address space
    fb_offset: int = 0

    # System aperture
    agp_start: int = 0
    agp_end: int = 0

    # Dummy page for fault handling
    dummy_page_bus_addr: int = 0


# ============================================================================
# Register access helpers
# ============================================================================

def _mmhub_reg(dev: WindowsDevice, config: GMCConfig, reg: int) -> int:
    """Read an MMHUB register via SOC15 addressing."""
    offset = (config.mmhub_base[0] + reg) * 4
    return dev.read_reg32(offset)


def _mmhub_wreg(dev: WindowsDevice, config: GMCConfig, reg: int, val: int) -> None:
    """Write an MMHUB register."""
    offset = (config.mmhub_base[0] + reg) * 4
    dev.write_reg32(offset, val)


def _gfxhub_reg(dev: WindowsDevice, config: GMCConfig, reg: int) -> int:
    """Read a GFXHUB register via SOC15 addressing."""
    offset = (config.gfxhub_base[0] + reg) * 4
    return dev.read_reg32(offset)


def _gfxhub_wreg(dev: WindowsDevice, config: GMCConfig, reg: int, val: int) -> None:
    """Write a GFXHUB register."""
    offset = (config.gfxhub_base[0] + reg) * 4
    dev.write_reg32(offset, val)


# ============================================================================
# MMHUB initialization (following mmhub_v4_1_0.c)
# ============================================================================

def _mmhub_init_gart_aperture(
    dev: WindowsDevice, config: GMCConfig
) -> None:
    """Program GART page table base for VMID 0 on MMHUB.

    Reference: mmhub_v4_1_0_init_gart_aperture_regs()
    """
    # Page table base = GART table bus address | VALID flag
    pt_base = config.gart_table_bus_addr | AMDGPU_PTE_VALID

    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
                pt_base & 0xFFFFFFFF)
    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
                (pt_base >> 32) & 0xFFFFFFFF)

    # GART aperture range (addresses are >> 12 for 4KB pages)
    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32,
                (config.gart_start >> 12) & 0xFFFFFFFF)
    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32,
                (config.gart_start >> 44) & 0xFFFFFFFF)
    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
                (config.gart_end >> 12) & 0xFFFFFFFF)
    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
                (config.gart_end >> 44) & 0xFFFFFFFF)


def _mmhub_init_system_aperture(
    dev: WindowsDevice, config: GMCConfig
) -> None:
    """Program system aperture registers on MMHUB.

    Reference: mmhub_v4_1_0_init_system_aperture_regs()
    """
    # AGP aperture (disabled by setting bot > top)
    _mmhub_wreg(dev, config, regMMMC_VM_AGP_BASE, 0)
    _mmhub_wreg(dev, config, regMMMC_VM_AGP_BOT, config.agp_start >> 24)
    _mmhub_wreg(dev, config, regMMMC_VM_AGP_TOP, config.agp_end >> 24)

    # System aperture bounds
    fb_start = config.vram_start
    fb_end = config.vram_end
    _mmhub_wreg(dev, config, regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR,
                fb_start >> 18)
    _mmhub_wreg(dev, config, regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR,
                fb_end >> 18)

    # Default address for unmapped access (point to dummy page)
    _mmhub_wreg(dev, config, regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
                config.dummy_page_bus_addr >> 12)
    _mmhub_wreg(dev, config, regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
                config.dummy_page_bus_addr >> 44)

    # Fault default address
    _mmhub_wreg(dev, config, regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
                config.dummy_page_bus_addr >> 12)
    _mmhub_wreg(dev, config, regMMVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
                config.dummy_page_bus_addr >> 44)


def _mmhub_init_tlb(dev: WindowsDevice, config: GMCConfig) -> None:
    """Program L1 TLB registers on MMHUB.

    Reference: mmhub_v4_1_0_init_tlb_regs()
    """
    val = _mmhub_reg(dev, config, regMMMC_VM_MX_L1_TLB_CNTL)

    # Enable L1 TLB
    val |= L1_TLB_ENABLE
    # System access mode = 3 (full VM translation)
    val = (val & ~L1_TLB_SYSTEM_ACCESS_MODE_MASK) | (3 << 3)
    # Enable advanced driver model
    val |= L1_TLB_ENABLE_ADV_DRIVER_MODEL
    # Unmapped access = 0
    val &= ~L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS

    _mmhub_wreg(dev, config, regMMMC_VM_MX_L1_TLB_CNTL, val)


def _mmhub_init_cache(dev: WindowsDevice, config: GMCConfig) -> None:
    """Program L2 cache registers on MMHUB.

    Reference: mmhub_v4_1_0_init_cache_regs()
    """
    # L2 control: enable cache, default page to system memory
    val = _mmhub_reg(dev, config, regMMVM_L2_CNTL)
    val |= (1 << 0)   # ENABLE_L2_CACHE
    val |= (1 << 8)   # ENABLE_DEFAULT_PAGE_OUT_TO_SYSTEM_MEMORY
    val &= ~(1 << 6)  # Disable ENABLE_L2_FRAGMENT_PROCESSING
    _mmhub_wreg(dev, config, regMMVM_L2_CNTL, val)

    # L2 control 2: invalidate all TLBs and cache
    _mmhub_wreg(dev, config, regMMVM_L2_CNTL2,
                (1 << 0) |  # INVALIDATE_ALL_L1_TLBS
                (1 << 1))   # INVALIDATE_L2_CACHE

    # L2 control 3: bank select and fragment size
    val = _mmhub_reg(dev, config, regMMVM_L2_CNTL3)
    # BANK_SELECT = 9, L2_CACHE_BIGK_FRAGMENT_SIZE = 6
    val = (val & ~0x3F000) | (9 << 15)   # BANK_SELECT
    val = (val & ~0x1F00000) | (6 << 20)  # BIGK_FRAGMENT_SIZE
    _mmhub_wreg(dev, config, regMMVM_L2_CNTL3, val)

    # L2 control 5: small-K fragment size = 0
    val = _mmhub_reg(dev, config, regMMVM_L2_CNTL5)
    val &= ~0x7E0   # Clear L2_CACHE_SMALLK_FRAGMENT_SIZE field
    _mmhub_wreg(dev, config, regMMVM_L2_CNTL5, val)


def _mmhub_enable_system_domain(dev: WindowsDevice, config: GMCConfig) -> None:
    """Enable VMID 0 context (system domain) on MMHUB.

    VMID 0 uses PAGE_TABLE_DEPTH=0 (flat/direct GART translation).

    Reference: mmhub_v4_1_0_enable_system_domain()
    """
    _mmhub_wreg(dev, config, regMMVM_CONTEXT0_CNTL,
                VM_CONTEXT_ENABLE_CONTEXT)  # depth=0, retry=0


def _mmhub_disable_identity_aperture(
    dev: WindowsDevice, config: GMCConfig
) -> None:
    """Disable identity aperture by setting low > high.

    Reference: mmhub_v4_1_0_disable_identity_aperture()
    """
    # Low = max value (0xF_FFFFFFFF)
    _mmhub_wreg(dev, config,
                regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32, 0xFFFFFFFF)
    _mmhub_wreg(dev, config,
                regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32, 0x0000000F)
    # High = 0
    _mmhub_wreg(dev, config,
                regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0)
    _mmhub_wreg(dev, config,
                regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0)
    # Physical offset = 0
    _mmhub_wreg(dev, config,
                regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0)
    _mmhub_wreg(dev, config,
                regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0)


def _mmhub_setup_vmid_config(
    dev: WindowsDevice, config: GMCConfig,
    num_level: int = 3, block_size: int = 9
) -> None:
    """Configure VMIDs 1-15 for 4-level page table translation.

    Reference: mmhub_v4_1_0_setup_vmid_config()
    """
    # 48-bit address space max PFN
    max_pfn = (1 << 36) - 1  # 48-bit addr >> 12

    for vmid in range(1, 16):
        # Context control: enable, set page table depth, enable fault protection
        val = VM_CONTEXT_ENABLE_CONTEXT
        val |= (num_level & 0x3) << 1  # PAGE_TABLE_DEPTH
        val |= (1 << 7)   # RANGE_PROTECTION_FAULT_ENABLE_DEFAULT
        val |= (1 << 8)   # DUMMY_PAGE_PROTECTION_FAULT_ENABLE_DEFAULT
        val |= (1 << 9)   # PDE0_PROTECTION_FAULT_ENABLE_DEFAULT
        val |= (1 << 10)  # VALID_PROTECTION_FAULT_ENABLE_DEFAULT
        val |= (1 << 11)  # READ_PROTECTION_FAULT_ENABLE_DEFAULT
        val |= (1 << 12)  # WRITE_PROTECTION_FAULT_ENABLE_DEFAULT
        val |= (1 << 13)  # EXECUTE_PROTECTION_FAULT_ENABLE_DEFAULT
        # PAGE_TABLE_BLOCK_SIZE = block_size - 9
        val |= ((block_size - 9) & 0xF) << 24

        ctx_reg = regMMVM_CONTEXT1_CNTL + (vmid - 1) * MMHUB_CTX_DISTANCE
        _mmhub_wreg(dev, config, ctx_reg, val)

        # Page table start = 0
        start_lo = regMMVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 + \
            (vmid - 1) * MMHUB_CTX_ADDR_DISTANCE
        _mmhub_wreg(dev, config, start_lo, 0)
        _mmhub_wreg(dev, config, start_lo + 1, 0)

        # Page table end = max
        end_lo = regMMVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 + \
            (vmid - 1) * MMHUB_CTX_ADDR_DISTANCE
        _mmhub_wreg(dev, config, end_lo, max_pfn & 0xFFFFFFFF)
        _mmhub_wreg(dev, config, end_lo + 1, (max_pfn >> 32) & 0xFFFFFFFF)


def _mmhub_program_invalidation(dev: WindowsDevice, config: GMCConfig) -> None:
    """Program invalidation engine address ranges on MMHUB.

    Reference: mmhub_v4_1_0_program_invalidation()
    """
    for eng in range(18):
        lo_reg = regMMVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + \
            eng * MMHUB_ENG_ADDR_DISTANCE
        _mmhub_wreg(dev, config, lo_reg, 0xFFFFFFFF)
        _mmhub_wreg(dev, config, lo_reg + 1, 0x1F)


def mmhub_gart_enable(dev: WindowsDevice, config: GMCConfig) -> None:
    """Full MMHUB GART enable sequence.

    Reference: mmhub_v4_1_0_gart_enable()
    """
    _mmhub_init_gart_aperture(dev, config)
    _mmhub_init_system_aperture(dev, config)
    _mmhub_init_tlb(dev, config)
    _mmhub_init_cache(dev, config)
    _mmhub_enable_system_domain(dev, config)
    _mmhub_disable_identity_aperture(dev, config)
    _mmhub_setup_vmid_config(dev, config)
    _mmhub_program_invalidation(dev, config)


# ============================================================================
# GFXHUB initialization (following gfxhub_v12_0.c)
# Same structure as MMHUB but different register prefix.
# ============================================================================

def _gfxhub_init_gart_aperture(
    dev: WindowsDevice, config: GMCConfig
) -> None:
    """Program GART page table base for VMID 0 on GFXHUB."""
    pt_base = config.gart_table_bus_addr | AMDGPU_PTE_VALID

    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32,
                 pt_base & 0xFFFFFFFF)
    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32,
                 (pt_base >> 32) & 0xFFFFFFFF)

    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32,
                 (config.gart_start >> 12) & 0xFFFFFFFF)
    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32,
                 (config.gart_start >> 44) & 0xFFFFFFFF)
    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32,
                 (config.gart_end >> 12) & 0xFFFFFFFF)
    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32,
                 (config.gart_end >> 44) & 0xFFFFFFFF)


def _gfxhub_init_system_aperture(
    dev: WindowsDevice, config: GMCConfig
) -> None:
    """Program system aperture on GFXHUB."""
    _gfxhub_wreg(dev, config, regGCMC_VM_AGP_BASE, 0)
    _gfxhub_wreg(dev, config, regGCMC_VM_AGP_BOT, config.agp_start >> 24)
    _gfxhub_wreg(dev, config, regGCMC_VM_AGP_TOP, config.agp_end >> 24)

    _gfxhub_wreg(dev, config, regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR,
                 config.vram_start >> 18)
    _gfxhub_wreg(dev, config, regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR,
                 config.vram_end >> 18)

    _gfxhub_wreg(dev, config, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB,
                 config.dummy_page_bus_addr >> 12)
    _gfxhub_wreg(dev, config, regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB,
                 config.dummy_page_bus_addr >> 44)

    _gfxhub_wreg(dev, config, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_LO32,
                 config.dummy_page_bus_addr >> 12)
    _gfxhub_wreg(dev, config, regGCVM_L2_PROTECTION_FAULT_DEFAULT_ADDR_HI32,
                 config.dummy_page_bus_addr >> 44)


def _gfxhub_init_tlb(dev: WindowsDevice, config: GMCConfig) -> None:
    """Program L1 TLB on GFXHUB."""
    val = _gfxhub_reg(dev, config, regGCMC_VM_MX_L1_TLB_CNTL)
    val |= L1_TLB_ENABLE
    val = (val & ~L1_TLB_SYSTEM_ACCESS_MODE_MASK) | (3 << 3)
    val |= L1_TLB_ENABLE_ADV_DRIVER_MODEL
    val &= ~L1_TLB_SYSTEM_APERTURE_UNMAPPED_ACCESS
    _gfxhub_wreg(dev, config, regGCMC_VM_MX_L1_TLB_CNTL, val)


def _gfxhub_init_cache(dev: WindowsDevice, config: GMCConfig) -> None:
    """Program L2 cache on GFXHUB."""
    val = _gfxhub_reg(dev, config, regGCVM_L2_CNTL)
    val |= (1 << 0)
    val |= (1 << 8)
    val &= ~(1 << 6)
    _gfxhub_wreg(dev, config, regGCVM_L2_CNTL, val)

    _gfxhub_wreg(dev, config, regGCVM_L2_CNTL2,
                 (1 << 0) | (1 << 1))

    val = _gfxhub_reg(dev, config, regGCVM_L2_CNTL3)
    val = (val & ~0x3F000) | (9 << 15)
    val = (val & ~0x1F00000) | (6 << 20)
    _gfxhub_wreg(dev, config, regGCVM_L2_CNTL3, val)

    val = _gfxhub_reg(dev, config, regGCVM_L2_CNTL5)
    val &= ~0x7E0
    _gfxhub_wreg(dev, config, regGCVM_L2_CNTL5, val)


def _gfxhub_enable_system_domain(dev: WindowsDevice, config: GMCConfig) -> None:
    """Enable VMID 0 on GFXHUB."""
    _gfxhub_wreg(dev, config, regGCVM_CONTEXT0_CNTL,
                 VM_CONTEXT_ENABLE_CONTEXT)


def _gfxhub_disable_identity_aperture(
    dev: WindowsDevice, config: GMCConfig
) -> None:
    """Disable identity aperture on GFXHUB."""
    _gfxhub_wreg(dev, config,
                 regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_LO32, 0xFFFFFFFF)
    _gfxhub_wreg(dev, config,
                 regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR_HI32, 0x0000000F)
    _gfxhub_wreg(dev, config,
                 regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_LO32, 0)
    _gfxhub_wreg(dev, config,
                 regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR_HI32, 0)
    _gfxhub_wreg(dev, config,
                 regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_LO32, 0)
    _gfxhub_wreg(dev, config,
                 regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET_HI32, 0)


def _gfxhub_setup_vmid_config(
    dev: WindowsDevice, config: GMCConfig,
    num_level: int = 3, block_size: int = 9
) -> None:
    """Configure VMIDs 1-15 on GFXHUB."""
    max_pfn = (1 << 36) - 1

    for vmid in range(1, 16):
        val = VM_CONTEXT_ENABLE_CONTEXT
        val |= (num_level & 0x3) << 1
        val |= (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10)
        val |= (1 << 11) | (1 << 12) | (1 << 13)
        val |= ((block_size - 9) & 0xF) << 24

        ctx_reg = regGCVM_CONTEXT1_CNTL + (vmid - 1) * GFXHUB_CTX_DISTANCE
        _gfxhub_wreg(dev, config, ctx_reg, val)

        start_lo = regGCVM_CONTEXT1_PAGE_TABLE_START_ADDR_LO32 + \
            (vmid - 1) * GFXHUB_CTX_ADDR_DISTANCE
        _gfxhub_wreg(dev, config, start_lo, 0)
        _gfxhub_wreg(dev, config, start_lo + 1, 0)

        end_lo = regGCVM_CONTEXT1_PAGE_TABLE_END_ADDR_LO32 + \
            (vmid - 1) * GFXHUB_CTX_ADDR_DISTANCE
        _gfxhub_wreg(dev, config, end_lo, max_pfn & 0xFFFFFFFF)
        _gfxhub_wreg(dev, config, end_lo + 1, (max_pfn >> 32) & 0xFFFFFFFF)


def _gfxhub_program_invalidation(dev: WindowsDevice, config: GMCConfig) -> None:
    """Program invalidation engines on GFXHUB."""
    for eng in range(18):
        lo_reg = regGCVM_INVALIDATE_ENG0_ADDR_RANGE_LO32 + \
            eng * GFXHUB_ENG_ADDR_DISTANCE
        _gfxhub_wreg(dev, config, lo_reg, 0xFFFFFFFF)
        _gfxhub_wreg(dev, config, lo_reg + 1, 0x1F)


def gfxhub_gart_enable(dev: WindowsDevice, config: GMCConfig) -> None:
    """Full GFXHUB GART enable sequence.

    Reference: gfxhub_v12_0_gart_enable()
    """
    _gfxhub_init_gart_aperture(dev, config)
    _gfxhub_init_system_aperture(dev, config)
    _gfxhub_init_tlb(dev, config)
    _gfxhub_init_cache(dev, config)
    _gfxhub_enable_system_domain(dev, config)
    _gfxhub_disable_identity_aperture(dev, config)
    _gfxhub_setup_vmid_config(dev, config)
    _gfxhub_program_invalidation(dev, config)

    # Disable CP UTCL1 error halt (GFXHUB-specific)
    val = _gfxhub_reg(dev, config, regCP_DEBUG)
    val |= (1 << 15)  # CPG_UTCL1_ERROR_HALT_DISABLE
    _gfxhub_wreg(dev, config, regCP_DEBUG, val)


# ============================================================================
# TLB invalidation
# ============================================================================

def flush_gpu_tlb(
    dev: WindowsDevice,
    config: GMCConfig,
    vmid: int = 0,
    hub: str = "mmhub",
) -> None:
    """Invalidate GPU TLB for a specific VMID.

    Reference: gmc_v12_0_flush_gpu_tlb()
    """
    if hub == "mmhub":
        req_reg = regMMVM_INVALIDATE_ENG0_REQ
        ack_reg = regMMVM_INVALIDATE_ENG0_ACK
        sem_reg = regMMVM_INVALIDATE_ENG0_SEM
        read_fn = _mmhub_reg
        write_fn = _mmhub_wreg
    else:
        req_reg = regGCVM_INVALIDATE_ENG0_REQ
        ack_reg = regGCVM_INVALIDATE_ENG0_ACK
        sem_reg = regGCVM_INVALIDATE_ENG0_SEM
        read_fn = _gfxhub_reg
        write_fn = _gfxhub_wreg

    # Use engine 0 for invalidation
    # Acquire semaphore
    for _ in range(10):
        val = read_fn(dev, config, sem_reg)
        if val & 0x1:
            break
        write_fn(dev, config, sem_reg, 1)

    # Request invalidation: all types, for the given VMID
    req = (1 << 0)  # PER_VMID_INVALIDATE_REQ
    req |= (vmid & 0xF) << 16  # INVALIDATE_ENG_VMID
    write_fn(dev, config, req_reg, req)

    # Poll for completion
    for _ in range(100):
        ack = read_fn(dev, config, ack_reg)
        if ack & (1 << vmid):
            break

    # Release semaphore
    write_fn(dev, config, sem_reg, 0)


# ============================================================================
# GART table management
# ============================================================================

def build_gart_pte(bus_addr: int, system: bool = True) -> int:
    """Build a GART PTE for a system memory page.

    Returns a 64-bit PTE value for the given bus address.
    GFX12 GART uses UC mtype, executable, is_pte flag.
    """
    pte = AMDGPU_PTE_VALID
    pte |= AMDGPU_PTE_READABLE
    pte |= AMDGPU_PTE_WRITEABLE
    pte |= AMDGPU_PTE_EXECUTABLE
    pte |= AMDGPU_PTE_IS_PTE
    pte |= gfx12_pte_mtype(MTYPE_UC)
    if system:
        pte |= AMDGPU_PTE_SYSTEM
        pte |= AMDGPU_PTE_SNOOPED
    # Physical page address (bits 47:12)
    pte |= (bus_addr & 0x0000FFFFF000)
    return pte


# ============================================================================
# Top-level GMC initialization
# ============================================================================

def resolve_gmc_bases(ip_result: IPDiscoveryResult) -> GMCConfig:
    """Resolve MMHUB and GFXHUB base addresses from IP discovery.

    MMHUB uses HW_ID 34, GFXHUB is part of GC (HW_ID 11).
    """
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    mmhub_bases = [0] * 6
    gfxhub_bases = [0] * 6

    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.MMHUB and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(mmhub_bases) and addr != 0:
                    mmhub_bases[i] = addr

        if block.hw_id == HardwareID.GC and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(gfxhub_bases) and addr != 0:
                    gfxhub_bases[i] = addr

    return GMCConfig(mmhub_base=mmhub_bases, gfxhub_base=gfxhub_bases)


def init_gmc(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
    vram_size_bytes: int,
    gart_table_bus_addr: int,
    dummy_page_bus_addr: int,
) -> GMCConfig:
    """Full GMC initialization sequence for RDNA4.

    Follows the Linux amdgpu gmc_v12_0_hw_init() -> gart_enable() path:
    1. Resolve MMHUB and GFXHUB base addresses from IP discovery
    2. Compute VRAM and GART address layout
    3. Enable MMHUB GART (system domain + VMID config)
    4. Flush HDP and GPU TLB

    GFXHUB GART is enabled separately (typically by GFX IP init).

    Args:
        dev: Windows device backend.
        ip_result: Parsed IP discovery data.
        nbio_config: NBIO configuration (for HDP flush).
        vram_size_bytes: VRAM size in bytes.
        gart_table_bus_addr: Bus address of the GART page table (from DMA alloc).
        dummy_page_bus_addr: Bus address of a dummy page for fault handling.

    Returns:
        Configured GMCConfig for use by other modules.
    """
    from amd_gpu_driver.backends.windows.nbio_init import hdp_flush

    config = resolve_gmc_bases(ip_result)

    # Read FB location from MMHUB registers
    fb_base_reg = _mmhub_reg(dev, config, regMMMC_VM_FB_LOCATION_BASE)
    fb_top_reg = _mmhub_reg(dev, config, regMMMC_VM_FB_LOCATION_TOP)

    # FB location registers are >> 24
    config.vram_start = fb_base_reg << 24
    config.vram_end = (fb_top_reg << 24) | 0xFFFFFF  # Inclusive end
    config.vram_size = vram_size_bytes

    # If VBIOS didn't set FB location, use a reasonable default
    if config.vram_start == 0 and config.vram_end == 0:
        config.vram_start = 0
        config.vram_end = vram_size_bytes - 1

    # GART sits after VRAM in GPU MC address space
    config.gart_start = config.vram_end + 1
    config.gart_end = config.gart_start + config.gart_size - 1

    # AGP disabled (bot > top)
    config.agp_start = config.gart_end + 1
    config.agp_end = config.agp_start

    config.gart_table_bus_addr = gart_table_bus_addr
    config.dummy_page_bus_addr = dummy_page_bus_addr

    print(f"  GMC: VRAM 0x{config.vram_start:012X}-0x{config.vram_end:012X} "
          f"({vram_size_bytes // (1024*1024)}MB)")
    print(f"  GMC: GART 0x{config.gart_start:012X}-0x{config.gart_end:012X} "
          f"({config.gart_size // (1024*1024)}MB)")

    # Enable MMHUB GART
    mmhub_gart_enable(dev, config)

    # HDP flush
    hdp_flush(dev, nbio_config)

    # Flush MMHUB TLB for VMID 0
    flush_gpu_tlb(dev, config, vmid=0, hub="mmhub")

    print("  GMC: MMHUB GART enabled")
    return config
