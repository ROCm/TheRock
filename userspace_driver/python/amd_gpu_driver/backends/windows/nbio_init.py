"""NBIO v7.11 initialization for RDNA4 (Navi 48 / GFX1201).

Programs NBIO registers needed for GPU bring-up:
- Doorbell aperture enable
- Selfring doorbell aperture
- SDMA/IH doorbell ranges
- Interrupt control

Register offsets are SOC15-style (base_index + offset) resolved using
IP discovery base addresses. Access is via BAR0 MMIO escape commands.

Reference: Linux amdgpu nbio_v7_11.c
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult


# ── NBIO v7.11 register offsets (SOC15 index, NOT byte offsets) ──────

# Base index 2 registers (NBIO address space)
regRCC_DOORBELL_APER_EN = 0x00C0         # Doorbell aperture enable
regRCC_CONFIG_MEMSIZE = 0x00C3           # VRAM size in MB
regBIF_FB_EN = 0x0100                    # Frame buffer enable
regINTERRUPT_CNTL = 0x00F1              # Interrupt control
regINTERRUPT_CNTL2 = 0x00F2             # Interrupt control 2 (dummy page)
regDOORBELL_SELFRING_BASE_HIGH = 0x00F3  # Selfring doorbell base (upper)
regDOORBELL_SELFRING_BASE_LOW = 0x00F4   # Selfring doorbell base (lower)
regDOORBELL_SELFRING_CNTL = 0x00F5       # Selfring doorbell control
regHDP_MEM_COHERENCY_FLUSH = 0x00F7      # HDP coherency flush
regHDP_FLUSH_REQ = 0x0106                # HDP flush request
regHDP_FLUSH_DONE = 0x0107               # HDP flush done
regREMAP_HDP_MEM_FLUSH = 0x012D          # HDP remap (memory flush)
regREMAP_HDP_REG_FLUSH = 0x012E          # HDP remap (register flush)

# Base index 0 registers (PCIe config space)
regPCIE_INDEX2 = 0x000E  # PCIe index port
regPCIE_DATA2 = 0x000F   # PCIe data port

# Bit masks
BIF_DOORBELL_APER_EN__BIT = 0x00000001
BIF_FB_EN__FB_READ_EN = 0x00000001
BIF_FB_EN__FB_WRITE_EN = 0x00000002
SELFRING_EN__BIT = 0x00000001
SELFRING_MODE__BIT = 0x00000002
SELFRING_SIZE__MASK = 0x000000F0

# Interrupt control bits
IH_DUMMY_RD_OVERRIDE = 0x00000001
IH_REQ_NONSNOOP_EN = 0x00000008

# MMIO register hole for HDP remap
MMIO_REG_HOLE_OFFSET = 0x44000


@dataclass
class NBIOConfig:
    """Resolved NBIO register base addresses from IP discovery."""
    # Base addresses per base_index (in DWORD units, multiply by 4 for byte offset)
    base: list[int]  # base_index 0..N
    doorbell_phys_addr: int = 0  # Physical address of doorbell BAR
    doorbell_size: int = 0       # Size of doorbell aperture


def resolve_nbio_bases(ip_result: IPDiscoveryResult) -> NBIOConfig:
    """Resolve NBIO base addresses from IP discovery.

    NBIO uses base_index 0-5. The base addresses come from
    the IP discovery table's ip_v3/ip_v4 base_address array.
    """
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    bases = [0] * 6  # 6 possible base indexes

    for block in ip_result.ip_blocks:
        # NBIO is OSSSYS (HW_ID 40) in the IP discovery table
        # But some versions report as separate NBIO entries
        # For SOC21, NBIO base addresses come from the discovery table
        if block.hw_id == HardwareID.OSSSYS and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(bases) and addr != 0:
                    bases[i] = addr

    return NBIOConfig(base=bases)


def mmio_offset(config: NBIOConfig, base_index: int, reg: int) -> int:
    """Calculate MMIO byte offset for a SOC15 register.

    SOC15 registers are addressed as:
        byte_offset = (base_addr[base_index] + reg_offset) * 4
    """
    if base_index >= len(config.base):
        raise ValueError(f"Invalid base_index {base_index}")
    return (config.base[base_index] + reg) * 4


def read_nbio_reg(dev: WindowsDevice, config: NBIOConfig,
                  base_index: int, reg: int) -> int:
    """Read an NBIO register via BAR0 MMIO."""
    offset = mmio_offset(config, base_index, reg)
    return dev.read_reg32(offset)


def write_nbio_reg(dev: WindowsDevice, config: NBIOConfig,
                   base_index: int, reg: int, value: int) -> None:
    """Write an NBIO register via BAR0 MMIO."""
    offset = mmio_offset(config, base_index, reg)
    dev.write_reg32(offset, value)


# ── Initialization functions ─────────────────────────────────────────

def enable_doorbell_aperture(dev: WindowsDevice, config: NBIOConfig,
                             enable: bool = True) -> None:
    """Enable or disable the doorbell aperture.

    The doorbell aperture allows userspace to write doorbell registers
    which signal the GPU to start processing work. This must be enabled
    before any queue submission.

    Reference: nbio_v7_11_enable_doorbell_aperture()
    """
    val = read_nbio_reg(dev, config, 2, regRCC_DOORBELL_APER_EN)
    if enable:
        val |= BIF_DOORBELL_APER_EN__BIT
    else:
        val &= ~BIF_DOORBELL_APER_EN__BIT
    write_nbio_reg(dev, config, 2, regRCC_DOORBELL_APER_EN, val)


def enable_selfring_doorbell(dev: WindowsDevice, config: NBIOConfig,
                             doorbell_addr: int,
                             enable: bool = True) -> None:
    """Enable the selfring doorbell aperture.

    The selfring doorbell provides a GPA (Guest Physical Address) based
    doorbell aperture that the GPU uses internally for inter-engine
    communication.

    Reference: nbio_v7_11_enable_doorbell_selfring_aperture()
    """
    if enable:
        # Set base address (split into high/low 32-bit words)
        write_nbio_reg(dev, config, 2, regDOORBELL_SELFRING_BASE_LOW,
                       doorbell_addr & 0xFFFFFFFF)
        write_nbio_reg(dev, config, 2, regDOORBELL_SELFRING_BASE_HIGH,
                       (doorbell_addr >> 32) & 0xFFFFFFFF)

        # Enable with mode=1, size=0 (full aperture)
        val = SELFRING_EN__BIT | SELFRING_MODE__BIT
        write_nbio_reg(dev, config, 2, regDOORBELL_SELFRING_CNTL, val)
    else:
        write_nbio_reg(dev, config, 2, regDOORBELL_SELFRING_CNTL, 0)


def enable_framebuffer(dev: WindowsDevice, config: NBIOConfig,
                       enable: bool = True) -> None:
    """Enable or disable frame buffer (VRAM) access through NBIO.

    Must be enabled before any VRAM reads/writes.

    Reference: nbio_v7_11_mc_access_enable()
    """
    val = read_nbio_reg(dev, config, 2, regBIF_FB_EN)
    if enable:
        val |= BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN
    else:
        val &= ~(BIF_FB_EN__FB_READ_EN | BIF_FB_EN__FB_WRITE_EN)
    write_nbio_reg(dev, config, 2, regBIF_FB_EN, val)


def setup_interrupt_control(dev: WindowsDevice, config: NBIOConfig,
                            dummy_page_phys: int) -> None:
    """Configure NBIO interrupt control.

    Sets up the dummy read page address and interrupt control flags.
    The dummy page is a physical page used for interrupt coalescing.

    Reference: nbio_v7_11_ih_control()
    """
    # Write dummy page address (upper 24 bits of physical address)
    write_nbio_reg(dev, config, 2, regINTERRUPT_CNTL2,
                   dummy_page_phys >> 8)

    # Configure interrupt control
    val = read_nbio_reg(dev, config, 2, regINTERRUPT_CNTL)
    val |= IH_DUMMY_RD_OVERRIDE  # Enable dummy read override
    val |= IH_REQ_NONSNOOP_EN   # Non-snooped memory for IH
    write_nbio_reg(dev, config, 2, regINTERRUPT_CNTL, val)


def hdp_flush(dev: WindowsDevice, config: NBIOConfig) -> None:
    """Trigger an HDP (Host Data Path) flush.

    Ensures all CPU writes through HDP are visible to the GPU.
    Should be called before GPU reads data written by CPU.

    Reference: nbio_v7_11_hdp_flush()
    """
    write_nbio_reg(dev, config, 2, regHDP_MEM_COHERENCY_FLUSH, 0)


def read_vram_size_mb(dev: WindowsDevice, config: NBIOConfig) -> int:
    """Read VRAM size in MB from NBIO config register."""
    return read_nbio_reg(dev, config, 2, regRCC_CONFIG_MEMSIZE)


def read_revision_id(dev: WindowsDevice) -> int:
    """Read GPU revision ID via direct MMIO (no NBIO base needed).

    Uses the standard mmRCC_CONFIG_MEMSIZE register at a well-known
    offset. The revision ID is in PCI config space (already read by
    the kernel driver in StartDevice).
    """
    # Revision ID is already available from the driver's GET_INFO escape
    # This function is here for completeness / verification
    return dev.read_reg32(0x08)  # PCI config space revision_id offset


def init_nbio(dev: WindowsDevice, ip_result: IPDiscoveryResult) -> NBIOConfig:
    """Full NBIO initialization sequence for RDNA4.

    Follows the Linux amdgpu soc21_common_hw_init() sequence:
    1. Resolve NBIO base addresses from IP discovery
    2. Enable doorbell aperture
    3. Read and report VRAM size

    Selfring doorbell and doorbell ranges are configured later
    after GMC init (which may resize BARs).

    Returns the resolved NBIO configuration for use by other modules.
    """
    config = resolve_nbio_bases(ip_result)

    # Phase 1: Enable doorbell aperture
    enable_doorbell_aperture(dev, config, enable=True)

    # Phase 2: Enable framebuffer access
    enable_framebuffer(dev, config, enable=True)

    # Phase 3: Read VRAM size for verification
    vram_mb = read_vram_size_mb(dev, config)
    if vram_mb > 0:
        print(f"  NBIO: VRAM size = {vram_mb} MB")

    return config
