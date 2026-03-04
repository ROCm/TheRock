"""IH v7.0 initialization for RDNA4 (Navi 48 / GFX1201).

Programs the Interrupt Handler ring buffer for GPU-to-host interrupt
delivery. The IH ring is a circular buffer in DMA memory where the
GPU writes 32-byte interrupt entries.

Init sequence:
1. Allocate IH ring buffer (DMA memory)
2. Program IH registers (base, size, control)
3. Configure NBIO doorbell and interrupt routing
4. Call kernel ENABLE_MSI escape to register the ring with the ISR

IH entry format (8 DWORDs = 32 bytes):
  DW[0]: [7:0]=client_id, [15:8]=source_id, [23:16]=ring_id,
         [27:24]=vmid, [31]=vmid_src
  DW[1]: timestamp_lo
  DW[2]: [15:0]=timestamp_hi, [31]=timestamp_src
  DW[3]: [15:0]=pasid, [23:16]=node_id
  DW[4-7]: src_data[0-3]

Reference: Linux amdgpu ih_v7_0.c, osssys_7_0_0_offset.h
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult
    from amd_gpu_driver.backends.windows.nbio_init import NBIOConfig


# ============================================================================
# IH v7.0 register offsets (relative to OSSSYS base, DWORD units)
# From osssys_7_0_0_offset.h
# ============================================================================

regIH_RB_CNTL = 0x0080          # Ring buffer control
regIH_RB_RPTR = 0x0081          # Ring buffer read pointer
regIH_RB_WPTR = 0x0082          # Ring buffer write pointer
regIH_RB_BASE = 0x0083          # Ring buffer base address [39:8]
regIH_RB_BASE_HI = 0x0084       # Ring buffer base address [47:40]
regIH_RB_WPTR_ADDR_HI = 0x0085  # WPTR writeback address high
regIH_RB_WPTR_ADDR_LO = 0x0086  # WPTR writeback address low
regIH_DOORBELL_RPTR = 0x0087    # Doorbell RPTR config
regIH_CNTL = 0x00A8             # IH global control
regIH_CNTL2 = 0x00C1            # IH control 2
regIH_INT_FLOOD_CNTL = 0x00D5   # Interrupt flood control
regIH_MSI_STORM_CTRL = 0x00F1   # MSI storm control
regIH_CHICKEN = 0x018A          # IH chicken bits
regIH_STORM_CLIENT_LIST_CNTL = 0x00D8  # Storm client list

# Ring 1 registers (secondary ring)
regIH_RB_CNTL_RING1 = 0x008C
regIH_RB_RPTR_RING1 = 0x008D
regIH_RB_WPTR_RING1 = 0x008E
regIH_RB_BASE_RING1 = 0x008F
regIH_RB_BASE_HI_RING1 = 0x0090
regIH_DOORBELL_RPTR_RING1 = 0x0093

# IH_RB_CNTL bit fields
IH_RB_CNTL__MC_SPACE__SHIFT = 4
IH_RB_CNTL__RB_SIZE__SHIFT = 1
IH_RB_CNTL__WPTR_OVERFLOW_CLEAR = 1 << 31
IH_RB_CNTL__WPTR_OVERFLOW_ENABLE = 1 << 8
IH_RB_CNTL__WPTR_WRITEBACK_ENABLE = 1 << 12
IH_RB_CNTL__MC_SNOOP = 1 << 14
IH_RB_CNTL__RPTR_REARM = 1 << 15
IH_RB_CNTL__ENABLE_INTR = 1 << 0

# MC_SPACE values
MC_SPACE_NONE = 0
MC_SPACE_BUS_ADDR = 2
MC_SPACE_GPU_VA = 4

# IH entry size
IH_ENTRY_SIZE = 32  # 8 DWORDs per entry

# Default ring size
IH_RING_SIZE = 256 * 1024  # 256KB (8192 entries)

# IH source IDs for compute
IH_CLIENT_GC = 0x0A       # Graphics/Compute engine
IH_CLIENT_SDMA0 = 0x08    # SDMA engine 0
IH_CLIENT_SDMA1 = 0x09    # SDMA engine 1
IH_CLIENT_BIF = 0x00      # Bus Interface (PCIe)

# GC source IDs
IH_SRC_CP_EOP = 0xB5      # End-of-pipe fence (RELEASE_MEM completion)
IH_SRC_CP_BAD_OPCODE = 0xB4  # Bad PM4 opcode


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class IHConfig:
    """IH ring configuration."""
    # Base addresses from IP discovery (OSSSYS = IH)
    ih_base: list[int]  # DWORD base addresses

    # Ring buffer (DMA allocated)
    ring_bus_addr: int = 0     # Bus address of ring buffer
    ring_cpu_addr: int = 0     # CPU VA of ring buffer
    ring_dma_handle: int = 0   # Kernel allocation handle
    ring_size: int = IH_RING_SIZE

    # WPTR writeback (DMA allocated)
    wptr_bus_addr: int = 0     # Bus address of WPTR writeback
    wptr_cpu_addr: int = 0     # CPU VA of WPTR writeback
    wptr_dma_handle: int = 0   # Kernel allocation handle

    # Computed register byte offsets (for kernel ISR)
    rptr_reg_byte_offset: int = 0
    wptr_reg_byte_offset: int = 0


# ============================================================================
# Register access helpers
# ============================================================================

def _ih_reg(dev: WindowsDevice, config: IHConfig, reg: int) -> int:
    """Read an IH register via SOC15 addressing."""
    offset = (config.ih_base[0] + reg) * 4
    return dev.read_reg32(offset)


def _ih_wreg(dev: WindowsDevice, config: IHConfig, reg: int, val: int) -> None:
    """Write an IH register."""
    offset = (config.ih_base[0] + reg) * 4
    dev.write_reg32(offset, val)


# ============================================================================
# IH initialization (following ih_v7_0.c)
# ============================================================================

def resolve_ih_bases(ip_result: IPDiscoveryResult) -> IHConfig:
    """Resolve IH base addresses from IP discovery.

    IH uses OSSSYS (HW_ID 40) in the IP discovery table.
    """
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    bases = [0] * 6

    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.OSSSYS and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(bases) and addr != 0:
                    bases[i] = addr

    return IHConfig(ih_base=bases)


def _toggle_interrupts(dev: WindowsDevice, config: IHConfig, enable: bool) -> None:
    """Enable or disable IH interrupts.

    Reference: ih_v7_0_toggle_interrupts()
    """
    val = _ih_reg(dev, config, regIH_RB_CNTL)
    if enable:
        val |= IH_RB_CNTL__ENABLE_INTR
    else:
        val &= ~IH_RB_CNTL__ENABLE_INTR
    _ih_wreg(dev, config, regIH_RB_CNTL, val)


def _setup_ring(
    dev: WindowsDevice,
    config: IHConfig,
    use_bus_addr: bool = True,
) -> None:
    """Program IH ring 0 registers.

    Reference: ih_v7_0_irq_init() ring 0 setup
    """
    ring_size_log2 = int(math.log2(config.ring_size / 4))

    # Ring base address (>> 8 for register)
    _ih_wreg(dev, config, regIH_RB_BASE,
             (config.ring_bus_addr >> 8) & 0xFFFFFFFF)
    _ih_wreg(dev, config, regIH_RB_BASE_HI,
             (config.ring_bus_addr >> 40) & 0xFF)

    # Ring control
    val = 0
    # MC_SPACE = 2 for bus addresses, 4 for GPU VA
    mc_space = MC_SPACE_BUS_ADDR if use_bus_addr else MC_SPACE_GPU_VA
    val |= mc_space << IH_RB_CNTL__MC_SPACE__SHIFT
    # RB_SIZE (log2(size/4))
    val |= (ring_size_log2 & 0x3F) << IH_RB_CNTL__RB_SIZE__SHIFT
    # Enable features
    val |= IH_RB_CNTL__WPTR_OVERFLOW_CLEAR
    val |= IH_RB_CNTL__WPTR_OVERFLOW_ENABLE
    val |= IH_RB_CNTL__WPTR_WRITEBACK_ENABLE
    val |= IH_RB_CNTL__MC_SNOOP
    val |= IH_RB_CNTL__RPTR_REARM
    _ih_wreg(dev, config, regIH_RB_CNTL, val)

    # WPTR writeback address
    _ih_wreg(dev, config, regIH_RB_WPTR_ADDR_LO,
             config.wptr_bus_addr & 0xFFFFFFFF)
    _ih_wreg(dev, config, regIH_RB_WPTR_ADDR_HI,
             (config.wptr_bus_addr >> 32) & 0xFFFF)

    # Reset read and write pointers
    _ih_wreg(dev, config, regIH_RB_RPTR, 0)
    _ih_wreg(dev, config, regIH_RB_WPTR, 0)


def init_ih(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
) -> IHConfig:
    """Full IH initialization sequence for RDNA4.

    Allocates IH ring buffer, programs registers, and registers
    the ring with the kernel driver ISR via ENABLE_MSI escape.

    Reference: ih_v7_0_irq_init()

    Args:
        dev: Windows device backend.
        ip_result: Parsed IP discovery data.
        nbio_config: NBIO configuration (for doorbell setup).

    Returns:
        Configured IHConfig.
    """
    from amd_gpu_driver.backends.windows.nbio_init import setup_interrupt_control

    config = resolve_ih_bases(ip_result)

    # Allocate IH ring buffer (DMA memory)
    ring_cpu, ring_bus, ring_handle = dev.driver.alloc_dma(config.ring_size)
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle

    # Allocate WPTR writeback buffer (one page)
    wptr_cpu, wptr_bus, wptr_handle = dev.driver.alloc_dma(4096)
    config.wptr_cpu_addr = wptr_cpu
    config.wptr_bus_addr = wptr_bus
    config.wptr_dma_handle = wptr_handle

    # Compute register byte offsets for the kernel ISR
    config.rptr_reg_byte_offset = (config.ih_base[0] + regIH_RB_RPTR) * 4
    config.wptr_reg_byte_offset = (config.ih_base[0] + regIH_RB_WPTR) * 4

    # Allocate a dummy page for interrupt control
    dummy_cpu, dummy_bus, dummy_handle = dev.driver.alloc_dma(4096)

    # Disable interrupts during setup
    _toggle_interrupts(dev, config, enable=False)

    # Configure NBIO interrupt control (dummy page address)
    setup_interrupt_control(dev, nbio_config, dummy_bus)

    # Program IH ring registers
    _setup_ring(dev, config, use_bus_addr=True)

    # Configure flood control
    val = _ih_reg(dev, config, regIH_INT_FLOOD_CNTL)
    val |= (1 << 0)   # FLOOD_CNTL_ENABLE
    _ih_wreg(dev, config, regIH_INT_FLOOD_CNTL, val)

    # MSI storm control (delay = 3)
    _ih_wreg(dev, config, regIH_MSI_STORM_CTRL, 3)

    # Enable interrupts
    _toggle_interrupts(dev, config, enable=True)

    # Register IH ring with kernel driver ISR
    enabled, num_vectors = dev.driver.enable_msi(
        ih_ring_dma_handle=ring_handle,
        ih_ring_size=config.ring_size,
        rptr_reg_offset=config.rptr_reg_byte_offset,
        wptr_reg_offset=config.wptr_reg_byte_offset,
    )

    print(f"  IH: Ring at bus 0x{ring_bus:012X}, size={config.ring_size // 1024}KB")
    print(f"  IH: MSI enabled={enabled}, vectors={num_vectors}")

    return config
