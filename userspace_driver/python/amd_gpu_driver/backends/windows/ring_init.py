"""Ring bring-up for RDNA4 (Navi 48 / GFX1201).

Programs compute and SDMA rings for command submission:
1. GRBM_SELECT for pipe/queue targeting
2. MES (Micro Engine Scheduler) enable/disable
3. KIQ ring init via direct MMIO register writes
4. Compute queue MQD (Memory Queue Descriptor) creation
5. SDMA queue MQD creation
6. Doorbell-based wptr update for packet submission

Two approaches are supported:
- **Direct MMIO**: Program CP_HQD_* registers directly (used for KIQ bootstrap)
- **MES-managed**: Submit ADD_QUEUE command to MES sched pipe (for user queues)

For our userspace driver, we use the direct MMIO approach since we bypass
MES entirely — we directly program a compute queue and submit PM4 packets.

Reference: Linux amdgpu gfx_v12_0.c, mes_v12_0.c, sdma_v7_0.c, v12_structs.h
"""

from __future__ import annotations

import ctypes
import math
import struct
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult
    from amd_gpu_driver.backends.windows.gmc_init import GMCConfig
    from amd_gpu_driver.backends.windows.nbio_init import NBIOConfig


# ============================================================================
# GC 12.0 register offsets (from gc_12_0_0_offset.h)
# These are DWORD offsets; each register below has an explicit base index.
# ============================================================================

# GRBM
regGRBM_CNTL = 0x0DA0                  # base_index 0
regGRBM_GFX_CNTL = 0x0900              # base_index 1

# CP/RLC engine bring-up
regCP_STAT = 0x0F40                    # base_index 0
regCP_PFP_PRGRM_CNTR_START = 0x1E44    # base_index 0
regCP_ME_PRGRM_CNTR_START = 0x1E45     # base_index 0
regCP_PFP_PRGRM_CNTR_START_HI = 0x1E59 # base_index 0
regCP_ME_PRGRM_CNTR_START_HI = 0x1E79  # base_index 0
regCP_ME_CNTL = 0x0803                 # base_index 1
regCP_MEC_RS64_PRGRM_CNTR_START = 0x2900     # base_index 1
regCP_MEC_RS64_CNTL = 0x2904                 # base_index 1
regCP_MEC_RS64_PRGRM_CNTR_START_HI = 0x2938  # base_index 1
regCP_MEC_DOORBELL_RANGE_LOWER = 0x1DFC      # base_index 0
regCP_MEC_DOORBELL_RANGE_UPPER = 0x1DFD      # base_index 0
regRLC_CNTL = 0x4C00                 # base_index 1
regRLC_SRM_CNTL = 0x4C80             # base_index 1
regRLC_RLCS_BOOTLOAD_STATUS = 0x4E7C # base_index 1
regRLC_SPM_MC_CNTL = 0x0982          # base_index 1
regSH_MEM_BASES = 0x09E3             # base_index 1
regSH_MEM_CONFIG = 0x09E4            # base_index 1
regTCP_CNTL = 0x19A2                 # base_index 1

# CP MES control registers (base_index 1)
regCP_MES_CNTL = 0x2807
regCP_MES_PRGRM_CNTR_START = 0x2800
regCP_MES_PRGRM_CNTR_START_HI = 0x289D
regCP_MES_IC_BASE_LO = 0x5850
regCP_MES_IC_BASE_HI = 0x5851
regCP_MES_IC_BASE_CNTL = 0x5852
regCP_MES_MDBASE_LO = 0x5854
regCP_MES_MDBASE_HI = 0x5855
regCP_MES_MIBOUND_LO = 0x585B
regCP_MES_MDBOUND_LO = 0x585D
regCP_MES_IC_OP_CNTL = 0x2820

# CP HQD registers (base_index 0, used for direct queue programming)
regCP_MQD_BASE_ADDR = 0x1FA9
regCP_MQD_BASE_ADDR_HI = 0x1FAA
regCP_HQD_ACTIVE = 0x1FAB
regCP_HQD_VMID = 0x1FAC
regCP_HQD_PERSISTENT_STATE = 0x1FAD
regCP_HQD_PIPE_PRIORITY = 0x1FAE
regCP_HQD_QUEUE_PRIORITY = 0x1FAF
regCP_HQD_QUANTUM = 0x1FB0
regCP_HQD_PQ_BASE = 0x1FB1
regCP_HQD_PQ_BASE_HI = 0x1FB2
regCP_HQD_PQ_RPTR = 0x1FB3
regCP_HQD_PQ_RPTR_REPORT_ADDR = 0x1FB4
regCP_HQD_PQ_RPTR_REPORT_ADDR_HI = 0x1FB5
regCP_HQD_PQ_WPTR_POLL_ADDR = 0x1FB6
regCP_HQD_PQ_WPTR_POLL_ADDR_HI = 0x1FB7
regCP_HQD_PQ_DOORBELL_CONTROL = 0x1FB8
regCP_HQD_PQ_CONTROL = 0x1FBA
regCP_MQD_CONTROL = 0x1FCB
regCP_HQD_EOP_BASE_ADDR = 0x1FCE
regCP_HQD_EOP_BASE_ADDR_HI = 0x1FCF
regCP_HQD_EOP_CONTROL = 0x1FD0
regCP_HQD_IB_CONTROL = 0x1FBE
regCP_HQD_HQ_STATUS0 = 0x1FC9
regCP_HQD_AQL_CONTROL = 0x1FDE
regCP_HQD_PQ_WPTR_LO = 0x1FBB
regCP_HQD_PQ_WPTR_HI = 0x1FBC

# RLC scheduler register
regRLC_CP_SCHEDULERS = 0x4E45  # base_index 1

# CP_MES_CNTL bit fields
CP_MES_CNTL__MES_INVALIDATE_ICACHE = 1 << 4
CP_MES_CNTL__MES_PIPE0_RESET = 1 << 16
CP_MES_CNTL__MES_PIPE1_RESET = 1 << 17
CP_MES_CNTL__MES_PIPE0_ACTIVE = 1 << 26
CP_MES_CNTL__MES_PIPE1_ACTIVE = 1 << 27
CP_MES_CNTL__MES_HALT = 1 << 30

# CP_HQD_PQ_DOORBELL_CONTROL bit fields
DOORBELL_OFFSET__SHIFT = 2
DOORBELL_EN = 1 << 30
DOORBELL_SOURCE = 1 << 28
DOORBELL_HIT = 1 << 31

# CP_HQD_PQ_CONTROL bit fields
PQ_CONTROL__QUEUE_SIZE__SHIFT = 0
PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT = 8
PQ_CONTROL__QUEUE_FULL_EN = 1 << 14
PQ_CONTROL__SLOT_BASED_WPTR__SHIFT = 18
PQ_CONTROL__NO_UPDATE_RPTR = 1 << 27
PQ_CONTROL__UNORD_DISPATCH = 1 << 28
PQ_CONTROL__TUNNEL_DISPATCH = 1 << 29
PQ_CONTROL__PRIV_STATE = 1 << 30
PQ_CONTROL__KMD_QUEUE = 1 << 31

# CP_HQD_PERSISTENT_STATE
HQD_PERSISTENT_STATE__PRELOAD_REQ = 1 << 0
HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT = 8
HQD_PERSISTENT_STATE__PRELOAD_SIZE = 0x55

# CP_HQD_EOP_CONTROL
EOP_SIZE_SHIFT = 0

# GRBM_GFX_CNTL bit fields
GRBM_GFX_CNTL__PIPEID__SHIFT = 0
GRBM_GFX_CNTL__MEID__SHIFT = 2
GRBM_GFX_CNTL__VMID__SHIFT = 4
GRBM_GFX_CNTL__QUEUEID__SHIFT = 8

# CP engine control bit fields
CP_ME_CNTL__PFP_PIPE0_RESET = 0x00040000
CP_ME_CNTL__ME_PIPE0_RESET = 0x00100000
CP_ME_CNTL__PFP_HALT = 0x04000000
CP_ME_CNTL__ME_HALT = 0x10000000
CP_MEC_RS64_CNTL__MEC_PIPE0_RESET = 0x00010000
CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE = 0x04000000
CP_MEC_RS64_CNTL__MEC_HALT = 0x40000000

# Doorbell assignments (LAYOUT1 for GFX12)
DOORBELL_KIQ = 0x000
DOORBELL_HIQ = 0x001
DOORBELL_MEC_RING_START = 0x020
DOORBELL_SDMA_START = 0x100

# MES pipe IDs
MES_PIPE_SCHED = 0
MES_PIPE_KIQ = 1

# Default ring sizes
COMPUTE_RING_SIZE = 256 * 1024  # 256KB
SDMA_RING_SIZE = 256 * 1024     # 256KB
EOP_BUFFER_SIZE = 2048          # 2KB per EOP buffer

# v12_compute_mqd constants
MQD_HEADER = 0xC0310800
MQD_SIZE = 256 * 4  # 256 DWORDs = 1024 bytes

# v12_sdma_mqd size
SDMA_MQD_SIZE = 128 * 4  # 128 DWORDs = 512 bytes


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ComputeQueueConfig:
    """Configuration for a compute queue."""
    # GC base addresses from IP discovery
    gc_base: list[int]

    # Ring buffer (DMA allocated)
    ring_bus_addr: int = 0
    ring_cpu_addr: int = 0
    ring_dma_handle: int = 0
    ring_size: int = COMPUTE_RING_SIZE

    # MQD (Memory Queue Descriptor, DMA allocated)
    mqd_bus_addr: int = 0
    mqd_cpu_addr: int = 0
    mqd_dma_handle: int = 0

    # EOP buffer (End-of-Pipe events)
    eop_bus_addr: int = 0
    eop_cpu_addr: int = 0
    eop_dma_handle: int = 0

    # WPTR writeback (for doorbell)
    wptr_bus_addr: int = 0
    wptr_cpu_addr: int = 0
    wptr_dma_handle: int = 0

    # RPTR writeback
    rptr_bus_addr: int = 0
    rptr_cpu_addr: int = 0
    rptr_dma_handle: int = 0

    # Fence buffer (for completion signaling)
    fence_bus_addr: int = 0
    fence_cpu_addr: int = 0
    fence_dma_handle: int = 0

    # Doorbell
    doorbell_index: int = DOORBELL_MEC_RING_START
    doorbell_cpu_addr: int = 0  # Mapped doorbell BAR address

    # Queue identity (for GRBM_SELECT)
    me: int = 1     # ME=1 = MEC0 (compute)
    pipe: int = 0
    queue: int = 0

    # State
    active: bool = False
    wptr: int = 0   # Current write pointer (in DWORDs)
    aql: bool = False

    # Optional backend hooks for HDP flush and lifetime tracking.
    dev: object | None = None
    nbio_config: object | None = None
    memory_handles: list[object] = field(default_factory=list)


@dataclass
class SDMAQueueConfig:
    """Configuration for an SDMA queue."""
    # SDMA base addresses from IP discovery
    sdma_base: list[int]

    # Ring buffer
    ring_bus_addr: int = 0
    ring_cpu_addr: int = 0
    ring_dma_handle: int = 0
    ring_size: int = SDMA_RING_SIZE

    # MQD
    mqd_bus_addr: int = 0
    mqd_cpu_addr: int = 0
    mqd_dma_handle: int = 0

    # WPTR writeback
    wptr_bus_addr: int = 0
    wptr_cpu_addr: int = 0
    wptr_dma_handle: int = 0

    # RPTR writeback
    rptr_bus_addr: int = 0
    rptr_cpu_addr: int = 0
    rptr_dma_handle: int = 0

    # Fence buffer
    fence_bus_addr: int = 0
    fence_cpu_addr: int = 0
    fence_dma_handle: int = 0

    # Doorbell
    doorbell_index: int = DOORBELL_SDMA_START
    doorbell_cpu_addr: int = 0

    # Instance (0 or 1)
    instance: int = 0

    # State
    active: bool = False
    wptr: int = 0


# ============================================================================
# Register access helpers
# ============================================================================

def _gc_reg(dev: WindowsDevice, gc_base: list[int],
            reg: int, base_idx: int = 1) -> int:
    """Read a GC register via SOC15 addressing."""
    offset = (gc_base[base_idx] + reg) * 4
    return dev.read_reg32(offset)


def _gc_wreg(dev: WindowsDevice, gc_base: list[int],
             reg: int, val: int, base_idx: int = 1) -> None:
    """Write a GC register via SOC15 addressing."""
    offset = (gc_base[base_idx] + reg) * 4
    dev.write_reg32(offset, val)


def _gc_wreg_pair(
    dev: WindowsDevice,
    gc_base: list[int],
    reg_lo: int,
    reg_hi: int,
    value: int,
    *,
    base_idx: int,
) -> None:
    _gc_wreg(dev, gc_base, reg_lo, value & 0xFFFFFFFF, base_idx=base_idx)
    _gc_wreg(dev, gc_base, reg_hi, (value >> 32) & 0xFFFFFFFF,
             base_idx=base_idx)


def _alloc_queue_buffer(
    dev: WindowsDevice,
    config: ComputeQueueConfig,
    size: int,
) -> tuple[int, int, int]:
    """Allocate queue memory, preferring VRAM MC addresses for amdgpu_lite."""
    if hasattr(dev, "read_vram") and hasattr(dev, "alloc_memory"):
        from amd_gpu_driver.backends.base import MemoryLocation

        handle = dev.alloc_memory(size, MemoryLocation.VRAM)
        if handle.cpu_addr == 0:
            raise RuntimeError("Queue VRAM allocation is not CPU mapped")
        ctypes.memset(handle.cpu_addr, 0, handle.size)
        config.memory_handles.append(handle)
        return handle.cpu_addr, handle.gpu_addr, 0

    cpu, bus, handle = dev.driver.alloc_dma(size)
    ctypes.memset(cpu, 0, size)
    return cpu, bus, handle


# ============================================================================
# GRBM_SELECT — target a specific ME/pipe/queue for subsequent register writes
# ============================================================================

def grbm_select(dev: WindowsDevice, gc_base: list[int],
                me: int, pipe: int, queue: int, vmid: int = 0) -> None:
    """Select ME/pipe/queue via GRBM_GFX_CNTL for targeted register writes.

    ME values: 0=GFX, 1=MEC0(compute), 2=MEC1, 3=MES
    This must be called before writing CP_HQD_* registers to target
    a specific pipe and queue.

    Reference: soc21_grbm_select()
    """
    val = 0
    val |= (pipe & 0x3) << GRBM_GFX_CNTL__PIPEID__SHIFT
    val |= (me & 0x3) << GRBM_GFX_CNTL__MEID__SHIFT
    val |= (vmid & 0xF) << GRBM_GFX_CNTL__VMID__SHIFT
    val |= (queue & 0x7) << GRBM_GFX_CNTL__QUEUEID__SHIFT

    # GRBM_GFX_CNTL is at base_index 1 on GC 12.
    offset = (gc_base[1] + regGRBM_GFX_CNTL) * 4
    dev.write_reg32(offset, val)


def grbm_deselect(dev: WindowsDevice, gc_base: list[int]) -> None:
    """Deselect GRBM (set all to 0)."""
    grbm_select(dev, gc_base, 0, 0, 0, 0)


# ============================================================================
# Compute MQD initialization (v12_compute_mqd format)
# ============================================================================

def _init_compute_mqd(config: ComputeQueueConfig) -> None:
    """Initialize a v12_compute_mqd in DMA memory.

    Fills the MQD structure following mes_v12_0_mqd_init().
    The MQD is a 256-DWORD structure that describes the queue to hardware.

    Reference: mes_v12_0_mqd_init() in mes_v12_0.c
    """
    mqd = (ctypes.c_uint32 * 256).from_address(config.mqd_cpu_addr)

    # Clear
    ctypes.memset(config.mqd_cpu_addr, 0, MQD_SIZE)

    # Header (offset 0)
    mqd[0] = MQD_HEADER

    # compute_pipelinestat_enable (offset 11)
    mqd[11] = 1

    # Thread management: enable all SEs (offsets 23, 24, 26, 27)
    mqd[23] = 0xFFFFFFFF  # SE0
    mqd[24] = 0xFFFFFFFF  # SE1
    mqd[26] = 0xFFFFFFFF  # SE2
    mqd[27] = 0xFFFFFFFF  # SE3

    # compute_misc_reserved (offset 32) = 0x7
    mqd[32] = 0x00000007

    # cp_mqd_base_addr (offsets 128-129)
    mqd[128] = config.mqd_bus_addr & 0xFFFFFFFC
    mqd[129] = (config.mqd_bus_addr >> 32) & 0xFFFFFFFF

    # cp_hqd_active (offset 130) — set active
    mqd[130] = 1

    # cp_hqd_vmid (offset 131) = 0 (kernel/system VMID)
    mqd[131] = 0

    # cp_hqd_persistent_state (offset 132)
    mqd[132] = (
        HQD_PERSISTENT_STATE__PRELOAD_REQ |
        (HQD_PERSISTENT_STATE__PRELOAD_SIZE <<
         HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT)
    )

    # cp_hqd_pipe_priority (offset 133) and queue_priority (offset 134)
    mqd[133] = 0x2
    mqd[134] = 0xF

    # cp_hqd_quantum (offset 135)
    mqd[135] = 0x111

    # cp_hqd_pq_base (offsets 136-137) — ring buffer address >> 8
    pq_base = config.ring_bus_addr >> 8
    mqd[136] = pq_base & 0xFFFFFFFF
    mqd[137] = (pq_base >> 32) & 0xFFFFFFFF

    # cp_hqd_pq_rptr (offset 138) = 0
    mqd[138] = 0

    # cp_hqd_pq_rptr_report_addr (offsets 139-140)
    mqd[139] = config.rptr_bus_addr & 0xFFFFFFFF
    mqd[140] = (config.rptr_bus_addr >> 32) & 0xFFFFFFFF

    # cp_hqd_pq_wptr_poll_addr (offsets 141-142)
    mqd[141] = config.wptr_bus_addr & 0xFFFFFFFF
    mqd[142] = (config.wptr_bus_addr >> 32) & 0xFFFFFFFF

    # cp_hqd_pq_doorbell_control (offset 143)
    doorbell_ctrl = 0
    doorbell_ctrl |= (config.doorbell_index & 0x0FFFFFFC) << DOORBELL_OFFSET__SHIFT
    doorbell_ctrl |= DOORBELL_EN
    mqd[143] = doorbell_ctrl

    # cp_hqd_pq_control (offset 145)
    ring_size_log2 = int(math.log2(config.ring_size // 4)) - 1
    pq_control = 0
    pq_control |= (ring_size_log2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT
    pq_control |= (5 & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT
    if config.aql:
        pq_control |= PQ_CONTROL__QUEUE_FULL_EN
        pq_control |= 2 << PQ_CONTROL__SLOT_BASED_WPTR__SHIFT
        pq_control |= PQ_CONTROL__NO_UPDATE_RPTR
    mqd[145] = pq_control

    # cp_mqd_control (offset 162)
    mqd[162] = 1 << 8  # PRIV_STATE

    # cp_hqd_eop_base_addr (offsets 165-166) — EOP buffer address >> 8
    eop_base = config.eop_bus_addr >> 8
    mqd[165] = eop_base & 0xFFFFFFFF
    mqd[166] = (eop_base >> 32) & 0xFFFFFFFF

    # cp_hqd_eop_control (offset 167) — EOP_SIZE = log2(2048/4) - 1 = 8
    mqd[167] = 8

    # cp_hqd_ib_control (offset 149), cp_hqd_hq_status0 (offset 160)
    mqd[149] = 3 << 20
    mqd[160] = 0x20004000

    # cp_hqd_pq_wptr_lo/hi (offsets 182-183) = 0
    mqd[182] = 0
    mqd[183] = 0

    # cp_hqd_aql_control (offset 181)
    mqd[181] = 1 if config.aql else 0

    # reserved_184 (offset 184) — set bit 15 for unmapped doorbell handling
    mqd[184] = 1 << 15


# ============================================================================
# Direct MMIO queue programming (bypass MES)
# ============================================================================

def _activate_compute_queue_mmio(
    dev: WindowsDevice,
    config: ComputeQueueConfig,
) -> None:
    """Program CP_HQD_* registers directly for a compute queue.

    This bypasses MES entirely — we write the HQD registers via MMIO
    after selecting the target pipe/queue with GRBM_SELECT.

    Reference: mes_v12_0_queue_init_register()
    """
    gc_base = config.gc_base

    # Select the target ME/pipe/queue
    grbm_select(dev, gc_base, config.me, config.pipe, config.queue)

    # Deactivate queue first
    _gc_wreg(dev, gc_base, regCP_HQD_ACTIVE, 0, base_idx=0)

    # Set VMID
    _gc_wreg(dev, gc_base, regCP_HQD_VMID, 0, base_idx=0)

    # Disable doorbell initially
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL, 0, base_idx=0)

    # MQD base address
    _gc_wreg(dev, gc_base, regCP_MQD_BASE_ADDR,
             config.mqd_bus_addr & 0xFFFFFFFC, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_MQD_BASE_ADDR_HI,
             (config.mqd_bus_addr >> 32) & 0xFFFFFFFF, base_idx=0)

    # MQD control (VMID = 0)
    _gc_wreg(dev, gc_base, regCP_MQD_CONTROL, 1 << 8, base_idx=0)

    # Ring buffer base (address >> 8)
    pq_base = config.ring_bus_addr >> 8
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_BASE,
             pq_base & 0xFFFFFFFF, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_BASE_HI,
             (pq_base >> 32) & 0xFFFFFFFF, base_idx=0)

    # RPTR report address
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR_REPORT_ADDR,
             config.rptr_bus_addr & 0xFFFFFFFF, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI,
             (config.rptr_bus_addr >> 32) & 0xFFFFFFFF, base_idx=0)

    # PQ control (ring size, flags)
    ring_size_log2 = int(math.log2(config.ring_size // 4)) - 1
    pq_control = 0
    pq_control |= (ring_size_log2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT
    pq_control |= (5 & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT
    if config.aql:
        pq_control |= PQ_CONTROL__QUEUE_FULL_EN
        pq_control |= 2 << PQ_CONTROL__SLOT_BASED_WPTR__SHIFT
        pq_control |= PQ_CONTROL__NO_UPDATE_RPTR
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_CONTROL, pq_control, base_idx=0)

    # WPTR poll address
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_POLL_ADDR,
             config.wptr_bus_addr & 0xFFFFFFFF, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_POLL_ADDR_HI,
             (config.wptr_bus_addr >> 32) & 0xFFFFFFFF, base_idx=0)

    # Reset RPTR and WPTR
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR, 0, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_LO, 0, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_HI, 0, base_idx=0)

    # Doorbell control (enable doorbell)
    doorbell_ctrl = 0
    doorbell_ctrl |= (config.doorbell_index & 0x0FFFFFFC) << DOORBELL_OFFSET__SHIFT
    doorbell_ctrl |= DOORBELL_EN
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL, doorbell_ctrl,
             base_idx=0)

    # Persistent state (preload)
    persistent = (
        HQD_PERSISTENT_STATE__PRELOAD_REQ |
        (HQD_PERSISTENT_STATE__PRELOAD_SIZE <<
         HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT)
    )
    _gc_wreg(dev, gc_base, regCP_HQD_PERSISTENT_STATE, persistent,
             base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PIPE_PRIORITY, 0x2, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_QUEUE_PRIORITY, 0xF, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_QUANTUM, 0x111, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_IB_CONTROL, 3 << 20, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_HQ_STATUS0, 0x20004000, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_AQL_CONTROL, 1 if config.aql else 0,
             base_idx=0)

    # EOP buffer
    eop_base = config.eop_bus_addr >> 8
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_BASE_ADDR,
             eop_base & 0xFFFFFFFF, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_BASE_ADDR_HI,
             (eop_base >> 32) & 0xFFFFFFFF, base_idx=0)
    # EOP_SIZE = log2(2048/4) - 1 = 8
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_CONTROL, 8, base_idx=0)

    # Activate the queue
    _gc_wreg(dev, gc_base, regCP_HQD_ACTIVE, 1, base_idx=0)

    # Deselect GRBM
    grbm_deselect(dev, gc_base)


# ============================================================================
# Compute ring submission
# ============================================================================

def submit_compute_packets(
    config: ComputeQueueConfig,
    packet_data: bytes,
) -> None:
    """Submit PM4 packets to the compute queue's ring buffer.

    Writes packet data to the ring, updates the wptr writeback location,
    and rings the doorbell to notify the GPU.

    The wptr is tracked in DWORDs. Doorbell write signals the GPU to
    start processing from the previous wptr to the new wptr.

    Reference: gfx_v12_0_ring_set_wptr_compute()
    """
    ring_mask = config.ring_size - 1  # byte mask
    byte_offset = (config.wptr * 4) & ring_mask

    # Write packet data to ring buffer (handle wrap-around)
    space_to_end = config.ring_size - byte_offset
    if len(packet_data) <= space_to_end:
        ctypes.memmove(config.ring_cpu_addr + byte_offset,
                       packet_data, len(packet_data))
    else:
        ctypes.memmove(config.ring_cpu_addr + byte_offset,
                       packet_data[:space_to_end], space_to_end)
        remainder = len(packet_data) - space_to_end
        ctypes.memmove(config.ring_cpu_addr,
                       packet_data[space_to_end:], remainder)

    # Advance wptr (in DWORDs)
    config.wptr += len(packet_data) // 4

    # Update wptr writeback location (64-bit write)
    ctypes.c_uint64.from_address(config.wptr_cpu_addr).value = config.wptr

    if config.dev is not None and config.nbio_config is not None:
        from amd_gpu_driver.backends.windows.nbio_init import hdp_flush
        hdp_flush(config.dev, config.nbio_config)

    # Ring the doorbell (64-bit write to doorbell BAR)
    if config.doorbell_cpu_addr != 0:
        ctypes.c_uint64.from_address(config.doorbell_cpu_addr).value = config.wptr


def read_fence_value(config: ComputeQueueConfig) -> int:
    """Read the current fence value from the fence buffer."""
    return ctypes.c_uint64.from_address(config.fence_cpu_addr).value


def wait_fence(
    config: ComputeQueueConfig,
    expected: int,
    timeout_ms: int = 5000,
) -> bool:
    """Poll the fence buffer until the expected value appears."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        val = read_fence_value(config)
        if val >= expected:
            return True
        time.sleep(0.001)
    return False


# ============================================================================
# SDMA MQD initialization
# ============================================================================

def _init_sdma_mqd(config: SDMAQueueConfig) -> None:
    """Initialize a v12_sdma_mqd in DMA memory.

    Reference: sdma_v7_0_mqd_init()
    """
    mqd = (ctypes.c_uint32 * 128).from_address(config.mqd_cpu_addr)

    # Clear
    ctypes.memset(config.mqd_cpu_addr, 0, SDMA_MQD_SIZE)

    ring_size_log2 = int(math.log2(config.ring_size // 4))

    # sdmax_rlcx_rb_cntl (offset 0)
    rb_cntl = 0
    rb_cntl |= ring_size_log2 << 1         # RB_SIZE
    rb_cntl |= 1 << 12                     # RPTR_WRITEBACK_ENABLE
    rb_cntl |= 4 << 16                     # RPTR_WRITEBACK_TIMER
    rb_cntl |= 1 << 28                     # MCU_WPTR_POLL_ENABLE
    mqd[0] = rb_cntl

    # Ring base (offset 1-2) — address >> 8
    rb_base = config.ring_bus_addr >> 8
    mqd[1] = rb_base & 0xFFFFFFFF
    mqd[2] = (rb_base >> 32) & 0xFFFFFFFF

    # RPTR/WPTR (offsets 3-6) = 0
    mqd[3] = 0
    mqd[4] = 0
    mqd[5] = 0
    mqd[6] = 0

    # RPTR writeback address (offsets 7-8)
    mqd[7] = config.rptr_bus_addr & 0xFFFFFFFF
    mqd[8] = (config.rptr_bus_addr >> 32) & 0xFFFFFFFF

    # Doorbell enable (offset 15)
    mqd[15] = 1  # DOORBELL_ENABLE

    # Doorbell offset (offset 17) — shifted left by 2
    mqd[17] = config.doorbell_index << 2

    # Dummy reg (offset 23)
    mqd[23] = 0xF

    # WPTR poll address (offsets 24-25)
    mqd[24] = config.wptr_bus_addr & 0xFFFFFFFF
    mqd[25] = (config.wptr_bus_addr >> 32) & 0xFFFFFFFF

    # AQL control (offset 26)
    mqd[26] = 0x4000


def submit_sdma_packets(
    config: SDMAQueueConfig,
    packet_data: bytes,
) -> None:
    """Submit SDMA packets to the SDMA queue's ring buffer.

    SDMA wptr is in bytes (shifted << 2 from DWORD index).

    Reference: sdma_v7_0_ring_set_wptr()
    """
    ring_mask = config.ring_size - 1
    byte_offset = (config.wptr * 4) & ring_mask

    space_to_end = config.ring_size - byte_offset
    if len(packet_data) <= space_to_end:
        ctypes.memmove(config.ring_cpu_addr + byte_offset,
                       packet_data, len(packet_data))
    else:
        ctypes.memmove(config.ring_cpu_addr + byte_offset,
                       packet_data[:space_to_end], space_to_end)
        remainder = len(packet_data) - space_to_end
        ctypes.memmove(config.ring_cpu_addr,
                       packet_data[space_to_end:], remainder)

    config.wptr += len(packet_data) // 4

    # SDMA uses wptr << 2 (byte offset) for the writeback/doorbell
    wptr_bytes = config.wptr << 2
    ctypes.c_uint64.from_address(config.wptr_cpu_addr).value = wptr_bytes

    if config.doorbell_cpu_addr != 0:
        ctypes.c_uint64.from_address(config.doorbell_cpu_addr).value = wptr_bytes


# ============================================================================
# Top-level initialization
# ============================================================================

def resolve_gc_bases(ip_result: IPDiscoveryResult) -> list[int]:
    """Resolve GC base addresses from IP discovery."""
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    bases = [0] * 6
    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.GC and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(bases) and addr != 0:
                    bases[i] = addr
    return bases


def _wait_rlc_autoload(dev: WindowsDevice, gc_base: list[int]) -> None:
    deadline = time.monotonic() + 5.0
    last_status = 0
    while time.monotonic() < deadline:
        last_status = _gc_reg(dev, gc_base, regRLC_RLCS_BOOTLOAD_STATUS,
                              base_idx=1)
        cp_stat = _gc_reg(dev, gc_base, regCP_STAT, base_idx=0)
        if (last_status & 0x80000000) or cp_stat == 0:
            print(f"  GFX: RLC autoload status=0x{last_status:08X}, "
                  f"CP_STAT=0x{cp_stat:08X}")
            return
        time.sleep(0.001)
    print(f"  GFX: WARNING RLC autoload not confirmed "
          f"(status=0x{last_status:08X})")


def _pulse_reset_bits(
    dev: WindowsDevice,
    gc_base: list[int],
    reg: int,
    reset_mask: int,
    *,
    base_idx: int,
) -> None:
    val = _gc_reg(dev, gc_base, reg, base_idx=base_idx)
    _gc_wreg(dev, gc_base, reg, val | reset_mask, base_idx=base_idx)
    val = _gc_reg(dev, gc_base, reg, base_idx=base_idx)
    _gc_wreg(dev, gc_base, reg, val & ~reset_mask, base_idx=base_idx)


def _config_mec_from_ucode(
    dev: WindowsDevice,
    gc_base: list[int],
    ucode_start: dict[str, int],
) -> None:
    missing = [name for name in ("PFP", "ME", "MEC")
               if name not in ucode_start]
    if missing:
        print(f"  GFX: Skipping CP program counters, missing {missing}")
        return

    grbm_select(dev, gc_base, me=0, pipe=0, queue=0)
    _gc_wreg_pair(
        dev, gc_base, regCP_PFP_PRGRM_CNTR_START,
        regCP_PFP_PRGRM_CNTR_START_HI, ucode_start["PFP"] >> 2,
        base_idx=0,
    )
    _gc_wreg_pair(
        dev, gc_base, regCP_ME_PRGRM_CNTR_START,
        regCP_ME_PRGRM_CNTR_START_HI, ucode_start["ME"] >> 2,
        base_idx=0,
    )
    grbm_deselect(dev, gc_base)

    _pulse_reset_bits(
        dev, gc_base, regCP_ME_CNTL,
        CP_ME_CNTL__PFP_PIPE0_RESET | CP_ME_CNTL__ME_PIPE0_RESET,
        base_idx=1,
    )
    val = _gc_reg(dev, gc_base, regCP_ME_CNTL, base_idx=1)
    val &= ~(CP_ME_CNTL__PFP_HALT | CP_ME_CNTL__ME_HALT)
    _gc_wreg(dev, gc_base, regCP_ME_CNTL, val, base_idx=1)

    grbm_select(dev, gc_base, me=1, pipe=0, queue=0)
    _gc_wreg_pair(
        dev, gc_base, regCP_MEC_RS64_PRGRM_CNTR_START,
        regCP_MEC_RS64_PRGRM_CNTR_START_HI, ucode_start["MEC"] >> 2,
        base_idx=1,
    )
    grbm_deselect(dev, gc_base)

    _pulse_reset_bits(
        dev, gc_base, regCP_MEC_RS64_CNTL,
        CP_MEC_RS64_CNTL__MEC_PIPE0_RESET,
        base_idx=1,
    )
    print("  GFX: CP PFP/ME/MEC program counters configured")


def _enable_mec(dev: WindowsDevice, gc_base: list[int]) -> None:
    val = _gc_reg(dev, gc_base, regCP_MEC_RS64_CNTL, base_idx=1)
    val &= ~(CP_MEC_RS64_CNTL__MEC_PIPE0_RESET |
             CP_MEC_RS64_CNTL__MEC_HALT)
    val |= CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE
    _gc_wreg(dev, gc_base, regCP_MEC_RS64_CNTL, val, base_idx=1)
    time.sleep(0.050)


def init_gfx_for_compute(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    psp_config: object | None = None,
) -> list[int]:
    """Initialize the minimal GC/MEC state needed for direct compute queues."""
    gc_base = resolve_gc_bases(ip_result)
    _wait_rlc_autoload(dev, gc_base)

    ucode_start = getattr(psp_config, "ucode_start", {}) if psp_config else {}
    if ucode_start:
        _config_mec_from_ucode(dev, gc_base, ucode_start)

    tcp_cntl = _gc_reg(dev, gc_base, regTCP_CNTL, base_idx=1)
    _gc_wreg(dev, gc_base, regTCP_CNTL, tcp_cntl | 0x20000000, base_idx=1)
    _gc_wreg(dev, gc_base, regRLC_CNTL, 0x1, base_idx=1)
    rlc_srm = _gc_reg(dev, gc_base, regRLC_SRM_CNTL, base_idx=1)
    _gc_wreg(dev, gc_base, regRLC_SRM_CNTL, rlc_srm | 0x3, base_idx=1)
    _gc_wreg(dev, gc_base, regRLC_SPM_MC_CNTL, 0xF, base_idx=1)

    grbm_cntl = _gc_reg(dev, gc_base, regGRBM_CNTL, base_idx=0)
    _gc_wreg(dev, gc_base, regGRBM_CNTL,
             (grbm_cntl & ~0xFFF) | 0xFF, base_idx=0)

    sh_mem_config = (3 << 2) | (3 << 14)
    sh_mem_bases = (1 << 16) | 2
    for vmid in range(16):
        grbm_select(dev, gc_base, me=0, pipe=0, queue=0, vmid=vmid)
        _gc_wreg(dev, gc_base, regSH_MEM_CONFIG, sh_mem_config, base_idx=1)
        _gc_wreg(dev, gc_base, regSH_MEM_BASES, sh_mem_bases, base_idx=1)
    grbm_deselect(dev, gc_base)

    _gc_wreg(dev, gc_base, regCP_MEC_DOORBELL_RANGE_LOWER, 0, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_MEC_DOORBELL_RANGE_UPPER,
             (0x8A * 2) << 2, base_idx=0)

    _enable_mec(dev, gc_base)
    mec_cntl = _gc_reg(dev, gc_base, regCP_MEC_RS64_CNTL, base_idx=1)
    print(f"  GFX: MEC enabled (CP_MEC_RS64_CNTL=0x{mec_cntl:08X})")
    return gc_base


def init_compute_queue(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
    *,
    pipe: int = 0,
    queue: int = 0,
    doorbell_index: int = DOORBELL_MEC_RING_START,
    ring_size: int = COMPUTE_RING_SIZE,
) -> ComputeQueueConfig:
    """Initialize a compute queue via direct MMIO register programming.

    Allocates all required DMA buffers, initializes the MQD, programs
    CP_HQD_* registers via GRBM_SELECT, and activates the queue.

    This bypasses MES entirely — suitable for a bare-metal approach
    where we own the GPU exclusively.

    Args:
        dev: Windows device backend.
        ip_result: Parsed IP discovery data.
        nbio_config: NBIO configuration (for doorbell BAR info).
        pipe: Compute pipe (0-7).
        queue: Queue within the pipe (0-7).
        doorbell_index: Doorbell index for this queue.
        ring_size: Ring buffer size in bytes (power of 2).

    Returns:
        Configured ComputeQueueConfig ready for packet submission.
    """
    gc_base = resolve_gc_bases(ip_result)
    config = ComputeQueueConfig(gc_base=gc_base)
    config.ring_size = ring_size
    config.me = 1  # MEC0
    config.pipe = pipe
    config.queue = queue
    config.doorbell_index = doorbell_index
    config.dev = dev
    config.nbio_config = nbio_config

    # Allocate ring buffer
    ring_cpu, ring_bus, ring_handle = _alloc_queue_buffer(dev, config, ring_size)
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle

    # Allocate MQD
    mqd_cpu, mqd_bus, mqd_handle = _alloc_queue_buffer(dev, config, 4096)
    config.mqd_cpu_addr = mqd_cpu
    config.mqd_bus_addr = mqd_bus
    config.mqd_dma_handle = mqd_handle

    # Allocate EOP buffer
    eop_cpu, eop_bus, eop_handle = _alloc_queue_buffer(dev, config, 4096)
    config.eop_cpu_addr = eop_cpu
    config.eop_bus_addr = eop_bus
    config.eop_dma_handle = eop_handle

    # Allocate WPTR writeback (8 bytes in a page)
    wptr_cpu, wptr_bus, wptr_handle = _alloc_queue_buffer(dev, config, 4096)
    config.wptr_cpu_addr = wptr_cpu
    config.wptr_bus_addr = wptr_bus
    config.wptr_dma_handle = wptr_handle

    # Allocate RPTR writeback
    rptr_cpu, rptr_bus, rptr_handle = _alloc_queue_buffer(dev, config, 4096)
    config.rptr_cpu_addr = rptr_cpu
    config.rptr_bus_addr = rptr_bus
    config.rptr_dma_handle = rptr_handle

    # Allocate fence buffer
    fence_cpu, fence_bus, fence_handle = _alloc_queue_buffer(dev, config, 4096)
    config.fence_cpu_addr = fence_cpu
    config.fence_bus_addr = fence_bus
    config.fence_dma_handle = fence_handle

    # Map doorbell BAR for this queue (if available from NBIO)
    if nbio_config.doorbell_phys_addr != 0:
        doorbell_offset = doorbell_index * 4  # doorbell_index is a DWORD index
        try:
            db_addr, db_handle = dev.driver.map_bar(
                2, doorbell_offset, 8)  # BAR2 = doorbell
            config.doorbell_cpu_addr = db_addr
        except RuntimeError:
            # Doorbell mapping may fail if BAR2 is not doorbell
            pass

    # Initialize MQD
    _init_compute_mqd(config)

    # Program HQD registers via MMIO
    _activate_compute_queue_mmio(dev, config)

    config.active = True
    config.wptr = 0

    print(f"  Compute: Queue ME={config.me} pipe={config.pipe} "
          f"queue={config.queue} activated")
    print(f"  Compute: Ring at bus 0x{ring_bus:012X}, "
          f"size={ring_size // 1024}KB")
    print(f"  Compute: Doorbell index=0x{doorbell_index:X}")

    return config


def init_sdma_queue(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
    *,
    instance: int = 0,
    doorbell_index: int = DOORBELL_SDMA_START,
    ring_size: int = SDMA_RING_SIZE,
) -> SDMAQueueConfig:
    """Initialize an SDMA queue.

    Allocates DMA buffers and initializes the SDMA MQD. The MQD is
    programmed but queue activation depends on MES or direct SDMA
    register programming.

    Args:
        dev: Windows device backend.
        ip_result: Parsed IP discovery data.
        nbio_config: NBIO configuration.
        instance: SDMA instance (0 or 1).
        doorbell_index: Doorbell index.
        ring_size: Ring buffer size in bytes.

    Returns:
        Configured SDMAQueueConfig.
    """
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    sdma_bases = [0] * 6
    inst_count = 0
    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.SDMA0:
            if inst_count == instance:
                for i, addr in enumerate(block.base_addresses):
                    if i < len(sdma_bases) and addr != 0:
                        sdma_bases[i] = addr
                break
            inst_count += 1

    config = SDMAQueueConfig(sdma_base=sdma_bases)
    config.ring_size = ring_size
    config.instance = instance
    config.doorbell_index = doorbell_index

    # Allocate ring buffer
    ring_cpu, ring_bus, ring_handle = dev.driver.alloc_dma(ring_size)
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle
    ctypes.memset(ring_cpu, 0, ring_size)

    # Allocate MQD
    mqd_cpu, mqd_bus, mqd_handle = dev.driver.alloc_dma(4096)
    config.mqd_cpu_addr = mqd_cpu
    config.mqd_bus_addr = mqd_bus
    config.mqd_dma_handle = mqd_handle

    # Allocate WPTR writeback
    wptr_cpu, wptr_bus, wptr_handle = dev.driver.alloc_dma(4096)
    config.wptr_cpu_addr = wptr_cpu
    config.wptr_bus_addr = wptr_bus
    config.wptr_dma_handle = wptr_handle
    ctypes.memset(wptr_cpu, 0, 4096)

    # Allocate RPTR writeback
    rptr_cpu, rptr_bus, rptr_handle = dev.driver.alloc_dma(4096)
    config.rptr_cpu_addr = rptr_cpu
    config.rptr_bus_addr = rptr_bus
    config.rptr_dma_handle = rptr_handle
    ctypes.memset(rptr_cpu, 0, 4096)

    # Allocate fence buffer
    fence_cpu, fence_bus, fence_handle = dev.driver.alloc_dma(4096)
    config.fence_cpu_addr = fence_cpu
    config.fence_bus_addr = fence_bus
    config.fence_dma_handle = fence_handle
    ctypes.memset(fence_cpu, 0, 4096)

    # Map doorbell BAR
    if nbio_config.doorbell_phys_addr != 0:
        doorbell_offset = doorbell_index * 8
        try:
            db_addr, db_handle = dev.driver.map_bar(2, doorbell_offset, 8)
            config.doorbell_cpu_addr = db_addr
        except RuntimeError:
            pass

    # Initialize MQD
    _init_sdma_mqd(config)

    print(f"  SDMA{instance}: MQD initialized at bus 0x{mqd_bus:012X}")
    print(f"  SDMA{instance}: Ring at bus 0x{ring_bus:012X}, "
          f"size={ring_size // 1024}KB")

    return config


# ============================================================================
# NOP + Fence test
# ============================================================================

def test_compute_nop_fence(
    config: ComputeQueueConfig,
    fence_seq: int = 1,
) -> bool:
    """Submit NOP + RELEASE_MEM and verify fence completion.

    This is the first smoke test: submit a NOP packet followed by
    RELEASE_MEM that writes a fence value to memory. If the GPU is
    alive and the queue is working, the fence value will appear.

    Returns True if the fence was signaled within the timeout.
    """
    from amd_gpu_driver.commands.pm4 import PM4PacketBuilder

    # Clear fence
    ctypes.c_uint64.from_address(config.fence_cpu_addr).value = 0

    # Build NOP + RELEASE_MEM
    builder = PM4PacketBuilder()
    builder.nop(4)  # A few NOPs for padding
    builder.release_mem(
        addr=config.fence_bus_addr,
        value=fence_seq,
        cache_flush=True,
    )

    # Submit
    packet_data = builder.build()
    submit_compute_packets(config, packet_data)

    # Wait for fence
    return wait_fence(config, fence_seq, timeout_ms=5000)
