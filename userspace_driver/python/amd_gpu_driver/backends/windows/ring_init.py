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
# These are DWORD offsets relative to GC base_index 1
# ============================================================================

# GRBM
regGRBM_GFX_CNTL = 0x0900  # base_index 0

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

# CP HQD registers (base_index 1, used for direct queue programming)
regCP_MQD_BASE_ADDR = 0x1FA9
regCP_MQD_BASE_ADDR_HI = 0x1FAA
regCP_HQD_ACTIVE = 0x1FAB
regCP_HQD_VMID = 0x1FAC
regCP_HQD_PERSISTENT_STATE = 0x1FAD
regCP_HQD_PIPE_PRIORITY = 0x1FAE
regCP_HQD_QUEUE_PRIORITY = 0x1FAF
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
PQ_CONTROL__NO_UPDATE_RPTR = 1 << 27
PQ_CONTROL__UNORD_DISPATCH = 1 << 28
PQ_CONTROL__TUNNEL_DISPATCH = 1 << 29
PQ_CONTROL__PRIV_STATE = 1 << 30
PQ_CONTROL__KMD_QUEUE = 1 << 31

# CP_HQD_PERSISTENT_STATE: preload size
HQD_PERSISTENT_STATE__PRELOAD_SIZE = 0x55

# CP_HQD_EOP_CONTROL
EOP_SIZE_SHIFT = 0

# GRBM_GFX_CNTL bit fields
GRBM_GFX_CNTL__PIPEID__SHIFT = 0
GRBM_GFX_CNTL__MEID__SHIFT = 2
GRBM_GFX_CNTL__VMID__SHIFT = 4
GRBM_GFX_CNTL__QUEUEID__SHIFT = 8

# Doorbell assignments (LAYOUT1 for GFX12)
DOORBELL_KIQ = 0x000
DOORBELL_HIQ = 0x001
DOORBELL_MEC_RING_START = 0x008
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

    # GRBM_GFX_CNTL is at base_index 0
    offset = (gc_base[0] + regGRBM_GFX_CNTL) * 4
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
    mqd[132] = HQD_PERSISTENT_STATE__PRELOAD_SIZE

    # cp_hqd_pipe_priority (offset 133) and queue_priority (offset 134)
    mqd[133] = 0
    mqd[134] = 0

    # cp_hqd_quantum (offset 135)
    mqd[135] = 0

    # cp_hqd_pq_base (offsets 136-137) — ring buffer address >> 8
    pq_base = config.ring_bus_addr >> 8
    mqd[136] = pq_base & 0xFFFFFFFF
    mqd[137] = (pq_base >> 32) & 0xFFFFFFFF

    # cp_hqd_pq_rptr (offset 138) = 0
    mqd[138] = 0

    # cp_hqd_pq_rptr_report_addr (offsets 139-140)
    mqd[139] = config.rptr_bus_addr & 0xFFFFFFFC
    mqd[140] = (config.rptr_bus_addr >> 32) & 0xFFFF

    # cp_hqd_pq_wptr_poll_addr (offsets 141-142)
    mqd[141] = config.wptr_bus_addr & 0xFFFFFFF8
    mqd[142] = (config.wptr_bus_addr >> 32) & 0xFFFF

    # cp_hqd_pq_doorbell_control (offset 143)
    doorbell_ctrl = 0
    doorbell_ctrl |= (config.doorbell_index & 0x0FFFFFFC) << DOORBELL_OFFSET__SHIFT
    doorbell_ctrl |= DOORBELL_EN
    mqd[143] = doorbell_ctrl

    # cp_hqd_pq_control (offset 145)
    ring_size_log2 = int(math.log2(config.ring_size // 4)) - 1
    pq_control = 0
    pq_control |= (ring_size_log2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT
    # RPTR_BLOCK_SIZE = log2(4096/4) - 1 = 9
    pq_control |= (9 & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT
    pq_control |= PQ_CONTROL__UNORD_DISPATCH
    pq_control |= PQ_CONTROL__NO_UPDATE_RPTR
    pq_control |= PQ_CONTROL__PRIV_STATE
    pq_control |= PQ_CONTROL__KMD_QUEUE
    mqd[145] = pq_control

    # cp_mqd_control (offset 162)
    mqd[162] = 0  # VMID=0

    # cp_hqd_eop_base_addr (offsets 165-166) — EOP buffer address >> 8
    eop_base = config.eop_bus_addr >> 8
    mqd[165] = eop_base & 0xFFFFFFFF
    mqd[166] = (eop_base >> 32) & 0xFFFFFFFF

    # cp_hqd_eop_control (offset 167) — EOP_SIZE = log2(2048/4) - 1 = 8
    mqd[167] = 8

    # cp_hqd_pq_wptr_lo/hi (offsets 182-183) = 0
    mqd[182] = 0
    mqd[183] = 0

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
    _gc_wreg(dev, gc_base, regCP_HQD_ACTIVE, 0)

    # Set VMID
    _gc_wreg(dev, gc_base, regCP_HQD_VMID, 0)

    # Disable doorbell initially
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL, 0)

    # MQD base address
    _gc_wreg(dev, gc_base, regCP_MQD_BASE_ADDR,
             config.mqd_bus_addr & 0xFFFFFFFC)
    _gc_wreg(dev, gc_base, regCP_MQD_BASE_ADDR_HI,
             (config.mqd_bus_addr >> 32) & 0xFFFFFFFF)

    # MQD control (VMID = 0)
    _gc_wreg(dev, gc_base, regCP_MQD_CONTROL, 0)

    # Ring buffer base (address >> 8)
    pq_base = config.ring_bus_addr >> 8
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_BASE,
             pq_base & 0xFFFFFFFF)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_BASE_HI,
             (pq_base >> 32) & 0xFFFFFFFF)

    # RPTR report address
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR_REPORT_ADDR,
             config.rptr_bus_addr & 0xFFFFFFFC)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI,
             (config.rptr_bus_addr >> 32) & 0xFFFF)

    # PQ control (ring size, flags)
    ring_size_log2 = int(math.log2(config.ring_size // 4)) - 1
    pq_control = 0
    pq_control |= (ring_size_log2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT
    pq_control |= (9 & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT
    pq_control |= PQ_CONTROL__UNORD_DISPATCH
    pq_control |= PQ_CONTROL__NO_UPDATE_RPTR
    pq_control |= PQ_CONTROL__PRIV_STATE
    pq_control |= PQ_CONTROL__KMD_QUEUE
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_CONTROL, pq_control)

    # WPTR poll address
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_POLL_ADDR,
             config.wptr_bus_addr & 0xFFFFFFF8)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_POLL_ADDR_HI,
             (config.wptr_bus_addr >> 32) & 0xFFFF)

    # Reset RPTR and WPTR
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR, 0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_LO, 0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_HI, 0)

    # Doorbell control (enable doorbell)
    doorbell_ctrl = 0
    doorbell_ctrl |= (config.doorbell_index & 0x0FFFFFFC) << DOORBELL_OFFSET__SHIFT
    doorbell_ctrl |= DOORBELL_EN
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL, doorbell_ctrl)

    # Persistent state (preload)
    _gc_wreg(dev, gc_base, regCP_HQD_PERSISTENT_STATE,
             HQD_PERSISTENT_STATE__PRELOAD_SIZE)

    # EOP buffer
    eop_base = config.eop_bus_addr >> 8
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_BASE_ADDR,
             eop_base & 0xFFFFFFFF)
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_BASE_ADDR_HI,
             (eop_base >> 32) & 0xFFFFFFFF)
    # EOP_SIZE = log2(2048/4) - 1 = 8
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_CONTROL, 8)

    # Activate the queue
    _gc_wreg(dev, gc_base, regCP_HQD_ACTIVE, 1)

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

    # Allocate ring buffer
    ring_cpu, ring_bus, ring_handle = dev.driver.alloc_dma(ring_size)
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle
    # Zero the ring
    ctypes.memset(ring_cpu, 0, ring_size)

    # Allocate MQD
    mqd_cpu, mqd_bus, mqd_handle = dev.driver.alloc_dma(4096)  # page-aligned
    config.mqd_cpu_addr = mqd_cpu
    config.mqd_bus_addr = mqd_bus
    config.mqd_dma_handle = mqd_handle

    # Allocate EOP buffer
    eop_cpu, eop_bus, eop_handle = dev.driver.alloc_dma(4096)
    config.eop_cpu_addr = eop_cpu
    config.eop_bus_addr = eop_bus
    config.eop_dma_handle = eop_handle
    ctypes.memset(eop_cpu, 0, 4096)

    # Allocate WPTR writeback (8 bytes in a page)
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

    # Map doorbell BAR for this queue (if available from NBIO)
    if nbio_config.doorbell_phys_addr != 0:
        doorbell_offset = doorbell_index * 8  # Each doorbell is 8 bytes
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
