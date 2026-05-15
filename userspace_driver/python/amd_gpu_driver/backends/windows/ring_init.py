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

For early bring-up we can still use direct MMIO, but the preferred GFX12 path
is MES-managed legacy queue mapping through a directly bootstrapped MES KIQ.

Reference: Linux amdgpu gfx_v12_0.c, mes_v12_0.c, sdma_v7_0.c, v12_structs.h
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult
    from amd_gpu_driver.backends.windows.gmc_init import GMCConfig
    from amd_gpu_driver.backends.windows.nbio_init import NBIOConfig
    from amd_gpu_driver.backends.windows.smu_init import SMUConfig


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
regRLC_GPM_THREAD_ENABLE = 0x4C45    # base_index 1
regSH_MEM_BASES = 0x09E3             # base_index 1
regSH_MEM_CONFIG = 0x09E4            # base_index 1
regTCP_CNTL = 0x19A2                 # base_index 1
regGFX_IMU_RLC_BOOTLOADER_ADDR_HI = 0x5F81  # base_index 1
regGFX_IMU_RLC_BOOTLOADER_ADDR_LO = 0x5F82  # base_index 1
regGFX_IMU_RLC_BOOTLOADER_SIZE = 0x5F83     # base_index 1
regGFX_IMU_C2PMSG_16 = 0x4010               # base_index 1
regGFX_IMU_C2PMSG_ACCESS_CTRL0 = 0x4040     # base_index 1
regGFX_IMU_C2PMSG_ACCESS_CTRL1 = 0x4041     # base_index 1
regGFX_IMU_SCRATCH_10 = 0x4072              # base_index 1
regGFX_IMU_RLC_RAM_INDEX = 0x40AC           # base_index 1
regGFX_IMU_RLC_RAM_ADDR_HIGH = 0x40AD       # base_index 1
regGFX_IMU_RLC_RAM_ADDR_LOW = 0x40AE        # base_index 1
regGFX_IMU_RLC_RAM_DATA = 0x40AF            # base_index 1
regGFX_IMU_CORE_CTRL = 0x40B6               # base_index 1
regGFX_IMU_GFX_RESET_CTRL = 0x40BC          # base_index 1
regGFX_IMU_D_RAM_ADDR = 0x40FC              # base_index 1
regGFX_IMU_D_RAM_DATA = 0x40FD              # base_index 1
regCPG_PSP_DEBUG = 0x5C10                   # base_index 1
regCPC_PSP_DEBUG = 0x5C11                   # base_index 1
regGFX_IMU_I_RAM_ADDR = 0x5F90              # base_index 1
regGFX_IMU_I_RAM_DATA = 0x5F91              # base_index 1

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
regCP_HQD_IB_BASE_ADDR = 0x1FBB
regCP_HQD_IB_BASE_ADDR_HI = 0x1FBC
regCP_HQD_IB_RPTR = 0x1FBD
regCP_HQD_IB_CONTROL = 0x1FBE
regCP_MQD_CONTROL = 0x1FCB
regCP_HQD_EOP_BASE_ADDR = 0x1FCE
regCP_HQD_EOP_BASE_ADDR_HI = 0x1FCF
regCP_HQD_EOP_CONTROL = 0x1FD0
regCP_HQD_HQ_STATUS0 = 0x1FC9
regCP_HQD_GFX_CONTROL = 0x1E9F
regCP_HQD_AQL_CONTROL = 0x1FDE
regCP_HQD_PQ_WPTR_LO = 0x1FDF
regCP_HQD_PQ_WPTR_HI = 0x1FE0
regCP_HQD_DEQUEUE_STATUS = 0x1FE8

# RLC scheduler register
regRLC_CP_SCHEDULERS = 0x098A  # base_index 1
regCP_UNMAPPED_DOORBELL = 0x0880  # base_index 1

# CP_MES_CNTL bit fields
CP_MES_CNTL__MES_INVALIDATE_ICACHE = 1 << 4
CP_MES_CNTL__MES_PIPE0_RESET = 1 << 16
CP_MES_CNTL__MES_PIPE1_RESET = 1 << 17
CP_MES_CNTL__MES_PIPE0_ACTIVE = 1 << 26
CP_MES_CNTL__MES_PIPE1_ACTIVE = 1 << 27
CP_MES_CNTL__MES_HALT = 1 << 30
CP_MES_IC_OP_CNTL__INVALIDATE_CACHE = 1 << 0
CP_MES_IC_OP_CNTL__PRIME_ICACHE = 1 << 4

# CP_HQD_PQ_DOORBELL_CONTROL bit fields
DOORBELL_OFFSET__SHIFT = 2
DOORBELL_EN = 1 << 30
DOORBELL_SOURCE = 1 << 28
DOORBELL_HIT = 1 << 31

# CP_HQD_PQ_CONTROL bit fields
PQ_CONTROL__QUEUE_SIZE__SHIFT = 0
PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT = 8
PQ_CONTROL__QUEUE_FULL_EN = 1 << 14
PQ_CONTROL__PQ_EMPTY = 1 << 15
PQ_CONTROL__SLOT_BASED_WPTR__SHIFT = 18
PQ_CONTROL__MIN_AVAIL_SIZE__SHIFT = 20
PQ_CONTROL__NO_UPDATE_RPTR = 1 << 27
PQ_CONTROL__UNORD_DISPATCH = 1 << 28
PQ_CONTROL__TUNNEL_DISPATCH = 1 << 29
PQ_CONTROL__PRIV_STATE = 1 << 30
PQ_CONTROL__KMD_QUEUE = 1 << 31

# CP_HQD_PERSISTENT_STATE
HQD_PERSISTENT_STATE__PRELOAD_REQ = 1 << 0
HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT = 8
HQD_PERSISTENT_STATE__PRELOAD_SIZE = 0x55
HQD_PERSISTENT_STATE_DEFAULT = 0x0BE05501

# CP_HQD_EOP_CONTROL
EOP_SIZE_SHIFT = 0
CP_HQD_GFX_CONTROL__DB_UPDATED_MSG_EN = 1 << 15

# CP_UNMAPPED_DOORBELL
CP_UNMAPPED_DOORBELL__ENABLE = 1 << 0
CP_UNMAPPED_DOORBELL__PROC_LSB__SHIFT = 8
CP_UNMAPPED_DOORBELL__PROC_LSB_MASK = 0x00001F00

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
CP_MEC_RS64_CNTL__MEC_PIPE1_RESET = 0x00020000
CP_MEC_RS64_CNTL__MEC_PIPE2_RESET = 0x00040000
CP_MEC_RS64_CNTL__MEC_PIPE3_RESET = 0x00080000
CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE = 0x04000000
CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE = 0x08000000
CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE = 0x10000000
CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE = 0x20000000
CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE = 0x00000010
CP_MEC_RS64_CNTL__MEC_HALT = 0x40000000
RLC_GPM_THREAD_ENABLE__THREAD0_ENABLE = 1 << 0
RLC_GPM_THREAD_ENABLE__THREAD1_ENABLE = 1 << 1
GFX_IMU_RLC_RAM_INDEX__RAM_VALID = 1 << 31
CPC_PSP_DEBUG__GPA_OVERRIDE = 1 << 3
CPG_PSP_DEBUG__GPA_OVERRIDE = 1 << 3

# Doorbell assignments for SOC24/Navi-style doorbell layout. Kernel constants
# are expressed as 64-bit doorbell slots; CP HQD programming and WDOORBELL64
# use DWORD offsets, so these are shifted left by one.
DOORBELL_KIQ = 0x000
DOORBELL_HIQ = 0x002
DOORBELL_DIQ = 0x004
DOORBELL_MEC_RING_START = 0x006
DOORBELL_MEC_RING_STRIDE = 0x2
DOORBELL_MES_RING0 = 0x016
DOORBELL_MES_RING1 = 0x018
DOORBELL_SDMA_START = 0x100

# MES pipe IDs
MES_PIPE_SCHED = 0
MES_PIPE_KIQ = 1

# MES scheduler ABI (mes_v12_api_def.h)
MES_API_TYPE_SCHEDULER = 1
MES_API_FRAME_DWORDS = 64
MES_API_FRAME_BYTES = MES_API_FRAME_DWORDS * 4
MES_SCH_API_SET_HW_RSRC = 0
MES_SCH_API_ADD_QUEUE = 2
MES_SCH_API_QUERY_SCHEDULER_STATUS = 11
MES_QUEUE_TYPE_COMPUTE = 1
MES_QUEUE_TYPE_SCHQ = 3
MES_RING_SIZE = 4 * 1024
MES_EOP_SIZE = 2048
MES_STATUS_BUFFER_SIZE = 4096
MES_PROCESS_QUANTUM = 100000
MES_GANG_QUANTUM = 10000
MES_PRIORITY_NORMAL = 1
MES_ADD_QUEUE_FLAG_MAP_LEGACY_KQ = 1 << 13
MES_SET_HW_RESOURCES_FLAGS = (
    (1 << 0) |   # disable_reset
    (1 << 1) |   # use_different_vmid_compute
    (1 << 2) |   # disable_mes_log
    (1 << 6) |   # enable_level_process_quantum_check
    (1 << 10) |  # enable_reg_active_poll
    (1 << 19)    # unmapped_doorbell_handling = basic
)

# RLC backdoor autoload firmware IDs (SOC24 / GFX12).
SOC24_FIRMWARE_ID_INVALID = 0
SOC24_FIRMWARE_ID_RLC_G_UCODE = 1
SOC24_FIRMWARE_ID_RLC_TOC = 2
SOC24_FIRMWARE_ID_RLCG_SCRATCH = 3
SOC24_FIRMWARE_ID_RLC_SRM_ARAM = 4
SOC24_FIRMWARE_ID_RLX6_UCODE = 7
SOC24_FIRMWARE_ID_RLX6_DRAM_BOOT = 9
SOC24_FIRMWARE_ID_SDMA_UCODE_TH0 = 11
SOC24_FIRMWARE_ID_RS64_MES_P0 = 16
SOC24_FIRMWARE_ID_RS64_MES_P1 = 17
SOC24_FIRMWARE_ID_RS64_PFP = 18
SOC24_FIRMWARE_ID_RS64_ME = 19
SOC24_FIRMWARE_ID_RS64_MEC = 20
SOC24_FIRMWARE_ID_RS64_MES_P0_STACK = 21
SOC24_FIRMWARE_ID_RS64_MES_P1_STACK = 22
SOC24_FIRMWARE_ID_RS64_PFP_P0_STACK = 23
SOC24_FIRMWARE_ID_RS64_PFP_P1_STACK = 24
SOC24_FIRMWARE_ID_RS64_ME_P0_STACK = 25
SOC24_FIRMWARE_ID_RS64_ME_P1_STACK = 26
SOC24_FIRMWARE_ID_RS64_MEC_P0_STACK = 27
SOC24_FIRMWARE_ID_RS64_MEC_P1_STACK = 28
SOC24_FIRMWARE_ID_RS64_MEC_P2_STACK = 29
SOC24_FIRMWARE_ID_RS64_MEC_P3_STACK = 30
SOC24_FIRMWARE_ID_UMF_ZONE_PAD = 42
SOC24_FIRMWARE_ID_MAX = 43

RLC_TOC_OFFSET_DWUNIT = 8
RLC_SIZE_MULTIPLE = 1024
RLC_TOC_UMF_SIZE = 23 * 1024 * 1024
RLC_TOC_FORMAT_API = 165
RLC_AUTOLOAD_ALIGNMENT = 64 * 1024
IMU_TRANSFER_RAM_MASK = 0x001C0000
IMU_RLC_RAM_GOLDEN_12_0_1 = [
    (1, 0x5B88, 0x000001E4, 0x001C0000, "regCH_PIPE_STEER"),
    (1, 0x5B85, 0x000001E4, 0x001C0000, "regGL1X_PIPE_STEER"),
    (1, 0x5B84, 0x000001E4, 0x001C0000, "regGL1_PIPE_STEER"),
    (1, 0x5B80, 0x13571357, 0x001C0000, "regGL2_PIPE_STEER_0"),
    (1, 0x5B81, 0x64206420, 0x001C0000, "regGL2_PIPE_STEER_1"),
    (1, 0x5B82, 0x02460246, 0x001C0000, "regGL2_PIPE_STEER_2"),
    (1, 0x5B83, 0x75317531, 0x001C0000, "regGL2_PIPE_STEER_3"),
    (1, 0x2E0C, 0xC0D41183, 0x001C0000, "regGL2C_CTRL3"),
    (0, 0x000E, 0x0507D1C0, 0x001C0000, "regSDMA0_CHICKEN_BITS"),
    (0, 0x060E, 0x0507D1C0, 0x001C0000, "regSDMA1_CHICKEN_BITS"),
    (0, 0x0F62, 0x00600100, 0x001C0000, "regCP_RB_WPTR_POLL_CNTL"),
    (0, 0x17A3, 0x003F7FFF, 0x001C0000, "regGC_EA_CPWD_SDP_CREDITS"),
    (0, 0x1823, 0x003F7EBF, 0x001C0000, "regGC_EA_SE_SDP_CREDITS"),
    (0, 0x17A4, 0x02E00000, 0x001C0000, "regGC_EA_CPWD_SDP_TAG_RESERVE0"),
    (0, 0x17A5, 0x0001A078, 0x001C0000, "regGC_EA_CPWD_SDP_TAG_RESERVE1"),
    (0, 0x17A6, 0x00000000, 0x001C0000, "regGC_EA_CPWD_SDP_TAG_RESERVE2"),
    (0, 0x1824, 0x00000000, 0x001C0000, "regGC_EA_SE_SDP_TAG_RESERVE0"),
    (0, 0x1825, 0x00012030, 0x001C0000, "regGC_EA_SE_SDP_TAG_RESERVE1"),
    (0, 0x1826, 0x00000000, 0x001C0000, "regGC_EA_SE_SDP_TAG_RESERVE2"),
    (0, 0x17A7, 0x19041000, 0x001C0000, "regGC_EA_CPWD_SDP_VCC_RESERVE0"),
    (0, 0x17A8, 0x80000000, 0x001C0000, "regGC_EA_CPWD_SDP_VCC_RESERVE1"),
    (0, 0x1827, 0x1E080000, 0x001C0000, "regGC_EA_SE_SDP_VCC_RESERVE0"),
    (0, 0x1828, 0x80000000, 0x001C0000, "regGC_EA_SE_SDP_VCC_RESERVE1"),
    (0, 0x17A2, 0x00000880, 0x001C0000, "regGC_EA_CPWD_SDP_PRIORITY"),
    (0, 0x1822, 0x00008880, 0x001C0000, "regGC_EA_SE_SDP_PRIORITY"),
    (0, 0x17A1, 0x00000017, 0x001C0000, "regGC_EA_CPWD_SDP_ARB_FINAL"),
    (0, 0x1821, 0x00000077, 0x001C0000, "regGC_EA_SE_SDP_ARB_FINAL"),
    (0, 0x181F, 0x00000001, 0x001C0000, "regGC_EA_CPWD_SDP_ENABLE"),
    (0, 0x189F, 0x00000001, 0x001C0000, "regGC_EA_SE_SDP_ENABLE"),
    (0, 0x15CD, 0x00020000, 0x001C0000, "regGCVM_L2_PROTECTION_FAULT_CNTL2"),
    (0, 0x15B0, 0x0000000C, 0x001C0000, "regGCMC_VM_APT_CNTL"),
    (0, 0x15AD, 0x000FFFFF, 0x001C0000, "regGCMC_VM_CACHEABLE_DRAM_ADDRESS_END"),
    (0, 0x17AC, 0x00000091, 0x001C0000, "regGC_EA_CPWD_MISC"),
    (0, 0x182C, 0x00000091, 0x001C0000, "regGC_EA_SE_MISC"),
    (1, 0x2200, 0xE0000000, 0x001C0000, "regGRBM_GFX_INDEX"),
    (1, 0x1990, 0x00008500, 0x001C0000, "regGCR_GENERAL_CNTL"),
    (0, 0x1025, 0x00880007, 0x001C0000, "regPA_CL_ENHANCE"),
    (0, 0x12C5, 0x00000001, 0x001C0000, "regTD_CNTL"),
    (1, 0x2200, 0x00000000, 0x001C0000, "regGRBM_GFX_INDEX"),
    (1, 0x1880, 0x01E00000, 0x001C0000, "regRMI_GENERAL_CNTL"),
    (1, 0x2200, 0x00000001, 0x001C0000, "regGRBM_GFX_INDEX"),
    (1, 0x1880, 0x01E00000, 0x001C0000, "regRMI_GENERAL_CNTL"),
    (1, 0x2200, 0x00000100, 0x001C0000, "regGRBM_GFX_INDEX"),
    (1, 0x1880, 0x01E00000, 0x001C0000, "regRMI_GENERAL_CNTL"),
    (1, 0x2200, 0x00000101, 0x001C0000, "regGRBM_GFX_INDEX"),
    (1, 0x1880, 0x01E00000, 0x001C0000, "regRMI_GENERAL_CNTL"),
    (1, 0x2200, 0xE0000000, 0x001C0000, "regGRBM_GFX_INDEX"),
    (0, 0x13DE, 0x08200545, 0x001C0000, "regGB_ADDR_CONFIG"),
    (1, 0x3808, 0x00000000, 0x001C0000, "regGRBMH_CP_PERFMON_CNTL"),
    (1, 0x3C02, 0x000FFFFF, 0x001C0000, "regCB_PERFCOUNTER0_SELECT1"),
    (0, 0x1E0C, 0x00020000, 0x001C0000, "regCP_DEBUG_2"),
    (0, 0x1E21, 0x00500010, 0x001C0000, "regCP_CPC_DEBUG"),
    (0, 0x161B, 0x00000500, 0x001C0000, "regGCMC_VM_MX_L1_TLB_CNTL"),
    (0, 0x1619, 0x00000001, 0x001C0000, "regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR"),
    (0, 0x161A, 0x00000000, 0x001C0000, "regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR"),
    (0, 0x15B1, 0x00000000, 0x001C0000, "regGCMC_VM_LOCAL_FB_ADDRESS_START"),
    (0, 0x15B2, 0x0000000F, 0x001C0000, "regGCMC_VM_LOCAL_FB_ADDRESS_END"),
    (0, 0x1614, 0x00006000, 0x001C0000, "regGCMC_VM_FB_LOCATION_BASE"),
    (0, 0x1615, 0x0000600F, 0x001C0000, "regGCMC_VM_FB_LOCATION_TOP"),
    (0, 0x1624, 0x00000000, 0x001C0000, "regGCVM_CONTEXT0_CNTL"),
    (0, 0x1625, 0x00000000, 0x001C0000, "regGCVM_CONTEXT1_CNTL"),
    (0, 0x15A4, 0xFF800000, 0xE0000000, "regGCMC_VM_NB_TOP_OF_DRAM_SLOT1"),
    (0, 0x15A5, 0x00000001, 0x001C0000, "regGCMC_VM_NB_LOWER_TOP_OF_DRAM2"),
    (0, 0x15A6, 0x0000FFFF, 0x001C0000, "regGCMC_VM_NB_UPPER_TOP_OF_DRAM2"),
    (0, 0x1618, 0x00000000, 0x001C0000, "regGCMC_VM_AGP_BASE"),
    (0, 0x1617, 0x00000002, 0x001C0000, "regGCMC_VM_AGP_BOT"),
    (0, 0x1616, 0x00000000, 0x001C0000, "regGCMC_VM_AGP_TOP"),
    (0, 0x15CC, 0x00001FFC, 0x001C0000, "regGCVM_L2_PROTECTION_FAULT_CNTL"),
    (0, 0x161B, 0x00000551, 0x001C0000, "regGCMC_VM_MX_L1_TLB_CNTL"),
    (0, 0x15C4, 0x00080603, 0x001C0000, "regGCVM_L2_CNTL"),
    (0, 0x15C5, 0x00000003, 0x001C0000, "regGCVM_L2_CNTL2"),
    (0, 0x15C6, 0x00100003, 0x001C0000, "regGCVM_L2_CNTL3"),
    (0, 0x15E3, 0x00003FE0, 0x001C0000, "regGCVM_L2_CNTL5"),
    (0, 0x1619, 0x0003D000, 0x001C0000, "regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR"),
    (0, 0x161A, 0x0003D7FF, 0x001C0000, "regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR"),
    (0, 0x15A8, 0x00000000, 0x001C0000, "regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB"),
    (0, 0x15A9, 0x00000000, 0x001C0000, "regGCMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB"),
]

# Default ring sizes
COMPUTE_RING_SIZE = 4 * 1024    # 4KB, matching the lite direct queue
SDMA_RING_SIZE = 256 * 1024     # 256KB
EOP_BUFFER_SIZE = 4 * 1024      # 4KB per EOP buffer

# v12_compute_mqd constants
MQD_HEADER = 0xC0310800
MQD_SIZE = 256 * 4  # 256 DWORDs = 1024 bytes

# v12_sdma_mqd size
SDMA_MQD_SIZE = 128 * 4  # 128 DWORDs = 512 bytes


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class RLCTOCEntry:
    """One firmware placement entry from the GFX12 RLC autoload TOC."""
    id: int
    offset: int
    size: int
    size_x16: bool = False


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


@dataclass
class MESRingConfig:
    """A directly mapped MES ring used to submit scheduler API frames."""

    gc_base: list[int]
    mmhub_base: list[int]
    osssys_base: list[int]
    pipe: int = MES_PIPE_KIQ
    doorbell_index: int = DOORBELL_MES_RING1

    ring_bus_addr: int = 0
    ring_cpu_addr: int = 0
    ring_dma_handle: int = 0
    ring_size: int = MES_RING_SIZE

    mqd_bus_addr: int = 0
    mqd_cpu_addr: int = 0
    mqd_dma_handle: int = 0

    eop_bus_addr: int = 0
    eop_cpu_addr: int = 0
    eop_dma_handle: int = 0

    wptr_bus_addr: int = 0
    wptr_cpu_addr: int = 0
    wptr_dma_handle: int = 0

    rptr_bus_addr: int = 0
    rptr_cpu_addr: int = 0
    rptr_dma_handle: int = 0

    status_bus_addr: int = 0
    status_cpu_addr: int = 0
    status_dma_handle: int = 0

    sch_ctx_bus_addr: int = 0
    sch_ctx_cpu_addr: int = 0
    sch_ctx_dma_handle: int = 0

    query_status_fence_bus_addr: int = 0
    query_status_fence_cpu_addr: int = 0
    query_status_fence_dma_handle: int = 0

    doorbell_cpu_addr: int = 0
    wptr: int = 0
    sequence: int = 0
    ready: bool = False
    scheduler: object | None = None

    dev: object | None = None
    nbio_config: object | None = None
    memory_handles: list[object] = field(default_factory=list)


class _MESAPIStatus(ctypes.LittleEndianStructure):
    _fields_ = [
        ("api_completion_fence_addr", ctypes.c_uint64),
        ("api_completion_fence_value", ctypes.c_uint64),
    ]


class _MESAddQueuePacket(ctypes.LittleEndianStructure):
    _pack_ = 8
    _fields_ = [
        ("header", ctypes.c_uint32),
        ("process_id", ctypes.c_uint32),
        ("page_table_base_addr", ctypes.c_uint64),
        ("process_va_start", ctypes.c_uint64),
        ("process_va_end", ctypes.c_uint64),
        ("process_quantum", ctypes.c_uint64),
        ("process_context_addr", ctypes.c_uint64),
        ("gang_quantum", ctypes.c_uint64),
        ("gang_context_addr", ctypes.c_uint64),
        ("inprocess_gang_priority", ctypes.c_uint32),
        ("gang_global_priority_level", ctypes.c_uint32),
        ("doorbell_offset", ctypes.c_uint32),
        ("mqd_addr", ctypes.c_uint64),
        ("wptr_addr", ctypes.c_uint64),
        ("h_context", ctypes.c_uint64),
        ("h_queue", ctypes.c_uint64),
        ("queue_type", ctypes.c_uint32),
        ("gds_base", ctypes.c_uint32),
        ("gds_size", ctypes.c_uint32),
        ("gws_base", ctypes.c_uint32),
        ("gws_size", ctypes.c_uint32),
        ("oa_mask", ctypes.c_uint32),
        ("trap_handler_addr", ctypes.c_uint64),
        ("vm_context_cntl", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("api_status", _MESAPIStatus),
        ("tma_addr", ctypes.c_uint64),
        ("sch_id", ctypes.c_uint32),
        ("timestamp", ctypes.c_uint64),
        ("process_context_array_index", ctypes.c_uint32),
        ("gang_context_array_index", ctypes.c_uint32),
        ("pipe_id", ctypes.c_uint32),
        ("queue_id", ctypes.c_uint32),
        ("alignment_mode_setting", ctypes.c_uint32),
    ]


class _MESSetHwResourcesPacket(ctypes.LittleEndianStructure):
    _pack_ = 8
    _fields_ = [
        ("header", ctypes.c_uint32),
        ("vmid_mask_mmhub", ctypes.c_uint32),
        ("vmid_mask_gfxhub", ctypes.c_uint32),
        ("gds_size", ctypes.c_uint32),
        ("paging_vmid", ctypes.c_uint32),
        ("compute_hqd_mask", ctypes.c_uint32 * 8),
        ("gfx_hqd_mask", ctypes.c_uint32 * 2),
        ("sdma_hqd_mask", ctypes.c_uint32 * 2),
        ("aggregated_doorbells", ctypes.c_uint32 * 5),
        ("g_sch_ctx_gpu_mc_ptr", ctypes.c_uint64),
        ("query_status_fence_gpu_mc_ptr", ctypes.c_uint64),
        ("gc_base", ctypes.c_uint32 * 8),
        ("mmhub_base", ctypes.c_uint32 * 8),
        ("osssys_base", ctypes.c_uint32 * 8),
        ("api_status", _MESAPIStatus),
        ("flags", ctypes.c_uint32),
        ("oversubscription_timer", ctypes.c_uint32),
        ("doorbell_info", ctypes.c_uint64),
        ("event_intr_history_gpu_mc_ptr", ctypes.c_uint64),
        ("timestamp", ctypes.c_uint64),
        ("os_tdr_timeout_in_sec", ctypes.c_uint32),
    ]


class _MESQueryStatusPacket(ctypes.LittleEndianStructure):
    _pack_ = 8
    _fields_ = [
        ("header", ctypes.c_uint32),
        ("api_status", _MESAPIStatus),
    ]


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
    config: object,
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


def _doorbell_cpu_addr(
    dev: WindowsDevice,
    nbio_config: object,
    doorbell_index: int,
) -> int:
    """Resolve a CPU pointer for a 64-bit doorbell DWORD offset."""
    mapped = getattr(dev, "doorbell_addr", 0)
    if mapped:
        return mapped + doorbell_index * 4

    if getattr(nbio_config, "doorbell_phys_addr", 0) == 0:
        return 0
    driver = getattr(dev, "driver", None)
    if driver is None or not hasattr(driver, "map_bar"):
        return 0
    try:
        db_addr, _db_handle = driver.map_bar(2, doorbell_index * 4, 8)
        return db_addr
    except RuntimeError:
        return 0


def _mes_header(opcode: int) -> int:
    return (
        (MES_API_TYPE_SCHEDULER & 0xF) |
        ((opcode & 0xFF) << 4) |
        ((MES_API_FRAME_DWORDS & 0xFF) << 12)
    )


def _mes_frame(packet: ctypes.LittleEndianStructure) -> bytes:
    size = ctypes.sizeof(packet)
    if size > MES_API_FRAME_BYTES:
        raise ValueError(
            f"MES packet {type(packet).__name__} is {size} bytes, "
            f"larger than the {MES_API_FRAME_BYTES}-byte API frame"
        )
    frame = bytearray(MES_API_FRAME_BYTES)
    frame[:size] = ctypes.string_at(ctypes.addressof(packet), size)
    return bytes(frame)


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

    # compute_dispatch_initiator (offset 1)
    mqd[1] = 1

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
        (HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FF << 8)) |
        (HQD_PERSISTENT_STATE__PRELOAD_SIZE <<
         HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT) |
        HQD_PERSISTENT_STATE__PRELOAD_REQ
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
    mqd[139] = config.rptr_bus_addr & 0xFFFFFFFC
    mqd[140] = (config.rptr_bus_addr >> 32) & 0xFFFF

    # cp_hqd_pq_wptr_poll_addr (offsets 141-142)
    mqd[141] = config.wptr_bus_addr & 0xFFFFFFF8
    mqd[142] = (config.wptr_bus_addr >> 32) & 0xFFFF

    # cp_hqd_pq_doorbell_control (offset 143)
    doorbell_ctrl = 0
    doorbell_ctrl |= (config.doorbell_index & 0x03FFFFFF) << DOORBELL_OFFSET__SHIFT
    doorbell_ctrl |= DOORBELL_EN
    mqd[143] = doorbell_ctrl

    # cp_hqd_pq_control (offset 145)
    ring_size_log2 = int(math.log2(config.ring_size // 4)) - 1
    pq_control = 0
    pq_control |= (ring_size_log2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT
    pq_control |= (5 & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT
    pq_control |= PQ_CONTROL__PQ_EMPTY
    pq_control |= 3 << PQ_CONTROL__MIN_AVAIL_SIZE__SHIFT
    pq_control |= PQ_CONTROL__NO_UPDATE_RPTR
    pq_control |= PQ_CONTROL__UNORD_DISPATCH
    pq_control |= PQ_CONTROL__PRIV_STATE
    pq_control |= PQ_CONTROL__KMD_QUEUE
    if config.aql:
        pq_control |= PQ_CONTROL__QUEUE_FULL_EN
        pq_control |= 2 << PQ_CONTROL__SLOT_BASED_WPTR__SHIFT
    mqd[145] = pq_control

    # cp_mqd_control (offset 162)
    mqd[162] = 1 << 8  # PRIV_STATE

    # cp_hqd_eop_base_addr (offsets 165-166) — EOP buffer address >> 8
    eop_base = config.eop_bus_addr >> 8
    mqd[165] = eop_base & 0xFFFFFFFF
    mqd[166] = (eop_base >> 32) & 0xFFFFFFFF

    # cp_hqd_eop_control (offset 167)
    mqd[167] = int(math.log2(EOP_BUFFER_SIZE // 4)) - 1

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
    vmid = _gc_reg(dev, gc_base, regCP_HQD_VMID, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_VMID, vmid & ~0xF, base_idx=0)

    # Disable doorbell initially
    doorbell_ctl = _gc_reg(
        dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL,
             doorbell_ctl & ~DOORBELL_EN, base_idx=0)

    # MQD base address
    _gc_wreg(dev, gc_base, regCP_MQD_BASE_ADDR,
             config.mqd_bus_addr & 0xFFFFFFFC, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_MQD_BASE_ADDR_HI,
             (config.mqd_bus_addr >> 32) & 0xFFFFFFFF, base_idx=0)

    # MQD control. The direct lite queue follows the macOS path and leaves
    # this register clear while the MQD image carries its persistent bits.
    _gc_wreg(dev, gc_base, regCP_MQD_CONTROL, 0, base_idx=0)

    # Ring buffer base (address >> 8)
    pq_base = config.ring_bus_addr >> 8
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_BASE,
             pq_base & 0xFFFFFFFF, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_BASE_HI,
             (pq_base >> 32) & 0xFFFFFFFF, base_idx=0)

    # RPTR report address
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR_REPORT_ADDR,
             config.rptr_bus_addr & 0xFFFFFFFC, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR_REPORT_ADDR_HI,
             (config.rptr_bus_addr >> 32) & 0xFFFF, base_idx=0)

    # PQ control (ring size, flags)
    ring_size_log2 = int(math.log2(config.ring_size // 4)) - 1
    pq_control = 0
    pq_control |= (ring_size_log2 & 0x3F) << PQ_CONTROL__QUEUE_SIZE__SHIFT
    pq_control |= (5 & 0x3F) << PQ_CONTROL__RPTR_BLOCK_SIZE__SHIFT
    pq_control |= PQ_CONTROL__PQ_EMPTY
    pq_control |= 3 << PQ_CONTROL__MIN_AVAIL_SIZE__SHIFT
    pq_control |= PQ_CONTROL__NO_UPDATE_RPTR
    pq_control |= PQ_CONTROL__UNORD_DISPATCH
    pq_control |= PQ_CONTROL__PRIV_STATE
    pq_control |= PQ_CONTROL__KMD_QUEUE
    if config.aql:
        pq_control |= PQ_CONTROL__QUEUE_FULL_EN
        pq_control |= 2 << PQ_CONTROL__SLOT_BASED_WPTR__SHIFT
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_CONTROL, pq_control, base_idx=0)

    # WPTR poll address
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_POLL_ADDR,
             config.wptr_bus_addr & 0xFFFFFFF8, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_POLL_ADDR_HI,
             (config.wptr_bus_addr >> 32) & 0xFFFF, base_idx=0)

    # Reset RPTR and WPTR
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_RPTR, 0, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_LO, 0, base_idx=0)
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_WPTR_HI, 0, base_idx=0)

    # Doorbell control (enable doorbell)
    doorbell_ctrl = 0
    doorbell_ctrl |= (config.doorbell_index & 0x03FFFFFF) << DOORBELL_OFFSET__SHIFT
    doorbell_ctrl |= DOORBELL_EN
    _gc_wreg(dev, gc_base, regCP_HQD_PQ_DOORBELL_CONTROL, doorbell_ctrl,
             base_idx=0)

    # Persistent state (preload)
    persistent = (
        (HQD_PERSISTENT_STATE_DEFAULT & ~(0x3FF << 8)) |
        (HQD_PERSISTENT_STATE__PRELOAD_SIZE <<
         HQD_PERSISTENT_STATE__PRELOAD_SIZE__SHIFT) |
        HQD_PERSISTENT_STATE__PRELOAD_REQ
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
    _gc_wreg(dev, gc_base, regCP_HQD_EOP_CONTROL,
             int(math.log2(EOP_BUFFER_SIZE // 4)) - 1, base_idx=0)

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

    if config.dev is not None:
        grbm_select(
            config.dev, config.gc_base, config.me, config.pipe, config.queue)
        try:
            _gc_wreg(config.dev, config.gc_base, regCP_HQD_PQ_WPTR_LO,
                     config.wptr & 0xFFFFFFFF, base_idx=0)
            _gc_wreg(config.dev, config.gc_base, regCP_HQD_PQ_WPTR_HI,
                     (config.wptr >> 32) & 0xFFFFFFFF, base_idx=0)
        finally:
            grbm_deselect(config.dev, config.gc_base)

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
# MES scheduler API submission
# ============================================================================

def _mes_ring_as_compute_config(config: MESRingConfig) -> ComputeQueueConfig:
    queue = ComputeQueueConfig(gc_base=config.gc_base)
    queue.ring_bus_addr = config.ring_bus_addr
    queue.ring_cpu_addr = config.ring_cpu_addr
    queue.ring_dma_handle = config.ring_dma_handle
    queue.ring_size = config.ring_size
    queue.mqd_bus_addr = config.mqd_bus_addr
    queue.mqd_cpu_addr = config.mqd_cpu_addr
    queue.mqd_dma_handle = config.mqd_dma_handle
    queue.eop_bus_addr = config.eop_bus_addr
    queue.eop_cpu_addr = config.eop_cpu_addr
    queue.eop_dma_handle = config.eop_dma_handle
    queue.wptr_bus_addr = config.wptr_bus_addr
    queue.wptr_cpu_addr = config.wptr_cpu_addr
    queue.wptr_dma_handle = config.wptr_dma_handle
    queue.rptr_bus_addr = config.rptr_bus_addr
    queue.rptr_cpu_addr = config.rptr_cpu_addr
    queue.rptr_dma_handle = config.rptr_dma_handle
    queue.doorbell_index = config.doorbell_index
    queue.doorbell_cpu_addr = config.doorbell_cpu_addr
    queue.me = 3
    queue.pipe = config.pipe
    queue.queue = 0
    queue.active = config.ready
    queue.wptr = config.wptr
    queue.dev = config.dev
    queue.nbio_config = config.nbio_config
    queue.memory_handles = config.memory_handles
    return queue


def _init_mes_ring(
    dev: WindowsDevice,
    config: MESRingConfig,
    nbio_config: NBIOConfig,
    *,
    direct_activate: bool,
) -> None:
    config.dev = dev
    config.nbio_config = nbio_config

    ring_cpu, ring_bus, ring_handle = _alloc_queue_buffer(
        dev, config, config.ring_size)
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle

    mqd_cpu, mqd_bus, mqd_handle = _alloc_queue_buffer(dev, config, 4096)
    config.mqd_cpu_addr = mqd_cpu
    config.mqd_bus_addr = mqd_bus
    config.mqd_dma_handle = mqd_handle

    eop_cpu, eop_bus, eop_handle = _alloc_queue_buffer(
        dev, config, EOP_BUFFER_SIZE)
    config.eop_cpu_addr = eop_cpu
    config.eop_bus_addr = eop_bus
    config.eop_dma_handle = eop_handle

    wptr_cpu, wptr_bus, wptr_handle = _alloc_queue_buffer(dev, config, 4096)
    config.wptr_cpu_addr = wptr_cpu
    config.wptr_bus_addr = wptr_bus
    config.wptr_dma_handle = wptr_handle

    rptr_cpu, rptr_bus, rptr_handle = _alloc_queue_buffer(dev, config, 4096)
    config.rptr_cpu_addr = rptr_cpu
    config.rptr_bus_addr = rptr_bus
    config.rptr_dma_handle = rptr_handle

    status_cpu, status_bus, status_handle = _alloc_queue_buffer(
        dev, config, MES_STATUS_BUFFER_SIZE)
    config.status_cpu_addr = status_cpu
    config.status_bus_addr = status_bus
    config.status_dma_handle = status_handle

    sch_cpu, sch_bus, sch_handle = _alloc_queue_buffer(dev, config, 4096)
    config.sch_ctx_cpu_addr = sch_cpu
    config.sch_ctx_bus_addr = sch_bus
    config.sch_ctx_dma_handle = sch_handle

    query_cpu, query_bus, query_handle = _alloc_queue_buffer(dev, config, 4096)
    config.query_status_fence_cpu_addr = query_cpu
    config.query_status_fence_bus_addr = query_bus
    config.query_status_fence_dma_handle = query_handle

    config.doorbell_cpu_addr = _doorbell_cpu_addr(
        dev, nbio_config, config.doorbell_index)

    queue_view = _mes_ring_as_compute_config(config)
    _init_compute_mqd(queue_view)
    if direct_activate:
        _activate_compute_queue_mmio(dev, queue_view)
        config.ready = True
    config.wptr = 0


def _submit_mes_api_packet(
    config: MESRingConfig,
    packet: ctypes.LittleEndianStructure,
    *,
    timeout_ms: int = 5000,
) -> None:
    if not config.ready:
        raise RuntimeError("MES ring is not ready for API submission")

    config.sequence += 1
    seq = config.sequence

    packet.api_status.api_completion_fence_addr = config.status_bus_addr
    packet.api_status.api_completion_fence_value = seq

    query = _MESQueryStatusPacket()
    query.header = _mes_header(MES_SCH_API_QUERY_SCHEDULER_STATUS)
    query.api_status.api_completion_fence_addr = config.status_bus_addr + 8
    query.api_status.api_completion_fence_value = seq

    ctypes.c_uint64.from_address(config.status_cpu_addr).value = 0
    ctypes.c_uint64.from_address(config.status_cpu_addr + 8).value = 0

    queue_view = _mes_ring_as_compute_config(config)
    submit_compute_packets(queue_view, _mes_frame(packet) + _mes_frame(query))
    config.wptr = queue_view.wptr

    deadline = time.monotonic() + timeout_ms / 1000.0
    api_fence = 0
    query_fence = 0
    while time.monotonic() < deadline:
        api_fence = ctypes.c_uint64.from_address(config.status_cpu_addr).value
        query_fence = ctypes.c_uint64.from_address(
            config.status_cpu_addr + 8).value
        if api_fence >= seq and query_fence >= seq:
            return
        time.sleep(0.001)

    opcode = packet.header >> 4 & 0xFF
    raise TimeoutError(
        f"MES API opcode {opcode} timed out "
        f"(api={api_fence}, query={query_fence}, seq={seq})"
    )


def _mes_set_hw_resources(config: MESRingConfig, *, full: bool) -> None:
    packet = _MESSetHwResourcesPacket()
    packet.header = _mes_header(MES_SCH_API_SET_HW_RSRC)

    if full:
        packet.vmid_mask_mmhub = 0xFFFFFF00
        packet.vmid_mask_gfxhub = 0xFFFFFF00
        for i in range(4):
            packet.compute_hqd_mask[i] = 0xC
        packet.gfx_hqd_mask[0] = 0xFFFFFFFE
        packet.sdma_hqd_mask[0] = 0xFC
        packet.sdma_hqd_mask[1] = 0xFC

    packet.g_sch_ctx_gpu_mc_ptr = config.sch_ctx_bus_addr
    packet.query_status_fence_gpu_mc_ptr = config.query_status_fence_bus_addr
    for i, base in enumerate(config.gc_base[:8]):
        packet.gc_base[i] = base
    for i, base in enumerate(config.mmhub_base[:8]):
        packet.mmhub_base[i] = base
    for i, base in enumerate(config.osssys_base[:8]):
        packet.osssys_base[i] = base
    packet.flags = MES_SET_HW_RESOURCES_FLAGS
    packet.oversubscription_timer = 50

    _submit_mes_api_packet(config, packet)


def _mes_query_status(config: MESRingConfig) -> None:
    packet = _MESQueryStatusPacket()
    packet.header = _mes_header(MES_SCH_API_QUERY_SCHEDULER_STATUS)
    _submit_mes_api_packet(config, packet)


def _mes_map_legacy_queue(
    api_ring: MESRingConfig,
    queue: ComputeQueueConfig | MESRingConfig,
    *,
    queue_type: int,
) -> None:
    packet = _MESAddQueuePacket()
    packet.header = _mes_header(MES_SCH_API_ADD_QUEUE)
    packet.pipe_id = queue.pipe
    packet.queue_id = queue.queue if isinstance(queue, ComputeQueueConfig) else 0
    packet.doorbell_offset = queue.doorbell_index
    packet.mqd_addr = queue.mqd_bus_addr
    packet.wptr_addr = queue.wptr_bus_addr
    packet.queue_type = queue_type
    packet.flags = MES_ADD_QUEUE_FLAG_MAP_LEGACY_KQ
    _submit_mes_api_packet(api_ring, packet)


def _enable_unmapped_doorbell_handling(
    dev: WindowsDevice,
    gc_base: list[int],
    enable: bool = True,
) -> None:
    data = _gc_reg(dev, gc_base, regCP_UNMAPPED_DOORBELL, base_idx=1)
    data &= ~CP_UNMAPPED_DOORBELL__PROC_LSB_MASK
    data |= 0xD << CP_UNMAPPED_DOORBELL__PROC_LSB__SHIFT
    if enable:
        data |= CP_UNMAPPED_DOORBELL__ENABLE
    else:
        data &= ~CP_UNMAPPED_DOORBELL__ENABLE
    _gc_wreg(dev, gc_base, regCP_UNMAPPED_DOORBELL, data, base_idx=1)
    _gc_wreg(dev, gc_base, regCP_HQD_GFX_CONTROL,
             CP_HQD_GFX_CONTROL__DB_UPDATED_MSG_EN, base_idx=0)


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


def _resolve_hwip_bases(
    ip_result: IPDiscoveryResult,
    hw_id: int,
    *,
    count: int = 8,
) -> list[int]:
    bases = [0] * count
    for block in ip_result.ip_blocks:
        if block.hw_id == hw_id and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(bases) and addr != 0:
                    bases[i] = addr
            break
    return bases


def _wait_rlc_autoload(dev: WindowsDevice, gc_base: list[int]) -> None:
    deadline = time.monotonic() + 5.0
    last_status = 0
    cp_stat = 0
    while time.monotonic() < deadline:
        last_status = _gc_reg(dev, gc_base, regRLC_RLCS_BOOTLOAD_STATUS,
                              base_idx=1)
        cp_stat = _gc_reg(dev, gc_base, regCP_STAT, base_idx=0)
        if (last_status & 0x80000000) and cp_stat == 0:
            print(f"  GFX: RLC autoload status=0x{last_status:08X}, "
                  f"CP_STAT=0x{cp_stat:08X}")
            return
        time.sleep(0.001)
    print(f"  GFX: WARNING RLC autoload not confirmed "
          f"(status=0x{last_status:08X}, CP_STAT=0x{cp_stat:08X})")


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


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _firmware_exists(path: Path) -> bool:
    return path.exists() or Path(str(path) + ".zst").exists()


def _parse_rlc_toc(data: bytes) -> tuple[object, dict[int, RLCTOCEntry],
                                         list[RLCTOCEntry]]:
    from amd_gpu_driver.backends.windows.psp_init import parse_firmware_header

    header = parse_firmware_header(data)
    start = header.ucode_array_offset_bytes
    end = start + header.ucode_size_bytes
    if start <= 0 or end > len(data):
        raise ValueError(
            f"Invalid RLC TOC firmware payload offset=0x{start:X}, "
            f"size=0x{header.ucode_size_bytes:X}, file=0x{len(data):X}"
        )

    entries: dict[int, RLCTOCEntry] = {}
    all_entries: list[RLCTOCEntry] = []
    for off in range(start, end, 8):
        if off + 8 > len(data):
            break
        dw0, dw1 = struct.unpack_from("<II", data, off)
        fw_id = (dw0 >> 25) & 0x7F
        if fw_id == SOC24_FIRMWARE_ID_INVALID:
            break
        toc_offset = (dw0 & 0x01FFFFFF) * RLC_TOC_OFFSET_DWUNIT * 4
        size_x16 = bool((dw1 >> 12) & 0x1)
        size_field = (dw1 >> 14) & 0x3FFFF
        size = size_field * (RLC_SIZE_MULTIPLE * 4 if size_x16 else 4)
        entry = RLCTOCEntry(
            id=fw_id, offset=toc_offset, size=size, size_x16=size_x16)
        all_entries.append(entry)
        if fw_id < SOC24_FIRMWARE_ID_MAX:
            entries[fw_id] = entry

    if SOC24_FIRMWARE_ID_RLC_G_UCODE not in entries:
        raise ValueError("RLC TOC does not contain RLC_G_UCODE")
    return header, entries, all_entries


def _calc_rlc_autoload_size(entries: dict[int, RLCTOCEntry],
                            all_entries: list[RLCTOCEntry]) -> int:
    total = sum(entry.size for entry in entries.values())
    max_end = max(
        (entry.offset + entry.size for entry in all_entries), default=0)
    return _align_up(
        max(total, max_end, RLC_TOC_UMF_SIZE), RLC_AUTOLOAD_ALIGNMENT)


def _alloc_rlc_autoload_buffer(
    dev: WindowsDevice,
    total_size: int,
) -> tuple[object, int, int]:
    from amd_gpu_driver.backends.base import MemoryLocation

    alloc_size = _align_up(
        total_size + RLC_AUTOLOAD_ALIGNMENT - 1, 4096)
    handle = dev.alloc_memory(alloc_size, MemoryLocation.VRAM)
    if handle.cpu_addr == 0:
        dev.free_memory(handle)
        raise RuntimeError("RLC autoload VRAM buffer is not CPU mapped")

    delta = (-handle.gpu_addr) & (RLC_AUTOLOAD_ALIGNMENT - 1)
    ctypes.memset(handle.cpu_addr, 0, handle.size)
    return handle, handle.cpu_addr + delta, handle.gpu_addr + delta


def _copy_autoload_blob(
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    fw_id: int,
    blob: bytes,
    *,
    fw_size: int = 0,
) -> bool:
    entry = entries.get(fw_id)
    if entry is None or entry.size == 0:
        return False
    if entry.offset + entry.size > total_size:
        raise RuntimeError(
            f"RLC autoload TOC entry {fw_id} outside allocation: "
            f"offset=0x{entry.offset:X}, size=0x{entry.size:X}, "
            f"total=0x{total_size:X}"
        )

    copy_size = len(blob) if fw_size == 0 else fw_size
    if copy_size > len(blob):
        raise ValueError(
            f"Firmware blob for TOC entry {fw_id} is shorter than requested "
            f"copy: blob=0x{len(blob):X}, requested=0x{copy_size:X}"
        )
    copy_size = min(copy_size, entry.size)
    if copy_size:
        ctypes.memmove(base_cpu + entry.offset, blob[:copy_size], copy_size)
    if copy_size < entry.size:
        ctypes.memset(base_cpu + entry.offset + copy_size, 0,
                      entry.size - copy_size)
    return True


def _copy_autoload_slice(
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    fw_id: int,
    data: bytes,
    offset: int,
    size: int,
) -> bool:
    from amd_gpu_driver.backends.windows.psp_init import _slice

    if size == 0:
        return False
    return _copy_autoload_blob(
        base_cpu, total_size, entries, fw_id, _slice(data, offset, size),
        fw_size=size,
    )


def _flush_backend_writes(dev: WindowsDevice, psp_config: object | None) -> None:
    nbio_config = getattr(psp_config, "nbio_config", None)
    if nbio_config is None:
        return
    try:
        from amd_gpu_driver.backends.windows.nbio_init import hdp_flush
        hdp_flush(dev, nbio_config)
    except Exception:
        pass


def _ucode_start_dict(psp_config: object) -> dict[str, int]:
    ucode_start = getattr(psp_config, "ucode_start", None)
    if ucode_start is None:
        ucode_start = {}
        setattr(psp_config, "ucode_start", ucode_start)
    return ucode_start


def _copy_sdma_autoload_ucode(
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    fw_path: Path,
) -> bool:
    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        _u32,
        parse_firmware_header,
    )

    data = _read_firmware(fw_path)
    header = parse_firmware_header(data)
    if header.header_version_major >= 3:
        ucode_offset = _u32(data, 36) or header.ucode_array_offset_bytes
        ucode_size = _u32(data, 40)
    elif header.header_version_major == 2:
        ucode_offset = header.ucode_array_offset_bytes
        ucode_size = _u32(data, 36)
    else:
        ucode_offset = header.ucode_array_offset_bytes
        ucode_size = header.ucode_size_bytes
    return _copy_autoload_slice(
        base_cpu, total_size, entries, SOC24_FIRMWARE_ID_SDMA_UCODE_TH0,
        data, ucode_offset, ucode_size,
    )


def _copy_rs64_autoload_ucode(
    psp_config: object,
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    fw_path: Path,
    name: str,
    code_id: int,
    stack_ids: tuple[int, ...],
) -> bool:
    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        _u32,
        parse_firmware_header,
    )

    data = _read_firmware(fw_path)
    header = parse_firmware_header(data)
    if header.header_version_major < 2:
        raise ValueError(f"{fw_path} is not an RS64 firmware image")

    code_size = _u32(data, 36)
    code_offset = _u32(data, 40) or header.ucode_array_offset_bytes
    data_size = _u32(data, 44)
    data_offset = _u32(data, 48)
    start = _u32(data, 52) | (_u32(data, 56) << 32)
    _ucode_start_dict(psp_config)[name.upper()] = start

    copied = _copy_autoload_slice(
        base_cpu, total_size, entries, code_id, data, code_offset, code_size)
    for stack_id in stack_ids:
        copied |= _copy_autoload_slice(
            base_cpu, total_size, entries, stack_id,
            data, data_offset, data_size,
        )
    return copied


def _copy_rlc_autoload_ucode(
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    fw_path: Path,
) -> bool:
    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        _u32,
        parse_firmware_header,
    )

    data = _read_firmware(fw_path)
    header = parse_firmware_header(data)
    copied = _copy_autoload_slice(
        base_cpu, total_size, entries, SOC24_FIRMWARE_ID_RLC_G_UCODE,
        data, header.ucode_array_offset_bytes, header.ucode_size_bytes,
    )

    minor = header.header_version_minor
    if minor >= 1 and len(data) >= 156:
        copied |= _copy_autoload_slice(
            base_cpu, total_size, entries, SOC24_FIRMWARE_ID_RLCG_SCRATCH,
            data, _u32(data, 136), _u32(data, 132),
        )
        copied |= _copy_autoload_slice(
            base_cpu, total_size, entries, SOC24_FIRMWARE_ID_RLC_SRM_ARAM,
            data, _u32(data, 152), _u32(data, 148),
        )
    if minor >= 2 and len(data) >= 172:
        copied |= _copy_autoload_slice(
            base_cpu, total_size, entries, SOC24_FIRMWARE_ID_RLX6_UCODE,
            data, _u32(data, 160), _u32(data, 156),
        )
        copied |= _copy_autoload_slice(
            base_cpu, total_size, entries, SOC24_FIRMWARE_ID_RLX6_DRAM_BOOT,
            data, _u32(data, 168), _u32(data, 164),
        )
    return copied


def _mes_autoload_paths(fw_dir: Path,
                        gc_version: str) -> list[tuple[int, str, Path]]:
    uni_mes = fw_dir / f"gc_{gc_version}_uni_mes.bin"
    if _firmware_exists(uni_mes):
        return [(0, "MES", uni_mes), (1, "MES1", uni_mes)]
    return [
        (0, "MES", fw_dir / f"gc_{gc_version}_mes.bin"),
        (1, "MES1", fw_dir / f"gc_{gc_version}_mes1.bin"),
    ]


def _copy_mes_autoload_ucode(
    psp_config: object,
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    pipe: int,
    name: str,
    fw_path: Path,
) -> bool:
    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        _u32,
        parse_firmware_header,
    )

    data = _read_firmware(fw_path)
    header = parse_firmware_header(data)
    if header.header_version_major >= 2:
        code_size = _u32(data, 36)
        code_offset = _u32(data, 40) or header.ucode_array_offset_bytes
        data_size = _u32(data, 44)
        data_offset = _u32(data, 48)
        start = _u32(data, 52) | (_u32(data, 56) << 32)
        data_start = 0
    else:
        code_size = _u32(data, 36)
        code_offset = _u32(data, 40) or header.ucode_array_offset_bytes
        data_size = _u32(data, 48)
        data_offset = _u32(data, 52)
        start = _u32(data, 56) | (_u32(data, 60) << 32)
        data_start = 0
        if len(data) >= 72:
            data_start = _u32(data, 64) | (_u32(data, 68) << 32)

    _ucode_start_dict(psp_config)[name.upper()] = start
    if data_start:
        ucode_data_start = getattr(psp_config, "ucode_data_start", None)
        if ucode_data_start is None:
            ucode_data_start = {}
            setattr(psp_config, "ucode_data_start", ucode_data_start)
        ucode_data_start[name.upper()] = data_start

    code_id = (SOC24_FIRMWARE_ID_RS64_MES_P0 if pipe == 0
               else SOC24_FIRMWARE_ID_RS64_MES_P1)
    data_id = (SOC24_FIRMWARE_ID_RS64_MES_P0_STACK if pipe == 0
               else SOC24_FIRMWARE_ID_RS64_MES_P1_STACK)
    copied = _copy_autoload_slice(
        base_cpu, total_size, entries, code_id, data, code_offset, code_size)
    copied |= _copy_autoload_slice(
        base_cpu, total_size, entries, data_id, data, data_offset, data_size)
    return copied


def _copy_toc_autoload_ucode(
    base_cpu: int,
    total_size: int,
    entries: dict[int, RLCTOCEntry],
    toc_header: object,
    toc_data: bytes,
) -> bool:
    from amd_gpu_driver.backends.windows.psp_init import _slice

    entry = entries.get(SOC24_FIRMWARE_ID_RLC_TOC)
    if entry is None or entry.size == 0:
        return False
    payload = bytearray(
        _slice(toc_data, toc_header.ucode_array_offset_bytes, entry.size))
    if len(payload) >= 8:
        struct.pack_into(
            "<I", payload, len(payload) - 8,
            (RLC_TOC_FORMAT_API << 24) | 0x1,
        )
    return _copy_autoload_blob(
        base_cpu, total_size, entries, SOC24_FIRMWARE_ID_RLC_TOC,
        bytes(payload), fw_size=len(payload),
    )


def _program_imu_rlc_ram(
    dev: WindowsDevice,
    gc_base: list[int],
    psp_config: object,
) -> None:
    _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_INDEX, 0x2, base_idx=1)

    vram_start = getattr(psp_config, "vram_mc_base", 0)
    vram_size = getattr(dev, "vram_size", 0)
    vram_end = vram_start + vram_size - 1 if vram_start and vram_size else 0

    for segment, reg, data, addr_mask, name in IMU_RLC_RAM_GOLDEN_12_0_1:
        if name == "regGCMC_VM_AGP_BASE":
            data = 0x00FFFFFF
        elif name == "regGCMC_VM_AGP_TOP":
            data = 0
        elif name == "regGCMC_VM_FB_LOCATION_BASE" and vram_start:
            data = (vram_start >> 24) & 0xFFFFFFFF
        elif name == "regGCMC_VM_FB_LOCATION_TOP" and vram_end:
            data = (vram_end >> 24) & 0xFFFFFFFF

        reg_addr = (gc_base[segment] + reg) | addr_mask
        _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_ADDR_HIGH, 0, base_idx=1)
        _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_ADDR_LOW,
                 reg_addr, base_idx=1)
        _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_DATA,
                 data & 0xFFFFFFFF, base_idx=1)

    _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_ADDR_HIGH, 0, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_ADDR_LOW, 0, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_DATA, 0, base_idx=1)
    val = _gc_reg(dev, gc_base, regGFX_IMU_RLC_RAM_INDEX, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_RLC_RAM_INDEX,
             val | GFX_IMU_RLC_RAM_INDEX__RAM_VALID, base_idx=1)
    print(f"  GFX: IMU RLC RAM programmed ({len(IMU_RLC_RAM_GOLDEN_12_0_1)} entries)")


def _load_imu_microcode(
    dev: WindowsDevice,
    gc_base: list[int],
    fw_dir: Path,
    gc_version: str,
) -> None:
    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        _slice,
        _u32,
        parse_firmware_header,
    )

    path = fw_dir / f"gc_{gc_version}_imu.bin"
    data = _read_firmware(path)
    header = parse_firmware_header(data)
    iram_size = _u32(data, 32)
    iram_rel_offset = _u32(data, 36)
    iram_offset = (
        header.ucode_array_offset_bytes + iram_rel_offset
        if iram_rel_offset else header.ucode_array_offset_bytes
    )
    dram_size = _u32(data, 40)
    dram_rel_offset = _u32(data, 44)
    dram_offset = (
        header.ucode_array_offset_bytes + dram_rel_offset
        if dram_rel_offset else iram_offset + iram_size
    )

    iram = _slice(data, iram_offset, iram_size)
    dram = _slice(data, dram_offset, dram_size)

    _gc_wreg(dev, gc_base, regGFX_IMU_I_RAM_ADDR, 0, base_idx=1)
    for off in range(0, len(iram), 4):
        _gc_wreg(dev, gc_base, regGFX_IMU_I_RAM_DATA,
                 struct.unpack_from("<I", iram, off)[0], base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_I_RAM_ADDR,
             header.ucode_version, base_idx=1)

    _gc_wreg(dev, gc_base, regGFX_IMU_D_RAM_ADDR, 0, base_idx=1)
    for off in range(0, len(dram), 4):
        _gc_wreg(dev, gc_base, regGFX_IMU_D_RAM_DATA,
                 struct.unpack_from("<I", dram, off)[0], base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_D_RAM_ADDR,
             header.ucode_version, base_idx=1)

    print(f"  GFX: IMU firmware loaded from {path.name} "
          f"(IRAM=0x{iram_size:X}, DRAM=0x{dram_size:X})")


def _setup_imu(dev: WindowsDevice, gc_base: list[int]) -> None:
    _gc_wreg(dev, gc_base, regGFX_IMU_C2PMSG_ACCESS_CTRL0,
             0x00FFFFFF, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_C2PMSG_ACCESS_CTRL1,
             0x0000FFFF, base_idx=1)
    c2pmsg16 = _gc_reg(dev, gc_base, regGFX_IMU_C2PMSG_16, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_C2PMSG_16,
             c2pmsg16 | 0x1, base_idx=1)
    scratch10 = _gc_reg(dev, gc_base, regGFX_IMU_SCRATCH_10, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_SCRATCH_10,
             scratch10 | 0x20010007, base_idx=1)


def _start_imu(
    dev: WindowsDevice,
    gc_base: list[int],
    smu_config: object | None = None,
) -> bool:
    val = _gc_reg(dev, gc_base, regGFX_IMU_CORE_CTRL, base_idx=1)
    _gc_wreg(dev, gc_base, regGFX_IMU_CORE_CTRL,
             val & 0xFFFFFFFE, base_idx=1)
    if (
        smu_config is not None
        and os.environ.get("AMDGPU_LITE_ENABLE_GFX_IMU_MSG", "1") != "0"
    ):
        try:
            from amd_gpu_driver.backends.windows.smu_init import (
                enable_gfx_imu_no_wait,
            )
            enable_gfx_imu_no_wait(dev, smu_config)
        except Exception as e:
            print(f"  SMU: WARNING EnableGfxImu skipped - {e}")

    deadline = time.monotonic() + 5.0
    reset_ctrl = 0
    while time.monotonic() < deadline:
        reset_ctrl = _gc_reg(dev, gc_base, regGFX_IMU_GFX_RESET_CTRL,
                             base_idx=1)
        if (reset_ctrl & 0x1F) == 0x1F:
            print(f"  GFX: IMU started (GFX_RESET_CTRL=0x{reset_ctrl:08X})")
            return True
        time.sleep(0.001)
    print(f"  GFX: WARNING IMU start not confirmed "
          f"(GFX_RESET_CTRL=0x{reset_ctrl:08X})")
    return False


def _disable_gpa_mode(dev: WindowsDevice, gc_base: list[int]) -> None:
    val = _gc_reg(dev, gc_base, regCPC_PSP_DEBUG, base_idx=1)
    _gc_wreg(dev, gc_base, regCPC_PSP_DEBUG,
             val | CPC_PSP_DEBUG__GPA_OVERRIDE, base_idx=1)
    val = _gc_reg(dev, gc_base, regCPG_PSP_DEBUG, base_idx=1)
    _gc_wreg(dev, gc_base, regCPG_PSP_DEBUG,
             val | CPG_PSP_DEBUG__GPA_OVERRIDE, base_idx=1)


def _rlc_backdoor_autoload(
    dev: WindowsDevice,
    gc_base: list[int],
    psp_config: object | None,
    smu_config: object | None = None,
) -> bool:
    if psp_config is None:
        return False
    if getattr(psp_config, "rlc_backdoor_autoloaded", False):
        return True

    env = os.environ.get("AMDGPU_LITE_RLC_BACKDOOR_AUTO")
    if env != "1":
        return False
    if not hasattr(dev, "alloc_memory") or not hasattr(dev, "free_memory"):
        return False

    from amd_gpu_driver.backends.windows.psp_init import _read_firmware

    fw_dir = getattr(psp_config, "fw_dir", None)
    if fw_dir is None:
        return False
    fw_dir = Path(fw_dir)
    versions = getattr(psp_config, "ip_versions", {})
    gc_version = versions.get("gc", "12_0_1")
    sdma_version = versions.get("sdma", "7_0_1")

    toc_path = fw_dir / f"gc_{gc_version}_toc.bin"
    toc_data = _read_firmware(toc_path)
    toc_header, entries, all_entries = _parse_rlc_toc(toc_data)
    total_size = _calc_rlc_autoload_size(entries, all_entries)
    use_imu = os.environ.get("AMDGPU_LITE_RLC_AUTOLOAD_IMU", "1") != "0"

    handle, base_cpu, base_gpu = _alloc_rlc_autoload_buffer(dev, total_size)
    success = False
    try:
        if use_imu:
            _program_imu_rlc_ram(dev, gc_base, psp_config)

        _copy_sdma_autoload_ucode(
            base_cpu, total_size, entries,
            fw_dir / f"sdma_{sdma_version}.bin",
        )
        _copy_rs64_autoload_ucode(
            psp_config, base_cpu, total_size, entries,
            fw_dir / f"gc_{gc_version}_pfp.bin", "PFP",
            SOC24_FIRMWARE_ID_RS64_PFP,
            (
                SOC24_FIRMWARE_ID_RS64_PFP_P0_STACK,
                SOC24_FIRMWARE_ID_RS64_PFP_P1_STACK,
            ),
        )
        _copy_rs64_autoload_ucode(
            psp_config, base_cpu, total_size, entries,
            fw_dir / f"gc_{gc_version}_me.bin", "ME",
            SOC24_FIRMWARE_ID_RS64_ME,
            (
                SOC24_FIRMWARE_ID_RS64_ME_P0_STACK,
                SOC24_FIRMWARE_ID_RS64_ME_P1_STACK,
            ),
        )
        _copy_rs64_autoload_ucode(
            psp_config, base_cpu, total_size, entries,
            fw_dir / f"gc_{gc_version}_mec.bin", "MEC",
            SOC24_FIRMWARE_ID_RS64_MEC,
            (
                SOC24_FIRMWARE_ID_RS64_MEC_P0_STACK,
                SOC24_FIRMWARE_ID_RS64_MEC_P1_STACK,
                SOC24_FIRMWARE_ID_RS64_MEC_P2_STACK,
                SOC24_FIRMWARE_ID_RS64_MEC_P3_STACK,
            ),
        )
        _copy_rlc_autoload_ucode(
            base_cpu, total_size, entries,
            fw_dir / f"gc_{gc_version}_rlc.bin",
        )
        for pipe, name, path in _mes_autoload_paths(fw_dir, gc_version):
            _copy_mes_autoload_ucode(
                psp_config, base_cpu, total_size, entries, pipe, name, path)
        _copy_toc_autoload_ucode(
            base_cpu, total_size, entries, toc_header, toc_data)

        _flush_backend_writes(dev, psp_config)

        rlc_entry = entries[SOC24_FIRMWARE_ID_RLC_G_UCODE]
        vram_mc_base = getattr(psp_config, "vram_mc_base", 0)
        bootloader_addr = base_gpu + rlc_entry.offset
        if vram_mc_base and bootloader_addr >= vram_mc_base:
            bootloader_addr -= vram_mc_base
        _gc_wreg(
            dev, gc_base, regGFX_IMU_RLC_BOOTLOADER_ADDR_HI,
            (bootloader_addr >> 32) & 0xFFFFFFFF, base_idx=1,
        )
        _gc_wreg(
            dev, gc_base, regGFX_IMU_RLC_BOOTLOADER_ADDR_LO,
            bootloader_addr & 0xFFFFFFFF, base_idx=1,
        )
        _gc_wreg(
            dev, gc_base, regGFX_IMU_RLC_BOOTLOADER_SIZE,
            rlc_entry.size, base_idx=1,
        )

        if use_imu:
            _load_imu_microcode(dev, gc_base, fw_dir, gc_version)
            _setup_imu(dev, gc_base)
            _start_imu(dev, gc_base, smu_config)
            _disable_gpa_mode(dev, gc_base)
        else:
            data = _gc_reg(dev, gc_base, regRLC_GPM_THREAD_ENABLE, base_idx=1)
            data |= (RLC_GPM_THREAD_ENABLE__THREAD0_ENABLE |
                     RLC_GPM_THREAD_ENABLE__THREAD1_ENABLE)
            _gc_wreg(dev, gc_base, regRLC_GPM_THREAD_ENABLE, data, base_idx=1)
            _gc_wreg(dev, gc_base, regRLC_CNTL, 0x1, base_idx=1)

        handles = getattr(psp_config, "firmware_memory_handles", None)
        if handles is None:
            handles = []
            setattr(psp_config, "firmware_memory_handles", handles)
        handles.append(handle)
        setattr(psp_config, "rlc_backdoor_autoloaded", True)
        setattr(psp_config, "rlc_autoload_cpu_addr", base_cpu)
        setattr(psp_config, "rlc_autoload_gpu_addr", base_gpu)
        setattr(psp_config, "rlc_autoload_size", total_size)
        success = True
    finally:
        if not success:
            dev.free_memory(handle)

    print(
        f"  GFX: RLC backdoor autoload BO at MC 0x{base_gpu:012X}, "
        f"size=0x{total_size:X}"
    )
    print(
        f"  GFX: RLC bootloader addr=0x{bootloader_addr:012X}, "
        f"size=0x{rlc_entry.size:X}"
    )
    return True


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

    for pipe in range(4):
        grbm_select(dev, gc_base, me=1, pipe=pipe, queue=0)
        _gc_wreg_pair(
            dev, gc_base, regCP_MEC_RS64_PRGRM_CNTR_START,
            regCP_MEC_RS64_PRGRM_CNTR_START_HI, ucode_start["MEC"] >> 2,
            base_idx=1,
        )
    grbm_deselect(dev, gc_base)

    _pulse_reset_bits(
        dev, gc_base, regCP_MEC_RS64_CNTL,
        CP_MEC_RS64_CNTL__MEC_PIPE0_RESET |
        CP_MEC_RS64_CNTL__MEC_PIPE1_RESET |
        CP_MEC_RS64_CNTL__MEC_PIPE2_RESET |
        CP_MEC_RS64_CNTL__MEC_PIPE3_RESET,
        base_idx=1,
    )
    print("  GFX: CP PFP/ME/MEC program counters configured")


def _enable_mec(dev: WindowsDevice, gc_base: list[int]) -> None:
    val = _gc_reg(dev, gc_base, regCP_MEC_RS64_CNTL, base_idx=1)
    val &= ~(CP_MEC_RS64_CNTL__MEC_INVALIDATE_ICACHE |
             CP_MEC_RS64_CNTL__MEC_PIPE0_RESET |
             CP_MEC_RS64_CNTL__MEC_PIPE1_RESET |
             CP_MEC_RS64_CNTL__MEC_PIPE2_RESET |
             CP_MEC_RS64_CNTL__MEC_PIPE3_RESET |
             CP_MEC_RS64_CNTL__MEC_HALT)
    val |= (CP_MEC_RS64_CNTL__MEC_PIPE0_ACTIVE |
            CP_MEC_RS64_CNTL__MEC_PIPE1_ACTIVE |
            CP_MEC_RS64_CNTL__MEC_PIPE2_ACTIVE |
            CP_MEC_RS64_CNTL__MEC_PIPE3_ACTIVE)
    _gc_wreg(dev, gc_base, regCP_MEC_RS64_CNTL, val, base_idx=1)
    time.sleep(0.050)


def _mes_firmware_paths(
    fw_dir: Path,
    gc_version: str = "12_0_1",
) -> list[tuple[int, str, Path]]:
    uni_mes = fw_dir / f"gc_{gc_version}_uni_mes.bin"
    if _firmware_exists(uni_mes):
        return [(0, "MES", uni_mes), (1, "MES1", uni_mes)]
    return [
        (0, "MES", fw_dir / f"gc_{gc_version}_mes.bin"),
        (1, "MES1", fw_dir / f"gc_{gc_version}_mes1.bin"),
    ]


def _program_mes_ucode_buffer(
    dev: WindowsDevice,
    gc_base: list[int],
    pipe: int,
    fw_path: Path,
    *,
    prime_icache: bool,
) -> object:
    from amd_gpu_driver.backends.base import MemoryLocation
    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        _slice,
        _u32,
        parse_firmware_header,
    )

    data = _read_firmware(fw_path)
    header = parse_firmware_header(data)
    if header.header_version_major != 1:
        raise ValueError(f"{fw_path} is not a MES v1 firmware image")

    code_size = _u32(data, 36)
    code_offset = _u32(data, 40) or header.ucode_array_offset_bytes
    data_size = _u32(data, 48)
    data_offset = _u32(data, 52)
    code_aligned_size = _align_up(code_size, 64 * 1024)
    total_size = code_aligned_size + data_size

    handle = dev.alloc_memory(total_size, MemoryLocation.VRAM)
    if handle.cpu_addr == 0:
        dev.free_memory(handle)
        raise RuntimeError(f"MES VRAM firmware buffer is not CPU mapped: {fw_path}")

    ctypes.memset(handle.cpu_addr, 0, handle.size)
    ctypes.memmove(handle.cpu_addr, _slice(data, code_offset, code_size),
                   code_size)
    ctypes.memmove(handle.cpu_addr + code_aligned_size,
                   _slice(data, data_offset, data_size), data_size)

    code_gpu = handle.gpu_addr
    data_gpu = handle.gpu_addr + code_aligned_size
    grbm_select(dev, gc_base, me=3, pipe=pipe, queue=0)
    _gc_wreg(dev, gc_base, regCP_MES_IC_BASE_CNTL, 0, base_idx=1)
    _gc_wreg_pair(dev, gc_base, regCP_MES_IC_BASE_LO, regCP_MES_IC_BASE_HI,
                  code_gpu, base_idx=1)
    _gc_wreg(dev, gc_base, regCP_MES_MIBOUND_LO, 0x1FFFFF, base_idx=1)
    _gc_wreg_pair(dev, gc_base, regCP_MES_MDBASE_LO, regCP_MES_MDBASE_HI,
                  data_gpu, base_idx=1)
    _gc_wreg(dev, gc_base, regCP_MES_MDBOUND_LO, 0x7FFFF, base_idx=1)

    if prime_icache:
        op = _gc_reg(dev, gc_base, regCP_MES_IC_OP_CNTL, base_idx=1)
        op &= ~CP_MES_IC_OP_CNTL__PRIME_ICACHE
        op |= CP_MES_IC_OP_CNTL__INVALIDATE_CACHE
        _gc_wreg(dev, gc_base, regCP_MES_IC_OP_CNTL, op, base_idx=1)
        op = _gc_reg(dev, gc_base, regCP_MES_IC_OP_CNTL, base_idx=1)
        op |= CP_MES_IC_OP_CNTL__PRIME_ICACHE
        _gc_wreg(dev, gc_base, regCP_MES_IC_OP_CNTL, op, base_idx=1)

    grbm_deselect(dev, gc_base)
    print(f"  GFX: MES pipe {pipe} firmware staged at "
          f"code=0x{code_gpu:016X}, data=0x{data_gpu:016X}")
    return handle


def _program_mes_ucode_buffers(
    dev: WindowsDevice,
    gc_base: list[int],
    psp_config: object | None,
) -> None:
    if not hasattr(dev, "alloc_memory") or not hasattr(dev, "free_memory"):
        return
    if os.environ.get("AMDGPU_LITE_MES_BACKDOOR", "1") == "0":
        return
    fw_dir = getattr(psp_config, "fw_dir", None)
    if fw_dir is None:
        return

    handles = getattr(psp_config, "mes_memory_handles", None)
    if handles is None:
        handles = []
        setattr(psp_config, "mes_memory_handles", handles)
    if handles:
        return

    versions = getattr(psp_config, "ip_versions", {})
    gc_version = versions.get("gc", "12_0_1")
    for pipe, _key, path in _mes_firmware_paths(Path(fw_dir), gc_version):
        handle = _program_mes_ucode_buffer(
            dev, gc_base, pipe, path, prime_icache=(pipe == MES_PIPE_KIQ)
        )
        handles.append(handle)


def _enable_mes_from_ucode(
    dev: WindowsDevice,
    gc_base: list[int],
    ucode_start: dict[str, int],
) -> None:
    if "MES" not in ucode_start:
        print("  GFX: Skipping MES enable, missing MES firmware start")
        return

    schedulers = _gc_reg(dev, gc_base, regRLC_CP_SCHEDULERS, base_idx=1)
    schedulers &= 0xFFFFFF00
    schedulers |= (3 << 5) | (MES_PIPE_KIQ << 3) | 0 | 0x80
    _gc_wreg(dev, gc_base, regRLC_CP_SCHEDULERS, schedulers, base_idx=1)

    val = _gc_reg(dev, gc_base, regCP_MES_CNTL, base_idx=1)
    val &= ~(CP_MES_CNTL__MES_PIPE0_ACTIVE | CP_MES_CNTL__MES_PIPE1_ACTIVE)
    val |= (CP_MES_CNTL__MES_INVALIDATE_ICACHE |
            CP_MES_CNTL__MES_PIPE0_RESET |
            CP_MES_CNTL__MES_PIPE1_RESET |
            CP_MES_CNTL__MES_HALT)
    _gc_wreg(dev, gc_base, regCP_MES_CNTL, val, base_idx=1)

    active_mask = 0
    for pipe, key in [(0, "MES"), (1, "MES1")]:
        if key not in ucode_start:
            continue
        grbm_select(dev, gc_base, me=3, pipe=pipe, queue=0)
        _gc_wreg_pair(
            dev, gc_base, regCP_MES_PRGRM_CNTR_START,
            regCP_MES_PRGRM_CNTR_START_HI, ucode_start[key] >> 2,
            base_idx=1,
        )
        active_mask |= (CP_MES_CNTL__MES_PIPE0_ACTIVE if pipe == 0
                        else CP_MES_CNTL__MES_PIPE1_ACTIVE)

    grbm_deselect(dev, gc_base)
    val = _gc_reg(dev, gc_base, regCP_MES_CNTL, base_idx=1)
    val &= ~(CP_MES_CNTL__MES_INVALIDATE_ICACHE |
             CP_MES_CNTL__MES_PIPE0_RESET |
             CP_MES_CNTL__MES_PIPE1_RESET |
             CP_MES_CNTL__MES_HALT |
             CP_MES_CNTL__MES_PIPE0_ACTIVE |
             CP_MES_CNTL__MES_PIPE1_ACTIVE)
    val |= active_mask
    _gc_wreg(dev, gc_base, regCP_MES_CNTL, val, base_idx=1)
    time.sleep(0.001)
    mes_cntl = _gc_reg(dev, gc_base, regCP_MES_CNTL, base_idx=1)
    print(f"  GFX: MES enabled (CP_MES_CNTL=0x{mes_cntl:08X})")


def init_gfx_for_compute(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    psp_config: object | None = None,
    smu_config: SMUConfig | None = None,
) -> list[int]:
    """Initialize the minimal GC/MEC state needed for direct compute queues."""
    gc_base = resolve_gc_bases(ip_result)
    _rlc_backdoor_autoload(dev, gc_base, psp_config, smu_config)
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
    if ucode_start:
        if not getattr(psp_config, "rlc_backdoor_autoloaded", False):
            _program_mes_ucode_buffers(dev, gc_base, psp_config)
        _enable_mes_from_ucode(dev, gc_base, ucode_start)
    return gc_base


def init_mes_for_compute(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
) -> MESRingConfig:
    """Initialize a MES KIQ ring and scheduler ring for queue management."""
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    gc_base = resolve_gc_bases(ip_result)
    mmhub_base = _resolve_hwip_bases(ip_result, HardwareID.MMHUB)
    osssys_base = _resolve_hwip_bases(ip_result, HardwareID.OSSSYS)

    schedulers = _gc_reg(dev, gc_base, regRLC_CP_SCHEDULERS, base_idx=1)
    schedulers &= 0xFFFFFF00
    schedulers |= (3 << 5) | (MES_PIPE_KIQ << 3) | 0
    _gc_wreg(dev, gc_base, regRLC_CP_SCHEDULERS, schedulers, base_idx=1)
    _gc_wreg(dev, gc_base, regRLC_CP_SCHEDULERS, schedulers | 0x80,
             base_idx=1)

    kiq = MESRingConfig(
        gc_base=gc_base,
        mmhub_base=mmhub_base,
        osssys_base=osssys_base,
        pipe=MES_PIPE_KIQ,
        doorbell_index=DOORBELL_MES_RING1,
    )
    _init_mes_ring(dev, kiq, nbio_config, direct_activate=True)
    _mes_set_hw_resources(kiq, full=False)

    scheduler = MESRingConfig(
        gc_base=gc_base,
        mmhub_base=mmhub_base,
        osssys_base=osssys_base,
        pipe=MES_PIPE_SCHED,
        doorbell_index=DOORBELL_MES_RING0,
    )
    _init_mes_ring(dev, scheduler, nbio_config, direct_activate=False)
    _enable_unmapped_doorbell_handling(dev, gc_base, True)
    _mes_map_legacy_queue(kiq, scheduler, queue_type=MES_QUEUE_TYPE_SCHQ)
    scheduler.ready = True
    _mes_set_hw_resources(scheduler, full=True)
    _mes_query_status(scheduler)

    kiq.scheduler = scheduler
    print("  MES: KIQ and scheduler rings initialized")
    print(f"  MES: KIQ doorbell=0x{kiq.doorbell_index:X}, "
          f"scheduler doorbell=0x{scheduler.doorbell_index:X}")
    return kiq


def init_compute_queue(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
    *,
    pipe: int = 0,
    queue: int = 0,
    doorbell_index: int = DOORBELL_MEC_RING_START,
    ring_size: int = COMPUTE_RING_SIZE,
    mes_ring: MESRingConfig | None = None,
    use_mes: bool | None = None,
) -> ComputeQueueConfig:
    """Initialize a compute queue, preferring MES legacy queue mapping.

    Allocates all required DMA buffers, initializes the MQD, programs
    CP_HQD_* registers directly only when MES is unavailable or disabled.

    Args:
        dev: Windows device backend.
        ip_result: Parsed IP discovery data.
        nbio_config: NBIO configuration (for doorbell BAR info).
        pipe: Compute pipe (0-7).
        queue: Queue within the pipe (0-7).
        doorbell_index: Doorbell index for this queue.
        ring_size: Ring buffer size in bytes (power of 2).
        mes_ring: Initialized MES KIQ ring used for legacy queue mapping.
        use_mes: Override MES mapping behavior. If None, follows
                 AMDGPU_LITE_QUEUE_MODE.

    Returns:
        Configured ComputeQueueConfig ready for packet submission.
    """
    gc_base = resolve_gc_bases(ip_result)
    queue_mode = os.environ.get("AMDGPU_LITE_QUEUE_MODE", "auto").lower()
    if use_mes is None:
        use_mes = mes_ring is not None and queue_mode != "direct"
    if use_mes and "AMDGPU_LITE_MES_QUEUE" in os.environ:
        queue = int(os.environ["AMDGPU_LITE_MES_QUEUE"], 0)
    elif use_mes and mes_ring is not None and queue == 0:
        queue = 2

    config = ComputeQueueConfig(gc_base=gc_base)
    config.ring_size = ring_size
    config.me = 1  # MEC0
    config.pipe = pipe
    config.queue = queue
    config.doorbell_index = (
        doorbell_index + (pipe * 4 + queue) * DOORBELL_MEC_RING_STRIDE
        if doorbell_index == DOORBELL_MEC_RING_START else doorbell_index
    )
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
    eop_cpu, eop_bus, eop_handle = _alloc_queue_buffer(
        dev, config, EOP_BUFFER_SIZE)
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

    config.doorbell_cpu_addr = _doorbell_cpu_addr(
        dev, nbio_config, config.doorbell_index)

    # Initialize MQD
    _init_compute_mqd(config)

    activated_by = "direct MMIO"
    if use_mes:
        if mes_ring is None:
            raise RuntimeError("AMDGPU_LITE_QUEUE_MODE=mes but MES is not initialized")
        try:
            _mes_map_legacy_queue(
                mes_ring, config, queue_type=MES_QUEUE_TYPE_COMPUTE)
            activated_by = "MES legacy map"
        except Exception:
            if queue_mode == "mes":
                raise
            print("  Compute: MES queue map failed; falling back to direct MMIO")
            _activate_compute_queue_mmio(dev, config)
    else:
        if queue_mode == "mes":
            raise RuntimeError("MES queue mode requested but no MES ring was provided")
        _activate_compute_queue_mmio(dev, config)

    config.active = True
    config.wptr = 0

    print(f"  Compute: Queue ME={config.me} pipe={config.pipe} "
          f"queue={config.queue} activated via {activated_by}")
    print(f"  Compute: Ring at bus 0x{ring_bus:012X}, "
          f"size={ring_size // 1024}KB")
    print(f"  Compute: Doorbell index=0x{config.doorbell_index:X}")

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
