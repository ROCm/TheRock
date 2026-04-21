"""PSP command submission via the KM ring for gfx1201 (PSP v14.0.3).

Mirrors `amdgpu_psp.c::psp_ring_cmd_submit` + `psp_cmd_submit_buf` + the
`psp_v14_0_ring_get_wptr`/`set_wptr` pair. After `psp_ring.ring_create`
has set up the KM ring, caller uses `submit_ip_fw_load(...)` (or the
generic `submit(...)`) to send commands to SOS.

Flow per command (non-SRIOV):
  1. memset cmd_buf (1 KB) to 0.
  2. Build psp_gfx_cmd_resp struct in cmd_buf.
  3. Read wptr from C2PMSG_67 (DWORD count).
  4. Locate the next psp_gfx_rb_frame in the ring memory.
  5. Populate cmd_buf_addr, fence_addr, fence_value.
  6. Bump wptr by sizeof(rb_frame)/4 (= 16 DWORDs) mod ring_dw_size.
  7. Write new wptr back to C2PMSG_67 (this is the kick).
  8. Poll *fence_buf == fence_value (or timeout).
  9. Read resp.status at cmd_buf[+864].
"""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass

from .psp_bootloader import c2pmsg_dw
from .psp_ring import PSPRing


# ---- GFX_CMD_ID values (psp_gfx_if.h) ----
GFX_CMD_ID_LOAD_TA      = 0x00000001
GFX_CMD_ID_UNLOAD_TA    = 0x00000002
GFX_CMD_ID_INVOKE_CMD   = 0x00000003
GFX_CMD_ID_LOAD_ASD     = 0x00000004
GFX_CMD_ID_SETUP_TMR    = 0x00000005
GFX_CMD_ID_LOAD_IP_FW   = 0x00000006
GFX_CMD_ID_LOAD_TOC     = 0x00000020
GFX_CMD_ID_AUTOLOAD_RLC = 0x00000021   # "all graphics fw loaded, start RLC autoload"
GFX_CMD_ID_BOOT_CFG     = 0x00000022

# ---- GFX_FW_TYPE enum (from psp_gfx_if.h — authoritative) ----
# Values evolved across GPU families; these are correct for SOC21/SOC22
# (gfx11/gfx12 — Navi 31/48). Some of the small-value types (CP_ME,
# CP_PFP, RLC_G, etc.) originated for older cards but stayed stable.
GFX_FW_TYPE_NONE        = 0
GFX_FW_TYPE_CP_ME       = 1
GFX_FW_TYPE_CP_PFP      = 2
GFX_FW_TYPE_CP_CE       = 3
GFX_FW_TYPE_CP_MEC      = 4
GFX_FW_TYPE_CP_MEC_ME1  = 5
GFX_FW_TYPE_CP_MEC_ME2  = 6
GFX_FW_TYPE_RLC_V       = 7
GFX_FW_TYPE_RLC_G       = 8
# 9-17 are older-family SDMA/DMCU/VCN/UVD/VCE/ISP/ACP — unused here.
GFX_FW_TYPE_SMU         = 18
# RLC save/restore-list sub-firmwares (from rlc_firmware_header_v2_1):
GFX_FW_TYPE_RLC_RESTORE_LIST_GPM_MEM  = 20   # SRLG
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_MEM  = 21   # SRLS
GFX_FW_TYPE_RLC_RESTORE_LIST_SRM_CNTL = 22   # SRLC
GFX_FW_TYPE_TOC         = 24
GFX_FW_TYPE_RLC_P       = 25
GFX_FW_TYPE_RLC_IRAM    = 26
# Tap-delay sub-firmwares live in RLC v2_4 containers:
GFX_FW_TYPE_GLOBAL_TAP_DELAYS          = 27
GFX_FW_TYPE_SE0_TAP_DELAYS             = 28
GFX_FW_TYPE_SE1_TAP_DELAYS             = 29
GFX_FW_TYPE_GLOBAL_SE0_SE1_SKEW_DELAYS = 30
GFX_FW_TYPE_CP_MES                     = 33
# SOC21+ (gfx11/gfx12) types:
GFX_FW_TYPE_IMU_I       = 68
GFX_FW_TYPE_IMU_D       = 69
GFX_FW_TYPE_SDMA_UCODE_TH0 = 71
GFX_FW_TYPE_SDMA_UCODE_TH1 = 72
GFX_FW_TYPE_PPTABLE     = 73
GFX_FW_TYPE_RS64_MES    = 76
GFX_FW_TYPE_RS64_MES_STACK = 77
GFX_FW_TYPE_RS64_KIQ    = 78
GFX_FW_TYPE_RS64_KIQ_STACK = 79
GFX_FW_TYPE_RS64_PFP    = 87
GFX_FW_TYPE_RS64_ME     = 88
GFX_FW_TYPE_RS64_MEC    = 89
GFX_FW_TYPE_RS64_PFP_P0_STACK = 90
GFX_FW_TYPE_RS64_PFP_P1_STACK = 91
GFX_FW_TYPE_RS64_ME_P0_STACK  = 92
GFX_FW_TYPE_RS64_ME_P1_STACK  = 93
GFX_FW_TYPE_RS64_MEC_P0_STACK = 94
GFX_FW_TYPE_RS64_MEC_P1_STACK = 95
GFX_FW_TYPE_RS64_MEC_P2_STACK = 96
GFX_FW_TYPE_RS64_MEC_P3_STACK = 97
GFX_FW_TYPE_SE2_TAP_DELAYS    = 65
GFX_FW_TYPE_SE3_TAP_DELAYS    = 66
GFX_FW_TYPE_SE2_MUX_SELECT_RAM = 43
GFX_FW_TYPE_SE3_MUX_SELECT_RAM = 44

# ---- Layout constants ----
PSP_GFX_CMD_BUF_VERSION = 0x00000001
PSP_GFX_CMD_BUF_SIZE    = 1024   # sizeof(struct psp_gfx_cmd_resp)
PSP_GFX_RB_FRAME_SIZE   = 64     # sizeof(struct psp_gfx_rb_frame)
PSP_GFX_RB_FRAME_DW     = PSP_GFX_RB_FRAME_SIZE // 4   # 16
PSP_GFX_RESP_OFFSET     = 864    # offset of struct psp_gfx_resp within cmd
PSP_FENCE_VALUE_OFFSET  = 0      # our fence buf is just a single DWORD

# Response status codes we care about
TEE_SUCCESS                  = 0
TEE_ERROR_NOT_SUPPORTED      = 0xFFFF000A
PSP_ERR_UNKNOWN_COMMAND      = 0x00000100


@dataclass
class PSPCmdCtx:
    """Per-device PSP command context: pre-allocated cmd + fence buffers."""
    cmd_cpu: int
    cmd_bus: int
    cmd_handle: int
    fence_cpu: int
    fence_bus: int
    fence_handle: int
    fence_value: int = 0  # monotonically incremented per cmd


def alloc_cmd_ctx(driver) -> PSPCmdCtx:
    """Allocate DMA-backed cmd + fence buffers for PSP command submission.

    Both are 4 KB (matches PSP_CMD_BUFFER_SIZE / PSP_FENCE_BUFFER_SIZE).
    Caller keeps this live for the lifetime of the ring.
    """
    cmd_cpu, cmd_bus, cmd_handle = driver.alloc_dma(0x1000)
    fence_cpu, fence_bus, fence_handle = driver.alloc_dma(0x1000)
    ctypes.memset(cmd_cpu, 0, 0x1000)
    ctypes.memset(fence_cpu, 0, 0x1000)
    return PSPCmdCtx(
        cmd_cpu=cmd_cpu, cmd_bus=cmd_bus, cmd_handle=cmd_handle,
        fence_cpu=fence_cpu, fence_bus=fence_bus, fence_handle=fence_handle,
        fence_value=0,
    )


def free_cmd_ctx(driver, ctx: PSPCmdCtx) -> None:
    driver.free_dma(ctx.cmd_handle)
    driver.free_dma(ctx.fence_handle)
    ctx.cmd_cpu = ctx.cmd_bus = ctx.cmd_handle = 0
    ctx.fence_cpu = ctx.fence_bus = ctx.fence_handle = 0


def _build_ip_fw_cmd(cmd_cpu: int, fw_bus_addr: int, fw_size: int,
                     fw_type: int) -> None:
    """Zero cmd_buf and write a psp_gfx_cmd_resp for GFX_CMD_ID_LOAD_IP_FW."""
    ctypes.memset(cmd_cpu, 0, PSP_GFX_CMD_BUF_SIZE)
    # psp_gfx_cmd_resp header (first 28 bytes):
    #   [0]  buf_size
    #   [4]  buf_version
    #   [8]  cmd_id
    #   [12] resp_buf_addr_lo (0 — using cmd_buf inline)
    #   [16] resp_buf_addr_hi (0)
    #   [20] resp_offset      (0)
    #   [24] resp_buf_size    (0)
    #   [28] cmd (union) — for load_ip_fw:
    #        [+0]  fw_phy_addr_lo
    #        [+4]  fw_phy_addr_hi
    #        [+8]  fw_size
    #        [+12] fw_type
    hdr = struct.pack(
        "<IIIIIII",
        PSP_GFX_CMD_BUF_SIZE,
        PSP_GFX_CMD_BUF_VERSION,
        GFX_CMD_ID_LOAD_IP_FW,
        0, 0, 0, 0,
    )
    load_ip = struct.pack(
        "<IIII",
        fw_bus_addr & 0xFFFFFFFF,
        (fw_bus_addr >> 32) & 0xFFFFFFFF,
        fw_size,
        fw_type,
    )
    (ctypes.c_ubyte * len(hdr)).from_address(cmd_cpu)[:] = hdr
    (ctypes.c_ubyte * len(load_ip)).from_address(cmd_cpu + 28)[:] = load_ip


def _write_rb_frame(ring: PSPRing, frame_idx: int,
                    cmd_bus: int, fence_bus: int, fence_value: int) -> None:
    """Write a psp_gfx_rb_frame into ring memory at the given frame index.

    Frame layout (64 bytes):
      [0..4]   cmd_buf_addr_lo
      [4..8]   cmd_buf_addr_hi
      [8..12]  cmd_buf_size
      [12..16] fence_addr_lo
      [16..20] fence_addr_hi
      [20..24] fence_value
      [24..32] sid_lo/sid_hi (0)
      [32]     vmid (0)
      [33]     frame_type (0 = RBI, but unused for GPCOM frames)
      [34..64] reserved (0)
    """
    frame_cpu = ring.ring_cpu + frame_idx * PSP_GFX_RB_FRAME_SIZE
    ctypes.memset(frame_cpu, 0, PSP_GFX_RB_FRAME_SIZE)
    frame = struct.pack(
        "<IIIIII",
        cmd_bus & 0xFFFFFFFF,          # cmd_buf_addr_lo
        (cmd_bus >> 32) & 0xFFFFFFFF,  # cmd_buf_addr_hi
        PSP_GFX_CMD_BUF_SIZE,          # cmd_buf_size
        fence_bus & 0xFFFFFFFF,        # fence_addr_lo
        (fence_bus >> 32) & 0xFFFFFFFF,# fence_addr_hi
        fence_value,                   # fence_value
    )
    (ctypes.c_ubyte * len(frame)).from_address(frame_cpu)[:] = frame


def submit(client, driver, mp0_base_dw: int,
           ring: PSPRing, ctx: PSPCmdCtx,
           *, timeout_ms: int = 20000,
           verbose: bool = False) -> dict:
    """Kick the command currently sitting in ctx.cmd_cpu, wait for fence.

    Caller must have populated ctx.cmd_cpu with a valid psp_gfx_cmd_resp
    (e.g. via _build_ip_fw_cmd). Returns a dict with 'status' (resp.status),
    'fence_value', and 'raw_resp' (96 bytes of struct psp_gfx_resp).
    """
    c67 = c2pmsg_dw(mp0_base_dw, 67) * 4

    # 1. Advance fence value BEFORE writing the frame (mirrors atomic_inc_return).
    ctx.fence_value += 1
    fence_value = ctx.fence_value
    # Clear the fence slot.
    ctypes.memset(ctx.fence_cpu, 0, 4)

    # 2. Read current write pointer (DWORD count).
    wptr_dw = client.mmio_read32(5, c67)
    ring_size_dw = ring.ring_size // 4
    # 3. Resolve frame index: if wptr_dw == 0 (or wraps), start at frame 0.
    if (wptr_dw % ring_size_dw) == 0:
        frame_idx = 0
    else:
        frame_idx = wptr_dw // PSP_GFX_RB_FRAME_DW
    if verbose:
        print(f"  submit: wptr_dw=0x{wptr_dw:x} frame_idx={frame_idx} "
              f"fence_value={fence_value}")

    # 4. Write the frame.
    _write_rb_frame(ring, frame_idx, ctx.cmd_bus, ctx.fence_bus, fence_value)

    # 5. Bump the wptr by 16 DWORDs (one frame).
    new_wptr = (wptr_dw + PSP_GFX_RB_FRAME_DW) % ring_size_dw
    # 6. Kick PSP by writing new wptr to C2PMSG_67.
    client.mmio_write32(5, c67, new_wptr)

    # 7. Poll fence (bus-visible memory, same virtual buffer CPU-mapped).
    fence_view = ctypes.cast(ctx.fence_cpu, ctypes.POINTER(ctypes.c_uint32))
    deadline = time.time() + timeout_ms / 1000
    observed = 0
    while time.time() < deadline:
        observed = fence_view[0]
        if observed == fence_value:
            break
        time.sleep(0.0005)
    if observed != fence_value:
        raise TimeoutError(
            f"PSP cmd fence did not fire within {timeout_ms} ms "
            f"(expected {fence_value}, got {observed}; "
            f"wptr now 0x{client.mmio_read32(5, c67):x})"
        )

    # 8. Read back response (the 96-byte struct psp_gfx_resp at +864 of cmd_buf).
    resp_bytes = bytes((ctypes.c_ubyte * 96).from_address(
        ctx.cmd_cpu + PSP_GFX_RESP_OFFSET))
    status = struct.unpack_from("<I", resp_bytes, 0)[0]
    if verbose:
        print(f"  submit: fired at fence={observed}  resp.status=0x{status:08x}")

    return {
        "status": status,
        "fence_value": fence_value,
        "raw_resp": resp_bytes,
    }


def submit_ip_fw_load(client, driver, mp0_base_dw: int,
                      ring: PSPRing, ctx: PSPCmdCtx,
                      fw_bus_addr: int, fw_size: int, fw_type: int,
                      *, timeout_ms: int = 20000,
                      verbose: bool = False) -> dict:
    """Shortcut: build a GFX_CMD_ID_LOAD_IP_FW command and submit.

    Caller must have pre-loaded the firmware bytes at fw_bus_addr (a
    DMA-mapped buffer) before calling this.
    """
    _build_ip_fw_cmd(ctx.cmd_cpu, fw_bus_addr, fw_size, fw_type)
    return submit(client, driver, mp0_base_dw, ring, ctx,
                  timeout_ms=timeout_ms, verbose=verbose)
