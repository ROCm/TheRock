"""PSP KM ring creation for gfx1201 (PSP v14.0.3).

Mirrors `drivers/gpu/drm/amd/amdgpu/psp_v14_0.c::psp_v14_0_ring_create` plus
the common `amdgpu_psp.c::psp_ring_create` flow. After SOS is alive, driver
allocates a ring buffer in DMA-visible memory and hands its physical address
to SOS via C2PMSG_69/70/71/64. SOS then consumes `psp_gfx_cmd_resp` commands
from the ring (produced via the write pointer in C2PMSG_67).

Ring layout:
    - ring_mem: array of `psp_gfx_rb_frame` (64 B each), DMA-allocated.
    - ring_size: 0x8000 (32 KB, matches Linux PSP_RING_BUFFER_SIZE on v14).
    - The write pointer is a DWORD-offset into this array of frames, so
      `frame_index = (ring_wptr * 4) / sizeof(psp_gfx_rb_frame)`.

Handshake (non-SRIOV path):
    1. Wait C2PMSG_64 sign-of-life: bit 31 set, low 16 == 0.
    2. Write ring bus-addr-low  to C2PMSG_69.
    3. Write ring bus-addr-high to C2PMSG_70.
    4. Write ring_size          to C2PMSG_71.
    5. Write (PSP_RING_TYPE__KM << 16) to C2PMSG_64 (kick).
    6. 20 ms delay.
    7. Wait C2PMSG_64 bit 31 set, low 16 == 0 again (response).
"""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass

from .psp_bootloader import c2pmsg_dw


# ---- Ring commands (C2PMSG_64, SRIOV uses C2PMSG_101) ----
GFX_CTRL_CMD_ID_INIT_RBI_RING   = 0x00010000
GFX_CTRL_CMD_ID_INIT_GPCOM_RING = 0x00020000
GFX_CTRL_CMD_ID_DESTROY_RINGS   = 0x00030000
GFX_CTRL_CMD_ID_MAX             = 0x000F0000

# ---- Ring type ----
PSP_RING_TYPE__INVALID = 0
PSP_RING_TYPE__UM      = 1  # User mode ring (formerly RBI)
PSP_RING_TYPE__KM      = 2  # Kernel mode ring (formerly GPCOM)

# ---- MBOX response bit layout (from psp_gfx_if.h) ----
GFX_FLAG_RESPONSE      = 0x80000000
GFX_CMD_RESPONSE_MASK  = 0x80000000
GFX_CMD_STATUS_MASK    = 0x0000FFFF
GFX_CMD_ID_MASK        = 0x000F0000
GFX_CMD_RESERVED_MASK  = 0x7FF00000

# `MBOX_TOS_READY_FLAG = GFX_FLAG_RESPONSE; MASK = RESPONSE | STATUS`
# A "ready" / "response" means: bit 31 set, low-16 status == 0 (no error).
MBOX_TOS_FLAG = GFX_FLAG_RESPONSE
MBOX_TOS_MASK = GFX_CMD_RESPONSE_MASK | GFX_CMD_STATUS_MASK

# ---- Sizes ----
PSP_RING_FRAME_SIZE = 64       # sizeof(struct psp_gfx_rb_frame)
PSP_KM_RING_SIZE    = 0x8000   # 32 KB
PSP_FENCE_BUF_SIZE  = 0x1000
PSP_CMD_BUF_SIZE    = 0x1000
PSP_RESP_BUF_SIZE   = 0x1000


@dataclass
class PSPRing:
    """Live PSP KM ring handle — passes into psp command submission."""
    ring_cpu: int          # CPU VA of ring base (mapped in user process)
    ring_bus: int          # DART-mapped bus addr (PSP-visible)
    ring_handle: int       # DEXT handle for free_dma()
    ring_size: int         # bytes
    ring_type: int         # PSP_RING_TYPE__KM
    # Fence + cmd + resp buffers (allocated together, optional)
    fence_cpu: int = 0
    fence_bus: int = 0
    fence_handle: int = 0
    # write pointer (DWORD offset into ring_mem)
    wptr: int = 0


def _wait_c64_response(client, mp0_base_dw: int, timeout_ms: int,
                       expect_mask: int = MBOX_TOS_MASK,
                       expect_flag: int = MBOX_TOS_FLAG) -> int:
    """Wait for (C2PMSG_64 & mask) == flag.

    Mirrors psp_wait_for(psp, reg, value, mask, check_changed=0):
    returns when `(reg & mask) == value`.
    """
    reg_byte = c2pmsg_dw(mp0_base_dw, 64) * 4
    deadline = time.time() + timeout_ms / 1000
    v = 0
    while time.time() < deadline:
        v = client.mmio_read32(5, reg_byte)
        if (v & expect_mask) == expect_flag:
            return v
        time.sleep(0.005)
    raise TimeoutError(
        f"C2PMSG_64 did not reach flag=0x{expect_flag:08x} "
        f"mask=0x{expect_mask:08x} within {timeout_ms} ms (last=0x{v:08x})"
    )


def ring_create(client, driver, mp0_base_dw: int,
                *, ring_size: int = PSP_KM_RING_SIZE,
                verbose: bool = False) -> PSPRing:
    """Create a KM (kernel-mode / GPCOM) ring for the PSP.

    Requires SOS to be alive (C2PMSG_81 != 0). Returns a PSPRing that holds
    the DMA buffer + bus address + write pointer. Caller owns the ring and
    must call ring_destroy() (or rely on DEXT teardown) to reclaim.
    """
    # 1. Allocate ring buffer (DART-mapped so PSP sees the right phys addr).
    ring_cpu, ring_bus, ring_handle = driver.alloc_dma(ring_size)
    # Zero the ring — psp_gfx_rb_frame.reserved* fields must be 0.
    ctypes.memset(ring_cpu, 0, ring_size)

    # 2. Wait for SOS to be ready for ring creation.
    #    C2PMSG_64 should read 0x80000000 (bit 31 set, low 16 == 0).
    v = _wait_c64_response(client, mp0_base_dw, timeout_ms=2000)
    if verbose:
        print(f"  TOS ready: C2PMSG_64 = 0x{v:08x}")

    # 3-5. Write ring bus address + size.
    c69 = c2pmsg_dw(mp0_base_dw, 69) * 4
    c70 = c2pmsg_dw(mp0_base_dw, 70) * 4
    c71 = c2pmsg_dw(mp0_base_dw, 71) * 4
    c64 = c2pmsg_dw(mp0_base_dw, 64) * 4

    client.mmio_write32(5, c69, ring_bus & 0xFFFFFFFF)
    client.mmio_write32(5, c70, (ring_bus >> 32) & 0xFFFFFFFF)
    client.mmio_write32(5, c71, ring_size)

    # 6. Kick: write (ring_type << 16) into C2PMSG_64.
    kick = PSP_RING_TYPE__KM << 16  # 0x00020000 = GFX_CTRL_CMD_ID_INIT_GPCOM_RING
    client.mmio_write32(5, c64, kick)
    if verbose:
        print(f"  Ring kick: C2PMSG_64 <= 0x{kick:08x}  "
              f"(bus=0x{ring_bus:x}, size=0x{ring_size:x})")

    # 7. Per Linux driver "handshake issue — needs delay".
    time.sleep(0.020)

    # 8. Wait for response flag.
    v = _wait_c64_response(client, mp0_base_dw, timeout_ms=2000)
    status = v & GFX_CMD_STATUS_MASK
    if status != 0:
        driver.free_dma(ring_handle)
        raise RuntimeError(
            f"PSP ring create failed: C2PMSG_64 = 0x{v:08x} "
            f"(status=0x{status:04x})")
    if verbose:
        print(f"  Ring created OK: C2PMSG_64 = 0x{v:08x}")

    return PSPRing(
        ring_cpu=ring_cpu,
        ring_bus=ring_bus,
        ring_handle=ring_handle,
        ring_size=ring_size,
        ring_type=PSP_RING_TYPE__KM,
    )


def ring_destroy(client, driver, mp0_base_dw: int, ring: PSPRing,
                 *, verbose: bool = False) -> None:
    """Tear down the KM ring via GFX_CTRL_CMD_ID_DESTROY_RINGS.

    Safe to call even if the ring is in an unknown state — will time out
    after 2 s and free the DMA buffer regardless.
    """
    c64 = c2pmsg_dw(mp0_base_dw, 64) * 4
    try:
        client.mmio_write32(5, c64, GFX_CTRL_CMD_ID_DESTROY_RINGS)
        time.sleep(0.020)
        v = _wait_c64_response(client, mp0_base_dw, timeout_ms=2000)
        if verbose:
            print(f"  Ring destroyed: C2PMSG_64 = 0x{v:08x}")
    except TimeoutError as e:
        if verbose:
            print(f"  Ring destroy timed out: {e}")
    finally:
        driver.free_dma(ring.ring_handle)
        ring.ring_cpu = 0
        ring.ring_bus = 0
        ring.ring_handle = 0
