"""Phase 13: submit a minimal no-op compute dispatch through AMDDevice.

This validates more than raw CP packet execution: CP must program compute
state, fetch a one-instruction shader from VRAM, dispatch it, and then execute
the post-dispatch WRITE_DATA marker after CS_PARTIAL_FLUSH.

RELEASE_MEM is deliberately not used here; that packet is still unresolved on
the macOS/eGPU path, while WRITE_DATA is proven on the direct compute queue.
"""
from __future__ import annotations

import ctypes
import os
import struct
import time

from amd_gpu_driver.commands.pm4 import (
    CP_COHER_CNTL_SH_ICACHE_ACTION,
    CP_COHER_CNTL_SH_KCACHE_ACTION,
    CP_COHER_CNTL_TCL1_ACTION,
    CS_PARTIAL_FLUSH,
    EVENT_INDEX_CS_PARTIAL_FLUSH,
    PM4PacketBuilder,
)
from amd_gpu_driver.device import AMDDevice
from amd_gpu_driver.gpu.registers import (
    COMPUTE_PGM_LO,
    COMPUTE_PGM_RSRC1,
    COMPUTE_PGM_RSRC3_GFX12,
    COMPUTE_RESOURCE_LIMITS,
    COMPUTE_RESTART_X,
    COMPUTE_START_X,
    COMPUTE_TMPRING_SIZE,
    COMPUTE_USER_DATA_0,
    sh_reg_offset,
)


# GFX12 SOPP s_endpgm.
NOOP_SHADER_CODE = struct.pack("<I", 0xBF810000)

# Minimal gfx12 resources matching AMDGPU kernel descriptor defaults:
# VGPRS=3 means 32 VGPRs for wave32, plus WGP/MEM_ORDERED/FWD_PROGRESS.
NOOP_RSRC1 = 3 | (0xC0 << 12) | (1 << 29) | (1 << 30) | (1 << 31)
NOOP_RSRC2 = 1 << 7
DISPATCH_INITIATOR_GFX12_W32 = 0x5 | (1 << 6) | (1 << 15)

PRE_MARKER = 0x0D15C001
AFTER_ACQUIRE_MARKER = 0x0D15C00A
AFTER_REGS_MARKER = 0x0D15C00B
POST_MARKER = 0x0D15C002

MARKER_NAMES = {
    0: "zero",
    PRE_MARKER: "pre",
    AFTER_ACQUIRE_MARKER: "after-acquire",
    AFTER_REGS_MARKER: "after-regs",
    POST_MARKER: "post",
}

GC_B0 = 0x1260
GC_B1 = 0xA000
regGRBM_GFX_CNTL = 0x0900
regCP_HQD_ACTIVE = 0x1FAB
regCP_HQD_PQ_RPTR = 0x1FB3
regCP_HQD_PQ_WPTR_LO = 0x1FDF
regCP_STAT = 0x0F40
regGRBM_STATUS = 0x0DA4


def read64(addr: int) -> int:
    return ctypes.c_uint64.from_address(addr).value


def read_compute_status(backend: object) -> tuple[int, int, int, int, int]:
    def rd(base: int, off: int) -> int:
        return backend.read_reg32((base + off) * 4)

    def wr(base: int, off: int, value: int) -> None:
        backend.write_reg32((base + off) * 4, value)

    # Direct compute queue prototype uses ME=1 pipe=0 queue=0.
    wr(GC_B1, regGRBM_GFX_CNTL, 1 << 2)
    active = rd(GC_B0, regCP_HQD_ACTIVE)
    rptr = rd(GC_B0, regCP_HQD_PQ_RPTR)
    wptr = rd(GC_B0, regCP_HQD_PQ_WPTR_LO)
    cp_stat = rd(GC_B0, regCP_STAT)
    grbm = rd(GC_B0, regGRBM_STATUS)
    return active, rptr, wptr, cp_stat, grbm


def build_noop_dispatch(code_addr: int, kernarg_addr: int, marker_addr: int) -> bytes:
    pm4 = PM4PacketBuilder()
    skip_all_regs = os.environ.get("PHASE13_SKIP_ALL_REGS") == "1"

    # Proves the queue is still executing commands before the dispatch.
    pm4.write_data(marker_addr, [PRE_MARKER, 0])

    if os.environ.get("PHASE13_ENABLE_ACQUIRE") == "1" and os.environ.get("PHASE13_SKIP_ACQUIRE") != "1":
        pm4.acquire_mem(
            coher_cntl=(
                CP_COHER_CNTL_SH_ICACHE_ACTION
                | CP_COHER_CNTL_SH_KCACHE_ACTION
                | CP_COHER_CNTL_TCL1_ACTION
            ),
        )
        pm4.write_data(marker_addr, [AFTER_ACQUIRE_MARKER, 0])

    if not skip_all_regs:
        if os.environ.get("PHASE13_SKIP_PGM_ADDR") != "1":
            pgm_addr = code_addr >> 8
            pm4.set_sh_reg(
                sh_reg_offset(COMPUTE_PGM_LO),
                pgm_addr & 0xFFFFFFFF,
                (pgm_addr >> 32) & 0xFFFFFFFF,
            )
        if os.environ.get("PHASE13_SKIP_RSRC12") != "1":
            pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_RSRC1), NOOP_RSRC1, NOOP_RSRC2)
        if os.environ.get("PHASE13_SKIP_RSRC3") != "1":
            pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_RSRC3_GFX12), 0)
        if os.environ.get("PHASE13_SKIP_TMPRING") != "1":
            pm4.set_sh_reg(sh_reg_offset(COMPUTE_TMPRING_SIZE), 0)
        if os.environ.get("PHASE13_SKIP_RESTART") != "1":
            pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESTART_X), 0, 0, 0)
        if os.environ.get("PHASE13_SKIP_USER_DATA") != "1":
            pm4.set_sh_reg(
                sh_reg_offset(COMPUTE_USER_DATA_0),
                kernarg_addr & 0xFFFFFFFF,
                (kernarg_addr >> 32) & 0xFFFFFFFF,
            )
        if os.environ.get("PHASE13_SKIP_RESOURCE_LIMITS") != "1":
            pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESOURCE_LIMITS), 0)
        if os.environ.get("PHASE13_SKIP_THREADS") != "1":
            pm4.set_sh_reg(
                sh_reg_offset(COMPUTE_START_X),
                0, 0, 0,     # start x/y/z
                1, 1, 1,     # one thread per workgroup
                0, 0,
            )
        pm4.write_data(marker_addr, [AFTER_REGS_MARKER, 0])

    if os.environ.get("PHASE13_SKIP_DISPATCH") != "1":
        pm4.dispatch_direct(1, 1, 1, initiator=DISPATCH_INITIATOR_GFX12_W32)
        if os.environ.get("PHASE13_SKIP_EVENT") != "1":
            pm4.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH)

    # Completion marker that avoids RELEASE_MEM/EOP.
    pm4.write_data(marker_addr, [POST_MARKER, 0])
    return pm4.build()


def main() -> None:
    dev = AMDDevice(backend="macos")
    print(f"device={dev.name} gfx={dev.gfx_target}")

    code = dev.alloc(4096, location="vram", executable=True)
    kernarg = dev.alloc(4096, location="vram")
    marker = dev.alloc(4096, location="vram")
    code.fill(0)
    kernarg.fill(0)
    marker.fill(0)
    code.write(NOOP_SHADER_CODE)

    queue = dev.backend.create_compute_queue()
    packets = build_noop_dispatch(code.gpu_addr, kernarg.gpu_addr, marker.gpu_addr)
    print(
        f"code=0x{code.gpu_addr:x} kernarg=0x{kernarg.gpu_addr:x} "
        f"marker=0x{marker.gpu_addr:x} pm4_dwords={len(packets) // 4}"
    )

    dev.backend.submit_packets(queue, packets)

    deadline = time.time() + 5
    last = None
    poll_regs = os.environ.get("PHASE13_POLL_REGS") == "1"
    while time.time() < deadline:
        marker_value = read64(marker.cpu_addr)
        rptr_report = read64(queue.read_ptr_addr) & 0xFFFFFFFF
        wptr_mem = read64(queue.write_ptr_addr)
        if poll_regs:
            active, rptr_reg, wptr_reg, cp_stat, grbm = read_compute_status(dev.backend)
            reg_text = (
                f" active=0x{active:x} rptr_reg=0x{rptr_reg:x} "
                f"wptr_reg=0x{wptr_reg:x} CP_STAT=0x{cp_stat:x} GRBM=0x{grbm:x}"
            )
            snap = (marker_value, rptr_report, wptr_mem, active, rptr_reg, wptr_reg, cp_stat, grbm)
        else:
            reg_text = ""
            snap = (marker_value, rptr_report, wptr_mem)
        if snap != last:
            marker_name = MARKER_NAMES.get(marker_value, "unknown")
            print(
                f"marker=0x{marker_value:016x}({marker_name}) "
                f"rptr_report=0x{rptr_report:x} wptr_mem=0x{wptr_mem:x}"
                f"{reg_text}"
            )
            last = snap
        if marker_value == POST_MARKER:
            print("AMDDevice macOS no-op dispatch ✓")
            return
        time.sleep(0.05)

    marker_value = read64(marker.cpu_addr)
    marker_name = MARKER_NAMES.get(marker_value, "unknown")
    raise TimeoutError(
        f"no-op dispatch stopped at marker=0x{marker_value:016x}({marker_name})"
    )


if __name__ == "__main__":
    main()
