"""Compute dispatch and GPU bring-up orchestration for Windows.

Ties together all initialization modules to bring up the GPU from cold
state to running compute workloads:

Init sequence:
1. Open device via D3DKMTEscape
2. IP discovery — enumerate IP blocks and base addresses
3. NBIO init — doorbell aperture, framebuffer access
4. GMC init — memory controller, system aperture, GART
5. PSP init — firmware loading (SOS, RLC, MEC, SDMA)
6. IH init — interrupt handler ring
7. Compute ring init — MQD, HQD registers, doorbell
8. Self-test — NOP + RELEASE_MEM fence verification

After bring-up, provides:
- PM4 WRITE_DATA memory test (no shader needed)
- Compute kernel dispatch from ELF code objects (.co files)
- Inline noop shader dispatch (validates the compute pipeline)

Usage:
    ctx = full_gpu_bringup()
    test_write_data(ctx)
    # dispatch_elf_kernel(ctx, "kernel.co", grid=(4,1,1), block=(64,1,1), args=[buf])
    shutdown(ctx)

Reference: Linux amdgpu driver initialization path
"""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from amd_gpu_driver.backends.windows.device import WindowsDevice
from amd_gpu_driver.backends.windows.ip_discovery import (
    IPDiscoveryResult,
    parse_ip_discovery,
    read_discovery_table_via_mmio,
)
from amd_gpu_driver.backends.windows.nbio_init import NBIOConfig, init_nbio
from amd_gpu_driver.backends.windows.gmc_init import GMCConfig, init_gmc
from amd_gpu_driver.backends.windows.psp_init import (
    PSPConfig,
    init_psp,
    load_all_firmware,
)
from amd_gpu_driver.backends.windows.ih_init import IHConfig, init_ih
from amd_gpu_driver.backends.windows.ring_init import (
    ComputeQueueConfig,
    init_compute_queue,
    submit_compute_packets,
    read_fence_value,
    wait_fence,
    test_compute_nop_fence,
)
from amd_gpu_driver.commands.pm4 import (
    PM4PacketBuilder,
    SH_REG_BASE,
)
from amd_gpu_driver.gpu.registers import (
    COMPUTE_PGM_LO,
    COMPUTE_PGM_RSRC1,
    COMPUTE_PGM_RSRC2,
    COMPUTE_PGM_RSRC3,
    COMPUTE_RESOURCE_LIMITS,
    COMPUTE_RESTART_X,
    COMPUTE_START_X,
    COMPUTE_TMPRING_SIZE,
    COMPUTE_USER_DATA_0,
    sh_reg_offset,
)


# ============================================================================
# PM4 opcodes not in the existing builder
# ============================================================================

PACKET3_WRITE_DATA = 0x37

# WRITE_DATA dst_sel values
WRITE_DATA_DST_SEL_MEM_MAPPED = 1
WRITE_DATA_DST_SEL_MEM_ASYNC = 5
WRITE_DATA_WR_CONFIRM = 1 << 20
WRITE_DATA_ENGINE_SEL_ME = 0


# ============================================================================
# GPUContext — holds all initialized subsystem configs
# ============================================================================

@dataclass
class GPUContext:
    """Holds all configuration state for an initialized GPU."""
    dev: WindowsDevice
    ip_result: IPDiscoveryResult
    nbio_config: NBIOConfig
    gmc_config: GMCConfig
    psp_config: PSPConfig
    ih_config: IHConfig
    compute_queue: ComputeQueueConfig

    # Additional DMA allocations used during bring-up
    gart_table_dma_handle: int = 0
    dummy_page_dma_handle: int = 0
    _fence_seq: int = 0

    def next_fence_seq(self) -> int:
        """Return the next fence sequence number."""
        self._fence_seq += 1
        return self._fence_seq


# ============================================================================
# PM4 packet helpers
# ============================================================================

def _build_write_data_packet(
    addr: int,
    values: list[int],
) -> bytes:
    """Build a PM4 WRITE_DATA packet to write DWORDs to a memory address.

    WRITE_DATA (opcode 0x37):
      DW0: header
      DW1: [11:8]=DST_SEL(5=mem_async), [20]=WR_CONFIRM, [31:30]=ENGINE_SEL
      DW2: addr_lo (DWORD-aligned)
      DW3: addr_hi
      DW4+: data values

    Reference: si_pm4.h, PACKET3_WRITE_DATA
    """
    n_body = 3 + len(values)  # control + addr_lo + addr_hi + data
    header = (3 << 30) | (((n_body - 1) & 0x3FFF) << 16) | (PACKET3_WRITE_DATA << 8)

    dw1 = (WRITE_DATA_DST_SEL_MEM_ASYNC << 8) | WRITE_DATA_WR_CONFIRM
    dwords = [header, dw1, addr & 0xFFFFFFFF, (addr >> 32) & 0xFFFFFFFF]
    dwords.extend(v & 0xFFFFFFFF for v in values)

    return struct.pack(f"<{len(dwords)}I", *dwords)


def _build_dispatch_packets(
    code_gpu_addr: int,
    kernarg_gpu_addr: int,
    pgm_rsrc1: int,
    pgm_rsrc2: int,
    pgm_rsrc3: int,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    fence_addr: int,
    fence_value: int,
) -> bytes:
    """Build a complete PM4 command stream for compute dispatch.

    Sequence:
    1. ACQUIRE_MEM — invalidate caches
    2. SET_SH_REG — program address (COMPUTE_PGM_LO/HI)
    3. SET_SH_REG — program resources (RSRC1, RSRC2)
    4. SET_SH_REG — RSRC3
    5. SET_SH_REG — scratch (TMPRING_SIZE = 0)
    6. SET_SH_REG — restart coordinates
    7. SET_SH_REG — kernarg pointer (USER_DATA_0/1)
    8. SET_SH_REG — resource limits
    9. SET_SH_REG — start coordinates + workgroup dimensions
    10. DISPATCH_DIRECT
    11. RELEASE_MEM — write fence value

    Args:
        code_gpu_addr: GPU address of the kernel code entry point.
        kernarg_gpu_addr: GPU address of kernel arguments.
        pgm_rsrc1: COMPUTE_PGM_RSRC1 register value.
        pgm_rsrc2: COMPUTE_PGM_RSRC2 register value.
        pgm_rsrc3: COMPUTE_PGM_RSRC3 register value.
        grid: Dispatch dimensions (workgroups in x, y, z).
        block: Workgroup dimensions (threads in x, y, z).
        fence_addr: GPU address for completion fence write.
        fence_value: Value to write on completion.

    Returns:
        Serialized PM4 packet bytes.
    """
    from amd_gpu_driver.commands.pm4 import (
        CP_COHER_CNTL_SH_KCACHE_ACTION,
        CP_COHER_CNTL_TCL1_ACTION,
        CS_PARTIAL_FLUSH,
        EVENT_INDEX_CS_PARTIAL_FLUSH,
    )

    pm4 = PM4PacketBuilder()

    # 1. Cache invalidation
    pm4.acquire_mem(
        coher_cntl=CP_COHER_CNTL_SH_KCACHE_ACTION | CP_COHER_CNTL_TCL1_ACTION,
    )

    # 2. Program address (shifted right by 8 bits)
    pgm_addr = code_gpu_addr >> 8
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_PGM_LO),
        pgm_addr & 0xFFFFFFFF,
        (pgm_addr >> 32) & 0xFFFFFFFF,
    )

    # 3. Program resources (RSRC1 and RSRC2 are consecutive)
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_PGM_RSRC1),
        pgm_rsrc1,
        pgm_rsrc2,
    )

    # 4. RSRC3
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_RSRC3), pgm_rsrc3)

    # 5. Scratch (none)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_TMPRING_SIZE), 0)

    # 6. Restart coordinates (zero)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESTART_X), 0, 0, 0)

    # 7. Kernarg pointer
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_USER_DATA_0),
        kernarg_gpu_addr & 0xFFFFFFFF,
        (kernarg_gpu_addr >> 32) & 0xFFFFFFFF,
    )

    # 8. Resource limits (0 = no restrictions)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESOURCE_LIMITS), 0)

    # 9. Start coordinates + workgroup dimensions
    pm4.set_sh_reg(
        sh_reg_offset(COMPUTE_START_X),
        0, 0, 0,                           # start x, y, z
        block[0], block[1], block[2],       # num_thread x, y, z
        0, 0,                               # trailing zeros
    )

    # 10. Dispatch
    pm4.dispatch_direct(grid[0], grid[1], grid[2])

    # 11. CS_PARTIAL_FLUSH (ensures shader completion before fence)
    pm4.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH)

    # 12. RELEASE_MEM fence
    pm4.release_mem(
        addr=fence_addr,
        value=fence_value,
        cache_flush=True,
    )

    return pm4.build()


# ============================================================================
# Noop shader — minimal compute kernel for pipeline validation
# ============================================================================

# GFX12 s_endpgm = 0xBF810000 (SOPP format: opcode=1)
# This shader does nothing and returns immediately. Used to verify
# the dispatch pipeline works end-to-end (SET_SH_REG → DISPATCH_DIRECT
# → CP fetches/executes shader → RELEASE_MEM writes fence).
_NOOP_SHADER_CODE = struct.pack("<I", 0xBF810000)  # s_endpgm

# Kernel descriptor for the noop shader (64 bytes)
# All zero except:
#   kernel_code_entry_byte_offset = 64 (code starts after descriptor)
#   compute_pgm_rsrc1: float_mode=0xC0, dx10_clamp=1, ieee_mode=1
#   compute_pgm_rsrc2: enable_workgroup_id_x=1
_NOOP_RSRC1 = (0xC0 << 12) | (1 << 21) | (1 << 23)  # 0x00AC0000
_NOOP_RSRC2 = (1 << 7)  # enable_sgpr_workgroup_id_x


def _build_noop_kernel_image() -> tuple[bytes, int, int, int]:
    """Build a minimal noop kernel: descriptor + s_endpgm.

    Returns:
        (image_bytes, pgm_rsrc1, pgm_rsrc2, pgm_rsrc3)
        image_bytes contains the 64-byte kernel descriptor followed by code.
        The code entry point is at offset 64 from the start of the image.
    """
    from amd_gpu_driver.kernel.descriptor import KernelDescriptor

    kd = KernelDescriptor(
        group_segment_fixed_size=0,
        private_segment_fixed_size=0,
        kernarg_size=0,
        kernel_code_entry_byte_offset=64,
        compute_pgm_rsrc3=0,
        compute_pgm_rsrc1=_NOOP_RSRC1,
        compute_pgm_rsrc2=_NOOP_RSRC2,
    )
    descriptor_bytes = kd.to_bytes()
    assert len(descriptor_bytes) == 64

    # Pad code to 256-byte alignment (GPU requires code address alignment)
    code = _NOOP_SHADER_CODE
    padding = (256 - (64 + len(code)) % 256) % 256
    image = descriptor_bytes + code + b"\x00" * padding

    return image, _NOOP_RSRC1, _NOOP_RSRC2, 0


# ============================================================================
# GPU bring-up sequence
# ============================================================================

def full_gpu_bringup(
    device_index: int = 0,
    fw_dir: str | Path = ".",
) -> GPUContext:
    """Run the full GPU initialization sequence.

    Opens the device, discovers IP blocks, initializes all IP subsystems,
    creates a compute queue, and runs a NOP fence self-test.

    Args:
        device_index: GPU index (0 for first AMD GPU).
        fw_dir: Directory containing firmware .bin files (for PSP).

    Returns:
        GPUContext with all initialized subsystems.

    Raises:
        RuntimeError: If any initialization step fails.
    """
    print("=" * 60)
    print("AMD GPU Bring-up (Windows Userspace Driver)")
    print("=" * 60)

    # --- 1. Open device ---
    print("\n[1/8] Opening device...")
    dev = WindowsDevice()
    dev.open(device_index)
    print(f"  Device: {dev.name}")
    print(f"  VRAM: {dev.vram_size // (1024**3)} GB")

    # --- 2. IP discovery ---
    print("\n[2/8] Running IP discovery...")
    raw_table = read_discovery_table_via_mmio(
        dev.read_reg_indirect, dev.vram_size)
    ip_result = parse_ip_discovery(raw_table)
    print(f"  Found {len(ip_result.ip_blocks)} IP blocks")
    for block in ip_result.ip_blocks:
        print(f"    {block.hw_id.name}: "
              f"v{block.major}.{block.minor}.{block.revision}")

    # --- 3. NBIO init ---
    print("\n[3/8] Initializing NBIO...")
    nbio_config = init_nbio(dev, ip_result)

    # --- 4. GMC init ---
    print("\n[4/8] Initializing GMC...")
    # GMC needs pre-allocated DMA buffers for GART page table and dummy page
    gart_cpu, gart_bus, gart_handle = dev.driver.alloc_dma(
        1024 * 1024)  # 1MB GART table
    dummy_cpu, dummy_bus, dummy_handle = dev.driver.alloc_dma(4096)
    # Zero the GART table and dummy page
    ctypes.memset(gart_cpu, 0, 1024 * 1024)
    ctypes.memset(dummy_cpu, 0, 4096)

    gmc_config = init_gmc(
        dev, ip_result, nbio_config,
        vram_size_bytes=dev.vram_size,
        gart_table_bus_addr=gart_bus,
        dummy_page_bus_addr=dummy_bus,
    )

    # --- 5. PSP init + firmware loading ---
    print("\n[5/8] Initializing PSP (firmware)...")
    psp_config = init_psp(dev, ip_result, fw_dir=fw_dir)
    # Load firmware (SOS should be alive from VBIOS POST)
    try:
        load_all_firmware(dev, psp_config)
    except FileNotFoundError as e:
        print(f"  WARNING: Firmware loading skipped — {e}")
        print("  Continuing with VBIOS-initialized firmware state")
    except RuntimeError as e:
        print(f"  WARNING: Firmware loading failed — {e}")
        print("  Continuing with VBIOS-initialized firmware state")

    # --- 6. IH init ---
    print("\n[6/8] Initializing IH (interrupts)...")
    ih_config = init_ih(dev, ip_result, nbio_config)

    # --- 7. Compute ring ---
    print("\n[7/8] Creating compute queue...")
    compute_queue = init_compute_queue(dev, ip_result, nbio_config)

    # --- 8. NOP fence test ---
    print("\n[8/8] Running NOP + fence self-test...")
    if test_compute_nop_fence(compute_queue):
        print("  PASS: NOP + RELEASE_MEM fence completed")
    else:
        print("  FAIL: Fence timeout — GPU may not be processing commands")

    print("\n" + "=" * 60)
    print("GPU bring-up complete!")
    print("=" * 60)

    return GPUContext(
        dev=dev,
        ip_result=ip_result,
        nbio_config=nbio_config,
        gmc_config=gmc_config,
        psp_config=psp_config,
        ih_config=ih_config,
        compute_queue=compute_queue,
        gart_table_dma_handle=gart_handle,
        dummy_page_dma_handle=dummy_handle,
    )


# ============================================================================
# Memory write test (PM4 WRITE_DATA — no shader needed)
# ============================================================================

def test_write_data(ctx: GPUContext, num_dwords: int = 16) -> bool:
    """Verify PM4 WRITE_DATA by writing and reading back values.

    Uses WRITE_DATA to write known patterns to a DMA buffer,
    then reads back via CPU to verify. This tests the compute queue
    packet processing without requiring a compiled shader.

    Args:
        ctx: Initialized GPU context.
        num_dwords: Number of DWORDs to write (default 16 = 64 bytes).

    Returns:
        True if all values match.
    """
    print("\n--- WRITE_DATA memory test ---")
    cq = ctx.compute_queue

    # Allocate a test buffer
    buf_size = max(num_dwords * 4, 4096)
    buf_cpu, buf_bus, buf_handle = ctx.dev.driver.alloc_dma(buf_size)
    ctypes.memset(buf_cpu, 0, buf_size)

    # Write known pattern via PM4 WRITE_DATA
    test_values = [(0xCAFE0000 + i) & 0xFFFFFFFF for i in range(num_dwords)]

    write_pkt = _build_write_data_packet(buf_bus, test_values)

    # Also add a RELEASE_MEM fence to know when the writes are done
    fence_seq = ctx.next_fence_seq()
    ctypes.c_uint64.from_address(cq.fence_cpu_addr).value = 0

    pm4 = PM4PacketBuilder()
    pm4.release_mem(
        addr=cq.fence_bus_addr,
        value=fence_seq,
        cache_flush=True,
    )
    fence_pkt = pm4.build()

    # Submit: WRITE_DATA + RELEASE_MEM
    submit_compute_packets(cq, write_pkt + fence_pkt)

    # Wait for fence
    if not wait_fence(cq, fence_seq, timeout_ms=5000):
        print("  FAIL: Fence timeout after WRITE_DATA")
        ctx.dev.driver.free_dma(buf_handle)
        return False

    # Read back and verify
    result = (ctypes.c_uint32 * num_dwords).from_address(buf_cpu)
    mismatches = []
    for i in range(num_dwords):
        if result[i] != test_values[i]:
            mismatches.append(
                f"  [{i}] expected 0x{test_values[i]:08X}, "
                f"got 0x{result[i]:08X}")

    ctx.dev.driver.free_dma(buf_handle)

    if mismatches:
        print(f"  FAIL: {len(mismatches)}/{num_dwords} mismatches:")
        for m in mismatches[:5]:
            print(m)
        return False

    print(f"  PASS: {num_dwords} DWORDs written and verified")
    return True


# ============================================================================
# Noop shader dispatch test
# ============================================================================

def test_noop_dispatch(ctx: GPUContext) -> bool:
    """Dispatch a noop shader (s_endpgm) to validate the compute pipeline.

    This tests the full dispatch path:
    1. Upload noop kernel (descriptor + s_endpgm) to DMA memory
    2. Build SET_SH_REG + DISPATCH_DIRECT + RELEASE_MEM packets
    3. Submit to compute queue
    4. Wait for fence

    The shader does nothing, but the GPU must successfully fetch it,
    execute it, and signal completion. This validates that the compute
    unit is alive and the register programming is correct.

    Args:
        ctx: Initialized GPU context.

    Returns:
        True if the dispatch completed (fence received).
    """
    print("\n--- Noop shader dispatch test ---")
    cq = ctx.compute_queue

    # Build the noop kernel image
    image, rsrc1, rsrc2, rsrc3 = _build_noop_kernel_image()

    # Allocate DMA memory for the kernel code (256-byte aligned)
    code_size = max(len(image), 4096)
    code_cpu, code_bus, code_handle = ctx.dev.driver.alloc_dma(code_size)
    ctypes.memset(code_cpu, 0, code_size)
    ctypes.memmove(code_cpu, image, len(image))

    # Code entry point is at offset 64 (after descriptor)
    code_entry_addr = code_bus + 64

    # Build dispatch packets (1x1x1 grid, 1x1x1 block — single thread)
    fence_seq = ctx.next_fence_seq()
    ctypes.c_uint64.from_address(cq.fence_cpu_addr).value = 0

    packets = _build_dispatch_packets(
        code_gpu_addr=code_entry_addr,
        kernarg_gpu_addr=0,  # No kernargs for noop
        pgm_rsrc1=rsrc1,
        pgm_rsrc2=rsrc2,
        pgm_rsrc3=rsrc3,
        grid=(1, 1, 1),
        block=(1, 1, 1),
        fence_addr=cq.fence_bus_addr,
        fence_value=fence_seq,
    )

    submit_compute_packets(cq, packets)

    if not wait_fence(cq, fence_seq, timeout_ms=5000):
        print("  FAIL: Fence timeout after noop dispatch")
        ctx.dev.driver.free_dma(code_handle)
        return False

    print("  PASS: Noop shader dispatched and completed")
    ctx.dev.driver.free_dma(code_handle)
    return True


# ============================================================================
# ELF kernel dispatch
# ============================================================================

def dispatch_elf_kernel(
    ctx: GPUContext,
    co_path: str | Path,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    args: list[int],
    *,
    kernel_name: str | None = None,
    timeout_ms: int = 5000,
) -> bool:
    """Dispatch a compute kernel from a compiled ELF code object (.co file).

    Loads the ELF, uploads code to DMA memory, builds kernarg buffer,
    and dispatches via PM4. Waits for completion fence.

    Args:
        ctx: Initialized GPU context.
        co_path: Path to .co / .hsaco file compiled for the target GPU.
        grid: Dispatch dimensions (workgroups in x, y, z).
        block: Workgroup dimensions (threads in x, y, z).
        args: Kernel arguments as 64-bit integers (GPU addresses or scalars).
        kernel_name: Specific kernel name (None = first kernel found).
        timeout_ms: Fence timeout in milliseconds.

    Returns:
        True if the dispatch completed within the timeout.
    """
    from amd_gpu_driver.kernel.elf_parser import parse_elf_file
    from amd_gpu_driver.kernel.descriptor import KernelDescriptor

    co_path = Path(co_path)
    print(f"\n--- ELF kernel dispatch: {co_path.name} ---")

    # Parse the ELF
    co = parse_elf_file(str(co_path))

    # Find kernel symbol
    kernels = co.kernel_symbols()
    if not kernels:
        raise RuntimeError(f"No kernel symbols in {co_path}")

    if kernel_name is not None:
        matches = [k for k in kernels if k.name == kernel_name]
        if not matches:
            available = [k.name for k in kernels]
            raise RuntimeError(
                f"Kernel '{kernel_name}' not found. Available: {available}")
        kernel_sym = matches[0]
    else:
        kernel_sym = kernels[0]

    print(f"  Kernel: {kernel_sym.name}")

    # Find the kernel descriptor
    kd_name = kernel_sym.name + ".kd"
    kd_syms = [s for s in co.symbols if s.name == kd_name]

    if kd_syms and co.rodata_section is not None:
        kd_sym = kd_syms[0]
        rodata_data = co.get_section_data(co.rodata_section)
        kd_offset = kd_sym.st_value - co.rodata_section.sh_addr
        kd = KernelDescriptor.from_bytes(rodata_data, kd_offset)
        descriptor_va = kd_sym.st_value
    elif co.text_section is not None:
        text_data = co.code
        desc_offset = kernel_sym.st_value - co.text_section.sh_addr
        kd = KernelDescriptor.from_bytes(text_data, desc_offset)
        descriptor_va = kernel_sym.st_value
    else:
        raise RuntimeError("Cannot find kernel descriptor in ELF")

    print(f"  RSRC1=0x{kd.compute_pgm_rsrc1:08X} "
          f"RSRC2=0x{kd.compute_pgm_rsrc2:08X}")
    print(f"  Kernarg size={kd.kernarg_size} bytes, "
          f"LDS={kd.group_segment_fixed_size} bytes")

    # Get code data
    code_data = co.code
    if not code_data:
        raise RuntimeError("No .text section in ELF")

    text_va = (co.text_section.sh_addr
               if co.text_section else 0)
    code_entry_va = descriptor_va + kd.kernel_code_entry_byte_offset

    # Upload code to DMA memory (256-byte aligned)
    code_alloc_size = max(len(code_data), 4096)
    code_alloc_size = (code_alloc_size + 255) & ~255  # 256-byte align
    code_cpu, code_bus, code_handle = ctx.dev.driver.alloc_dma(code_alloc_size)
    ctypes.memset(code_cpu, 0, code_alloc_size)
    ctypes.memmove(code_cpu, code_data, len(code_data))

    # Compute the code entry GPU address
    # .text was loaded at code_bus, and code_entry_va is relative to text_va
    code_entry_addr = code_bus + (code_entry_va - text_va)
    print(f"  Code uploaded at bus 0x{code_bus:012X}")
    print(f"  Code entry at bus 0x{code_entry_addr:012X}")

    # Build kernarg buffer
    kernarg_data = bytearray(max(kd.kernarg_size, 64))
    offset = 0
    for arg in args:
        struct.pack_into("<Q", kernarg_data, offset, arg & 0xFFFFFFFFFFFFFFFF)
        offset += 8

    # Write implicit dispatch packet (if kernarg_size accommodates it)
    implicit_offset = (offset + 15) & ~15
    if implicit_offset + 18 <= len(kernarg_data):
        struct.pack_into(
            "<III HHH", kernarg_data, implicit_offset,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
        )

    # Upload kernarg to DMA memory
    ka_size = max(len(kernarg_data), 4096)
    ka_cpu, ka_bus, ka_handle = ctx.dev.driver.alloc_dma(ka_size)
    ctypes.memset(ka_cpu, 0, ka_size)
    ctypes.memmove(ka_cpu, bytes(kernarg_data), len(kernarg_data))

    # Build and submit dispatch
    cq = ctx.compute_queue
    fence_seq = ctx.next_fence_seq()
    ctypes.c_uint64.from_address(cq.fence_cpu_addr).value = 0

    packets = _build_dispatch_packets(
        code_gpu_addr=code_entry_addr,
        kernarg_gpu_addr=ka_bus,
        pgm_rsrc1=kd.compute_pgm_rsrc1,
        pgm_rsrc2=kd.compute_pgm_rsrc2,
        pgm_rsrc3=kd.compute_pgm_rsrc3,
        grid=grid,
        block=block,
        fence_addr=cq.fence_bus_addr,
        fence_value=fence_seq,
    )

    print(f"  Dispatching grid={grid} block={block}...")
    submit_compute_packets(cq, packets)

    # Wait for completion
    if not wait_fence(cq, fence_seq, timeout_ms=timeout_ms):
        print(f"  FAIL: Fence timeout after {timeout_ms}ms")
        ctx.dev.driver.free_dma(code_handle)
        ctx.dev.driver.free_dma(ka_handle)
        return False

    print("  PASS: Dispatch completed")
    ctx.dev.driver.free_dma(code_handle)
    ctx.dev.driver.free_dma(ka_handle)
    return True


# ============================================================================
# Fill buffer test (dispatch with result verification)
# ============================================================================

def test_fill_dispatch(
    ctx: GPUContext,
    co_path: str | Path,
    *,
    num_elements: int = 256,
    fill_value: int = 0xDEADBEEF,
    kernel_name: str = "fill_kernel",
) -> bool:
    """Dispatch a fill kernel and verify the output buffer.

    Requires a compiled fill_kernel for the target GPU:
      __global__ void fill_kernel(uint32_t* out, uint32_t val) {
          out[threadIdx.x + blockIdx.x * blockDim.x] = val;
      }

    Args:
        ctx: Initialized GPU context.
        co_path: Path to compiled fill kernel .co file.
        num_elements: Number of uint32 elements to fill.
        fill_value: Value to fill with.
        kernel_name: Name of the kernel in the .co file.

    Returns:
        True if all elements match the fill value.
    """
    print(f"\n--- Fill kernel test ({num_elements} elements) ---")

    # Allocate output buffer
    out_size = num_elements * 4
    out_cpu, out_bus, out_handle = ctx.dev.driver.alloc_dma(
        max(out_size, 4096))
    ctypes.memset(out_cpu, 0, max(out_size, 4096))

    # Dispatch
    block_size = 64
    grid_x = (num_elements + block_size - 1) // block_size
    success = dispatch_elf_kernel(
        ctx,
        co_path,
        grid=(grid_x, 1, 1),
        block=(block_size, 1, 1),
        args=[out_bus, fill_value],
        kernel_name=kernel_name,
    )

    if not success:
        ctx.dev.driver.free_dma(out_handle)
        return False

    # Verify output
    result = (ctypes.c_uint32 * num_elements).from_address(out_cpu)
    mismatches = 0
    first_mismatch = -1
    for i in range(num_elements):
        if result[i] != fill_value:
            mismatches += 1
            if first_mismatch < 0:
                first_mismatch = i

    ctx.dev.driver.free_dma(out_handle)

    if mismatches > 0:
        print(f"  FAIL: {mismatches}/{num_elements} mismatches "
              f"(first at index {first_mismatch}: "
              f"got 0x{result[first_mismatch]:08X})")
        return False

    print(f"  PASS: All {num_elements} elements = 0x{fill_value:08X}")
    return True


# ============================================================================
# DMA buffer allocation helper for external use
# ============================================================================

def alloc_gpu_buffer(ctx: GPUContext, size: int) -> tuple[int, int, int]:
    """Allocate a DMA buffer accessible by both CPU and GPU.

    Returns:
        (cpu_addr, bus_addr, handle) — cpu_addr for CPU access,
        bus_addr for GPU access (as kernarg or output pointer).
    """
    cpu_addr, bus_addr, handle = ctx.dev.driver.alloc_dma(
        max(size, 4096))
    ctypes.memset(cpu_addr, 0, max(size, 4096))
    return cpu_addr, bus_addr, handle


def free_gpu_buffer(ctx: GPUContext, handle: int) -> None:
    """Free a previously allocated DMA buffer."""
    ctx.dev.driver.free_dma(handle)


# ============================================================================
# Shutdown
# ============================================================================

def shutdown(ctx: GPUContext) -> None:
    """Clean shutdown: close device and release resources.

    Note: Does not free individual DMA buffers from tests.
    The kernel driver will reclaim all resources when the device is closed.
    """
    print("\nShutting down GPU...")
    ctx.dev.close()
    print("  Device closed")


# ============================================================================
# Demo entry point
# ============================================================================

def run_demo(
    device_index: int = 0,
    fw_dir: str = ".",
    kernel_co: str | None = None,
) -> None:
    """Run the full GPU bring-up and self-test sequence.

    Args:
        device_index: GPU index.
        fw_dir: Firmware directory.
        kernel_co: Optional path to a compiled fill kernel .co file.
    """
    ctx = full_gpu_bringup(device_index=device_index, fw_dir=fw_dir)

    print("\n" + "=" * 60)
    print("Running self-tests")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    # Test 1: WRITE_DATA
    results.append(("WRITE_DATA memory test", test_write_data(ctx)))

    # Test 2: Noop dispatch
    results.append(("Noop shader dispatch", test_noop_dispatch(ctx)))

    # Test 3: Fill kernel (if .co file provided)
    if kernel_co is not None:
        results.append((
            "Fill kernel dispatch",
            test_fill_dispatch(ctx, kernel_co),
        ))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
    else:
        print(f"\n{sum(1 for _, p in results if not p)} test(s) failed")

    shutdown(ctx)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AMD GPU bring-up and compute dispatch demo")
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device index (default: 0)")
    parser.add_argument(
        "--fw-dir", type=str, default=".",
        help="Directory containing firmware .bin files")
    parser.add_argument(
        "--kernel", type=str, default=None,
        help="Path to compiled fill kernel .co file (optional)")
    args = parser.parse_args()

    run_demo(
        device_index=args.device,
        fw_dir=args.fw_dir,
        kernel_co=args.kernel,
    )
