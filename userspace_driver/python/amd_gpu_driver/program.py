"""Program: ELF loading + compute dispatch."""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Any, Sequence

from amd_gpu_driver.backends.base import DeviceBackend, MemoryHandle, MemoryLocation, QueueHandle
from amd_gpu_driver.commands.pm4 import (
    CS_PARTIAL_FLUSH,
    EVENT_INDEX_CS_PARTIAL_FLUSH,
    PM4PacketBuilder,
    SH_REG_BASE,
)
from amd_gpu_driver.errors import KernelLoadError
from amd_gpu_driver.gpu.family import GPUFamilyConfig
from amd_gpu_driver.gpu.registers import (
    COMPUTE_DISPATCH_INITIATOR,
    COMPUTE_NUM_THREAD_X,
    COMPUTE_NUM_THREAD_Y,
    COMPUTE_NUM_THREAD_Z,
    COMPUTE_PGM_LO,
    COMPUTE_PGM_RSRC1,
    COMPUTE_PGM_RSRC2,
    COMPUTE_PGM_RSRC3,
    COMPUTE_RESOURCE_LIMITS,
    COMPUTE_RESTART_X,
    COMPUTE_START_X,
    COMPUTE_START_Y,
    COMPUTE_START_Z,
    COMPUTE_TMPRING_SIZE,
    COMPUTE_USER_DATA_0,
    sh_reg_offset,
)
from amd_gpu_driver.kernel.descriptor import KernelDescriptor
from amd_gpu_driver.kernel.elf_parser import AMDGPUCodeObject, parse_elf, parse_elf_file
from amd_gpu_driver.memory.buffer import Buffer
from amd_gpu_driver.sync.timeline import TimelineSemaphore


class Program:
    """A loaded GPU program (kernel) ready for dispatch."""

    def __init__(
        self,
        backend: DeviceBackend,
        code_object: AMDGPUCodeObject,
        descriptor: KernelDescriptor,
        code_mem: MemoryHandle,
        family: GPUFamilyConfig,
        kernel_name: str = "",
        descriptor_va: int = 0,
    ) -> None:
        self._backend = backend
        self._code_object = code_object
        self._descriptor = descriptor
        self._code_mem = code_mem
        self._family = family
        self._kernel_name = kernel_name
        self._descriptor_va = descriptor_va

    @property
    def name(self) -> str:
        return self._kernel_name

    @property
    def descriptor(self) -> KernelDescriptor:
        return self._descriptor

    @property
    def kernarg_size(self) -> int:
        return self._descriptor.kernarg_size

    @property
    def group_segment_size(self) -> int:
        return self._descriptor.group_segment_fixed_size

    @property
    def private_segment_size(self) -> int:
        return self._descriptor.private_segment_fixed_size

    def dispatch(
        self,
        queue: QueueHandle,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[int | Buffer | tuple[int, int]],
        *,
        timeline: TimelineSemaphore | None = None,
    ) -> None:
        """Build and submit a compute dispatch.

        Args:
            queue: Compute queue to submit to.
            grid: Global dispatch dimensions (workgroups in x, y, z).
            block: Workgroup dimensions (threads in x, y, z).
            args: Kernel arguments — Buffer for GPU addresses, int for 64-bit
                scalars, or (value, byte_size) tuples for sized scalars
                (e.g. (42, 4) for uint32_t).
            timeline: Optional timeline semaphore for completion signaling.
        """
        # Build kernarg buffer (explicit args + implicit dispatch packet)
        kernarg_data = self._build_kernargs(args, grid, block)
        kernarg_mem = self._backend.alloc_memory(
            max(len(kernarg_data), 4096),
            MemoryLocation.GTT,
            uncached=True,
        )
        # Write kernarg data
        ctypes.memmove(kernarg_mem.cpu_addr, kernarg_data, len(kernarg_data))

        # Compute code entry address.
        # kernel_code_entry_byte_offset is relative to the descriptor's
        # virtual address in the ELF.  We uploaded .text starting at
        # code_mem.gpu_addr, so we map the ELF VA to the GPU address.
        text_va = self._code_object.text_section.sh_addr if self._code_object.text_section else 0
        code_entry_va = self._descriptor_va + self._descriptor.kernel_code_entry_byte_offset
        code_entry_addr = self._code_mem.gpu_addr + (code_entry_va - text_va)

        # Build PM4 command stream
        pm4 = PM4PacketBuilder()

        # 1. Pre-dispatch cache invalidation (K-cache + L1 only, per tinygrad)
        from amd_gpu_driver.commands.pm4 import (
            CP_COHER_CNTL_SH_KCACHE_ACTION,
            CP_COHER_CNTL_TCL1_ACTION,
        )
        pm4.acquire_mem(
            coher_cntl=CP_COHER_CNTL_SH_KCACHE_ACTION | CP_COHER_CNTL_TCL1_ACTION,
        )

        # 2. Program address (shifted right by 8 bits, LO and HI consecutive)
        pgm_addr = code_entry_addr >> 8
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_PGM_LO),
            pgm_addr & 0xFFFFFFFF,
            (pgm_addr >> 32) & 0xFFFFFFFF,
        )

        # 3. Program resource registers (RSRC1 and RSRC2 are consecutive)
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_PGM_RSRC1),
            self._descriptor.compute_pgm_rsrc1,
            self._descriptor.compute_pgm_rsrc2,
        )

        # 4. RSRC3 (required for gfx942/CDNA3)
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_PGM_RSRC3),
            self._descriptor.compute_pgm_rsrc3,
        )

        # 5. Tmpring size (scratch)
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_TMPRING_SIZE), 0)

        # 6. Restart coordinates (zero)
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESTART_X), 0, 0, 0)

        # 7. Kernarg pointer (USER_DATA_0 and USER_DATA_1)
        kernarg_addr = kernarg_mem.gpu_addr
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_USER_DATA_0),
            kernarg_addr & 0xFFFFFFFF,
            (kernarg_addr >> 32) & 0xFFFFFFFF,
        )

        # 8. Resource limits (0 = no restrictions)
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESOURCE_LIMITS), 0)

        # 9. Start coordinates + workgroup dimensions (consecutive registers)
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_START_X),
            0, 0, 0,                           # start x, y, z
            block[0], block[1], block[2],       # num_thread x, y, z
            0, 0,                               # trailing zeros (thread holes)
        )

        # 10. Dispatch
        pm4.dispatch_direct(grid[0], grid[1], grid[2])

        # 11. CS_PARTIAL_FLUSH after dispatch (ensures shader completion)
        pm4.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH)

        # 12. Signal completion
        if timeline is not None:
            signal_value = timeline.next_value()
            signal_bytes = timeline.signal_packets(signal_value)
            packets = pm4.build() + signal_bytes
        else:
            packets = pm4.build()

        # Submit to queue
        self._backend.submit_packets(queue, packets)

    def _build_kernargs(
        self,
        args: Sequence[int | Buffer | tuple[int, int]],
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
    ) -> bytes:
        """Build the kernarg buffer from arguments + implicit dispatch packet.

        HIP kernels expect implicit arguments after the explicit ones.
        The implicit dispatch packet layout (at 16-byte aligned offset):
            +0x00: block_count_x (u32)   = grid[0]
            +0x04: block_count_y (u32)   = grid[1]
            +0x08: block_count_z (u32)   = grid[2]
            +0x0C: group_size_x (u16)    = block[0]
            +0x0E: group_size_y (u16)    = block[1]
            +0x10: group_size_z (u16)    = block[2]
            +0x12: remainder is reserved/zero

        Args can be:
            - Buffer: packed as 8-byte GPU address
            - int: packed as 8-byte uint64 (for pointers / large values)
            - (value, byte_size): packed at the specified size (2, 4, or 8 bytes)
        """
        _SIZE_FMTS = {2: "<H", 4: "<I", 8: "<Q"}
        parts: list[bytes] = []
        for arg in args:
            if isinstance(arg, Buffer):
                parts.append(struct.pack("<Q", arg.gpu_addr))
            elif isinstance(arg, tuple):
                val, size = arg
                fmt = _SIZE_FMTS.get(size)
                if fmt is None:
                    raise KernelLoadError(f"Unsupported kernarg size: {size} (must be 2, 4, or 8)")
                parts.append(struct.pack(fmt, val))
            elif isinstance(arg, int):
                parts.append(struct.pack("<Q", arg))
            else:
                raise KernelLoadError(f"Unsupported kernel argument type: {type(arg)}")

        explicit = b"".join(parts)

        # Align explicit args to 16 bytes for implicit packet
        implicit_offset = (len(explicit) + 15) & ~15

        # Build implicit dispatch packet
        implicit_packet = struct.pack(
            "<III HHH",
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
        )

        # Create full kernarg buffer
        result = bytearray(max(self._descriptor.kernarg_size, implicit_offset + len(implicit_packet)))
        result[:len(explicit)] = explicit
        result[implicit_offset:implicit_offset + len(implicit_packet)] = implicit_packet

        return bytes(result)

    def free(self) -> None:
        """Free the code memory."""
        self._backend.free_memory(self._code_mem)


def load_program(
    backend: DeviceBackend,
    path: str | Path,
    family: GPUFamilyConfig,
    kernel_name: str | None = None,
) -> Program:
    """Load a GPU program from an ELF code object file.

    Args:
        backend: Device backend to use.
        path: Path to .co / .hsaco file.
        family: GPU family config.
        kernel_name: Specific kernel to load (None = first kernel found).

    Returns:
        A Program ready for dispatch.
    """
    co = parse_elf_file(str(path))

    # Find kernel symbol
    kernels = co.kernel_symbols()
    if not kernels:
        raise KernelLoadError(f"No kernel symbols found in {path}")

    if kernel_name is not None:
        matches = [k for k in kernels if k.name == kernel_name]
        if not matches:
            available = [k.name for k in kernels]
            raise KernelLoadError(
                f"Kernel '{kernel_name}' not found. Available: {available}"
            )
        kernel_sym = matches[0]
    else:
        kernel_sym = kernels[0]

    # Find the kernel descriptor.
    # Modern AMDGPU ELFs put the descriptor as an OBJECT symbol named
    # "<kernel>.kd" in .rodata.  The FUNC symbol points at the code entry.
    kd_name = kernel_sym.name + ".kd"
    kd_syms = [s for s in co.symbols if s.name == kd_name]

    descriptor_va = 0
    if kd_syms and co.rodata_section is not None:
        kd_sym = kd_syms[0]
        rodata_data = co.get_section_data(co.rodata_section)
        # st_value is the virtual address; convert to offset within section
        kd_offset = kd_sym.st_value - co.rodata_section.sh_addr
        kd = KernelDescriptor.from_bytes(rodata_data, kd_offset)
        descriptor_va = kd_sym.st_value
    elif co.text_section is not None:
        # Older format: descriptor embedded in .text at the symbol offset
        text_data = co.code
        descriptor_offset = kernel_sym.st_value - co.text_section.sh_addr
        kd = KernelDescriptor.from_bytes(text_data, descriptor_offset)
        descriptor_va = kernel_sym.st_value
    elif co.rodata_section is not None:
        rodata_data = co.get_section_data(co.rodata_section)
        kd = KernelDescriptor.from_bytes(rodata_data, 0)
        descriptor_va = co.rodata_section.sh_addr
    else:
        raise KernelLoadError("Cannot find kernel descriptor section")

    # Allocate executable VRAM for the code
    code_data = co.code
    if not code_data:
        raise KernelLoadError("No .text section in code object")

    code_mem = backend.alloc_memory(
        len(code_data),
        MemoryLocation.VRAM,
        executable=True,
        public=True,
    )

    # Upload code to GPU memory
    if code_mem.cpu_addr:
        ctypes.memmove(code_mem.cpu_addr, code_data, len(code_data))

    return Program(
        backend=backend,
        code_object=co,
        descriptor=kd,
        code_mem=code_mem,
        family=family,
        kernel_name=kernel_sym.name,
        descriptor_va=descriptor_va,
    )
