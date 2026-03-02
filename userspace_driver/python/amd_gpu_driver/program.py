"""Program: ELF loading + compute dispatch."""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Any, Sequence

from amd_gpu_driver.backends.base import DeviceBackend, MemoryHandle, MemoryLocation, QueueHandle
from amd_gpu_driver.commands.pm4 import (
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
    ) -> None:
        self._backend = backend
        self._code_object = code_object
        self._descriptor = descriptor
        self._code_mem = code_mem
        self._family = family
        self._kernel_name = kernel_name

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
        args: Sequence[int | Buffer],
        *,
        timeline: TimelineSemaphore | None = None,
    ) -> None:
        """Build and submit a compute dispatch.

        Args:
            queue: Compute queue to submit to.
            grid: Global dispatch dimensions (workgroups in x, y, z).
            block: Workgroup dimensions (threads in x, y, z).
            args: Kernel arguments - Buffer GPU addresses or scalar uint64 values.
            timeline: Optional timeline semaphore for completion signaling.
        """
        # Build kernarg buffer
        kernarg_data = self._build_kernargs(args)
        kernarg_mem = self._backend.alloc_memory(
            max(len(kernarg_data), 4096),
            MemoryLocation.GTT,
            uncached=True,
        )
        # Write kernarg data
        ctypes.memmove(kernarg_mem.cpu_addr, kernarg_data, len(kernarg_data))

        # Compute code entry address
        code_entry_addr = self._code_mem.gpu_addr + self._descriptor.kernel_code_entry_byte_offset

        # Build PM4 command stream
        pm4 = PM4PacketBuilder()

        # 1. Cache invalidation
        pm4.acquire_mem()

        # 2. Program address (shifted right by 8 bits)
        pgm_addr = code_entry_addr >> 8
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_PGM_LO),
            pgm_addr & 0xFFFFFFFF,
            (pgm_addr >> 32) & 0xFFFFFFFF,
        )

        # 3. Program resource registers
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_PGM_RSRC1),
            self._descriptor.compute_pgm_rsrc1,
        )
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_PGM_RSRC2),
            self._descriptor.compute_pgm_rsrc2,
        )

        # 4. Kernarg pointer (USER_DATA_0 and USER_DATA_1)
        kernarg_addr = kernarg_mem.gpu_addr
        pm4.set_sh_reg(
            sh_reg_offset(COMPUTE_USER_DATA_0),
            kernarg_addr & 0xFFFFFFFF,
            (kernarg_addr >> 32) & 0xFFFFFFFF,
        )

        # 5. Thread dimensions
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_START_X), 0)
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_START_Y), 0)
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_START_Z), 0)

        pm4.set_sh_reg(sh_reg_offset(COMPUTE_NUM_THREAD_X), block[0])
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_NUM_THREAD_Y), block[1])
        pm4.set_sh_reg(sh_reg_offset(COMPUTE_NUM_THREAD_Z), block[2])

        # 6. Scratch (tmpring) size
        if self._descriptor.private_segment_fixed_size > 0:
            # Compute scratch size per wave
            scratch_per_item = self._descriptor.private_segment_fixed_size
            waves_per_sh = 1  # Simplified
            scratch_size = scratch_per_item * self._family.wave_size * waves_per_sh
            pm4.set_sh_reg(sh_reg_offset(COMPUTE_TMPRING_SIZE), scratch_size)

        # 7. Dispatch
        pm4.dispatch_direct(grid[0], grid[1], grid[2], initiator=1)

        # 8. Signal completion
        if timeline is not None:
            signal_value = timeline.next_value()
            signal_bytes = timeline.signal_packets(signal_value)
            # Combine PM4 packets
            packets = pm4.build() + signal_bytes
        else:
            packets = pm4.build()

        # Submit to queue
        self._backend.submit_packets(queue, packets)

    def _build_kernargs(self, args: Sequence[int | Buffer]) -> bytes:
        """Build the kernarg buffer from a list of arguments."""
        parts: list[bytes] = []
        for arg in args:
            if isinstance(arg, Buffer):
                # Pass GPU address as uint64
                parts.append(struct.pack("<Q", arg.gpu_addr))
            elif isinstance(arg, int):
                parts.append(struct.pack("<Q", arg))
            else:
                raise KernelLoadError(f"Unsupported kernel argument type: {type(arg)}")

        result = b"".join(parts)
        # Pad to kernarg_size if needed
        if len(result) < self._descriptor.kernarg_size:
            result += b"\x00" * (self._descriptor.kernarg_size - len(result))
        return result

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

    # Parse kernel descriptor from the symbol's section
    # The descriptor is at the symbol's value offset
    descriptor_offset = kernel_sym.st_value
    if co.text_section is not None:
        # Descriptor is embedded in .text at the symbol offset
        text_data = co.code
        kd = KernelDescriptor.from_bytes(text_data, descriptor_offset)
    elif co.rodata_section is not None:
        rodata_data = co.get_section_data(co.rodata_section)
        kd = KernelDescriptor.from_bytes(rodata_data, descriptor_offset)
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
    )
