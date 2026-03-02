"""Parse the AMDHSA kernel descriptor (llvm_amdhsa_kernel_descriptor_t).

The kernel descriptor is a 64-byte structure embedded in the .rodata or .text
section of an AMDGPU ELF code object. It contains the register setup and
resource requirements for a compute kernel.

Reference: LLVM AMDGPUUsage documentation, kernel descriptor structure.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

KERNEL_DESCRIPTOR_SIZE = 64


@dataclass
class KernelDescriptor:
    """The 64-byte amdhsa kernel descriptor."""

    group_segment_fixed_size: int = 0  # LDS size in bytes
    private_segment_fixed_size: int = 0  # Scratch size per work-item
    kernarg_size: int = 0  # Size of kernel arguments in bytes
    reserved0: int = 0
    kernel_code_entry_byte_offset: int = 0  # Offset from descriptor to code entry
    reserved1: int = 0
    reserved2: int = 0
    compute_pgm_rsrc3: int = 0
    compute_pgm_rsrc1: int = 0
    compute_pgm_rsrc2: int = 0
    kernel_code_properties: int = 0  # Bitfield
    kernarg_preload: int = 0
    reserved3: int = 0

    STRUCT_FMT = "<IIIIQIIIIIHHI"

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> KernelDescriptor:
        """Parse a kernel descriptor from raw bytes."""
        size = struct.calcsize(cls.STRUCT_FMT)
        if len(data) - offset < size:
            from amd_gpu_driver.errors import KernelLoadError
            raise KernelLoadError(
                f"Insufficient data for kernel descriptor: "
                f"need {size} bytes, have {len(data) - offset}"
            )
        fields = struct.unpack(cls.STRUCT_FMT, data[offset : offset + size])
        return cls(
            group_segment_fixed_size=fields[0],
            private_segment_fixed_size=fields[1],
            kernarg_size=fields[2],
            reserved0=fields[3],
            kernel_code_entry_byte_offset=fields[4],
            reserved1=fields[5],
            reserved2=fields[6],
            compute_pgm_rsrc3=fields[7],
            compute_pgm_rsrc1=fields[8],
            compute_pgm_rsrc2=fields[9],
            kernel_code_properties=fields[10],
            kernarg_preload=fields[11],
            reserved3=fields[12],
        )

    def to_bytes(self) -> bytes:
        """Serialize back to bytes."""
        return struct.pack(
            self.STRUCT_FMT,
            self.group_segment_fixed_size,
            self.private_segment_fixed_size,
            self.kernarg_size,
            self.reserved0,
            self.kernel_code_entry_byte_offset,
            self.reserved1,
            self.reserved2,
            self.compute_pgm_rsrc3,
            self.compute_pgm_rsrc1,
            self.compute_pgm_rsrc2,
            self.kernel_code_properties,
            self.kernarg_preload,
            self.reserved3,
        )

    # --- Decoded fields from compute_pgm_rsrc1 ---

    @property
    def granulated_workitem_vgpr_count(self) -> int:
        """Number of VGPRs, granulated. Actual VGPRs = (value + 1) * granularity."""
        return self.compute_pgm_rsrc1 & 0x3F

    @property
    def granulated_wavefront_sgpr_count(self) -> int:
        """Number of SGPRs, granulated."""
        return (self.compute_pgm_rsrc1 >> 6) & 0xF

    @property
    def float_mode(self) -> int:
        return (self.compute_pgm_rsrc1 >> 12) & 0xFF

    @property
    def enable_dx10_clamp(self) -> bool:
        return bool((self.compute_pgm_rsrc1 >> 21) & 1)

    @property
    def enable_ieee_mode(self) -> bool:
        return bool((self.compute_pgm_rsrc1 >> 23) & 1)

    # --- Decoded fields from compute_pgm_rsrc2 ---

    @property
    def enable_sgpr_private_segment_wave_byte_offset(self) -> bool:
        return bool(self.compute_pgm_rsrc2 & 1)

    @property
    def user_sgpr_count(self) -> int:
        return (self.compute_pgm_rsrc2 >> 1) & 0x1F

    @property
    def enable_trap_handler(self) -> bool:
        return bool((self.compute_pgm_rsrc2 >> 6) & 1)

    @property
    def enable_sgpr_workgroup_id_x(self) -> bool:
        return bool((self.compute_pgm_rsrc2 >> 7) & 1)

    @property
    def enable_sgpr_workgroup_id_y(self) -> bool:
        return bool((self.compute_pgm_rsrc2 >> 8) & 1)

    @property
    def enable_sgpr_workgroup_id_z(self) -> bool:
        return bool((self.compute_pgm_rsrc2 >> 9) & 1)

    @property
    def lds_size(self) -> int:
        """LDS size granularity from RSRC2. Actual LDS = value * 128 * 4 bytes."""
        return (self.compute_pgm_rsrc2 >> 15) & 0x1FF

    # --- Decoded fields from kernel_code_properties ---

    @property
    def enable_sgpr_private_segment_buffer(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 0))

    @property
    def enable_sgpr_dispatch_ptr(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 1))

    @property
    def enable_sgpr_queue_ptr(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 2))

    @property
    def enable_sgpr_kernarg_segment_ptr(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 3))

    @property
    def enable_sgpr_dispatch_id(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 4))

    @property
    def enable_sgpr_flat_scratch_init(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 5))

    @property
    def enable_sgpr_private_segment_size(self) -> bool:
        return bool(self.kernel_code_properties & (1 << 6))
