"""Debug: inspect kernel descriptor and attempt dispatch."""
import ctypes
import struct
import time
from pathlib import Path

from amd_gpu_driver import AMDDevice
from amd_gpu_driver.backends.base import MemoryLocation
from amd_gpu_driver.commands.pm4 import (
    PM4PacketBuilder, SH_REG_BASE,
    CP_COHER_CNTL_SH_KCACHE_ACTION, CP_COHER_CNTL_TCL1_ACTION,
    CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH,
)
from amd_gpu_driver.gpu.registers import (
    COMPUTE_PGM_LO, COMPUTE_PGM_RSRC1, COMPUTE_PGM_RSRC2, COMPUTE_PGM_RSRC3,
    COMPUTE_USER_DATA_0, COMPUTE_START_X,
    COMPUTE_TMPRING_SIZE, COMPUTE_RESOURCE_LIMITS, COMPUTE_RESTART_X,
    COMPUTE_DISPATCH_INITIATOR, sh_reg_offset,
)
from amd_gpu_driver.kernel.elf_parser import parse_elf_file
from amd_gpu_driver.kernel.descriptor import KernelDescriptor
from amd_gpu_driver.sync.timeline import TimelineSemaphore


def main():
    co_path = Path(__file__).parent.parent / "tests" / "fixtures" / "fill_kernel_gfx942.co"
    print(f"Loading: {co_path}")

    co = parse_elf_file(str(co_path))
    print(f"Sections: {[(s.name, hex(s.sh_addr), hex(s.sh_offset), s.sh_size) for s in co.sections if s.name]}")
    print(f"Symbols: {[(s.name, hex(s.st_value), s.st_size, s.type) for s in co.symbols if s.name]}")

    # Find kernel descriptor
    kd_syms = [s for s in co.symbols if s.name == "fill_kernel.kd"]
    func_syms = [s for s in co.symbols if s.name == "fill_kernel" and s.type == 2]
    print(f"\nKD symbol: {kd_syms[0].name} at VA 0x{kd_syms[0].st_value:x}" if kd_syms else "No .kd symbol!")
    print(f"FUNC symbol: {func_syms[0].name} at VA 0x{func_syms[0].st_value:x}" if func_syms else "No FUNC symbol!")

    # Parse descriptor
    kd_sym = kd_syms[0]
    rodata = co.get_section_data(co.rodata_section)
    kd_offset = kd_sym.st_value - co.rodata_section.sh_addr
    print(f"\nrodata section at VA 0x{co.rodata_section.sh_addr:x}, file offset 0x{co.rodata_section.sh_offset:x}")
    print(f"KD at rodata offset 0x{kd_offset:x}")
    kd = KernelDescriptor.from_bytes(rodata, kd_offset)

    print(f"\nKernel Descriptor:")
    print(f"  group_segment_fixed_size: {kd.group_segment_fixed_size}")
    print(f"  private_segment_fixed_size: {kd.private_segment_fixed_size}")
    print(f"  kernarg_size: {kd.kernarg_size}")
    print(f"  kernel_code_entry_byte_offset: 0x{kd.kernel_code_entry_byte_offset:x}")
    print(f"  compute_pgm_rsrc1: 0x{kd.compute_pgm_rsrc1:08x}")
    print(f"  compute_pgm_rsrc2: 0x{kd.compute_pgm_rsrc2:08x}")
    print(f"  compute_pgm_rsrc3: 0x{kd.compute_pgm_rsrc3:08x}")
    print(f"  kernel_code_properties: 0x{kd.kernel_code_properties:04x}")
    print(f"  user_sgpr_count: {kd.user_sgpr_count}")
    print(f"  enable_sgpr_kernarg_segment_ptr: {kd.enable_sgpr_kernarg_segment_ptr}")
    print(f"  enable_sgpr_dispatch_ptr: {kd.enable_sgpr_dispatch_ptr}")
    print(f"  enable_sgpr_workgroup_id_x: {kd.enable_sgpr_workgroup_id_x}")
    print(f"  enable_sgpr_workgroup_id_y: {kd.enable_sgpr_workgroup_id_y}")
    print(f"  enable_sgpr_workgroup_id_z: {kd.enable_sgpr_workgroup_id_z}")

    # Compute code entry
    text_va = co.text_section.sh_addr
    descriptor_va = kd_sym.st_value
    code_entry_va = descriptor_va + kd.kernel_code_entry_byte_offset
    print(f"\n  .text VA: 0x{text_va:x}")
    print(f"  descriptor VA: 0x{descriptor_va:x}")
    print(f"  code_entry VA: 0x{code_entry_va:x}")
    print(f"  code_entry offset in .text: 0x{code_entry_va - text_va:x}")

    # Cross-check: FUNC symbol should == code_entry_va
    if func_syms:
        print(f"  FUNC symbol VA: 0x{func_syms[0].st_value:x} (should match code_entry VA)")

    # Now try actual dispatch
    dev = AMDDevice()
    program = dev.load_program(str(co_path))
    print(f"\nLoaded program: {program.name}")
    print(f"  kernarg_size: {program.kernarg_size}")
    print(f"  descriptor.kernel_code_entry_byte_offset: 0x{program.descriptor.kernel_code_entry_byte_offset:x}")

    # Allocate output buffer
    out_buf = dev.alloc(64 * 4, location="vram")
    out_buf.fill(0x00)

    backend = dev.backend
    queue = backend.create_compute_queue()

    # Allocate signal memory directly (no timeline, just SDMA fence for simplicity)
    sig = backend.alloc_memory(4096, MemoryLocation.GTT, uncached=True)
    ctypes.memset(sig.cpu_addr, 0, 16)

    # Build PM4 manually for debugging
    pm4 = PM4PacketBuilder()
    pm4.acquire_mem(
        coher_cntl=CP_COHER_CNTL_SH_KCACHE_ACTION | CP_COHER_CNTL_TCL1_ACTION,
    )

    # Code entry address
    code_mem = program._code_mem
    text_va = co.text_section.sh_addr
    code_entry_va = program._descriptor_va + kd.kernel_code_entry_byte_offset
    code_entry_addr = code_mem.gpu_addr + (code_entry_va - text_va)
    pgm_addr = code_entry_addr >> 8

    print(f"\n  code_mem.gpu_addr: 0x{code_mem.gpu_addr:x}")
    print(f"  code_entry_addr: 0x{code_entry_addr:x}")
    print(f"  pgm_addr (>>8): 0x{pgm_addr:x}")

    # Kernarg
    kernarg_data = struct.pack("<QI", out_buf.gpu_addr, 0xBEEF) + b'\x00' * (kd.kernarg_size - 12)
    kernarg_mem = backend.alloc_memory(max(len(kernarg_data), 4096), MemoryLocation.GTT, uncached=True)
    ctypes.memmove(kernarg_mem.cpu_addr, kernarg_data, len(kernarg_data))
    print(f"  kernarg_mem.gpu_addr: 0x{kernarg_mem.gpu_addr:x}")
    print(f"  kernarg_data: {kernarg_data[:16].hex()}")

    # Set registers (matching tinygrad's sequence for gfx942)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_LO), pgm_addr & 0xFFFFFFFF, (pgm_addr >> 32) & 0xFFFFFFFF)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_RSRC1), kd.compute_pgm_rsrc1, kd.compute_pgm_rsrc2)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_PGM_RSRC3), kd.compute_pgm_rsrc3)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_TMPRING_SIZE), 0)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESTART_X), 0, 0, 0)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_USER_DATA_0), kernarg_mem.gpu_addr & 0xFFFFFFFF, (kernarg_mem.gpu_addr >> 32) & 0xFFFFFFFF)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_RESOURCE_LIMITS), 0)
    pm4.set_sh_reg(sh_reg_offset(COMPUTE_START_X), 0, 0, 0, 64, 1, 1, 0, 0)

    pm4.dispatch_direct(1, 1, 1)

    # CS_PARTIAL_FLUSH after dispatch
    pm4.event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_CS_PARTIAL_FLUSH)

    # Release mem to write signal (with cache flush)
    pm4.release_mem(addr=sig.gpu_addr, value=1, cache_flush=True)

    packets = pm4.build()
    dwords = struct.unpack(f"<{len(packets)//4}I", packets)
    print(f"\nPM4 packet ({len(dwords)} dwords):")
    for i, d in enumerate(dwords):
        print(f"  [{i:3d}] 0x{d:08x}")

    backend.submit_packets(queue, packets)

    # Poll
    for i in range(50):
        time.sleep(0.1)
        val = ctypes.c_uint64.from_address(sig.cpu_addr).value
        if val > 0:
            print(f"\n  Signal received: {val} at {(i+1)*100}ms")
            break
    else:
        val = ctypes.c_uint64.from_address(sig.cpu_addr).value
        print(f"\n  TIMEOUT! Signal value: {val}")

    # Check output
    data = out_buf.read(16)
    print(f"  Output (first 16 bytes): {data.hex()}")
    if data != b'\x00' * 16:
        values = struct.unpack("<4I", data)
        print(f"  Values: {[hex(v) for v in values]}")

    backend.destroy_queue(queue)
    program.free()
    backend.free_memory(sig)
    backend.free_memory(kernarg_mem)
    out_buf.free()
    dev.close()


if __name__ == "__main__":
    main()
