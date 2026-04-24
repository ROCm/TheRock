"""Phase 15: smoke Program.dispatch on macOS with a synthetic no-op kernel."""
from __future__ import annotations

import struct
from types import SimpleNamespace

from amd_gpu_driver.device import AMDDevice
from amd_gpu_driver.kernel.descriptor import KernelDescriptor
from amd_gpu_driver.program import Program
from amd_gpu_driver.sync.timeline import TimelineSemaphore


NOOP_SHADER_CODE = struct.pack("<I", 0xBF810000)
NOOP_RSRC1 = 3 | (0xC0 << 12) | (1 << 29) | (1 << 30) | (1 << 31)
NOOP_RSRC2 = 1 << 7


def main() -> None:
    dev = AMDDevice(backend="macos")
    print(f"device={dev.name} gfx={dev.gfx_target}")

    code = dev.alloc(4096, location="vram", executable=True)
    code.fill(0)
    code.write(NOOP_SHADER_CODE)

    descriptor = KernelDescriptor(
        group_segment_fixed_size=0,
        private_segment_fixed_size=0,
        kernarg_size=0,
        kernel_code_entry_byte_offset=0,
        compute_pgm_rsrc1=NOOP_RSRC1,
        compute_pgm_rsrc2=NOOP_RSRC2,
        compute_pgm_rsrc3=0,
    )
    code_object = SimpleNamespace(text_section=SimpleNamespace(sh_addr=0))
    program = Program(
        dev.backend,
        code_object,
        descriptor,
        code.handle,
        dev.backend.family,
        kernel_name="synthetic_noop",
    )

    queue = dev.backend.create_compute_queue()
    timeline = TimelineSemaphore(dev.backend)
    program.dispatch(queue, grid=(1, 1, 1), block=(1, 1, 1), args=[], timeline=timeline)
    timeline.cpu_wait(timeout_ms=5000)
    print(f"timeline value={timeline.gpu_value}")
    print("Program.dispatch macOS no-op ✓")


if __name__ == "__main__":
    main()
