"""GPU bring-up orchestration for amdgpu_lite on Linux.

Reuses the same init modules as the Windows backend (ip_discovery, nbio_init,
gmc_init, psp_init, ih_init, ring_init) but with AmdgpuLiteDevice providing
the register access and DMA allocation.

Init sequence:
1. Open /dev/amdgpu_lite0, map MMIO + doorbell BARs
2. IP discovery — enumerate IP blocks via SMN indirect reads
3. NBIO init — doorbell aperture, framebuffer access
4. GMC init — memory controller, system aperture, GART
5. PSP init — firmware loading (SOS, RLC, MEC, SDMA)
6. IH init — interrupt handler ring
7. Compute ring — MQD construction + direct MMIO HQD programming
8. Self-test — NOP + RELEASE_MEM fence verification

Usage:
    ctx = full_gpu_bringup()
    # ctx.compute_queue is ready for PM4 packet submission
    shutdown(ctx)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

from amd_gpu_driver.backends.amdgpu_lite.device import AmdgpuLiteDevice

# Reuse the shared init modules from the windows backend.
# Despite the package name, these are hardware-level register programming
# that works with any device providing read_reg32/write_reg32/alloc_dma.
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
    wait_fence,
    test_compute_nop_fence,
)


@dataclass
class GPUContext:
    """Holds all configuration state for an initialized GPU."""

    dev: AmdgpuLiteDevice
    ip_result: IPDiscoveryResult
    nbio_config: NBIOConfig
    gmc_config: GMCConfig
    psp_config: PSPConfig | None
    ih_config: IHConfig | None
    compute_queue: ComputeQueueConfig | None
    gart_table_dma_handle: int = 0
    dummy_page_dma_handle: int = 0
    _fence_seq: int = 0

    def next_fence_seq(self) -> int:
        self._fence_seq += 1
        return self._fence_seq


# Default firmware directories on Linux
LINUX_FW_DIRS = [
    "/lib/firmware/amdgpu",
    "/usr/lib/firmware/amdgpu",
    "/usr/share/firmware/amdgpu",
]


def _find_fw_dir() -> str:
    """Find the firmware directory on this system."""
    for d in LINUX_FW_DIRS:
        p = Path(d)
        if p.is_dir() and any(p.glob("*.bin")):
            return str(p)
    return "."


def full_gpu_bringup(
    device_index: int = 0,
    fw_dir: str | Path | None = None,
) -> GPUContext:
    """Run the full GPU initialization sequence via amdgpu_lite.

    Opens the device, discovers IP blocks, initializes all IP subsystems,
    creates a compute queue, and runs a NOP fence self-test.

    Args:
        device_index: GPU index (0 for first AMD GPU).
        fw_dir: Directory containing firmware .bin files.
                If None, searches standard Linux firmware paths.

    Returns:
        GPUContext with all initialized subsystems.
    """
    if fw_dir is None:
        fw_dir = _find_fw_dir()

    print("=" * 60)
    print("AMD GPU Bring-up (Linux amdgpu_lite)")
    print("=" * 60)

    # --- 1. Open device ---
    print("\n[1/8] Opening device...")
    dev = AmdgpuLiteDevice()
    dev.open(device_index)
    print(f"  Device: {dev.name}")
    print(f"  VRAM: {dev.vram_size // (1024**2)} MB")
    print(f"  MMIO BAR mapped at: {dev.mmio_addr:#018x}")
    print(f"  Doorbell BAR mapped at: {dev.doorbell_addr:#018x}")

    # --- 2. IP discovery ---
    print("\n[2/8] Running IP discovery...")
    # Read discovery table from top of VRAM via BAR aperture.
    # SMN indirect reads don't work on RDNA4 with amdgpu_lite (returns zeros),
    # but direct VRAM BAR reads work when resizable BAR covers full VRAM.
    disc_offset = dev.vram_size - 65536
    raw_table = dev.read_vram(disc_offset, 65536)
    ip_result = parse_ip_discovery(raw_table)
    print(f"  Found {len(ip_result.ip_blocks)} IP blocks")
    for block in ip_result.ip_blocks:
        hw_name = block.hw_id.name if hasattr(block.hw_id, 'name') else f"0x{block.hw_id:x}"
        print(f"    {hw_name}: "
              f"v{block.major}.{block.minor}.{block.revision}")

    # --- 3. NBIO init ---
    print("\n[3/8] Initializing NBIO...")
    nbio_config = init_nbio(dev, ip_result)

    # --- 4. GMC init ---
    print("\n[4/8] Initializing GMC...")
    # Use the kernel module's pre-allocated GART table
    info = dev.info
    assert info is not None
    gart_bus = info.gart_table_bus_addr
    gart_handle = 0  # Kernel-managed, no handle needed

    if gart_bus == 0:
        # Kernel didn't allocate GART — allocate our own
        print("  Allocating GART table (1MB)...")
        gart_cpu, gart_bus, gart_handle = dev.alloc_dma(1024 * 1024)
        ctypes.memset(gart_cpu, 0, 1024 * 1024)

    # Allocate dummy page for fault handling
    dummy_cpu, dummy_bus, dummy_handle = dev.alloc_dma(4096)
    ctypes.memset(dummy_cpu, 0, 4096)

    gmc_config = init_gmc(
        dev, ip_result, nbio_config,
        vram_size_bytes=dev.vram_size,
        gart_table_bus_addr=gart_bus,
        dummy_page_bus_addr=dummy_bus,
    )

    # --- 5. PSP init + firmware loading ---
    print("\n[5/8] Initializing PSP (firmware)...")
    psp_config = None
    try:
        psp_config = init_psp(dev, ip_result, fw_dir=str(fw_dir))
        try:
            load_all_firmware(dev, psp_config)
        except FileNotFoundError as e:
            print(f"  WARNING: Firmware loading skipped — {e}")
            print("  Continuing with VBIOS-initialized firmware state")
    except RuntimeError as e:
        # On a VBIOS-POST'd GPU (e.g. passthrough), PSP ring creation
        # may fail because firmware is already loaded and running.
        # This is expected — we can proceed without PSP control.
        print(f"  PSP init skipped — {e}")
        print("  Continuing with VBIOS-initialized firmware state")
        psp_config = PSPConfig(mp0_base=[0] * 6)

    # --- 6. IH init ---
    print("\n[6/8] Initializing IH (interrupts)...")
    ih_config = None
    try:
        ih_config = init_ih(dev, ip_result, nbio_config)
    except Exception as e:
        print(f"  IH init failed: {e}")
        print("  Continuing without interrupt support")

    # --- 7. Compute ring ---
    print("\n[7/8] Creating compute queue...")
    compute_queue = None
    try:
        compute_queue = init_compute_queue(dev, ip_result, nbio_config)

        # Fix up doorbell address: NBIO init may not have found the doorbell
        # physical address from registers, but the kernel module already mapped
        # the doorbell BAR for us.
        if compute_queue.doorbell_cpu_addr == 0 and dev.doorbell_addr != 0:
            doorbell_offset = compute_queue.doorbell_index * 8
            compute_queue.doorbell_cpu_addr = dev.doorbell_addr + doorbell_offset
    except Exception as e:
        print(f"  Compute queue init failed: {e}")

    # --- 8. NOP fence test ---
    print("\n[8/8] Running NOP + fence self-test...")
    if compute_queue is not None:
        if test_compute_nop_fence(compute_queue):
            print("  PASS: NOP + RELEASE_MEM fence completed")
        else:
            print("  FAIL: Fence timeout")
            print("  Note: GFX12 requires MES for compute queue activation.")
            print("  Direct MMIO HQD programming (SOC21 offsets) does not work.")
            print("  Next step: implement MES-based queue add.")
    else:
        print("  SKIP: No compute queue available")

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


def shutdown(ctx: GPUContext) -> None:
    """Clean shutdown."""
    print("\nShutting down GPU...")
    if ctx.dummy_page_dma_handle:
        ctx.dev.free_dma(ctx.dummy_page_dma_handle)
    if ctx.gart_table_dma_handle:
        ctx.dev.free_dma(ctx.gart_table_dma_handle)
    ctx.dev.close()
    print("  Device closed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AMD GPU bring-up via amdgpu_lite (Linux)")
    parser.add_argument(
        "--device", type=int, default=0,
        help="GPU device index (default: 0)")
    parser.add_argument(
        "--fw-dir", type=str, default=None,
        help="Directory containing firmware .bin files")
    args = parser.parse_args()

    ctx = full_gpu_bringup(
        device_index=args.device,
        fw_dir=args.fw_dir,
    )
    shutdown(ctx)
