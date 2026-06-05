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
import os
import time
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
from amd_gpu_driver.backends.windows.gmc_init import (
    GMCConfig,
    flush_gpu_tlb,
    gfxhub_gart_enable,
    init_gmc,
)
from amd_gpu_driver.backends.windows.psp_init import (
    PSPConfig,
    init_psp,
    load_all_firmware,
)
from amd_gpu_driver.backends.windows.smu_init import (
    SMUConfig,
    init_smu,
    load_smu_firmware_direct,
)
from amd_gpu_driver.backends.windows.ih_init import IHConfig, init_ih
from amd_gpu_driver.backends.windows.ring_init import (
    ComputeQueueConfig,
    MESRingConfig,
    init_gfx_for_compute,
    init_mes_for_compute,
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
    smu_config: SMUConfig | None
    ih_config: IHConfig | None
    mes_ring: MESRingConfig | None
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
        gart_start=info.gart_gpu_va_start,
    )
    gfxhub_gart_enable(dev, gmc_config)
    flush_gpu_tlb(dev, gmc_config, vmid=0, hub="gfxhub")
    print("  GMC: GFXHUB GART enabled")

    # --- macOS-proven cold-boot recipe (LITE_MES_RECIPE=1) ---
    if os.environ.get("LITE_MES_RECIPE") == "1":
        return _recipe_bringup(
            dev, ip_result, nbio_config, gmc_config, fw_dir,
            gart_handle, dummy_handle,
        )

    # --- 5. PSP init + firmware loading ---
    print("\n[5/8] Initializing PSP (firmware)...")
    psp_config = None
    try:
        psp_config = init_psp(
            dev,
            ip_result,
            fw_dir=str(fw_dir),
            vram_mc_base=gmc_config.vram_start,
            vram_base_offset=gmc_config.fb_offset,
            vram_bar_phys_addr=info.bars[info.vram_bar_index].phys_addr,
            nbio_config=nbio_config,
        )
        try:
            mp1_version = psp_config.ip_versions.get("mp1", "14_0_2")
            if os.environ.get("AMDGPU_LITE_DIRECT_SMU_LOAD", "0") != "0":
                load_smu_firmware_direct(dev, fw_dir, mp1_version, force=True)
            load_all_firmware(dev, psp_config)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  WARNING: Firmware loading incomplete — {e}")
            print("  Continuing with firmware state loaded so far")
    except RuntimeError as e:
        # On a VBIOS-POST'd GPU (e.g. passthrough), PSP ring creation
        # may fail because firmware is already loaded and running.
        # This is expected — we can proceed without PSP control.
        print(f"  PSP init skipped — {e}")
        print("  Continuing with VBIOS-initialized firmware state")
        psp_config = PSPConfig(mp0_base=[0] * 6)

    smu_config = None
    print("\n[5a/8] Initializing SMU...")
    try:
        smu_config = init_smu(
            dev,
            ip_result,
            vram_mc_base=gmc_config.vram_start,
        )
    except Exception as e:
        print(f"  SMU init failed: {e}")
        print("  Continuing with existing SMU state")

    print("\n[5b/8] Initializing GFX/MEC...")
    try:
        init_gfx_for_compute(dev, ip_result, psp_config, smu_config)
    except Exception as e:
        print(f"  GFX/MEC init failed: {e}")
        print("  Continuing with existing firmware state")

    print("\n[5c/8] Initializing MES scheduler...")
    mes_ring = None
    try:
        mes_ring = init_mes_for_compute(dev, ip_result, nbio_config)
    except Exception as e:
        print(f"  MES init failed: {e}")
        print("  Continuing with direct queue fallback")

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
        compute_queue = init_compute_queue(
            dev, ip_result, nbio_config, mes_ring=mes_ring)
    except Exception as e:
        print(f"  Compute queue init failed: {e}")

    # --- 8. NOP fence test ---
    print("\n[8/8] Running NOP + fence self-test...")
    if compute_queue is not None:
        if test_compute_nop_fence(compute_queue):
            print("  PASS: NOP + RELEASE_MEM fence completed")
        else:
            print("  FAIL: Fence timeout")
            print("  Note: MES queue activation or GFX firmware bring-up is incomplete.")
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
        smu_config=smu_config,
        ih_config=ih_config,
        mes_ring=mes_ring,
        compute_queue=compute_queue,
        gart_table_dma_handle=gart_handle,
        dummy_page_dma_handle=dummy_handle,
    )


# gfx1201 GC BASE_IDX=1 register DWORD base (validated on this silicon; matches
# the macOS phase-9 hardcode). IP discovery prints the discovered value for
# cross-check in _recipe_bringup.
_GC_B1_DW = 0xA000
_REG_RLC_RLCS_BOOTLOAD_STATUS = 0x4e7c  # bit31 = BOOTLOAD_COMPLETE
_REG_GFX_IMU_GFX_RESET_CTRL = 0x40bc    # == 0x7F when all 7 GFX blocks released
_REG_GFX_IMU_CORE_CTRL = 0x40b6         # == 0x8 when IMU running
_REG_RLC_CNTL = 0x4c00                  # == 0x1 when RLC enabled


def _gc_b1_rd(dev: AmdgpuLiteDevice, dw_off: int) -> int:
    return dev.read_reg32((_GC_B1_DW + dw_off) * 4)


def _gc_b1_wr(dev: AmdgpuLiteDevice, dw_off: int, val: int) -> None:
    dev.write_reg32((_GC_B1_DW + dw_off) * 4, val & 0xFFFFFFFF)


def _mes_pipe_liveness(dev: AmdgpuLiteDevice, pipe: int) -> tuple[int, int]:
    """Read CP_MES_INSTR_PNTR + HEADER_DUMP for a specific MES pipe.

    Selects me=3 (MES), the given pipe via GRBM_GFX_CNTL, reads, deselects.
    Used to tell whether the KIQ pipe (pipe 1) is actually executing vs only
    pipe 0 (HEADER_DUMP read without a pipe-select is ambiguous).
    """
    _gc_b1_wr(dev, 0x0900, (3 << 2) | (pipe & 0x3))  # GRBM_GFX_CNTL me=3,pipe
    ip = _gc_b1_rd(dev, 0x2813)   # CP_MES_INSTR_PNTR
    hdr = _gc_b1_rd(dev, 0x280d)  # CP_MES_HEADER_DUMP
    _gc_b1_wr(dev, 0x0900, 0)
    return ip, hdr


def _poll_bootload_complete(dev: AmdgpuLiteDevice, timeout_ms: int = 5000) -> int:
    """Poll RLC_RLCS_BOOTLOAD_STATUS until bit31 set or timeout."""
    import time
    deadline = time.time() + timeout_ms / 1000.0
    bl = 0
    while time.time() < deadline:
        bl = _gc_b1_rd(dev, _REG_RLC_RLCS_BOOTLOAD_STATUS)
        if bl & 0x80000000:
            break
        time.sleep(0.01)
    return bl


def _gc_base_idx1(ip_result: IPDiscoveryResult) -> int | None:
    """The discovered GC BASE_IDX=1 base (for cross-checking _GC_B1_DW)."""
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID
    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.GC and block.instance_number == 0:
            if len(block.base_addresses) > 1:
                return block.base_addresses[1]
    return None


def _recipe_mes_start(dev, ip_result, psp_config, smu_config) -> None:
    """Start the MES engine by reusing ring_init.init_gfx_for_compute.

    init_gfx_for_compute does MEC RS64 enable + MES VRAM-backdoor firmware
    staging (CP_MES_IC_BASE/MDBASE) + MES enable (CP_MES_CNTL) -- but only if
    psp_config.ucode_start carries PFP/ME/MEC/MES/MES1 entry-point addresses.
    The LITE_MES_RECIPE PSP path does not populate ucode_start (and the PSP
    RS64_MES load was rejected), so fill it from the gfx firmware headers here.
    init_gfx_for_compute's _rlc_backdoor_autoload is a no-op without
    AMDGPU_LITE_RLC_BACKDOOR_AUTO=1, and _wait_rlc_autoload passes because our
    PSP AUTOLOAD_RLC already set RLC_RLCS_BOOTLOAD_STATUS bit31.
    """
    import struct
    from amd_gpu_driver.backends.windows.psp_init import _read_firmware

    fw_dir = psp_config.fw_dir
    gc = psp_config.ip_versions.get("gc", "12_0_1")

    def _rs64_entry(name: str) -> int:
        # gfx_firmware_header_v2_0: ucode_start_addr_lo/hi at +52/+56.
        blob = _read_firmware(fw_dir / f"gc_{gc}_{name}.bin")
        lo, hi = struct.unpack_from("<II", blob, 52)
        return (hi << 32) | lo

    def _mes_entry() -> int:
        # mes_firmware_header_v1_0: ucode_start_addr_lo/hi at +56/+60.
        blob = _read_firmware(fw_dir / f"gc_{gc}_uni_mes.bin")
        lo, hi = struct.unpack_from("<II", blob, 56)
        return (hi << 32) | lo

    mes_entry = _mes_entry()
    psp_config.ucode_start.update({
        "PFP": _rs64_entry("pfp"),
        "ME": _rs64_entry("me"),
        "MEC": _rs64_entry("mec"),
        "MES": mes_entry,
        "MES1": mes_entry,
    })
    us = psp_config.ucode_start
    print(f"  ucode_start keys={sorted(us.keys())} "
          f"MES=0x{us['MES']:x} MES1=0x{us.get('MES1', 0):x}")
    init_gfx_for_compute(dev, ip_result, psp_config, smu_config)


def _recipe_bringup(
    dev: AmdgpuLiteDevice,
    ip_result: IPDiscoveryResult,
    nbio_config: NBIOConfig,
    gmc_config: GMCConfig,
    fw_dir: str | Path,
    gart_handle: int,
    dummy_handle: int,
) -> GPUContext:
    """Cold-boot via the macOS gfx_bring_up order, gated by LITE_MES_RECIPE.

    Sequence (all PSP before SMU mailbox, per gfx_bringup.py):
      init_psp -> LOAD_TOC(SOS) -> LOAD_IP_FW(SMU) -> gfx slices (RLC_G last)
      -> AUTOLOAD_RLC -> SMU SetDriverDramAddr + EnableAllSmuFeatures(0)
      -> poll RLC_RLCS_BOOTLOAD_STATUS bit31.

    LITE_STOP_AFTER={toc,autoload,smu,bootload,mes,queue} gates how far to go.
    MES start / compute queue (milestone 4+) are added once bootload passes.
    """
    info = dev.info
    assert info is not None
    stop_after = os.environ.get("LITE_STOP_AFTER", "")

    # The recipe's SMU step matches macOS: enable all features (non-fatal) and
    # do NOT DisallowGfxOff. Respect explicit overrides.
    os.environ.setdefault("AMDGPU_LITE_ENABLE_SMU_FEATURES", "1")
    os.environ.setdefault("AMDGPU_LITE_DISALLOW_GFXOFF", "0")

    print("\n[recipe 1/5] PSP init + LOAD_TOC/IP_FW + AUTOLOAD_RLC...")
    psp_config = init_psp(
        dev, ip_result,
        fw_dir=str(fw_dir),
        vram_mc_base=gmc_config.vram_start,
        vram_base_offset=gmc_config.fb_offset,
        vram_bar_phys_addr=info.bars[info.vram_bar_index].phys_addr,
        nbio_config=nbio_config,
    )
    gc_disc = _gc_base_idx1(ip_result)
    print(f"  bases: MP0[0]=0x{psp_config.mp0_base[0]:x} "
          f"GC_B1(used)=0x{_GC_B1_DW:x} GC_B1(discovered)="
          f"{'0x%x' % gc_disc if gc_disc is not None else 'n/a'}")

    load_all_firmware(dev, psp_config)  # recipe path (LITE_MES_RECIPE=1)
    if stop_after == "toc" or stop_after == "autoload":
        print(f"  LITE_STOP_AFTER={stop_after}: stopping after PSP firmware phase")
        return _recipe_context(dev, ip_result, nbio_config, gmc_config,
                               psp_config, None, gart_handle, dummy_handle)

    print("\n[recipe 2/5] SMU mailbox (SetDriverDramAddr + EnableAllSmuFeatures)...")
    smu_config = None
    try:
        smu_config = init_smu(
            dev, ip_result,
            disable_gfxoff=False,
            vram_mc_base=gmc_config.vram_start,
        )
        print(f"  MP1[0]=0x{smu_config.mp1_base[0]:x} {smu_config.messages.name}")
    except Exception as e:
        print(f"  SMU step failed (non-fatal): {e}")
    if stop_after == "smu":
        print("  LITE_STOP_AFTER=smu: stopping after SMU mailbox")
        return _recipe_context(dev, ip_result, nbio_config, gmc_config,
                               psp_config, smu_config, gart_handle, dummy_handle)

    print("\n[recipe 3/5] Poll RLC_RLCS_BOOTLOAD_STATUS bit31...")
    bl = _poll_bootload_complete(dev)
    reset_ctrl = _gc_b1_rd(dev, _REG_GFX_IMU_GFX_RESET_CTRL)
    core_ctrl = _gc_b1_rd(dev, _REG_GFX_IMU_CORE_CTRL)
    rlc_cntl = _gc_b1_rd(dev, _REG_RLC_CNTL)
    print(f"  BOOTLOAD_STATUS=0x{bl:08x} RESET_CTRL=0x{reset_ctrl:08x} "
          f"CORE_CTRL=0x{core_ctrl:x} RLC_CNTL=0x{rlc_cntl:x}")
    if bl & 0x80000000:
        print("  PASS: BOOTLOAD_COMPLETE set -- RLC/IMU autoload succeeded")
    else:
        print("  FAIL: BOOTLOAD_COMPLETE not set within timeout")
    if stop_after == "bootload" or not (bl & 0x80000000):
        return _recipe_context(dev, ip_result, nbio_config, gmc_config,
                               psp_config, smu_config, gart_handle, dummy_handle)

    # --- [recipe 4/5] MES start: reuse ring_init's gfx-compute path ---
    # init_gfx_for_compute does MEC config/enable + MES VRAM-backdoor fw staging
    # (CP_MES_IC_BASE) + MES enable -- gated on psp_config.ucode_start, which our
    # recipe did not populate. _recipe_mes_start fills it from the gfx fw headers
    # then calls init_gfx_for_compute. Its _rlc_backdoor_autoload is a no-op
    # (AMDGPU_LITE_RLC_BACKDOOR_AUTO unset) and _wait_rlc_autoload passes since our
    # PSP AUTOLOAD already set bootload bit31.
    print("\n[recipe 4/5] MES start (init_gfx_for_compute, ucode_start populated)...")
    _recipe_mes_start(dev, ip_result, psp_config, smu_config)

    hdr0 = _gc_b1_rd(dev, 0x280d)   # CP_MES_HEADER_DUMP
    ip0 = _gc_b1_rd(dev, 0x2813)    # CP_MES_INSTR_PNTR
    time.sleep(0.2)
    hdr1 = _gc_b1_rd(dev, 0x280d)
    ip1 = _gc_b1_rd(dev, 0x2813)
    mes_cntl = _gc_b1_rd(dev, 0x2807)   # CP_MES_CNTL
    mec_cntl = _gc_b1_rd(dev, 0x2904)   # CP_MEC_RS64_CNTL
    print(f"  MES alive check: HEADER_DUMP 0x{hdr0:08x}->0x{hdr1:08x} "
          f"INSTR_PNTR 0x{ip0:08x}->0x{ip1:08x} "
          f"CP_MES_CNTL=0x{mes_cntl:08x} CP_MEC_RS64_CNTL=0x{mec_cntl:08x}")
    mes_alive = (hdr0 != hdr1) or (ip1 != 0)
    print(f"  MES {'RUNNING' if mes_alive else 'NOT visibly executing'}")
    # Per-pipe liveness: is the KIQ pipe (pipe 1) actually executing, or only
    # pipe 0? (the KIQ runs on me=3,pipe=1; if pipe 1 is dead the KIQ never
    # services its ring even with an active HQD + delivered doorbell.)
    for _p in (0, 1):
        _ipa, _ha = _mes_pipe_liveness(dev, _p)
        time.sleep(0.05)
        _ipb, _hb = _mes_pipe_liveness(dev, _p)
        _live = (_ipa != _ipb) or (_ha != _hb) or (_ipb != 0)
        print(f"  MES pipe {_p}: INSTR_PNTR 0x{_ipa:08x}->0x{_ipb:08x} "
              f"HEADER_DUMP 0x{_ha:08x}->0x{_hb:08x} "
              f"-> {'LIVE' if _live else 'DEAD'}")
    if stop_after == "mes":
        print("  LITE_STOP_AFTER=mes: stopping after MES start")
        return _recipe_context(dev, ip_result, nbio_config, gmc_config,
                               psp_config, smu_config, gart_handle, dummy_handle)

    # --- [recipe 5/5] MES rings + compute queue + NOP+fence dispatch ---
    print("\n[recipe 5/5] MES rings + compute queue + NOP+fence...")
    # H3 (kiq-activation-diff): ring_init enables the doorbell aperture +
    # CP_MEC_DOORBELL_RANGE but does NOT program the GDC S2A doorbell monitor
    # entries that the proven try_phase9 path uses (lines 635-641). Without S2A
    # routing, a CPU write to the doorbell BAR may never reach the CP front-end,
    # so the KIQ never fetches the SET_HW_RESOURCES frame -> "MES API opcode 0
    # timed out". Program them here (NBIO BASE_IDX=2 = 0xD20) before the KIQ ring
    # is built/rung. Toggle off with LITE_NO_S2A=1 for A/B comparison.
    if os.environ.get("LITE_NO_S2A") != "1":
        _NBIO_B2 = 0xD20
        # RCC_DOORBELL_APER_EN routes doorbell-BAR writes to the GC doorbell
        # monitor. ring_init never enables it (grep: absent from the python
        # backend AND the amdgpu_lite kernel module), so without it the KIQ
        # doorbell never reaches the CP -> SET_HW_RESOURCES / the PM4 NOP are
        # never fetched (KIQ NOP test: CONSUMED=False despite CP_HQD_ACTIVE=1).
        # try_phase9 enables it at line 626. APER_EN + S2A entries together are
        # the proven doorbell delivery setup (CP_MEC_DOORBELL_RANGE is done by
        # init_gfx_for_compute). Disable for A/B with LITE_NO_S2A=1.
        # GDC S2A doorbell routing entries. VERIFIED against stock amdgpu live on
        # this gfx1201 (dri/1/amdgpu_regs): entries 0-4 are programmed; try_phase9
        # only set 0 and 3. Entry 1 has awaddr_31_28=0 so it routes LOW doorbell
        # addresses (the KIQ doorbell lands at byte 0x60) -- without it the KIQ
        # doorbell never reaches the CP (NOP CONSUMED=False). Program all 5 to the
        # golden values; read pre-write to learn the Linux VBIOS POST state.
        _S2A = [
            (0x01cb, 0x30000007),  # ENTRY_0
            (0x01cc, 0x00057801),  # ENTRY_1 (awaddr_31_28=0: low addrs)
            (0x01cd, 0x3051001d),  # ENTRY_2
            (0x01ce, 0x3000000d),  # ENTRY_3
            (0x01cf, 0x40118809),  # ENTRY_4
        ]
        _pre = [dev.read_reg32((_NBIO_B2 + off) * 4) for off, _ in _S2A]
        print("  S2A pre-write:  " + " ".join(
            f"E{i}=0x{v:08x}" for i, v in enumerate(_pre)))
        for off, val in _S2A:
            dev.write_reg32((_NBIO_B2 + off) * 4, val)
        dev.write_reg32((_NBIO_B2 + 0x00c0) * 4, 1)  # RCC_DOORBELL_APER_EN
        # Doorbell selfring GPA aperture (nbio_v7_11_enable_doorbell_selfring_aperture):
        # PF1 regs at NBIO BASE_IDX=2: BASE_LOW=0xf4 BASE_HIGH=0xf3 CNTL=0xf5;
        # CNTL = EN|MODE = 0x3; base = doorbell BAR phys.
        _db_base = dev.info.bars[dev.info.doorbell_bar_index].phys_addr
        dev.write_reg32((_NBIO_B2 + 0x00f4) * 4, _db_base & 0xFFFFFFFF)
        dev.write_reg32((_NBIO_B2 + 0x00f3) * 4, (_db_base >> 32) & 0xFFFFFFFF)
        dev.write_reg32((_NBIO_B2 + 0x00f5) * 4, 0x3)
        _post = [dev.read_reg32((_NBIO_B2 + off) * 4) for off, _ in _S2A]
        aper = dev.read_reg32((_NBIO_B2 + 0x00c0) * 4)
        print("  S2A post-write: " + " ".join(
            f"E{i}=0x{v:08x}" for i, v in enumerate(_post)))
        print(f"  doorbell: APER_EN=0x{aper:x} selfring base=0x{_db_base:x} CNTL=0x3")
    # --- DIRECT MEC compute HQD path (LITE_DIRECT_QUEUE=1, Tinygrad-modeled) ---
    # Skip the MES KIQ entirely (the #17 blocker). Tinygrad dispatches on gfx1201
    # via a direct me=1 MEC HQD with NO MES servicing; our init_compute_queue
    # (use_mes=False) -> _activate_compute_queue_mmio is structurally the same. The
    # MEC is already enabled by init_gfx_for_compute (milestone 4) and the doorbell
    # fabric programmed above is exactly what Tinygrad's doorbell_enable does.
    if os.environ.get("LITE_DIRECT_QUEUE") == "1":
        print("\n[recipe 5/5] DIRECT MEC compute queue (no MES KIQ)...")
        ih_config = None
        try:
            ih_config = init_ih(dev, ip_result, nbio_config)
        except Exception as e:  # noqa: BLE001
            print(f"  IH init skipped (non-fatal): {e}")
        compute_queue = None
        try:
            compute_queue = init_compute_queue(
                dev, ip_result, nbio_config, use_mes=False)
            ok = test_compute_nop_fence(compute_queue)
            print("  PASS: DIRECT-MEC NOP+fence completed" if ok
                  else "  FAIL: direct-queue NOP+fence did not complete")
            # Multi-dispatch (#21): N more back-to-back NOP-fences on the same
            # queue. Tinygrad has no ceiling, so a ceiling here = our setup bug
            # (NO_UPDATE_RPTR / ring-wrap / wptr -- try LITE_HQD_TG_PARITY=1).
            n = int(os.environ.get("LITE_MULTI_DISPATCH", "0") or "0")
            if ok and n > 0:
                print(f"  [#21] multi-dispatch: {n} more NOP-fences...")
                done, ceiling = 1, None
                for seq in range(2, 2 + n):
                    if test_compute_nop_fence(compute_queue, fence_seq=seq):
                        done += 1
                    else:
                        ceiling = done + 1
                        break
                if ceiling is None:
                    print(f"  [#21] PASS: sustained {done} dispatches, no ceiling")
                else:
                    print(f"  [#21] CEILING at dispatch {ceiling} "
                          f"(fence_seq={ceiling} timed out)")
        except Exception as e:  # noqa: BLE001
            print(f"  direct compute queue / dispatch failed: {e}")
            import traceback
            traceback.print_exc()
        return GPUContext(
            dev=dev, ip_result=ip_result, nbio_config=nbio_config,
            gmc_config=gmc_config, psp_config=psp_config, smu_config=smu_config,
            ih_config=ih_config, mes_ring=None, compute_queue=compute_queue,
            gart_table_dma_handle=gart_handle, dummy_page_dma_handle=dummy_handle,
        )

    mes_ring = None
    compute_queue = None
    ih_config = None
    try:
        mes_ring = init_mes_for_compute(dev, ip_result, nbio_config)
    except Exception as e:  # noqa: BLE001
        print(f"  init_mes_for_compute failed: {e}")
        import traceback
        traceback.print_exc()
        return _recipe_context(dev, ip_result, nbio_config, gmc_config,
                               psp_config, smu_config, gart_handle, dummy_handle)
    # IH is not needed for the NOP-fence (wait_fence is a CPU memory poll); init
    # it best-effort so a later interrupt-driven path has it.
    try:
        ih_config = init_ih(dev, ip_result, nbio_config)
    except Exception as e:  # noqa: BLE001
        print(f"  IH init skipped (non-fatal for NOP fence): {e}")
    try:
        compute_queue = init_compute_queue(
            dev, ip_result, nbio_config, mes_ring=mes_ring)
        ok = test_compute_nop_fence(compute_queue)
        if ok:
            print("  PASS: MILESTONE 4 -- MES-scheduled NOP+fence completed")
        else:
            print("  FAIL: NOP+fence did not complete (fence never reached seq)")
    except Exception as e:  # noqa: BLE001
        print(f"  compute queue / dispatch failed: {e}")
        import traceback
        traceback.print_exc()

    return GPUContext(
        dev=dev, ip_result=ip_result, nbio_config=nbio_config,
        gmc_config=gmc_config, psp_config=psp_config, smu_config=smu_config,
        ih_config=ih_config, mes_ring=mes_ring, compute_queue=compute_queue,
        gart_table_dma_handle=gart_handle, dummy_page_dma_handle=dummy_handle,
    )


def _recipe_context(dev, ip_result, nbio_config, gmc_config, psp_config,
                    smu_config, gart_handle, dummy_handle) -> GPUContext:
    return GPUContext(
        dev=dev,
        ip_result=ip_result,
        nbio_config=nbio_config,
        gmc_config=gmc_config,
        psp_config=psp_config,
        smu_config=smu_config,
        ih_config=None,
        mes_ring=None,
        compute_queue=None,
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
