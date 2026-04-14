"""macOS GPU bringup — cold-boot RDNA4 from reset state.

Orchestrates the GPU initialization sequence using MMIO register
access provided by the ROCmGPU DEXT. Reuses GPU-generation-specific
register definitions from the Windows backend (which are OS-agnostic).

Bringup sequence:
  1. Function-Level Reset (FLR)
  2. Enable bus mastering + memory space
  3. Map PCI BARs (MMIO, doorbell, VRAM)
  4. IP Discovery (parse hardware block table from VRAM)
  5. NBIO init (North Bridge I/O configuration)
  6. GMC init (Graphics Memory Controller, page tables)
  7. PSP init (Platform Security Processor, firmware loading)
  8. IH init (Interrupt Handler ring)
  9. Ring init (compute/SDMA queue hardware setup)

The register-programming modules (gmc_init, nbio_init, etc.) work with
any device that provides read_reg32()/write_reg32() methods, so they
are shared across Windows and macOS backends.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.macos.device import MacOSDevice

logger = logging.getLogger(__name__)

# BAR indices for RDNA GPUs
BAR_MMIO = 0       # BAR0: MMIO registers
BAR_VRAM = 0       # BAR0 (with resizable BAR) or BAR1: VRAM aperture
BAR_DOORBELL = 2   # BAR2: Doorbell aperture


def gpu_bringup(
    device: MacOSDevice,
    *,
    firmware_path: str | None = None,
    skip_reset: bool = False,
) -> None:
    """Cold-boot the GPU from reset state.

    Args:
        device: MacOSDevice with open IOKit connection.
        firmware_path: Path to firmware directory (containing *.bin files).
            If None, uses bundled firmware location.
        skip_reset: Skip FLR (useful if GPU was already reset externally).
    """
    client = device.client

    # ================================================================
    # Phase 1: Hardware reset and PCI setup
    # ================================================================

    if not skip_reset:
        logger.info("Phase 1: Function-Level Reset")
        client.reset()
        time.sleep(0.5)  # Wait for reset to complete

    # Enable bus mastering and memory space
    cmd_reg = client.cfg_read(0x04, width=2)
    cmd_reg |= 0x0006  # BIT1=Memory Space, BIT2=Bus Master
    client.cfg_write(0x04, cmd_reg, width=2)
    logger.info("Bus mastering enabled (cmd=0x%04x)", cmd_reg | 0x0006)

    # Read device ID for identification
    info = client.get_info()
    logger.info(
        "Device: vendor=0x%04x device=0x%04x rev=0x%02x vram=%dMB",
        info.vendor_id, info.device_id, info.revision_id,
        info.vram_size // (1024 * 1024),
    )

    # ================================================================
    # Phase 2: Map PCI BARs
    # ================================================================

    logger.info("Phase 2: Mapping PCI BARs")

    # Map MMIO BAR (BAR0) — always needed for register access
    mmio_addr, mmio_size = client.map_bar(BAR_MMIO)
    logger.info("  BAR0 (MMIO): addr=0x%x size=%dMB", mmio_addr, mmio_size // (1024*1024))

    # Map doorbell BAR (BAR2) — needed for queue submission
    try:
        db_addr, db_size = client.map_bar(BAR_DOORBELL)
        logger.info("  BAR2 (doorbell): addr=0x%x size=%dKB", db_addr, db_size // 1024)
        if device._queues:
            device._queues.set_doorbell_bar(db_addr, db_size)
    except RuntimeError:
        logger.warning("  BAR2 (doorbell): not available")

    # Map VRAM BAR if it's separate from MMIO (or if BAR0 is resizable)
    # On RDNA4 with resizable BAR, BAR0 covers both MMIO and VRAM
    for bar_idx in range(6):
        bar = info.bars[bar_idx]
        if bar.prefetchable and bar.size > mmio_size and bar_idx != BAR_MMIO:
            try:
                vram_addr, vram_size = client.map_bar(bar_idx)
                logger.info(
                    "  BAR%d (VRAM): addr=0x%x size=%dMB",
                    bar_idx, vram_addr, vram_size // (1024*1024),
                )
                if device._memory:
                    device._memory.set_vram_bar(vram_addr, vram_size)
                break
            except RuntimeError:
                logger.warning("  BAR%d (VRAM): mapping failed", bar_idx)

    # ================================================================
    # Phase 3: Verify basic register access
    # ================================================================

    logger.info("Phase 3: Verifying register access")

    # Read a known register to verify MMIO works
    # 0x0 is typically the GPU revision register
    rev_reg = device.read_reg32(0x0)
    logger.info("  Register 0x0 = 0x%08x", rev_reg)

    # Read ASIC family from golden register
    # GC_CAC_ID (0x2800) or similar identification register
    gc_cac = device.read_reg32(0x2800)
    logger.info("  Register 0x2800 (GC_CAC_ID) = 0x%08x", gc_cac)

    # ================================================================
    # Phase 4: IP Discovery
    # ================================================================

    logger.info("Phase 4: IP Discovery")
    ip_blocks = _ip_discovery(device, info)

    # ================================================================
    # Phase 5-8: Hardware initialization
    #
    # These phases use the shared register-programming modules from
    # the Windows backend. Each module takes a 'device' object that
    # provides read_reg32()/write_reg32() methods.
    #
    # TODO: Wire up when hardware is available for testing.
    # The init modules need to be called with the correct IP block
    # base addresses from the discovery table.
    # ================================================================

    logger.info("Phase 5: NBIO init — TODO (requires hardware testing)")
    # from amd_gpu_driver.backends.windows.nbio_init import nbio_init
    # nbio_init(device, ip_blocks)

    logger.info("Phase 6: GMC init — TODO (requires hardware testing)")
    # from amd_gpu_driver.backends.windows.gmc_init import gmc_init
    # gmc_init(device, ip_blocks)

    logger.info("Phase 7: PSP init — TODO (requires firmware + hardware)")
    # from amd_gpu_driver.backends.windows.psp_init import psp_init
    # psp_init(device, ip_blocks, firmware_path)

    logger.info("Phase 8: IH init — TODO (requires hardware testing)")
    # from amd_gpu_driver.backends.windows.ih_init import ih_init
    # ih_init(device, ip_blocks)

    logger.info("Phase 9: Ring init — TODO (requires GMC + PSP)")
    # from amd_gpu_driver.backends.windows.ring_init import ring_init
    # ring_init(device, ip_blocks)

    logger.info("Bringup complete (phases 5-9 pending hardware testing)")


def _ip_discovery(device: MacOSDevice, info) -> dict:
    """Parse the IP discovery table from VRAM.

    The IP discovery table is located at VRAM_SIZE - 64KB and contains
    a catalog of all hardware IP blocks (GC, SDMA, MMHUB, etc.) with
    their version numbers and base register addresses.

    Returns a dict mapping IP block names to their info.
    """
    # IP discovery table is at the end of VRAM
    # It can be read via MMIO (if the table is in the MMIO aperture)
    # or via BAR mapping (if VRAM BAR is large enough)

    # For now, try to read via SMN indirect access
    # The IP discovery table signature is at a known offset
    IP_DISCOVERY_OFFSET = 0x10000  # 64KB from end of VRAM

    ip_blocks: dict[str, dict] = {}

    try:
        # Try to read the IP discovery signature
        # The table starts with a header: signature (4 bytes) = "IPHD"
        sig_addr = info.vram_size - IP_DISCOVERY_OFFSET
        logger.info("  Looking for IP discovery at VRAM offset 0x%x", sig_addr)

        # Read via the ip_discovery module if available
        try:
            from amd_gpu_driver.backends.windows.ip_discovery import parse_ip_discovery
            ip_blocks = parse_ip_discovery(device)
            logger.info("  Found %d IP blocks", len(ip_blocks))
        except ImportError:
            logger.info("  ip_discovery module not available, using defaults")
            ip_blocks = _default_ip_blocks(info.device_id)
        except Exception as e:
            logger.warning("  IP discovery failed: %s, using defaults", e)
            ip_blocks = _default_ip_blocks(info.device_id)

    except Exception as e:
        logger.warning("  IP discovery error: %s", e)
        ip_blocks = _default_ip_blocks(info.device_id)

    return ip_blocks


def _default_ip_blocks(device_id: int) -> dict:
    """Default IP block configuration for known devices.

    Used when IP discovery table cannot be parsed (e.g., before
    GMC init makes VRAM accessible).
    """
    # RDNA4 defaults
    if device_id in (0x7551, 0x7550):
        return {
            "GC": {"version": (12, 0, 0), "base": 0x0},
            "SDMA0": {"version": (7, 0, 0), "base": 0x0},
            "MMHUB": {"version": (4, 1, 0), "base": 0x0},
            "GFXHUB": {"version": (12, 0, 0), "base": 0x0},
            "IH": {"version": (7, 0, 0), "base": 0x0},
            "NBIO": {"version": (7, 11, 0), "base": 0x0},
            "PSP": {"version": (14, 0, 0), "base": 0x0},
        }
    # RDNA3 defaults
    elif device_id in (0x744C, 0x7448):
        return {
            "GC": {"version": (11, 0, 0), "base": 0x0},
            "SDMA0": {"version": (6, 0, 0), "base": 0x0},
            "MMHUB": {"version": (3, 3, 0), "base": 0x0},
        }
    else:
        return {}
