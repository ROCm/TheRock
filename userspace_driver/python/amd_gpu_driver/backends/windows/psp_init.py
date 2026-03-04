"""PSP v14.0 initialization for RDNA4 (Navi 48 / GFX1201).

The PSP (Platform Security Processor) is an ARM co-processor on the GPU
that manages firmware loading and security. After VBIOS POST, the PSP
SOS (Secure OS) is already running. We need to:

1. Verify SOS is alive (C2PMSG_81 != 0)
2. Create the PSP GPCOM ring for command submission
3. Submit IP firmware (GFX, SDMA, RLC, MES, SMU) via the ring
4. Trigger RLC autoload
5. Wait for firmware ready signals

The PSP mailbox protocol uses MP0 registers for communication:
- C2PMSG_35: Bootloader command/status (bit 31 = ready)
- C2PMSG_36: Firmware address (>> 20 for 1MB alignment)
- C2PMSG_64: Ring command/status
- C2PMSG_67: Ring write pointer
- C2PMSG_69/70/71: Ring address low/high/size
- C2PMSG_81: SOS Sign of Life (non-zero = alive)

Reference: Linux amdgpu psp_v14_0.c, amdgpu_psp.c
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult


# ============================================================================
# PSP v14.0 register offsets (MP0 base, DWORD offsets)
# From mp_14_0_2_offset.h
# ============================================================================

# Bootloader mailbox
regMPASP_SMN_C2PMSG_35 = 0x0063   # Command/status (bit 31 = ready)
regMPASP_SMN_C2PMSG_36 = 0x0064   # Firmware address (>> 20)
regMPASP_SMN_C2PMSG_81 = 0x0091   # SOS Sign of Life

# TOS ring mailbox
regMPASP_SMN_C2PMSG_64 = 0x0080   # Ring command/status
regMPASP_SMN_C2PMSG_67 = 0x0083   # Ring write pointer
regMPASP_SMN_C2PMSG_69 = 0x0085   # Ring address low
regMPASP_SMN_C2PMSG_70 = 0x0086   # Ring address high
regMPASP_SMN_C2PMSG_71 = 0x0087   # Ring size

# Response flags
GFX_FLAG_RESPONSE = 0x80000000
GFX_CMD_STATUS_MASK = 0x0000FFFF
GFX_CMD_RESPONSE_MASK = 0x8000FFFF

# Bootloader commands
PSP_BL__LOAD_KEY_DATABASE = 0x80000
PSP_BL__LOAD_TOS_SPL_TABLE = 0x10000000
PSP_BL__LOAD_SYSDRV = 0x10000
PSP_BL__LOAD_SOCDRV = 0xB0000
PSP_BL__LOAD_INTFDRV = 0xD0000
PSP_BL__LOAD_HADDRV = 0xC0000   # Debug/HAD driver
PSP_BL__LOAD_RASDRV = 0xE0000
PSP_BL__LOAD_IPKEYMGRDRV = 0xF0000
PSP_BL__LOAD_SOSDRV = 0x20000

# PSP ring type
PSP_RING_TYPE_KM = 2

# PSP ring commands
GFX_CMD_ID_LOAD_IP_FW = 0x00006
GFX_CMD_ID_AUTOLOAD_RLC = 0x00017

# PSP ring size
PSP_RING_SIZE = 0x10000  # 64KB

# GPCOM ring entry size
PSP_RING_ENTRY_SIZE = 16  # 4 DWORDs per entry


# ============================================================================
# PSP firmware types
# ============================================================================

class PSPFWType(IntEnum):
    """PSP firmware type IDs for the ring command."""
    PSP_SOS = 1
    PSP_SYS_DRV = 2
    PSP_KDB = 3
    PSP_TOC = 4
    PSP_SPL = 5
    PSP_RL = 6
    PSP_SOC_DRV = 7
    PSP_INTF_DRV = 8
    PSP_DBG_DRV = 9
    PSP_RAS_DRV = 10
    PSP_IPKEYMGR_DRV = 11


class UCODEType(IntEnum):
    """GPU IP firmware type IDs for PSP ring load commands."""
    SDMA0 = 0
    SDMA1 = 1
    CP_CE = 2
    CP_PFP = 3
    CP_ME = 4
    CP_MEC1 = 5
    CP_MEC2 = 6
    RLC_G = 7
    RLC_V = 8
    RLC_RESTORE_LIST_CNTL = 9
    RLC_RESTORE_LIST_GPM_MEM = 10
    RLC_RESTORE_LIST_SRM_MEM = 11
    SMC = 12
    UVD = 13
    UVD1 = 14
    VCE = 15
    ISP = 16
    DMCU_ERAM = 17
    DMCU_INTV = 18
    VCN0_RAM = 19
    VCN1_RAM = 20
    DMCUB = 21
    VPE = 22
    UMSCH_MM_UCODE = 23
    UMSCH_MM_DATA = 24
    UMSCH_MM_CMD_BUFFER = 25
    P2S_TABLE = 26
    RLC_P = 27
    RLC_IRAM = 28
    RLC_DRAM = 29
    RS64_ME = 30
    RS64_ME_P0_DATA = 31
    RS64_ME_P1_DATA = 32
    RS64_PFP = 33
    RS64_PFP_P0_DATA = 34
    RS64_PFP_P1_DATA = 35
    RS64_MEC = 36
    RS64_MEC_P0_DATA = 37
    RS64_MEC_P1_DATA = 38
    RS64_MEC_P2_DATA = 39
    RS64_MEC_P3_DATA = 40
    MES = 41
    MES_KIQ = 42
    MES_STACK = 43
    MES_THREAD1_STACK = 44
    MES_KIQ_STACK = 45
    RLC_SRM_DRAM_SR = 46
    IMU_I = 47
    IMU_D = 48
    CP_RS64_PFP_P0_STACK = 49
    CP_RS64_PFP_P1_STACK = 50
    CP_RS64_ME_P0_STACK = 51
    CP_RS64_ME_P1_STACK = 52
    CP_RS64_MEC_P0_STACK = 53
    CP_RS64_MEC_P1_STACK = 54
    CP_RS64_MEC_P2_STACK = 55
    CP_RS64_MEC_P3_STACK = 56
    RLC_AUTOLOAD = 57
    MES_DATA = 58
    MES_KIQ_DATA = 59
    UMSCH_MM_DATA_UCODE = 60
    ISP_DATA = 61


# ============================================================================
# Firmware binary parsing
# ============================================================================

@dataclass
class FirmwareHeader:
    """Common firmware header from amdgpu_ucode.h."""
    size_bytes: int
    header_size_bytes: int
    header_version_major: int
    header_version_minor: int
    ip_version_major: int
    ip_version_minor: int
    ucode_version: int
    ucode_size_bytes: int
    ucode_array_offset_bytes: int
    crc32: int


@dataclass
class FWBinDesc:
    """Firmware binary descriptor (v2.0 format)."""
    fw_type: int
    fw_version: int
    offset_bytes: int
    size_bytes: int


@dataclass
class FirmwareBundle:
    """Parsed firmware file with sub-firmware descriptors."""
    header: FirmwareHeader
    raw_data: bytes
    sub_fws: list[FWBinDesc] = field(default_factory=list)

    def get_sub_fw(self, fw_type: int) -> tuple[bytes, FWBinDesc] | None:
        """Get sub-firmware data and descriptor by type."""
        for desc in self.sub_fws:
            if desc.fw_type == fw_type:
                start = self.header.ucode_array_offset_bytes + desc.offset_bytes
                end = start + desc.size_bytes
                return self.raw_data[start:end], desc
        return None


def parse_firmware_header(data: bytes) -> FirmwareHeader:
    """Parse a common firmware header."""
    fields = struct.unpack_from("<IIHHHHIIIi", data, 0)
    return FirmwareHeader(
        size_bytes=fields[0],
        header_size_bytes=fields[1],
        header_version_major=fields[2],
        header_version_minor=fields[3],
        ip_version_major=fields[4],
        ip_version_minor=fields[5],
        ucode_version=fields[6],
        ucode_size_bytes=fields[7],
        ucode_array_offset_bytes=fields[8],
        crc32=fields[9],
    )


def parse_firmware_file(path: Path) -> FirmwareBundle:
    """Parse a firmware file (SOS, TA, or IP firmware)."""
    data = path.read_bytes()
    header = parse_firmware_header(data)
    bundle = FirmwareBundle(header=header, raw_data=data)

    # v2.0 format has sub-firmware descriptors
    if header.header_version_major >= 2:
        # After the common header (40 bytes), there's a count
        count_offset = 40  # sizeof(common_firmware_header)
        fw_count = struct.unpack_from("<I", data, count_offset)[0]

        # Sub-firmware descriptors follow
        desc_offset = count_offset + 4
        for i in range(fw_count):
            off = desc_offset + i * 16
            fw_type, fw_version, fw_offset, fw_size = \
                struct.unpack_from("<IIII", data, off)
            bundle.sub_fws.append(FWBinDesc(
                fw_type=fw_type,
                fw_version=fw_version,
                offset_bytes=fw_offset,
                size_bytes=fw_size,
            ))

    return bundle


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PSPConfig:
    """PSP configuration and state."""
    # Base addresses from IP discovery (MP0 = PSP)
    mp0_base: list[int]  # DWORD base addresses

    # PSP ring
    ring_bus_addr: int = 0
    ring_cpu_addr: int = 0
    ring_dma_handle: int = 0
    ring_size: int = PSP_RING_SIZE
    ring_wptr: int = 0

    # Firmware buffer (1MB DMA for loading firmware to PSP)
    fw_buf_bus_addr: int = 0
    fw_buf_cpu_addr: int = 0
    fw_buf_dma_handle: int = 0

    # State
    sos_alive: bool = False
    ring_created: bool = False

    # Firmware directory
    fw_dir: Path = Path(".")


# ============================================================================
# Register access
# ============================================================================

def _psp_reg(dev: WindowsDevice, config: PSPConfig, reg: int) -> int:
    """Read a PSP register."""
    offset = (config.mp0_base[0] + reg) * 4
    return dev.read_reg32(offset)


def _psp_wreg(dev: WindowsDevice, config: PSPConfig, reg: int, val: int) -> None:
    """Write a PSP register."""
    offset = (config.mp0_base[0] + reg) * 4
    dev.write_reg32(offset, val)


def _wait_for_reg(
    dev: WindowsDevice, config: PSPConfig,
    reg: int, expected: int, mask: int,
    timeout_ms: int = 10000
) -> bool:
    """Poll a register until (value & mask) == expected."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        val = _psp_reg(dev, config, reg)
        if (val & mask) == expected:
            return True
        time.sleep(0.001)  # 1ms between polls
    return False


# ============================================================================
# PSP initialization
# ============================================================================

def resolve_psp_bases(ip_result: IPDiscoveryResult) -> PSPConfig:
    """Resolve PSP (MP0) base addresses from IP discovery."""
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    bases = [0] * 6

    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.MP0 and block.instance_number == 0:
            for i, addr in enumerate(block.base_addresses):
                if i < len(bases) and addr != 0:
                    bases[i] = addr

    return PSPConfig(mp0_base=bases)


def is_sos_alive(dev: WindowsDevice, config: PSPConfig) -> bool:
    """Check if the PSP SOS (Secure OS) is already running.

    After VBIOS POST, SOS should already be alive. This reads the
    Sign of Life register (C2PMSG_81).
    """
    sol = _psp_reg(dev, config, regMPASP_SMN_C2PMSG_81)
    return sol != 0


def wait_for_bootloader(dev: WindowsDevice, config: PSPConfig) -> bool:
    """Wait for the PSP bootloader to be ready (bit 31 of C2PMSG_35)."""
    return _wait_for_reg(dev, config, regMPASP_SMN_C2PMSG_35,
                         0x80000000, 0x80000000, timeout_ms=10000)


def create_psp_ring(dev: WindowsDevice, config: PSPConfig) -> None:
    """Create the PSP GPCOM ring for firmware command submission.

    Protocol:
    1. Wait for TOS ready (C2PMSG_64 has response + status=0)
    2. Write ring address and size
    3. Send ring create command
    4. Wait for completion

    Reference: psp_v14_0_ring_create()
    """
    # Wait for TOS ready
    if not _wait_for_reg(dev, config, regMPASP_SMN_C2PMSG_64,
                         GFX_FLAG_RESPONSE, GFX_CMD_RESPONSE_MASK,
                         timeout_ms=20000):
        raise RuntimeError("PSP TOS not ready (C2PMSG_64 timeout)")

    # Write ring address (low 32 bits, high 32 bits)
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_69,
              config.ring_bus_addr & 0xFFFFFFFF)
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_70,
              (config.ring_bus_addr >> 32) & 0xFFFFFFFF)

    # Write ring size
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_71, config.ring_size)

    # Send ring create command (ring_type << 16)
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_64,
              PSP_RING_TYPE_KM << 16)

    # Wait 20ms for hardware handshake
    time.sleep(0.020)

    # Wait for response
    if not _wait_for_reg(dev, config, regMPASP_SMN_C2PMSG_64,
                         GFX_FLAG_RESPONSE, GFX_CMD_RESPONSE_MASK,
                         timeout_ms=20000):
        raise RuntimeError("PSP ring creation failed (C2PMSG_64 timeout)")

    config.ring_created = True
    config.ring_wptr = _psp_reg(dev, config, regMPASP_SMN_C2PMSG_67)


def submit_psp_cmd(
    dev: WindowsDevice,
    config: PSPConfig,
    cmd_id: int,
    fw_type: int = 0,
    fw_bus_addr: int = 0,
    fw_size: int = 0,
) -> None:
    """Submit a command to the PSP GPCOM ring.

    The ring entry format is 4 DWORDs:
    DW[0]: fence_addr_lo (or 0)
    DW[1]: fence_addr_hi (or 0)
    DW[2]: fence_value (or 0)
    DW[3]: cmd_id | (fw_type << 16)

    For LOAD_IP_FW commands, additional data is written to the
    firmware buffer and the address is passed via C2PMSG_36.
    """
    import ctypes

    # Write ring entry at current wptr
    entry_offset = (config.ring_wptr * PSP_RING_ENTRY_SIZE) % config.ring_size
    entry_addr = config.ring_cpu_addr + entry_offset

    # Build entry: fence_lo, fence_hi, fence_val, cmd
    cmd_word = cmd_id | (fw_type << 16)

    # Write the 4 DWORDs to the ring buffer
    buf = (ctypes.c_uint32 * 4).from_address(entry_addr)
    buf[0] = 0  # fence_addr_lo
    buf[1] = 0  # fence_addr_hi
    buf[2] = 0  # fence_value
    buf[3] = cmd_word

    # Advance write pointer
    config.ring_wptr += 1

    # Update write pointer register
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_67, config.ring_wptr)

    # Wait for response
    if not _wait_for_reg(dev, config, regMPASP_SMN_C2PMSG_64,
                         GFX_FLAG_RESPONSE, GFX_CMD_RESPONSE_MASK,
                         timeout_ms=30000):
        raise RuntimeError(
            f"PSP command 0x{cmd_id:X} (fw_type={fw_type}) timed out"
        )


def load_ip_firmware(
    dev: WindowsDevice,
    config: PSPConfig,
    ucode_type: int,
    fw_data: bytes,
) -> None:
    """Load an IP firmware blob via the PSP ring.

    Copies firmware data to the DMA firmware buffer, then submits
    a LOAD_IP_FW command through the PSP ring.
    """
    import ctypes

    if len(fw_data) > 1024 * 1024:
        raise ValueError(f"Firmware too large: {len(fw_data)} bytes (max 1MB)")

    # Copy firmware data to the DMA buffer
    ctypes.memmove(config.fw_buf_cpu_addr, fw_data, len(fw_data))

    # Write firmware address to C2PMSG_36 (>> 20 for 1MB alignment)
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_36,
              (config.fw_buf_bus_addr >> 20) & 0xFFFFFFFF)

    # Submit LOAD_IP_FW command
    submit_psp_cmd(
        dev, config,
        cmd_id=GFX_CMD_ID_LOAD_IP_FW,
        fw_type=ucode_type,
        fw_bus_addr=config.fw_buf_bus_addr,
        fw_size=len(fw_data),
    )


def trigger_rlc_autoload(dev: WindowsDevice, config: PSPConfig) -> None:
    """Trigger RLC autoload after all firmware is loaded.

    This tells the PSP to instruct the RLC to auto-initialize
    all IP blocks using the loaded firmware.
    """
    submit_psp_cmd(dev, config, cmd_id=GFX_CMD_ID_AUTOLOAD_RLC)


# ============================================================================
# Top-level initialization
# ============================================================================

def init_psp(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    fw_dir: Path | str = Path("."),
) -> PSPConfig:
    """Initialize the PSP and prepare for firmware loading.

    On a POST'd GPU, this:
    1. Checks that SOS is already alive
    2. Allocates PSP ring and firmware buffer
    3. Creates the PSP GPCOM ring

    Does NOT load firmware — that's done by load_all_firmware().

    Args:
        dev: Windows device backend.
        ip_result: Parsed IP discovery data.
        fw_dir: Directory containing firmware .bin files.

    Returns:
        PSPConfig ready for firmware loading.
    """
    config = resolve_psp_bases(ip_result)
    config.fw_dir = Path(fw_dir)

    # Check if SOS is already alive (should be after VBIOS POST)
    config.sos_alive = is_sos_alive(dev, config)
    if not config.sos_alive:
        raise RuntimeError(
            "PSP SOS is not alive. VBIOS POST may have failed, "
            "or the GPU needs a full firmware load (not supported "
            "in this lightweight driver)."
        )

    print("  PSP: SOS is alive (POST'd by VBIOS)")

    # Allocate PSP ring buffer
    ring_cpu, ring_bus, ring_handle = dev.driver.alloc_dma(config.ring_size)
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle

    # Allocate firmware staging buffer (1MB)
    fw_cpu, fw_bus, fw_handle = dev.driver.alloc_dma(1024 * 1024)
    config.fw_buf_cpu_addr = fw_cpu
    config.fw_buf_bus_addr = fw_bus
    config.fw_buf_dma_handle = fw_handle

    # Create PSP GPCOM ring
    create_psp_ring(dev, config)
    print("  PSP: GPCOM ring created")

    return config


def load_all_firmware(
    dev: WindowsDevice,
    config: PSPConfig,
    ip_version: str = "14_0_2",
) -> None:
    """Load all required IP firmware via the PSP ring.

    Loads firmware in the correct order for RDNA4 with RLC autoload:
    1. SMU firmware
    2. SDMA firmware
    3. GFX firmware (PFP, ME, MEC)
    4. RLC firmware (triggers autoload)
    5. MES firmware

    Args:
        dev: Windows device backend.
        config: PSP config from init_psp().
        ip_version: IP version string for firmware file names.
    """
    fw_dir = config.fw_dir

    # Determine GC IP version for firmware files
    # GFX1201 = GC 12.0.1
    gc_version = "12_0_1"

    # Load SMU firmware first (required for autoload)
    smu_path = fw_dir / f"smu_{ip_version}.bin"
    if smu_path.exists():
        smu_fw = smu_path.read_bytes()
        load_ip_firmware(dev, config, UCODEType.SMC, smu_fw)
        print(f"  PSP: Loaded SMU firmware ({len(smu_fw)} bytes)")

    # Load SDMA firmware
    sdma_path = fw_dir / f"sdma_{gc_version}.bin"
    if sdma_path.exists():
        sdma_fw = sdma_path.read_bytes()
        load_ip_firmware(dev, config, UCODEType.SDMA0, sdma_fw)
        print(f"  PSP: Loaded SDMA firmware ({len(sdma_fw)} bytes)")

    # Load GFX firmware components
    for ucode_name, ucode_type in [
        ("pfp", UCODEType.RS64_PFP),
        ("me", UCODEType.RS64_ME),
        ("mec", UCODEType.RS64_MEC),
    ]:
        path = fw_dir / f"gc_{gc_version}_{ucode_name}.bin"
        if path.exists():
            fw_data = path.read_bytes()
            load_ip_firmware(dev, config, ucode_type, fw_data)
            print(f"  PSP: Loaded {ucode_name.upper()} firmware ({len(fw_data)} bytes)")

    # Load RLC firmware (this is the one that triggers autoload)
    rlc_path = fw_dir / f"gc_{gc_version}_rlc.bin"
    if rlc_path.exists():
        rlc_fw = rlc_path.read_bytes()
        load_ip_firmware(dev, config, UCODEType.RLC_G, rlc_fw)
        print(f"  PSP: Loaded RLC firmware ({len(rlc_fw)} bytes)")

        # Trigger RLC autoload after RLC_G is loaded
        trigger_rlc_autoload(dev, config)
        print("  PSP: RLC autoload triggered")

    # Load MES firmware
    for mes_name, mes_type in [
        ("mes", UCODEType.MES),
        ("mes1", UCODEType.MES_KIQ),
    ]:
        path = fw_dir / f"gc_{gc_version}_{mes_name}.bin"
        if path.exists():
            mes_fw = path.read_bytes()
            load_ip_firmware(dev, config, mes_type, mes_fw)
            print(f"  PSP: Loaded {mes_name.upper()} firmware ({len(mes_fw)} bytes)")

    print("  PSP: All firmware loaded")
