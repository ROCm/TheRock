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

import ctypes
import os
import subprocess
import struct
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
GFX_CMD_ID_AUTOLOAD_RLC = 0x00021
GFX_CMD_BUF_VERSION = 1
GFX_CMD_RESP_SIZE = 1024
PSP_FENCE_BUFFER_SIZE = 4096

# PSP ring size
PSP_RING_SIZE = 0x10000  # 64KB

# GPCOM ring frame size. The write pointer is tracked in DWORDs.
PSP_RING_FRAME_SIZE = 64


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


class GFXFWType(IntEnum):
    """PSP GFX firmware type IDs for GFX_CMD_ID_LOAD_IP_FW."""
    RLC_V = 7
    RLC_G = 8
    SDMA0 = 9
    SDMA1 = 10
    SMU = 18
    RLC_RESTORE_LIST_GPM_MEM = 20
    RLC_RESTORE_LIST_SRM_MEM = 21
    RLC_RESTORE_LIST_SRM_CNTL = 22
    RLC_P = 25
    RLC_IRAM = 26
    RLC_DRAM_BOOT = 48
    IMU_I = 68
    IMU_D = 69
    SDMA_UCODE_TH0 = 71
    SDMA_UCODE_TH1 = 72
    RS64_MES = 76
    RS64_MES_STACK = 77
    RS64_KIQ = 78
    RS64_KIQ_STACK = 79
    CP_MES_KIQ = 81
    MES_KIQ_STACK = 82
    RS64_PFP = 87
    RS64_ME = 88
    RS64_MEC = 89
    RS64_PFP_P0_STACK = 90
    RS64_PFP_P1_STACK = 91
    RS64_ME_P0_STACK = 92
    RS64_ME_P1_STACK = 93
    RS64_MEC_P0_STACK = 94
    RS64_MEC_P1_STACK = 95
    RS64_MEC_P2_STACK = 96
    RS64_MEC_P3_STACK = 97


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
    fields = struct.unpack_from("<IIHHHHIIII", data, 0)
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


def _read_firmware(path: Path) -> bytes:
    """Read a firmware file, including linux-firmware .zst files."""
    if path.exists():
        return path.read_bytes()

    zst_path = Path(str(path) + ".zst")
    if zst_path.exists():
        result = subprocess.run(
            ["zstdcat", str(zst_path)],
            check=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout

    raise FileNotFoundError(path)


def _optional_firmware(path: Path) -> bytes | None:
    try:
        return _read_firmware(path)
    except FileNotFoundError:
        return None


def _slice(data: bytes, offset: int, size: int) -> bytes:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(
            f"Firmware slice outside blob: offset={offset}, size={size}, "
            f"blob={len(data)}"
        )
    return data[offset:offset + size]


def _version_string(major: int, minor: int, revision: int) -> str:
    return f"{major}_{minor}_{revision}"


def parse_firmware_file(path: Path) -> FirmwareBundle:
    """Parse a firmware file (SOS, TA, or IP firmware)."""
    data = _read_firmware(path)
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

    # Command and fence buffers for the PSP GPCOM ring command-buffer ABI
    cmd_buf_bus_addr: int = 0
    cmd_buf_cpu_addr: int = 0
    cmd_buf_dma_handle: int = 0
    fence_bus_addr: int = 0
    fence_cpu_addr: int = 0
    fence_dma_handle: int = 0
    fence_value: int = 0

    # Backing memory handles for VRAM allocations used by amdgpu_lite.
    ring_mem_handle: Any | None = None
    fw_buf_mem_handle: Any | None = None
    cmd_buf_mem_handle: Any | None = None
    fence_mem_handle: Any | None = None

    # State
    sos_alive: bool = False
    ring_created: bool = False

    # Firmware directory
    fw_dir: Path = Path(".")

    # Firmware versions and entry points resolved from IP discovery/files.
    ip_versions: dict[str, str] = field(default_factory=dict)
    ucode_start: dict[str, int] = field(default_factory=dict)


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


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _alloc_psp_buffer(
    dev: WindowsDevice,
    size: int,
    *,
    alignment: int = 4096,
) -> tuple[int, int, int, Any | None]:
    """Allocate a PSP-visible buffer.

    amdgpu_lite uses VRAM MC addresses for PSP-visible boot buffers, which
    matches the macOS/Tinygrad path. Other backends keep the old DMA fallback.
    """
    alloc_size = _align_up(size + alignment - 1, 4096)

    if hasattr(dev, "read_vram") and hasattr(dev, "alloc_memory"):
        from amd_gpu_driver.backends.base import MemoryLocation

        handle = dev.alloc_memory(alloc_size, MemoryLocation.VRAM)
        if handle.cpu_addr == 0:
            raise RuntimeError("PSP VRAM allocation is not CPU mapped")
        delta = (-handle.gpu_addr) & (alignment - 1)
        ctypes.memset(handle.cpu_addr, 0, handle.size)
        return (
            handle.cpu_addr + delta,
            handle.gpu_addr + delta,
            0,
            handle,
        )

    cpu_addr, bus_addr, dma_handle = dev.driver.alloc_dma(alloc_size)
    delta = (-bus_addr) & (alignment - 1)
    ctypes.memset(cpu_addr, 0, alloc_size)
    return cpu_addr + delta, bus_addr + delta, dma_handle, None


def _resolve_ip_versions(ip_result: IPDiscoveryResult) -> dict[str, str]:
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    versions: dict[str, str] = {}
    for block in ip_result.ip_blocks:
        version = _version_string(block.major, block.minor, block.revision)
        if block.hw_id == HardwareID.GC and block.instance_number == 0:
            versions["gc"] = version
        elif block.hw_id == HardwareID.SDMA0 and "sdma" not in versions:
            versions["sdma"] = version
        elif block.hw_id == HardwareID.MP1 and block.instance_number == 0:
            versions["mp1"] = version
        elif block.hw_id == HardwareID.MP0 and block.instance_number == 0:
            versions["mp0"] = version
    return versions


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
    if _psp_reg(dev, config, regMPASP_SMN_C2PMSG_71) != 0:
        # Destroy any stale ring left by an earlier userspace process.
        status = _psp_reg(dev, config, regMPASP_SMN_C2PMSG_64)
        if (status & GFX_FLAG_RESPONSE) == 0:
            raise RuntimeError(
                f"PSP TOS has a stale busy ring (C2PMSG_64=0x{status:08X})"
            )
        _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_64, 0x00030000)
        time.sleep(0.020)

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
    if os.environ.get("AMDGPU_LITE_PSP_CMD_BUFFER") == "1":
        _submit_psp_cmd_buffer(dev, config, cmd_id, fw_type, fw_bus_addr,
                               fw_size)
    else:
        _submit_psp_cmd_legacy(dev, config, cmd_id, fw_type, fw_bus_addr,
                               fw_size)


def _submit_psp_cmd_buffer(
    dev: WindowsDevice,
    config: PSPConfig,
    cmd_id: int,
    fw_type: int = 0,
    fw_bus_addr: int = 0,
    fw_size: int = 0,
) -> None:
    """Submit a command to the PSP GPCOM command-buffer ring."""
    if config.cmd_buf_cpu_addr == 0 or config.fence_cpu_addr == 0:
        raise RuntimeError("PSP command/fence buffers are not allocated")

    ctypes.memset(config.cmd_buf_cpu_addr, 0, GFX_CMD_RESP_SIZE)
    ctypes.memset(config.fence_cpu_addr, 0, 4)

    cmd = (ctypes.c_uint32 * (GFX_CMD_RESP_SIZE // 4)).from_address(
        config.cmd_buf_cpu_addr
    )
    cmd[0] = GFX_CMD_RESP_SIZE
    cmd[1] = GFX_CMD_BUF_VERSION
    cmd[2] = cmd_id

    if cmd_id == GFX_CMD_ID_LOAD_IP_FW:
        cmd[7] = fw_bus_addr & 0xFFFFFFFF
        cmd[8] = (fw_bus_addr >> 32) & 0xFFFFFFFF
        cmd[9] = fw_size
        cmd[10] = fw_type

    config.fence_value += 1
    fence = (ctypes.c_uint32).from_address(config.fence_cpu_addr)
    fence.value = 0

    ring_wptr = _psp_reg(dev, config, regMPASP_SMN_C2PMSG_67)
    frame_offset = (ring_wptr * 4) % config.ring_size
    frame = (ctypes.c_uint32 * (PSP_RING_FRAME_SIZE // 4)).from_address(
        config.ring_cpu_addr + frame_offset
    )
    for i in range(PSP_RING_FRAME_SIZE // 4):
        frame[i] = 0
    frame[0] = config.cmd_buf_bus_addr & 0xFFFFFFFF
    frame[1] = (config.cmd_buf_bus_addr >> 32) & 0xFFFFFFFF
    frame[2] = GFX_CMD_RESP_SIZE
    frame[3] = config.fence_bus_addr & 0xFFFFFFFF
    frame[4] = (config.fence_bus_addr >> 32) & 0xFFFFFFFF
    frame[5] = config.fence_value

    config.ring_wptr = ring_wptr + (PSP_RING_FRAME_SIZE // 4)
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_67, config.ring_wptr)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if fence.value == config.fence_value:
            status = cmd[216]  # struct psp_gfx_cmd_resp.resp.status
            if status != 0:
                raise RuntimeError(
                    f"PSP command 0x{cmd_id:X} fw_type={fw_type} "
                    f"failed with status 0x{status:X}"
                )
            return
        time.sleep(0.001)

    raise RuntimeError(
        f"PSP command 0x{cmd_id:X} fw_type={fw_type} timed out "
        f"(fence={fence.value}, expected={config.fence_value})"
    )


def _submit_psp_cmd_legacy(
    dev: WindowsDevice,
    config: PSPConfig,
    cmd_id: int,
    fw_type: int = 0,
    fw_bus_addr: int = 0,
    fw_size: int = 0,
) -> None:
    """Submit a command using the 16-byte GPCOM ring frame.

    This is the protocol used by the current lite Windows path and by the
    PSP state we inherit after firmware POST in the Linux VM.
    """
    if config.fence_cpu_addr == 0:
        raise RuntimeError("PSP fence buffer is not allocated")

    config.fence_value += 1
    fence = (ctypes.c_uint32).from_address(config.fence_cpu_addr)
    fence.value = 0

    if cmd_id == GFX_CMD_ID_LOAD_IP_FW:
        _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_36,
                  (fw_bus_addr >> 20) & 0xFFFFFFFF)

    entry_offset = (config.ring_wptr * 16) % config.ring_size
    entry = (ctypes.c_uint32 * 4).from_address(
        config.ring_cpu_addr + entry_offset
    )
    entry[0] = config.fence_bus_addr & 0xFFFFFFFF
    entry[1] = (config.fence_bus_addr >> 32) & 0xFFFFFFFF
    entry[2] = config.fence_value
    entry[3] = cmd_id | (fw_type << 16)

    config.ring_wptr += 1
    _psp_wreg(dev, config, regMPASP_SMN_C2PMSG_67, config.ring_wptr)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if fence.value == config.fence_value:
            return
        time.sleep(0.001)

    raise RuntimeError(
        f"PSP legacy command 0x{cmd_id:X} fw_type={fw_type} timed out "
        f"(fence={fence.value}, expected={config.fence_value}, "
        f"size={fw_size})"
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


def _load_fw_desc(
    dev: WindowsDevice,
    config: PSPConfig,
    fw_type: GFXFWType,
    fw_data: bytes,
    label: str,
) -> None:
    if not fw_data:
        return
    load_ip_firmware(dev, config, int(fw_type), fw_data)
    print(f"  PSP: Loaded {label} ({len(fw_data)} bytes, type={int(fw_type)})")


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _load_smu_firmware(dev: WindowsDevice, config: PSPConfig, path: Path) -> bool:
    data = _optional_firmware(path)
    if data is None:
        return False
    header = parse_firmware_header(data)
    _load_fw_desc(
        dev,
        config,
        GFXFWType.SMU,
        _slice(data, header.ucode_array_offset_bytes, header.ucode_size_bytes),
        path.name,
    )
    return True


def _load_sdma_firmware(dev: WindowsDevice, config: PSPConfig, path: Path) -> bool:
    data = _optional_firmware(path)
    if data is None:
        return False
    header = parse_firmware_header(data)

    if header.header_version_major == 1:
        fw = _slice(data, header.ucode_array_offset_bytes,
                    header.ucode_size_bytes)
        _load_fw_desc(dev, config, GFXFWType.SDMA0, fw, f"{path.name}:SDMA0")
        _load_fw_desc(dev, config, GFXFWType.SDMA1, fw, f"{path.name}:SDMA1")
    elif header.header_version_major == 2:
        ctx_size = _u32(data, 36)
        ctl_offset = _u32(data, 48)
        ctl_size = _u32(data, 52)
        _load_fw_desc(
            dev, config, GFXFWType.SDMA_UCODE_TH1,
            _slice(data, ctl_offset, ctl_size), f"{path.name}:TH1"
        )
        _load_fw_desc(
            dev, config, GFXFWType.SDMA_UCODE_TH0,
            _slice(data, header.ucode_array_offset_bytes, ctx_size),
            f"{path.name}:TH0",
        )
    else:
        ucode_offset = _u32(data, 36)
        ucode_size = _u32(data, 40)
        _load_fw_desc(
            dev, config, GFXFWType.SDMA_UCODE_TH0,
            _slice(data, ucode_offset, ucode_size), f"{path.name}:TH0"
        )
    return True


def _load_gfx_rs64_firmware(
    dev: WindowsDevice,
    config: PSPConfig,
    path: Path,
    name: str,
    code_type: GFXFWType,
    stack_type: GFXFWType,
) -> bool:
    data = _optional_firmware(path)
    if data is None:
        return False
    header = parse_firmware_header(data)
    if header.header_version_major < 2:
        raise ValueError(f"{path} is not an RS64 firmware image")

    ucode_size = _u32(data, 36)
    ucode_offset = header.ucode_array_offset_bytes or _u32(data, 40)
    data_size = _u32(data, 44)
    data_offset = _u32(data, 48)
    start_lo = _u32(data, 52)
    start_hi = _u32(data, 56)
    config.ucode_start[name.upper()] = start_lo | (start_hi << 32)

    _load_fw_desc(
        dev, config, code_type, _slice(data, ucode_offset, ucode_size),
        f"{path.name}:{name.upper()}"
    )
    _load_fw_desc(
        dev, config, stack_type, _slice(data, data_offset, data_size),
        f"{path.name}:{name.upper()}_STACK"
    )
    return True


def _load_imu_firmware(dev: WindowsDevice, config: PSPConfig, path: Path) -> bool:
    data = _optional_firmware(path)
    if data is None:
        return False
    header = parse_firmware_header(data)
    imu_i_size = _u32(data, 32)
    imu_i_offset = _u32(data, 36) or header.ucode_array_offset_bytes
    imu_d_size = _u32(data, 40)
    imu_d_offset = _u32(data, 44) or (imu_i_offset + imu_i_size)
    _load_fw_desc(
        dev, config, GFXFWType.IMU_I,
        _slice(data, imu_i_offset, imu_i_size), f"{path.name}:IMU_I"
    )
    _load_fw_desc(
        dev, config, GFXFWType.IMU_D,
        _slice(data, imu_d_offset, imu_d_size), f"{path.name}:IMU_D"
    )
    return True


def _load_rlc_firmware(dev: WindowsDevice, config: PSPConfig, path: Path) -> bool:
    data = _optional_firmware(path)
    if data is None:
        return False
    header = parse_firmware_header(data)
    minor = header.header_version_minor

    if minor >= 1 and len(data) >= 156:
        for label, fw_type, size_off, data_off in [
            ("RLC_RESTORE_LIST_SRM_CNTL",
             GFXFWType.RLC_RESTORE_LIST_SRM_CNTL, 116, 120),
            ("RLC_RESTORE_LIST_GPM_MEM",
             GFXFWType.RLC_RESTORE_LIST_GPM_MEM, 132, 136),
            ("RLC_RESTORE_LIST_SRM_MEM",
             GFXFWType.RLC_RESTORE_LIST_SRM_MEM, 148, 152),
        ]:
            size = _u32(data, size_off)
            offset = _u32(data, data_off)
            if size:
                _load_fw_desc(
                    dev, config, fw_type, _slice(data, offset, size),
                    f"{path.name}:{label}"
                )

    if minor >= 2 and len(data) >= 172:
        for label, fw_type, size_off, data_off in [
            ("RLC_IRAM", GFXFWType.RLC_IRAM, 156, 160),
            ("RLC_DRAM_BOOT", GFXFWType.RLC_DRAM_BOOT, 164, 168),
        ]:
            size = _u32(data, size_off)
            offset = _u32(data, data_off)
            if size:
                _load_fw_desc(
                    dev, config, fw_type, _slice(data, offset, size),
                    f"{path.name}:{label}"
                )

    if minor >= 3 and len(data) >= 204:
        for label, fw_type, size_off, data_off in [
            ("RLC_P", GFXFWType.RLC_P, 180, 184),
            ("RLC_V", GFXFWType.RLC_V, 196, 200),
        ]:
            size = _u32(data, size_off)
            offset = _u32(data, data_off)
            if size:
                _load_fw_desc(
                    dev, config, fw_type, _slice(data, offset, size),
                    f"{path.name}:{label}"
                )

    _load_fw_desc(
        dev, config, GFXFWType.RLC_G,
        _slice(data, header.ucode_array_offset_bytes, header.ucode_size_bytes),
        f"{path.name}:RLC_G",
    )
    return True


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
    config.ip_versions = _resolve_ip_versions(ip_result)

    # Check if SOS is already alive (should be after VBIOS POST)
    config.sos_alive = is_sos_alive(dev, config)
    if not config.sos_alive:
        raise RuntimeError(
            "PSP SOS is not alive. VBIOS POST may have failed, "
            "or the GPU needs a full firmware load (not supported "
            "in this lightweight driver)."
        )

    print("  PSP: SOS is alive (POST'd by VBIOS)")

    # Allocate PSP-visible buffers. amdgpu_lite uses VRAM MC addresses here,
    # matching the macOS userspace bring-up path.
    ring_cpu, ring_bus, ring_handle, ring_mem = _alloc_psp_buffer(
        dev, config.ring_size
    )
    config.ring_cpu_addr = ring_cpu
    config.ring_bus_addr = ring_bus
    config.ring_dma_handle = ring_handle
    config.ring_mem_handle = ring_mem

    # Allocate firmware staging buffer (1MB, PSP-aligned)
    fw_cpu, fw_bus, fw_handle, fw_mem = _alloc_psp_buffer(
        dev, 1024 * 1024, alignment=1024 * 1024
    )
    config.fw_buf_cpu_addr = fw_cpu
    config.fw_buf_bus_addr = fw_bus
    config.fw_buf_dma_handle = fw_handle
    config.fw_buf_mem_handle = fw_mem

    cmd_cpu, cmd_bus, cmd_handle, cmd_mem = _alloc_psp_buffer(
        dev, GFX_CMD_RESP_SIZE
    )
    config.cmd_buf_cpu_addr = cmd_cpu
    config.cmd_buf_bus_addr = cmd_bus
    config.cmd_buf_dma_handle = cmd_handle
    config.cmd_buf_mem_handle = cmd_mem

    fence_cpu, fence_bus, fence_handle, fence_mem = _alloc_psp_buffer(
        dev, PSP_FENCE_BUFFER_SIZE
    )
    config.fence_cpu_addr = fence_cpu
    config.fence_bus_addr = fence_bus
    config.fence_dma_handle = fence_handle
    config.fence_mem_handle = fence_mem

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
    versions = config.ip_versions
    gc_version = versions.get("gc", "12_0_1")
    sdma_version = versions.get("sdma", gc_version)
    mp1_version = versions.get("mp1", ip_version)

    loaded_any = False

    loaded_any |= _load_smu_firmware(
        dev, config, fw_dir / f"smu_{mp1_version}.bin"
    )
    loaded_any |= _load_sdma_firmware(
        dev, config, fw_dir / f"sdma_{sdma_version}.bin"
    )

    for fw_name, code_type, stack_type in [
        ("pfp", GFXFWType.RS64_PFP, GFXFWType.RS64_PFP_P0_STACK),
        ("me", GFXFWType.RS64_ME, GFXFWType.RS64_ME_P0_STACK),
        ("mec", GFXFWType.RS64_MEC, GFXFWType.RS64_MEC_P0_STACK),
    ]:
        loaded_any |= _load_gfx_rs64_firmware(
            dev,
            config,
            fw_dir / f"gc_{gc_version}_{fw_name}.bin",
            fw_name,
            code_type,
            stack_type,
        )

    loaded_any |= _load_imu_firmware(
        dev, config, fw_dir / f"gc_{gc_version}_imu.bin"
    )

    if _load_rlc_firmware(dev, config, fw_dir / f"gc_{gc_version}_rlc.bin"):
        loaded_any = True
        trigger_rlc_autoload(dev, config)
        print("  PSP: RLC autoload triggered")

    if not loaded_any:
        raise FileNotFoundError(
            f"No usable firmware found in {fw_dir} "
            f"(gc={gc_version}, sdma={sdma_version}, mp1={mp1_version})"
        )

    print("  PSP: All firmware loaded")
