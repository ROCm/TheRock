"""SMU mailbox bring-up helpers for RDNA4.

The SMU is reached through the MP1 C2PMSG mailbox.  The helpers here are
backend-neutral: they only require read_reg32/write_reg32 and, when available,
alloc_memory for the driver table.
"""

from __future__ import annotations

import ctypes
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amd_gpu_driver.backends.windows.device import WindowsDevice
    from amd_gpu_driver.backends.windows.ip_discovery import IPDiscoveryResult


# MP1 v14 C2PMSG register offsets. These are equivalent to base_index 1
# offsets 0x82/0x92/0x9a; using the base_index 0 form matches the existing
# WDDM-lite code and works with the IP-discovery base arrays on Navi48.
regMP1_SMN_C2PMSG_66 = 0x0282  # message
regMP1_SMN_C2PMSG_82 = 0x0292  # parameter / readback
regMP1_SMN_C2PMSG_90 = 0x029A  # response

PPSMC_Result_OK = 0x1
PPSMC_Result_CmdRejectedBusy = 0xFC
PPSMC_Result_CmdRejectedPrereq = 0xFD
PPSMC_Result_UnknownCmd = 0xFE
PPSMC_Result_Failed = 0xFF

PPSMC_MSG_GetSmuVersion = 0x02
ENABLE_IMU_ARG_GFXOFF_ENABLE = 1

SMU_FEATURE_DPM_GFXCLK_BIT = 1
SMU_FEATURE_GFXOFF_BIT = 18
SMU_FEATURE_GFX_IMU_BIT = 36
SMU_FEATURE_PWR_GFX = 4

SMU_DRIVER_TABLE_SIZE = 0x4000

# SMU14 direct PMFW loader constants. Linux amdgpu uses WREG32_PCIE to copy
# smu_14_0_x.bin into MP1 SRAM for non-PSP firmware load types.
MP1_SRAM = 0x03C00004
MP1_PUBLIC = 0x03B00000
smnMP1_FIRMWARE_FLAGS = 0x03010024
smnMP1_FIRMWARE_FLAGS_14_0_0 = 0x03010028
smnMP1_PUB_CTRL = 0x03010D10
MP1_FIRMWARE_INTERRUPTS_ENABLED = 0x1

# Navi48/NBIO v7.11 PCIE_INDEX2/DATA2 offsets. These are byte offsets into
# BAR0 because NBIF base index 0 is zero on the lite path.
regPCIE_INDEX2_BYTE = 0x000E * 4
regPCIE_DATA2_BYTE = 0x000F * 4


@dataclass(frozen=True)
class SMUMessageMap:
    """ASIC-specific PMFW message IDs used by the lite bring-up path."""

    set_driver_dram_addr_high: int
    set_driver_dram_addr_low: int
    set_allowed_features_mask_low: int | None
    set_allowed_features_mask_high: int | None
    enable_all_smu_features: int | None
    enable_smu_features_low: int | None
    enable_smu_features_high: int | None
    get_enabled_smu_features_low: int
    get_enabled_smu_features_high: int | None
    allow_gfxoff: int
    disallow_gfxoff: int
    enable_gfx_imu: int | None
    name: str


SMU14_0_0_MESSAGES = SMUMessageMap(
    set_driver_dram_addr_high=0x0D,
    set_driver_dram_addr_low=0x0E,
    set_allowed_features_mask_low=None,
    set_allowed_features_mask_high=None,
    enable_all_smu_features=None,
    enable_smu_features_low=None,
    enable_smu_features_high=None,
    get_enabled_smu_features_low=0x12,
    get_enabled_smu_features_high=None,
    allow_gfxoff=0x19,
    disallow_gfxoff=0x1A,
    enable_gfx_imu=0x16,
    name="smu14_0_0",
)

SMU14_0_2_MESSAGES = SMUMessageMap(
    set_driver_dram_addr_high=0x0E,
    set_driver_dram_addr_low=0x0F,
    set_allowed_features_mask_low=0x04,
    set_allowed_features_mask_high=0x05,
    enable_all_smu_features=0x06,
    enable_smu_features_low=0x08,
    enable_smu_features_high=0x09,
    get_enabled_smu_features_low=0x0C,
    get_enabled_smu_features_high=0x0D,
    allow_gfxoff=0x28,
    disallow_gfxoff=0x29,
    enable_gfx_imu=None,
    name="smu14_0_2",
)


@dataclass
class SMUConfig:
    """Resolved SMU mailbox state."""

    mp1_base: list[int]
    mp1_version: tuple[int, int, int] = (0, 0, 0)
    messages: SMUMessageMap = SMU14_0_0_MESSAGES
    driver_table_bus_addr: int = 0
    driver_table_cpu_addr: int = 0
    driver_table_handle: Any | None = None
    vram_mc_base: int = 0
    version: int = 0


def _message_map_for_mp1(major: int, minor: int, revision: int) -> SMUMessageMap:
    if major == 14 and minor == 0 and revision >= 2:
        return SMU14_0_2_MESSAGES
    return SMU14_0_0_MESSAGES


def resolve_smu_bases(ip_result: IPDiscoveryResult) -> SMUConfig:
    """Resolve MP1 base addresses from IP discovery."""
    from amd_gpu_driver.backends.windows.ip_discovery import HardwareID

    bases = [0] * 16
    mp1_version = (0, 0, 0)
    for block in ip_result.ip_blocks:
        if block.hw_id == HardwareID.MP1 and block.instance_number == 0:
            mp1_version = (int(block.major), int(block.minor), int(block.revision))
            for i, addr in enumerate(block.base_addresses):
                if i < len(bases) and addr != 0:
                    bases[i] = addr
    if bases[0] == 0:
        raise RuntimeError("SMU MP1 base not found in IP discovery")
    return SMUConfig(
        mp1_base=bases,
        mp1_version=mp1_version,
        messages=_message_map_for_mp1(*mp1_version),
    )


def _smu_reg(dev: WindowsDevice, config: SMUConfig, reg: int) -> int:
    return dev.read_reg32((config.mp1_base[0] + reg) * 4)


def _smu_wreg(dev: WindowsDevice, config: SMUConfig, reg: int, value: int) -> None:
    dev.write_reg32((config.mp1_base[0] + reg) * 4, value)


def _pcie_rreg(dev: WindowsDevice, address: int) -> int:
    dev.write_reg32(regPCIE_INDEX2_BYTE, address & 0xFFFFFFFF)
    dev.read_reg32(regPCIE_INDEX2_BYTE)
    return dev.read_reg32(regPCIE_DATA2_BYTE)


def _pcie_wreg(dev: WindowsDevice, address: int, value: int) -> None:
    dev.write_reg32(regPCIE_INDEX2_BYTE, address & 0xFFFFFFFF)
    dev.read_reg32(regPCIE_INDEX2_BYTE)
    dev.write_reg32(regPCIE_DATA2_BYTE, value & 0xFFFFFFFF)
    dev.read_reg32(regPCIE_DATA2_BYTE)


def _smu_fw_flags(dev: WindowsDevice) -> int:
    return _pcie_rreg(dev, MP1_PUBLIC | (smnMP1_FIRMWARE_FLAGS & 0xFFFFFFFF))


def is_smu_firmware_running(dev: WindowsDevice) -> bool:
    """Return whether PMFW reports interrupts enabled via MP1 firmware flags."""
    flags = _smu_fw_flags(dev)
    return flags != 0xFFFFFFFF and (flags & MP1_FIRMWARE_INTERRUPTS_ENABLED) != 0


def load_smu_firmware_direct(
    dev: WindowsDevice,
    fw_dir: str | Path,
    mp1_version: str = "14_0_2",
    *,
    force: bool = True,
) -> bool:
    """Load SMU14 PMFW directly through MP1 SRAM, matching Linux amdgpu.

    This path is used by amdgpu_lite/RLC-backdoor bring-up. PSP
    GFX_CMD_ID_LOAD_IP_FW with GFX_FW_TYPE_SMU is a PSP-load path and Navi48
    rejects it in this mode.
    """
    if not hasattr(dev, "read_vram"):
        return False

    if os.environ.get("AMDGPU_LITE_DIRECT_SMU_LOAD", "0") == "0":
        return False

    if not force and is_smu_firmware_running(dev):
        print("  SMU: firmware already running")
        return True

    from amd_gpu_driver.backends.windows.psp_init import (
        _read_firmware,
        parse_firmware_header,
    )

    path = Path(fw_dir) / f"smu_{mp1_version}.bin"
    data = _read_firmware(path)
    header = parse_firmware_header(data)
    start = header.ucode_array_offset_bytes
    end = start + header.ucode_size_bytes
    if start <= 0 or end > len(data):
        raise ValueError(
            f"Invalid SMU firmware payload offset=0x{start:X}, "
            f"size=0x{header.ucode_size_bytes:X}, file=0x{len(data):X}"
        )

    dword_count = header.ucode_size_bytes // 4
    if dword_count < 3:
        raise ValueError(f"SMU firmware payload too small: {header.ucode_size_bytes}")

    src = struct.unpack_from(f"<{dword_count}I", data, start)
    addr = MP1_SRAM
    for value in src[1:-1]:
        _pcie_wreg(dev, addr, value)
        addr += 4

    pub_ctrl = MP1_PUBLIC | (smnMP1_PUB_CTRL & 0xFFFFFFFF)
    _pcie_wreg(dev, pub_ctrl, 1)
    _pcie_wreg(dev, pub_ctrl, 0)

    deadline = time.monotonic() + 2.0
    flags = 0
    while time.monotonic() < deadline:
        flags = _smu_fw_flags(dev)
        if flags & MP1_FIRMWARE_INTERRUPTS_ENABLED:
            print(
                f"  SMU: direct PMFW load complete "
                f"(v0x{header.ucode_version:08X}, flags=0x{flags:08X})"
            )
            return True
        time.sleep(0.001)

    raise RuntimeError(f"SMU direct PMFW load timed out (flags=0x{flags:08X})")


def _wait_for_response(
    dev: WindowsDevice,
    config: SMUConfig,
    *,
    timeout_ms: int,
) -> int:
    deadline = time.monotonic() + timeout_ms / 1000.0
    last = 0
    while time.monotonic() < deadline:
        last = _smu_reg(dev, config, regMP1_SMN_C2PMSG_90)
        if last != 0:
            return last
        time.sleep(0.001)
    return last


def send_smu_msg(
    dev: WindowsDevice,
    config: SMUConfig,
    msg: int,
    param: int = 0,
    *,
    read_back_arg: bool = False,
    timeout_ms: int = 1000,
    label: str | None = None,
) -> int | None:
    """Send one SMU mailbox message.

    Returns the readback argument when requested. Raises RuntimeError for
    timeout or non-OK SMU status.
    """
    # Match Linux smu_msg_v1's send phase: clear response, write args, and
    # ring the message register. A freshly initialized mailbox may have a zero
    # response register even though it is ready for the first command.
    _smu_wreg(dev, config, regMP1_SMN_C2PMSG_90, 0)
    _smu_wreg(dev, config, regMP1_SMN_C2PMSG_82, param)
    _smu_wreg(dev, config, regMP1_SMN_C2PMSG_66, msg)

    resp = _wait_for_response(dev, config, timeout_ms=timeout_ms)
    if resp == 0:
        what = label or f"0x{msg:X}"
        raise RuntimeError(f"SMU message {what} timed out")
    if resp != PPSMC_Result_OK:
        what = label or f"0x{msg:X}"
        raise RuntimeError(f"SMU message {what} failed with response 0x{resp:X}")

    if read_back_arg:
        return _smu_reg(dev, config, regMP1_SMN_C2PMSG_82)
    return None


def send_smu_msg_no_wait(
    dev: WindowsDevice,
    config: SMUConfig,
    msg: int,
    param: int = 0,
) -> None:
    """Send one SMU mailbox message without waiting for a response."""
    _smu_wreg(dev, config, regMP1_SMN_C2PMSG_90, 0)
    _smu_wreg(dev, config, regMP1_SMN_C2PMSG_82, param)
    _smu_wreg(dev, config, regMP1_SMN_C2PMSG_66, msg)


def enable_gfx_imu_no_wait(dev: WindowsDevice, config: SMUConfig) -> None:
    if config.messages.enable_gfx_imu is None:
        print(f"  SMU: EnableGfxImu skipped for {config.messages.name}")
        return
    send_smu_msg_no_wait(
        dev,
        config,
        config.messages.enable_gfx_imu,
        ENABLE_IMU_ARG_GFXOFF_ENABLE,
    )
    print("  SMU: EnableGfxImu sent")


def _driver_table_gpu_addr(handle: Any, config: SMUConfig) -> int:
    from amd_gpu_driver.backends.base import MemoryLocation

    gpu_addr = int(handle.gpu_addr)
    if (
        config.vram_mc_base
        and handle.location == MemoryLocation.VRAM
        and gpu_addr < config.vram_mc_base
    ):
        gpu_addr += config.vram_mc_base
    return gpu_addr


def _allocate_driver_table(dev: WindowsDevice, config: SMUConfig) -> None:
    """Allocate and publish a small SMU driver table when the backend can."""
    if not hasattr(dev, "alloc_memory"):
        return

    from amd_gpu_driver.backends.base import MemoryLocation

    handle = dev.alloc_memory(SMU_DRIVER_TABLE_SIZE, MemoryLocation.VRAM)
    if handle.cpu_addr == 0:
        if hasattr(dev, "free_memory"):
            dev.free_memory(handle)
        return
    ctypes.memset(handle.cpu_addr, 0, handle.size)
    config.driver_table_handle = handle
    config.driver_table_cpu_addr = handle.cpu_addr
    config.driver_table_bus_addr = _driver_table_gpu_addr(handle, config)

    send_smu_msg(
        dev,
        config,
        config.messages.set_driver_dram_addr_high,
        (config.driver_table_bus_addr >> 32) & 0xFFFFFFFF,
        label="SetDriverDramAddrHigh",
    )
    send_smu_msg(
        dev,
        config,
        config.messages.set_driver_dram_addr_low,
        config.driver_table_bus_addr & 0xFFFFFFFF,
        label="SetDriverDramAddrLow",
    )
    print(f"  SMU: driver table at MC 0x{config.driver_table_bus_addr:012X}")


def _try_smu_msg(
    dev: WindowsDevice,
    config: SMUConfig,
    msg: int,
    param: int = 0,
    *,
    label: str,
    timeout_ms: int = 1000,
) -> bool:
    try:
        send_smu_msg(dev, config, msg, param, timeout_ms=timeout_ms, label=label)
        print(f"  SMU: {label} OK")
        return True
    except RuntimeError as e:
        print(f"  SMU: WARNING {label} skipped - {e}")
        return False


def _feature_mask(feature_bit: int) -> tuple[int, int]:
    if feature_bit < 32:
        return 1 << feature_bit, 0
    return 0, 1 << (feature_bit - 32)


def _default_allowed_feature_masks() -> tuple[int, int]:
    gfxclk_low, gfxclk_high = _feature_mask(SMU_FEATURE_DPM_GFXCLK_BIT)
    imu_low, imu_high = _feature_mask(SMU_FEATURE_GFX_IMU_BIT)
    return gfxclk_low | imu_low, gfxclk_high | imu_high


def _env_u32(name: str, default: int) -> int:
    return int(os.environ.get(name, f"0x{default:X}"), 0) & 0xFFFFFFFF


def _enable_smu_features_via_allowed_mask(
    dev: WindowsDevice,
    config: SMUConfig,
) -> None:
    """Try the Linux PPTable-era allowed-mask sequence for diagnostics."""
    if os.environ.get("AMDGPU_LITE_ENABLE_SMU_FEATURES", "1") == "0":
        print("  SMU: feature enable skipped by AMDGPU_LITE_ENABLE_SMU_FEATURES=0")
        return

    messages = config.messages
    if (
        messages.set_allowed_features_mask_low is None
        or messages.set_allowed_features_mask_high is None
    ):
        print(f"  SMU: allowed feature mask skipped for {messages.name}")
        return

    default_low, default_high = _default_allowed_feature_masks()
    allowed_low = _env_u32("AMDGPU_LITE_SMU_ALLOWED_LOW", default_low)
    allowed_high = _env_u32("AMDGPU_LITE_SMU_ALLOWED_HIGH", default_high)

    print(
        f"  SMU: allowing features high=0x{allowed_high:08X} "
        f"low=0x{allowed_low:08X}"
    )
    if not _try_smu_msg(
        dev,
        config,
        messages.set_allowed_features_mask_high,
        allowed_high,
        label="SetAllowedFeaturesMaskHigh",
        timeout_ms=2000,
    ):
        return
    if not _try_smu_msg(
        dev,
        config,
        messages.set_allowed_features_mask_low,
        allowed_low,
        label="SetAllowedFeaturesMaskLow",
        timeout_ms=2000,
    ):
        return

    if messages.enable_all_smu_features is not None:
        _try_smu_msg(
            dev,
            config,
            messages.enable_all_smu_features,
            0,
            label="EnableAllSmuFeatures",
            timeout_ms=5000,
        )
        return

    if messages.enable_smu_features_high is not None:
        _try_smu_msg(
            dev,
            config,
            messages.enable_smu_features_high,
            allowed_high,
            label="EnableSmuFeaturesHigh",
            timeout_ms=2000,
        )
    if messages.enable_smu_features_low is not None:
        _try_smu_msg(
            dev,
            config,
            messages.enable_smu_features_low,
            allowed_low,
            label="EnableSmuFeaturesLow",
            timeout_ms=2000,
        )


def _enable_smu_features(dev: WindowsDevice, config: SMUConfig) -> None:
    """Enable PMFW features needed before GFX/IMU bring-up.

    Linux amdgpu's full SMU path programs a PPTable before SetAllowedFeaturesMask.
    On the PSP/TMR-managed lite path that command is rejected as a prerequisite
    failure, matching SCPM-style firmware ownership. Tinygrad's userspace path
    uses the direct EnableAllSmuFeatures mailbox after publishing the driver
    table, but Navi48 lite bring-up has shown that this can wedge the mailbox
    before GFX is available. Keep feature enablement opt-in until the full SMU
    table sequence is present.
    """
    if os.environ.get("AMDGPU_LITE_ENABLE_SMU_FEATURES", "0") == "0":
        print("  SMU: feature enable skipped by AMDGPU_LITE_ENABLE_SMU_FEATURES=0")
        return

    if os.environ.get("AMDGPU_LITE_SMU_USE_ALLOWED_MASK", "0") != "0":
        _enable_smu_features_via_allowed_mask(dev, config)
        return

    if config.messages.enable_all_smu_features is None:
        print(f"  SMU: EnableAllSmuFeatures skipped for {config.messages.name}")
        return

    default_param = (
        SMU_FEATURE_PWR_GFX
        if os.environ.get("AMDGPU_LITE_SMU_GFX_FEATURES_ONLY", "0") != "0"
        else 0
    )
    param = _env_u32("AMDGPU_LITE_ENABLE_ALL_SMU_FEATURES_PARAM", default_param)
    _try_smu_msg(
        dev,
        config,
        config.messages.enable_all_smu_features,
        param,
        label=f"EnableAllSmuFeatures(param=0x{param:X})",
        timeout_ms=5000,
    )


def init_smu(
    dev: WindowsDevice,
    ip_result: IPDiscoveryResult,
    *,
    disable_gfxoff: bool = True,
    vram_mc_base: int = 0,
) -> SMUConfig:
    """Initialize the minimal SMU state needed before GC register bring-up."""
    config = resolve_smu_bases(ip_result)
    config.vram_mc_base = vram_mc_base
    major, minor, revision = config.mp1_version
    print(
        f"  SMU: MP1 {major}.{minor}.{revision} "
        f"using {config.messages.name} messages"
    )

    try:
        version = send_smu_msg(
            dev,
            config,
            PPSMC_MSG_GetSmuVersion,
            read_back_arg=True,
            label="GetSmuVersion",
        )
        config.version = int(version or 0)
        print(f"  SMU: firmware version 0x{config.version:08X}")
    except RuntimeError as e:
        print(f"  SMU: init skipped - {e}")
        return config

    try:
        _allocate_driver_table(dev, config)
    except RuntimeError as e:
        print(f"  SMU: WARNING driver table setup skipped - {e}")

    _enable_smu_features(dev, config)

    disable_gfxoff = (
        disable_gfxoff
        and os.environ.get("AMDGPU_LITE_DISALLOW_GFXOFF", "1") != "0"
    )
    if disable_gfxoff:
        _try_smu_msg(
            dev,
            config,
            config.messages.disallow_gfxoff,
            0,
            label="DisallowGfxOff",
            timeout_ms=2000,
        )

    try:
        enabled_low = send_smu_msg(
            dev,
            config,
            config.messages.get_enabled_smu_features_low,
            read_back_arg=True,
            label="GetEnabledSmuFeaturesLow",
        )
        if config.messages.get_enabled_smu_features_high is None:
            print(f"  SMU: enabled features low=0x{int(enabled_low or 0):08X}")
        else:
            enabled_high = send_smu_msg(
                dev,
                config,
                config.messages.get_enabled_smu_features_high,
                read_back_arg=True,
                label="GetEnabledSmuFeaturesHigh",
            )
            print(
                f"  SMU: enabled features "
                f"high=0x{int(enabled_high or 0):08X} "
                f"low=0x{int(enabled_low or 0):08X}"
            )
    except RuntimeError:
        pass

    return config
