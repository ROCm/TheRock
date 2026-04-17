"""PSP bootloader firmware loading for gfx1201 (PSP v14.0.3).

Mirrors `drivers/gpu/drm/amd/amdgpu/psp_v14_0.c::psp_v14_0_bootloader_load_*`.
Parses the combined-SOS firmware file (psp_firmware_header_v2_0), extracts
each sub-component (KDB, SYS_DRV, SOS, SOC_DRV, INTF_DRV, DBG_DRV, RAS_DRV,
IPKEYMGR_DRV, TOC, SPL, RL) and loads them via the PSP bootloader mailbox:

  1. DMA-allocate a 1 MB staging buffer.
  2. For each component (KDB → SYS_DRV → SOC_DRV → INTF_DRV → DBG_DRV →
     RAS_DRV → IPKEYMGR_DRV → SOS):
       a. Wait for bootloader ready (C2PMSG_35 bit 31 set and
          C2PMSG_35 low bits zero = command complete).
       b. memcpy component bytes into the DMA buffer.
       c. Write (bus_addr >> 20) to C2PMSG_36.
       d. Write the component's PSP_BL__LOAD_* command to C2PMSG_35.
       e. Wait 20 ms then poll for bootloader-ready again.
  3. After SOS load, poll C2PMSG_81 (sign-of-life) for non-zero.

PSP_BL__LOAD_* command values (from `amdgpu_psp.h::enum psp_bootloader_cmd`).
PSP firmware subtypes (`enum psp_fw_type` in `amdgpu_ucode.h`).
"""

from __future__ import annotations

import ctypes
import struct
import time
from dataclasses import dataclass


# ---- PSP bootloader commands (C2PMSG_35) ----

PSP_BL__LOAD_SYSDRV        = 0x10000
PSP_BL__LOAD_SOSDRV        = 0x20000
PSP_BL__LOAD_KEY_DATABASE  = 0x80000
PSP_BL__LOAD_SOCDRV        = 0xB0000
PSP_BL__LOAD_DBGDRV        = 0xC0000
PSP_BL__LOAD_HADDRV        = PSP_BL__LOAD_DBGDRV  # renamed in psp v14
PSP_BL__LOAD_INTFDRV       = 0xD0000
PSP_BL__LOAD_RASDRV        = 0xE0000
PSP_BL__LOAD_IPKEYMGRDRV   = 0xF0000
PSP_BL__LOAD_TOS_SPL_TABLE = 0x10000000

# ---- PSP firmware sub-component types (enum psp_fw_type) ----
FW_TYPE_UNKNOWN      = 0
FW_TYPE_PSP_SOS      = 1
FW_TYPE_PSP_SYS_DRV  = 2
FW_TYPE_PSP_KDB      = 3
FW_TYPE_PSP_TOC      = 4
FW_TYPE_PSP_SPL      = 5
FW_TYPE_PSP_RL       = 6
FW_TYPE_PSP_SOC_DRV  = 7
FW_TYPE_PSP_INTF_DRV = 8
FW_TYPE_PSP_DBG_DRV  = 9
FW_TYPE_PSP_RAS_DRV  = 10
FW_TYPE_PSP_IPKEYMGR_DRV = 11

FW_TYPE_NAME = {
    1: "SOS", 2: "SYS_DRV", 3: "KDB", 4: "TOC", 5: "SPL", 6: "RL",
    7: "SOC_DRV", 8: "INTF_DRV", 9: "DBG_DRV", 10: "RAS_DRV", 11: "IPKEYMGR_DRV",
}

# ---- MP0 (PSP) C2PMSG register DWORD offsets relative to MP0 base ----
C2PMSG_BASE_DW = 0x40  # C2PMSG_N lives at MP0_base + 0x40 + N

PSP_SOS_ALIVE_TIMEOUT_MS   = 2000
PSP_BL_READY_TIMEOUT_MS    = 2000
PSP_FW_BUF_SIZE            = 1 << 20  # 1 MB, matches Linux PSP_1_MEG


@dataclass
class PSPComponent:
    """One sub-component extracted from a combined PSP firmware blob."""
    fw_type: int
    name: str
    version: int
    data: bytes


def c2pmsg_dw(mp0_base_dw: int, n: int) -> int:
    """Return the absolute BAR5 DWORD offset for C2PMSG_N at the given MP0 base."""
    return mp0_base_dw + C2PMSG_BASE_DW + n


def parse_psp_firmware(data: bytes) -> list[PSPComponent]:
    """Parse an AMD PSP firmware file (psp_firmware_header_v2_0 or v2_1).

    Returns the list of sub-components (KDB, SYS_DRV, SOS, ...).
    """
    # common_firmware_header (32 bytes)
    (
        size_bytes, header_size_bytes,
        hver_maj, hver_min,
        ipver_maj, ipver_min,
        ucode_version, ucode_size_bytes,
        ucode_array_offset_bytes, crc32,
    ) = struct.unpack_from("<IIHHHHIIII", data, 0)

    if hver_maj != 2:
        raise ValueError(
            f"Unsupported PSP firmware header major version {hver_maj}; "
            "this module only handles v2_0/v2_1 (combined-SOS)."
        )

    # psp_fw_bin_count follows at offset 32, array at offset 36 (v2_0) or 40 (v2_1)
    psp_fw_bin_count = struct.unpack_from("<I", data, 32)[0]
    # For v2_1 there's also psp_aux_fw_bin_index (uint32) before the array;
    # detect by checking header_size_bytes. In v2_0 header_size = 36 (32 + 4).
    if header_size_bytes == 36:
        desc_off = 36
    elif header_size_bytes == 40:
        desc_off = 40
    else:
        # Fall back to desc right after count
        desc_off = 36

    components: list[PSPComponent] = []
    for i in range(psp_fw_bin_count):
        d = desc_off + i * 16
        fw_type, fw_version, off_in_blob, sub_size = struct.unpack_from("<IIII", data, d)
        # offset_bytes is relative to the start of the ucode blob, which itself
        # starts at ucode_array_offset_bytes from the start of the file.
        start = ucode_array_offset_bytes + off_in_blob
        end = start + sub_size
        if end > len(data):
            raise ValueError(
                f"component {FW_TYPE_NAME.get(fw_type, fw_type)} "
                f"extends past end of file ({end} > {len(data)})"
            )
        components.append(PSPComponent(
            fw_type=fw_type,
            name=FW_TYPE_NAME.get(fw_type, f"T{fw_type}"),
            version=fw_version,
            data=bytes(data[start:end]),
        ))
    return components


def is_sos_alive(client, mp0_base_dw: int) -> bool:
    """C2PMSG_81 != 0 means the SOS is running."""
    v = client.mmio_read32(5, c2pmsg_dw(mp0_base_dw, 81) * 4)
    return v != 0


def wait_bootloader_ready(client, mp0_base_dw: int,
                          timeout_ms: int = PSP_BL_READY_TIMEOUT_MS) -> int:
    """Wait until PSP bootloader signals readiness via C2PMSG_35 bit 31.

    Mirrors `psp_v14_0_wait_for_bootloader()`: loop until bit 31 of
    C2PMSG_35 is set. Bit 31 = "bootloader alive / ready / command done".
    Returns the final register value for diagnostics.
    """
    deadline = time.time() + timeout_ms / 1000
    reg = c2pmsg_dw(mp0_base_dw, 35) * 4
    while time.time() < deadline:
        v = client.mmio_read32(5, reg)
        if v & 0x80000000:
            return v
        time.sleep(0.010)
    raise TimeoutError(
        f"PSP bootloader not ready after {timeout_ms} ms (C2PMSG_35 = 0x{v:08x})"
    )


def load_bootloader_component(
    client, driver, mp0_base_dw: int,
    comp: PSPComponent, bl_cmd: int,
    fw_buf_cpu: int, fw_buf_bus: int,
    *, verbose: bool = False,
    is_sos: bool = False,
    sos_timeout_ms: int = 5000,
) -> dict:
    """Load one PSP bootloader component (KDB, SYS_DRV, SOS, etc.).

    Mirrors psp_v14_0_bootloader_load_component() line-by-line — with one
    special case: SOS itself uses psp_v14_0_bootloader_load_sos(), which
    after issuing the cmd waits on C2PMSG_81 (SOS sign-of-life) instead
    of C2PMSG_35 (bootloader ready). Set `is_sos=True` to switch to that
    post-wait path.
    """
    c35 = c2pmsg_dw(mp0_base_dw, 35) * 4
    c36 = c2pmsg_dw(mp0_base_dw, 36) * 4
    c81 = c2pmsg_dw(mp0_base_dw, 81) * 4

    diag = {"name": comp.name, "bl_cmd": bl_cmd, "size": len(comp.data)}

    if is_sos_alive(client, mp0_base_dw):
        diag["skipped"] = "sos_alive"
        return diag

    wait_bootloader_ready(client, mp0_base_dw)
    diag["c35_pre"] = client.mmio_read32(5, c35)

    # Zero + copy into the 1 MB staging buffer (client process's CPU mapping).
    ctypes.memset(fw_buf_cpu, 0, PSP_FW_BUF_SIZE)
    if len(comp.data) > PSP_FW_BUF_SIZE:
        raise ValueError(
            f"component {comp.name} is {len(comp.data)} bytes "
            f"(>{PSP_FW_BUF_SIZE} limit)")
    (ctypes.c_ubyte * len(comp.data)).from_address(fw_buf_cpu)[:] = comp.data

    # Program MC address (must be 1 MB aligned; we alloc'd a full MB).
    client.mmio_write32(5, c36, fw_buf_bus >> 20)
    # Kick the bootloader with the load command.
    client.mmio_write32(5, c35, bl_cmd)

    # Handshake delay per psp_v14_0_bootloader_load_sos (20 ms).
    time.sleep(0.020)

    if is_sos:
        # SOS handoff: the bootloader accepts the cmd and transfers control
        # to SOS, which then populates C2PMSG_81 with a non-zero
        # sign-of-life value. C2PMSG_35 will *not* go back to bit-31-set
        # after this point — the bootloader is done, SOS is now driving.
        deadline = time.time() + sos_timeout_ms / 1000
        v81 = 0
        while time.time() < deadline:
            v81 = client.mmio_read32(5, c81)
            if v81 != 0:
                break
            time.sleep(0.002)
        if v81 == 0:
            raise TimeoutError(
                f"SOS did not come alive after {sos_timeout_ms} ms "
                f"(C2PMSG_81 still 0, C2PMSG_35=0x{client.mmio_read32(5, c35):08x})"
            )
        diag["c81_post"] = v81
        diag["c35_post"] = client.mmio_read32(5, c35)
        if verbose:
            print(f"    {comp.name}: cmd=0x{bl_cmd:x}  "
                  f"C2PMSG_81=0x{v81:08x} ← SOS ALIVE")
    else:
        # Non-SOS components: wait for bootloader-ready signal again.
        c35_final = wait_bootloader_ready(client, mp0_base_dw)
        diag["c35_post"] = c35_final
        diag["c81_post"] = client.mmio_read32(5, c81)
        if verbose:
            print(f"    {comp.name}: cmd=0x{bl_cmd:x} "
                  f"c35_post=0x{c35_final:08x} c81=0x{diag['c81_post']:08x}")
    return diag


def load_sos(client, driver, mp0_base_dw: int, firmware_path: str,
             *, verbose: bool = False) -> None:
    """Full Linux-amdgpu-style PSP bootloader chain for gfx1201.

    Order (from psp_v14_0_hw_start): KDB -> SYS_DRV -> SOC_DRV -> INTF_DRV ->
    DBG_DRV -> RAS_DRV -> IPKEYMGR_DRV -> SOS. Each is a no-op if the SOS
    comes alive early.
    """
    with open(firmware_path, "rb") as f:
        blob = f.read()
    components = {c.fw_type: c for c in parse_psp_firmware(blob)}

    # 1 MB DMA staging buffer. Our driver shim returns (cpu_addr, bus_addr,
    # handle); the bus address is IOMMU-translated (DART-mapped) so the PSP
    # reads the right physical memory.
    fw_cpu, fw_bus, fw_handle = driver.alloc_dma(PSP_FW_BUF_SIZE)

    try:
        # Order per drivers/gpu/drm/amd/amdgpu/amdgpu_psp.c::psp_hw_start:
        #   KDB → SPL → SYS_DRV → SOC_DRV → INTF_DRV → DBG_DRV → RAS_DRV →
        #   IPKEYMGR_DRV → SOS
        load_order = [
            (FW_TYPE_PSP_KDB,          PSP_BL__LOAD_KEY_DATABASE),
            (FW_TYPE_PSP_SPL,          PSP_BL__LOAD_TOS_SPL_TABLE),
            (FW_TYPE_PSP_SYS_DRV,      PSP_BL__LOAD_SYSDRV),
            (FW_TYPE_PSP_SOC_DRV,      PSP_BL__LOAD_SOCDRV),
            (FW_TYPE_PSP_INTF_DRV,     PSP_BL__LOAD_INTFDRV),
            (FW_TYPE_PSP_DBG_DRV,      PSP_BL__LOAD_HADDRV),
            (FW_TYPE_PSP_RAS_DRV,      PSP_BL__LOAD_RASDRV),
            (FW_TYPE_PSP_IPKEYMGR_DRV, PSP_BL__LOAD_IPKEYMGRDRV),
            (FW_TYPE_PSP_SOS,          PSP_BL__LOAD_SOSDRV),
        ]
        for fw_type, cmd in load_order:
            comp = components.get(fw_type)
            if comp is None:
                continue
            load_bootloader_component(
                client, driver, mp0_base_dw, comp, cmd, fw_cpu, fw_bus,
                verbose=verbose,
                is_sos=(fw_type == FW_TYPE_PSP_SOS))
    finally:
        driver.free_dma(fw_handle)

    # Final sign-of-life check.
    deadline = time.time() + PSP_SOS_ALIVE_TIMEOUT_MS / 1000
    while time.time() < deadline:
        if is_sos_alive(client, mp0_base_dw):
            return
        time.sleep(0.005)
    raise TimeoutError(
        f"PSP SOS did not come alive after bootloader chain "
        f"(C2PMSG_81 still zero)"
    )
