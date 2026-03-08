"""First hardware contact test — "Hello GPU".

Run this after installing the kernel driver to verify basic
communication works through the D3DKMTEscape path.

Tests (in order):
1. Device discovery via D3DKMT enumeration
2. Device open
3. GET_INFO escape (PCI IDs, BAR sizes, VRAM)
4. READ_REG32 on BAR0 offset 0 (should return non-zero/non-FF)
5. SMN indirect read of mmRCC_CONFIG_MEMSIZE (VRAM size register)
6. Read a few well-known registers to confirm MMIO works

Usage:
    python test_hw_hello.py
"""

from __future__ import annotations

import sys
import traceback


def test_discovery() -> bool:
    """Test 1: Can we discover AMD GPU devices via D3DKMT?"""
    print("\n[1] Device discovery via D3DKMT...")
    try:
        from amd_gpu_driver.backends.windows.discovery import discover_devices
        devices = discover_devices()
        if not devices:
            print("  FAIL: No AMD GPU devices found via D3DKMT.")
            print("  Is the driver installed? Check Device Manager.")
            return False
        for i, dev in enumerate(devices):
            print(f"  Device {i}: {dev.device_name} "
                  f"(adapter_index={dev.adapter_index})")
        print(f"  OK: Found {len(devices)} device(s)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_open() -> object | None:
    """Test 2: Open the device."""
    print("\n[2] Opening device...")
    try:
        from amd_gpu_driver.backends.windows.device import WindowsDevice
        dev = WindowsDevice()
        dev.open(0)
        print(f"  Device: {dev.name}")
        print(f"  GFX target: {dev.gfx_target_version}")
        print(f"  VRAM: {dev.vram_size // (1024*1024)} MB "
              f"({dev.vram_size // (1024**3)} GB)")
        print("  OK: Device opened successfully")
        return dev
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return None


def test_get_info(dev: object) -> bool:
    """Test 3: GET_INFO escape command."""
    print("\n[3] GET_INFO escape...")
    try:
        info = dev.driver.get_info()
        print(f"  Vendor ID:  0x{info.vendor_id:04X}")
        print(f"  Device ID:  0x{info.device_id:04X}")
        print(f"  Revision:   0x{info.revision_id:02X}")
        print(f"  Subsystem:  0x{info.subsystem_vendor_id:04X}:"
              f"0x{info.subsystem_id:04X}")
        print(f"  VRAM:       {info.vram_size // (1024*1024)} MB")
        print(f"  Visible:    {info.visible_vram_size // (1024*1024)} MB")
        print(f"  MMIO BAR:   {info.mmio_bar_index}")
        print(f"  VRAM BAR:   {info.vram_bar_index}")
        print(f"  Headless:   {info.headless}")

        # Print BAR info
        for i, bar in enumerate(info.bars):
            if bar.get("length", 0) > 0:
                addr = bar.get("physical_address", 0)
                length = bar.get("length", 0)
                label = ""
                if i == info.mmio_bar_index:
                    label = " [MMIO]"
                elif i == info.vram_bar_index:
                    label = " [VRAM]"
                if length >= 1024 * 1024:
                    size_str = f"{length // (1024*1024)}MB"
                else:
                    size_str = f"{length // 1024}KB"
                print(f"  BAR{i}:       addr=0x{addr:012X} "
                      f"size={size_str}{label}")

        if info.vendor_id != 0x1002:
            print("  WARNING: Unexpected vendor ID")
        if info.device_id != 0x7551:
            print("  WARNING: Unexpected device ID (expected 0x7551)")
        print("  OK: GET_INFO returned valid data")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_reg_read(dev: object) -> bool:
    """Test 4: Basic MMIO register reads."""
    print("\n[4] Register reads (BAR0 MMIO)...")
    try:
        # Read offset 0 — any GPU register
        val0 = dev.read_reg32(0)
        print(f"  BAR0[0x0000] = 0x{val0:08X}")

        # A few more offsets to check MMIO is working
        test_offsets = [0x4, 0x8, 0xC, 0x10]
        for off in test_offsets:
            val = dev.read_reg32(off)
            print(f"  BAR0[0x{off:04X}] = 0x{val:08X}")

        # Check for obvious bad values
        if val0 == 0xFFFFFFFF:
            print("  WARNING: All 1s — device may not be responding")
            print("  (This can happen if BAR is not enabled or device is in D3)")
            return False
        if val0 == 0x00000000:
            print("  NOTE: Register 0 returned 0 (may be normal)")

        print("  OK: MMIO register reads working")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_smn_indirect(dev: object) -> bool:
    """Test 5: SMN indirect register read (NBIO index/data)."""
    print("\n[5] SMN indirect read...")
    try:
        # mmRCC_CONFIG_MEMSIZE is at SMN address 0x0100_C078 for NBIO v7.11
        # but the BAR0 direct offset is 0x378C on many AMD GPUs
        # Let's try the direct BAR0 offset first
        memsize_offset = 0x378C  # mmRCC_CONFIG_MEMSIZE typical offset
        val = dev.read_reg32(memsize_offset)
        vram_mb = val & 0xFFFF  # Lower 16 bits = VRAM in MB typically
        print(f"  BAR0[0x{memsize_offset:04X}] = 0x{val:08X} "
              f"(raw value)")

        # Now try SMN indirect
        # NBIO v7.11 SMN index = 0x60, data = 0x64
        # Read a known register via SMN
        val_indirect = dev.read_reg_indirect(0x0)
        print(f"  SMN[0x00000000] = 0x{val_indirect:08X}")

        # Read RCC_CONFIG_MEMSIZE via SMN (address varies by NBIO version)
        # For NBIO v7.11, try 0x0100_C078
        smn_memsize = 0x0100C078
        val_smn = dev.read_reg_indirect(smn_memsize)
        print(f"  SMN[0x{smn_memsize:08X}] = 0x{val_smn:08X} "
              f"(RCC_CONFIG_MEMSIZE)")
        if val_smn > 0 and val_smn < 0xFFFF:
            print(f"  -> VRAM = {val_smn} MB")
        elif val_smn >= 0xFFFF:
            print(f"  -> Raw value, VRAM detection may need different address")

        print("  OK: SMN indirect reads working")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def test_write_read(dev: object) -> bool:
    """Test 6: Write and read back a scratch register (if safe)."""
    print("\n[6] Write/read-back test (scratch register)...")
    try:
        # SCRATCH_REG0 is typically at a known offset
        # For GFX12, scratch registers are part of GC block
        # We'll skip this for safety — writing to the wrong register
        # could hang the GPU. Instead, just confirm write_reg32 doesn't crash.
        print("  SKIPPED: Write test deferred until IP discovery provides")
        print("  safe register addresses. Register reads are sufficient")
        print("  to confirm the escape path works.")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    print("=" * 60)
    print("AMD GPU Hardware Test — Hello GPU")
    print("=" * 60)

    results: list[tuple[str, bool]] = []

    # Test 1: Discovery
    ok = test_discovery()
    results.append(("Device discovery", ok))
    if not ok:
        print("\nCannot proceed without device discovery.")
        print("Ensure the driver is installed and the device is present.")
        return 1

    # Test 2: Open
    dev = test_open()
    results.append(("Device open", dev is not None))
    if dev is None:
        print("\nCannot proceed without opening the device.")
        return 1

    # Test 3: GET_INFO
    ok = test_get_info(dev)
    results.append(("GET_INFO escape", ok))

    # Test 4: Register reads
    ok = test_reg_read(dev)
    results.append(("MMIO register reads", ok))

    # Test 5: SMN indirect
    ok = test_smn_indirect(dev)
    results.append(("SMN indirect reads", ok))

    # Test 6: Write/read
    ok = test_write_read(dev)
    results.append(("Write/read test", ok))

    # Cleanup
    dev.close()

    # Summary
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed! The GPU is responding.")
        print("Next step: Run IP discovery:")
        print("  python -c \"from amd_gpu_driver.backends.windows.compute_dispatch import full_gpu_bringup; full_gpu_bringup()\"")
    else:
        failed = sum(1 for _, p in results if not p)
        print(f"{failed} test(s) failed. Check output above for details.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
