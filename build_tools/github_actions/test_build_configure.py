# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for build_configure helper functions."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Patch env vars before import (the module reads them at import time)
os.environ.setdefault("amdgpu_families", "")
os.environ.setdefault("package_version", "0.0.0")
os.environ.setdefault("extra_cmake_options", "")

from build_configure import host_march_for_families


# --- Individual Zen 5 targets ---


def test_gfx1150_returns_znver5():
    assert host_march_for_families("gfx1150") == "znver5"


def test_gfx1151_returns_znver5():
    assert host_march_for_families("gfx1151") == "znver5"


def test_gfx1152_returns_znver5():
    assert host_march_for_families("gfx1152") == "znver5"


def test_all_strix_zen5_together_returns_znver5():
    # All three map to the same arch so the result is unambiguous.
    assert host_march_for_families("gfx1150,gfx1151,gfx1152") == "znver5"


def test_semicolon_separator():
    assert host_march_for_families("gfx1150;gfx1151;gfx1152") == "znver5"


# --- Family aliases must NOT be blindly accepted ---


def test_gfx115X_igpu_family_returns_none():
    # gfx115X-igpu includes gfx1153 (Hawk Point / Zen 4), which is not znver5.
    # The family alias is intentionally absent from AMDGPU_HOST_MARCH_MAP.
    assert host_march_for_families("gfx115X-igpu") is None


# --- Non-Zen5 / unknown targets ---


def test_unknown_family_returns_none():
    assert host_march_for_families("gfx1100") is None


def test_gfx1153_hawk_point_returns_none():
    # Hawk Point is Zen 4 — must never receive -march=znver5.
    assert host_march_for_families("gfx1153") is None


def test_mixed_known_and_unknown_returns_none():
    # Any unknown token disables injection for the whole build.
    assert host_march_for_families("gfx1100,gfx1151") is None


def test_empty_returns_none():
    assert host_march_for_families("") is None
    assert host_march_for_families(None) is None


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
