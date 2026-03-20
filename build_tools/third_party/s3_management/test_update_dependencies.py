# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from update_dependencies import is_wheel_allowed


# ---------------------------------------------------------------------------
# Allowed wheels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pkg",
    [
        # linux_x86_64, various supported CPython versions
        "numpy-2.0.0-cp310-cp310-linux_x86_64.whl",
        "numpy-2.0.0-cp311-cp311-linux_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-linux_x86_64.whl",
        "numpy-2.0.0-cp313-cp313-linux_x86_64.whl",
        # manylinux variants
        "numpy-2.0.0-cp310-cp310-manylinux_2_17_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-manylinux2014_x86_64.whl",
        "pillow-10.0.0-cp311-cp311-manylinux_2_28_x86_64.whl",
        # pure-Python / platform-independent
        "sympy-1.13.0-py3-none-any.whl",
        "filelock-3.15.0-py3-none-any.whl",
        # Windows x64 — was not excluded by the old blacklist
        "torch-2.3.0-cp312-cp312-win_amd64.whl",
        "numpy-2.0.0-cp310-cp310-win_amd64.whl",
    ],
)
def test_allowed(pkg: str) -> None:
    assert is_wheel_allowed(pkg), f"Expected allowed: {pkg}"


# ---------------------------------------------------------------------------
# Rejected platform tags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pkg",
    [
        # win32 and win_arm64
        "numpy-2.0.0-cp312-cp312-win32.whl",
        "numpy-2.0.0-cp312-cp312-win_arm64.whl",
        # musllinux
        "numpy-2.0.0-cp312-cp312-musllinux_1_1_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-musllinux_1_2_aarch64.whl",
        # macOS — including the tricky _x86_64 suffix variant
        "numpy-2.0.0-cp312-cp312-macosx_10_9_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-macosx_11_0_arm64.whl",
        "numpy-2.0.0-cp312-cp312-macosx_12_0_universal2.whl",
        # aarch64 / ARM
        "numpy-2.0.0-cp312-cp312-manylinux_2_17_aarch64.whl",
        "numpy-2.0.0-cp312-cp312-linux_aarch64.whl",
        # i686 / 32-bit x86
        "numpy-2.0.0-cp312-cp312-manylinux_2_17_i686.whl",
        "numpy-2.0.0-cp312-cp312-linux_i686.whl",
        # iOS
        "numpy-2.0.0-cp312-cp312-iphoneos_17_0_arm64.whl",
        "numpy-2.0.0-cp312-cp312-iphonesimulator_17_0_x86_64.whl",
        # RISC-V
        "numpy-2.0.0-cp312-cp312-linux_riscv64.whl",
    ],
)
def test_rejected_platform(pkg: str) -> None:
    assert not is_wheel_allowed(pkg), f"Expected rejected: {pkg}"


# ---------------------------------------------------------------------------
# Rejected Python tags
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pkg",
    [
        # Too old
        "numpy-1.26.0-cp39-cp39-linux_x86_64.whl",
        # PyPy
        "numpy-2.0.0-pp310-pypy310_pp73-linux_x86_64.whl",
        "numpy-2.0.0-pp310-pypy310_pp73-manylinux_2_17_x86_64.whl",
        # Free-threaded and future versions
        "numpy-2.0.0-cp313t-cp313t-linux_x86_64.whl",
        "numpy-2.0.0-cp314-cp314-linux_x86_64.whl",
        "numpy-2.0.0-cp314t-cp314t-linux_x86_64.whl",
        # Python 2 and py2.py3 universal tags
        "six-1.16.0-py2-none-any.whl",
        "six-1.16.0-py2.py3-none-any.whl",
    ],
)
def test_rejected_python(pkg: str) -> None:
    assert not is_wheel_allowed(pkg), f"Expected rejected: {pkg}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pkg",
    [
        # Not a wheel
        "numpy-2.0.0.tar.gz",
        "numpy-2.0.0-cp312-cp312-linux_x86_64.zip",
        # Malformed (too few parts)
        "numpy-2.0.0-linux_x86_64.whl",
        # Empty string
        "",
    ],
)
def test_rejected_non_wheel_or_malformed(pkg: str) -> None:
    assert not is_wheel_allowed(pkg), f"Expected rejected: {pkg}"
