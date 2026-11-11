# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the structured Python package publishing policy."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.python_package_publishing import (
    build_variant_index_suffix,
    get_python_index,
)


# ---------------------------------------------------------------------------
# get_python_index
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "index_type, suffix, expected",
    [
        ("whl-next", None, "whl-next"),
        ("whl", None, "whl"),
        ("whl-next", "asan", "whl-next-asan"),
    ],
)
def test_get_python_index(index_type: str, suffix: str | None, expected: str) -> None:
    assert get_python_index(index_type, suffix) == expected


def test_get_python_index_default() -> None:
    assert get_python_index() == "whl-next"


@pytest.mark.parametrize(
    "index_type, suffix",
    [
        ("wheels", None),
        ("whl-next-asan", None),
        # ASAN is only published to whl-next-asan.
        ("whl", "asan"),
        # No tsan/debug indexes until they are supported.
        ("whl-next", "tsan"),
        ("whl-next", "debug"),
    ],
)
def test_get_python_index_rejects_unknown(index_type: str, suffix: str | None) -> None:
    with pytest.raises(ValueError):
        get_python_index(index_type, suffix)


# ---------------------------------------------------------------------------
# build_variant_index_suffix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build_variant, expected",
    [
        ("release", None),
        ("asan", "asan"),
        ("asan-debug", "asan"),
        ("host-asan", "asan"),
        ("host-asan-debug", "asan"),
    ],
)
def test_build_variant_index_suffix(build_variant: str, expected: str | None) -> None:
    assert build_variant_index_suffix(build_variant) == expected


@pytest.mark.parametrize(
    "build_variant, message",
    [
        # Not a build variant at all.
        ("foo", "is invalid"),
        # Known build variant that is not published yet.
        ("tsan", "not supported for publishing yet"),
    ],
)
def test_build_variant_index_suffix_rejects_unknown(
    build_variant: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_variant_index_suffix(build_variant)
