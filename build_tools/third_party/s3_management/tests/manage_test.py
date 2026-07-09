# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from manage import S3Index, S3Object, _make_list_prefix, update_pep503_index


# ---------------------------------------------------------------------------
# _make_list_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix, package_name, expected",
    [
        ("v4/whl", "torch", "v4/whl/torch-"),
        ("v4/whl", "amd_torch_device_gfx942", "v4/whl/amd_torch_device_gfx942-"),
        ("v4/whl", "torchaudio", "v4/whl/torchaudio-"),
        ("v4/whl", None, "v4/whl"),
        ("v2/gfx94X-dcgpu", None, "v2/gfx94X-dcgpu"),
        # Empty string must not narrow the prefix; treated as full sweep.
        ("v4/whl", "", "v4/whl"),
    ],
)
def test_make_list_prefix(prefix: str, package_name: str | None, expected: str) -> None:
    assert _make_list_prefix(prefix, package_name) == expected


def test_update_pep503_index_rejects_package_name_with_update_root_index() -> None:
    with pytest.raises(ValueError, match="package_name and update_root_index=True cannot be used together"):
        update_pep503_index(prefix="v4/whl", package_name="torch", update_root_index=True)


def test_make_list_prefix_torch_does_not_match_torchaudio() -> None:
    # The hyphen delimiter ensures torch- cannot match torchaudio- wheels.
    prefix = _make_list_prefix("v4/whl", "torch")
    assert not "torchaudio-2.10.0.whl".startswith(prefix.split("/")[-1] + "audio")
    assert "torch-2.10.0.whl".startswith(prefix.split("/")[-1])
    assert not "torchaudio-2.10.0.whl".startswith(prefix.split("/")[-1])


# ---------------------------------------------------------------------------
# S3Index.obj_to_package_name
# ---------------------------------------------------------------------------


def _make_obj(key: str) -> S3Object:
    return S3Object(key=key, orig_key=key, checksum=None, size=None, pep658=None)


@pytest.mark.parametrize(
    "key, expected",
    [
        (
            "v4/whl/torch-2.10.0%2Brocm7.14.0a20260617-cp310-cp310-linux_x86_64.whl",
            "torch",
        ),
        (
            "v4/whl/amd_torch_device_gfx942-2.10.0%2Brocm7.14.0a20260617-py3-none-linux_x86_64.whl",
            "amd_torch_device_gfx942",
        ),
        (
            "v4/whl/torchaudio-2.10.0%2Brocm7.14.0a20260617-cp310-cp310-linux_x86_64.whl",
            "torchaudio",
        ),
        (
            "v4/whl/torchvision-0.21.0%2Brocm7.14.0a20260617-cp312-cp312-linux_x86_64.whl",
            "torchvision",
        ),
        (
            "v2/gfx94X-dcgpu/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
            "torch",
        ),
    ],
)
def test_obj_to_package_name(key: str, expected: str) -> None:
    idx = S3Index([], prefix="v4/whl")
    assert idx.obj_to_package_name(_make_obj(key)) == expected
