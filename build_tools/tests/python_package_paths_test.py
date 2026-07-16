# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for structured product-local Python package path computation.

All tests operate on in-memory inputs or a local temp directory. No AWS
credentials or network access are required.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.python_package_paths import (
    PlannedCopy,
    PlannedUpload,
    is_accepted_artifact,
    package_name_from_filename,
    pep503_normalize,
    plan_key_copies,
    plan_local_uploads,
    structured_key,
)
from _therock_utils.storage_location import StorageLocation


# ---------------------------------------------------------------------------
# pep503_normalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Foo_Bar", "foo-bar"),
        ("torch.scatter", "torch-scatter"),
        ("a--b__c", "a-b-c"),
        ("rocm_sdk_core", "rocm-sdk-core"),
        ("torch", "torch"),
        ("rocm-sdk-core", "rocm-sdk-core"),
    ],
)
def test_pep503_normalize(raw: str, expected: str) -> None:
    assert pep503_normalize(raw) == expected


# ---------------------------------------------------------------------------
# package_name_from_filename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("torch-2.10.0-cp310-cp310-linux_x86_64.whl", "torch"),
        ("rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl", "rocm-sdk-core"),
        # Wheel dist names escape to underscores (PEP 427); spec-aware parsing
        # normalizes them to the canonical dashed form.
        (
            "amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl",
            "amd-torch-device-gfx942",
        ),
        (
            "rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl",
            "rocm-sdk-device-gfx1100",
        ),
        (
            "rocm_sdk_libraries_gfx942-7.13.0-py3-none-linux_x86_64.whl",
            "rocm-sdk-libraries-gfx942",
        ),
        # sdist with single-token name.
        ("rocm-7.13.0.tar.gz", "rocm"),
        # sdist with hyphenated name: spec-aware parsing handles this correctly
        # where first-hyphen tokenization would produce only "llnl".
        ("llnl-hatchet-2024.1.tar.gz", "llnl-hatchet"),
        ("llnl-hatchet-2024.1.zip", "llnl-hatchet"),
    ],
)
def test_package_name_from_filename(filename: str, expected: str) -> None:
    assert package_name_from_filename(filename) == expected


# ---------------------------------------------------------------------------
# is_accepted_artifact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("torch-2.10.0-cp310-cp310-linux_x86_64.whl", True),
        ("rocm-7.13.0.tar.gz", True),
        ("something-1.0.zip", True),
        ("index.html", False),
        ("torch-2.10.0.whl.metadata", False),
        ("README.md", False),
    ],
)
def test_is_accepted_artifact(filename: str, expected: bool) -> None:
    assert is_accepted_artifact(filename) is expected


# ---------------------------------------------------------------------------
# structured_key
# ---------------------------------------------------------------------------


def test_structured_key_composition() -> None:
    key = structured_key("pytorch", "whl", "torch-2.10.0-cp310-cp310-linux_x86_64.whl")
    assert key == "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl"


def test_structured_key_whl_next() -> None:
    key = structured_key(
        "pytorch", "whl-next", "torch-2.10.0-cp310-cp310-linux_x86_64.whl"
    )
    assert key == "pytorch/whl-next/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl"


def test_structured_key_underscore_filename_dashed_dir() -> None:
    # Filename keeps underscores; the package directory is dashed.
    key = structured_key(
        "core", "whl", "rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl"
    )
    assert key == (
        "core/whl/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl"
    )


def test_structured_key_rejects_bad_index() -> None:
    with pytest.raises(ValueError, match="index="):
        structured_key("pytorch", "wheels", "torch-2.10.0.whl")


# ---------------------------------------------------------------------------
# plan_local_uploads
# ---------------------------------------------------------------------------


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path


def test_plan_local_uploads(tmp_path: Path) -> None:
    _touch(tmp_path, "torch-2.10.0-cp310-cp310-linux_x86_64.whl")
    _touch(tmp_path, "amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl")
    # Non-artifacts must be ignored.
    _touch(tmp_path, "index.html")
    _touch(tmp_path, "torch-2.10.0.whl.metadata")

    plans = plan_local_uploads(tmp_path, "therock-dev-python", "pytorch", "whl")

    assert all(isinstance(p, PlannedUpload) for p in plans)
    dests = {p.dest.relative_path for p in plans}
    assert dests == {
        "pytorch/whl/torch/torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        "pytorch/whl/amd-torch-device-gfx942/"
        "amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl",
    }
    for p in plans:
        assert p.dest.bucket == "therock-dev-python"
        assert p.source.parent == tmp_path


def test_plan_local_uploads_empty_dir(tmp_path: Path) -> None:
    assert plan_local_uploads(tmp_path, "b", "pytorch", "whl") == []


# ---------------------------------------------------------------------------
# plan_key_copies
# ---------------------------------------------------------------------------


def test_plan_key_copies() -> None:
    source_keys = [
        "12345-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl",
        "12345-linux/python/rocm-7.13.0.tar.gz",
        "12345-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl",
        # Ignored: not an accepted artifact.
        "12345-linux/python/index.html",
    ]
    plans = plan_key_copies(
        source_keys,
        source_bucket="therock-dev-artifacts",
        dest_bucket="therock-dev-python",
        product="core",
        index="whl",
    )

    assert all(isinstance(p, PlannedCopy) for p in plans)
    mapping = {p.source.relative_path: p.dest.relative_path for p in plans}
    assert mapping == {
        "12345-linux/python/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl": (
            "core/whl/rocm-sdk-core/rocm_sdk_core-7.13.0-py3-none-linux_x86_64.whl"
        ),
        "12345-linux/python/rocm-7.13.0.tar.gz": "core/whl/rocm/rocm-7.13.0.tar.gz",
        "12345-linux/python/rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl": (
            "core/whl/rocm-sdk-device-gfx1100/"
            "rocm_sdk_device_gfx1100-7.13.0-py3-none-linux_x86_64.whl"
        ),
    }
    for p in plans:
        assert p.source.bucket == "therock-dev-artifacts"
        assert p.dest.bucket == "therock-dev-python"


# ---------------------------------------------------------------------------
# Round-trip: producer output is consumable by the generator.
# ---------------------------------------------------------------------------


def test_generated_keys_round_trip_through_discover_packages() -> None:
    sys.path.insert(
        0,
        os.fspath(Path(__file__).parent.parent / "third_party" / "s3_management"),
    )
    from manage_structured import discover_packages

    filenames = [
        "torch-2.10.0-cp310-cp310-linux_x86_64.whl",
        "amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl",
        "llnl-hatchet-2024.1.tar.gz",
    ]
    keys = [structured_key("pytorch", "whl", f) for f in filenames]

    packages = discover_packages(keys, root="pytorch/whl")
    names = sorted(p.name for p in packages)
    assert names == ["amd-torch-device-gfx942", "llnl-hatchet", "torch"]
