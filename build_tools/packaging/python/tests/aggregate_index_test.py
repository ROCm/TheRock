# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for ROCm aggregate Python index ownership tooling."""

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from aggregate_index import (
    ManifestError,
    load_ownership_manifest,
    parse_ownership_manifest,
)


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "python_indexes": [
            {
                "public_base": "/rocm/whl-next",
                "packages": {
                    "torch": {"owner_path": "pytorch/whl-next"},
                    "rocm-sdk-core": {"owner_path": "core/whl-next"},
                    "apex": {"owner_path": "pytorch/whl-next"},
                },
            }
        ],
    }


def test_checked_in_manifest_loads() -> None:
    manifest_path = Path(__file__).parents[1] / "rocm_whl_next_ownership.yaml"
    manifest = load_ownership_manifest(manifest_path)
    index = manifest.python_indexes[0]

    assert manifest.schema_version == 1
    assert index.public_base == "/rocm/whl-next"
    assert len(index.packages) == 103
    assert "jax-rocm7-plugin" not in index.packages
    assert "jax-rocm7-pjrt" not in index.packages
    assert index.packages["jax-rocm10-plugin"].owner_public_base == "/rocm/jax/whl-next"
    assert index.packages["rocm-sdk-core"].owner_public_base == "/rocm/core/whl-next"
    assert index.packages["torch"].owner_public_base == "/rocm/pytorch/whl-next"


def test_parse_sorts_packages_by_owner_path_then_name() -> None:
    data = _valid_manifest()
    manifest = parse_ownership_manifest(data)
    index = manifest.python_indexes[0]

    assert list(index.packages) == ["rocm-sdk-core", "apex", "torch"]
    assert [package.name for package in index.ordered_packages()] == [
        "rocm-sdk-core",
        "apex",
        "torch",
    ]


@pytest.mark.parametrize(
    "data, match",
    [
        ({}, "missing required key"),
        (
            {"schema_version": 2, "python_indexes": []},
            "schema_version must be 1",
        ),
        (
            {"schema_version": True, "python_indexes": []},
            "schema_version must be an integer",
        ),
        (
            {"schema_version": 1, "python_indexes": []},
            "exactly one /rocm/whl-next index",
        ),
        (
            {
                "schema_version": 1,
                "python_indexes": [
                    {
                        "public_base": "/rocm/whl-next",
                        "packages": {},
                    }
                ],
            },
            "packages must not be empty",
        ),
    ],
)
def test_rejects_malformed_top_level_data(data: object, match: str) -> None:
    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "public_base, match",
    [
        ("/rocm/whl-next/", "must not end"),
        ("/other/whl-next", "contained under /rocm"),
        ("/rocm/../whl-next", "path segment"),
        ("/rocm/whl", "must be '/rocm/whl-next'"),
    ],
)
def test_rejects_invalid_public_base(public_base: str, match: str) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    index["public_base"] = public_base

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "package_name, match",
    [
        ("Torch", "package name must be PEP 503 normalized"),
        ("torch_gpu", "package name must be PEP 503 normalized"),
    ],
)
def test_rejects_unnormalized_package_names(package_name: str, match: str) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages.clear()
    packages[package_name] = {"owner_path": "pytorch/whl-next"}

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


@pytest.mark.parametrize(
    "owner_path, match",
    [
        ("/rocm/core/whl-next", "must be relative"),
        ("core/../whl-next", "path segment"),
        ("https://example.com/rocm/core/whl-next", "URL scheme"),
        ("core//whl-next", "path segment"),
        ("core/whl next", "unsupported path segment"),
    ],
)
def test_rejects_invalid_owner_paths(owner_path: str, match: str) -> None:
    data = _valid_manifest()
    index = data["python_indexes"][0]
    assert isinstance(index, dict)
    packages = index["packages"]
    assert isinstance(packages, dict)
    packages["torch"] = {"owner_path": owner_path}

    with pytest.raises(ManifestError, match=match):
        parse_ownership_manifest(data)


def test_rejects_unknown_keys() -> None:
    data = _valid_manifest()
    data["extra"] = "unsupported"

    with pytest.raises(ManifestError, match="unknown key"):
        parse_ownership_manifest(data)


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text(
        """
schema_version: 1
python_indexes:
  - public_base: /rocm/whl-next
    packages:
      torch:
        owner_path: pytorch/whl-next
      torch:
        owner_path: core/whl-next
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="duplicate key 'torch'"):
        load_ownership_manifest(manifest_path)


def test_rejects_malformed_yaml(tmp_path: Path) -> None:
    manifest_path = tmp_path / "ownership.yaml"
    manifest_path.write_text("schema_version: [", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_ownership_manifest(manifest_path)
