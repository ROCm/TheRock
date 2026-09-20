# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for deterministic artifact equivalence manifests."""

import json

from artifact_equivalence_manifest import create_manifest


def test_manifest_is_relative_deterministic_and_content_sensitive(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        (root / "nested").mkdir(parents=True)
        (root / "b.tar.xz").write_bytes(b"same-b")
        (root / "nested" / "a.tar.xz").write_bytes(b"same-a")

    first = create_manifest(first_root)
    second = create_manifest(second_root)
    assert first == second
    assert [item["path"] for item in first["files"]] == [
        "b.tar.xz",
        "nested/a.tar.xz",
    ]
    assert str(tmp_path) not in json.dumps(first)

    (second_root / "b.tar.xz").write_bytes(b"different")
    assert create_manifest(first_root) != create_manifest(second_root)
