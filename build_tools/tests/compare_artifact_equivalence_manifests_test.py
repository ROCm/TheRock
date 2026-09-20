# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from build_tools.compare_artifact_equivalence_manifests import compare
import pytest


def test_equal_manifests_are_equivalent():
    manifest = {
        "schema": "therock.artifact_equivalence.v1",
        "files": [{"path": "a.zip", "size_bytes": 3, "sha256": "abc"}],
    }
    result = compare({"stage": manifest}, {"stage": manifest})
    assert result["equivalent"] is True
    assert result["comparisons"][0]["status"] == "equivalent"


def test_missing_or_changed_manifest_is_not_equivalent():
    manifest = {"schema": "therock.artifact_equivalence.v1", "files": []}
    changed = {
        "schema": "therock.artifact_equivalence.v1",
        "files": [{"path": "a.zip", "size_bytes": 1, "sha256": "x"}],
    }
    result = compare({"same": manifest, "missing": manifest}, {"same": changed})
    assert result["equivalent"] is False
    assert {item["status"] for item in result["comparisons"]} == {"different"}


def test_empty_manifest_sets_are_rejected():
    with pytest.raises(ValueError, match="must be non-empty"):
        compare({}, {})
