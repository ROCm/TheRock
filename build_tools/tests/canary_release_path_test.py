# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the canary's non-publishing local release phases."""

from canary_release_path import mirror


def test_mirror_streams_files_and_reports_explicit_scope(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "artifact.tar.xz").write_bytes(b"artifact")

    result = mirror(source, destination)

    assert (destination / "nested" / "artifact.tar.xz").read_bytes() == b"artifact"
    assert result["scope"] == "local-filesystem-equivalent-no-network-no-credentials"
    assert result["files"][0]["path"] == "nested/artifact.tar.xz"
