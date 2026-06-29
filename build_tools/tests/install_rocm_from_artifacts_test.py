#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for install_rocm_from_artifacts.py."""

from datetime import datetime
from pathlib import Path
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import install_rocm_from_artifacts as mod


class TestReleaseDiscovery(unittest.TestCase):
    def test_extract_version_ignores_test_tarball(self) -> None:
        self.assertIsNone(
            mod.extract_version_from_asset_name(
                "therock-dist-linux-gfx94X-dcgpu-tests-7.13.0.tar.gz",
                "gfx94X-dcgpu",
                "linux",
            )
        )

    def test_fetch_and_sort_nightly_releases_ignores_test_tarballs(self) -> None:
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {
                        "Key": (
                            "therock-dist-linux-gfx94X-dcgpu-tests-"
                            "7.13.0a20260102.tar.gz"
                        ),
                        "LastModified": datetime(2026, 1, 2),
                        "Size": 20,
                    },
                    {
                        "Key": "therock-dist-linux-gfx94X-dcgpu-7.13.0a20260101.tar.gz",
                        "LastModified": datetime(2026, 1, 1),
                        "Size": 10,
                    },
                ]
            }
        ]
        s3_client = mock.Mock()
        s3_client.get_paginator.return_value = paginator

        with mock.patch.object(mod, "s3_client", s3_client):
            releases = mod._fetch_and_sort_nightly_releases("gfx94X-dcgpu", "linux")

        self.assertEqual(
            [release["asset_name"] for release in releases],
            ["therock-dist-linux-gfx94X-dcgpu-7.13.0a20260101.tar.gz"],
        )

    def test_list_available_nightly_gpu_families_ignores_test_tarballs(self) -> None:
        paginator = mock.Mock()
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz"},
                    {"Key": ("therock-dist-linux-gfx94X-dcgpu-tests-" "7.13.0.tar.gz")},
                ]
            }
        ]
        s3_client = mock.Mock()
        s3_client.get_paginator.return_value = paginator

        with mock.patch.object(mod, "s3_client", s3_client):
            families = mod.list_available_nightly_gpu_families("linux")

        self.assertEqual(families, {"gfx94X-dcgpu"})


if __name__ == "__main__":
    unittest.main()
