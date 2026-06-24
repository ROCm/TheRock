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

import install_rocm_from_artifacts
import install_rocm_from_artifacts as mod


class MirageArtifactSelectionTest(unittest.TestCase):
    """Verifies the --mirage flag wires through to fetch_artifacts."""

    def _run_main(self, extra_args):
        """Run main() with fetch_artifacts mocked, returning the captured argv."""
        captured = {}

        def fake_fetch(argv):
            captured["argv"] = argv

        with mock.patch.object(
            install_rocm_from_artifacts, "fetch_artifacts_main", fake_fetch
        ):
            install_rocm_from_artifacts.main(
                [
                    "--run-id",
                    "12345",
                    "--artifact-group",
                    "gfx942",
                    "--amdgpu-targets",
                    "gfx942",
                    "--dry-run",
                ]
                + extra_args
            )
        return captured["argv"]

    def test_mirage_flag_includes_mirage_artifacts(self):
        argv = self._run_main(["--mirage"])
        self.assertIn("mirage_run", argv)

    def test_no_mirage_flag_excludes_mirage_artifacts(self):
        argv = self._run_main(["--rocjitsu"])
        self.assertNotIn("mirage_run", argv)

    def test_mirage_with_tests_includes_test_artifact(self):
        argv = self._run_main(["--mirage", "--tests"])
        self.assertIn("mirage_run", argv)
        self.assertIn("mirage_test", argv)


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
