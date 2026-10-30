# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for copy_rocm_packages_between_streams.py."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions import copy_rocm_packages_between_streams as m


class CopyRocmPackagesBetweenStreamsTest(unittest.TestCase):
    def test_stream_index_url(self):
        self.assertEqual(
            m.stream_index_url("nightly"),
            "https://nightly.repo.amd.com/rocm/whl-next/",
        )

    def test_select_core_packages(self):
        packages = [
            "rocm",
            "rocm-sdk-core",
            "rocm-sdk-device-gfx942",
            "rocm-bootstrap",
            "torch",
            "amd-torch-device-gfx942",
        ]
        self.assertEqual(
            m.select_core_packages(packages),
            ["rocm", "rocm-sdk-core", "rocm-sdk-device-gfx942"],
        )

    def test_select_torch_packages(self):
        packages = [
            "rocm",
            "torch",
            "torchaudio",
            "amd-torch-device-gfx942",
            "apex",
            "triton",
            "numpy",
        ]
        self.assertEqual(
            m.select_torch_packages(packages),
            [
                "torch",
                "torchaudio",
                "amd-torch-device-gfx942",
                "apex",
                "triton",
            ],
        )

    def test_list_package_artifacts_filters_version(self):
        html = """
        <a href="rocm-10.3.0a20261026.tar.gz">old</a>
        <a href="rocm-10.3.0a20261027.tar.gz">match</a>
        <a href="rocm_sdk_core-10.3.0a20261027-py3-none-linux_x86_64.whl">whl</a>
        """
        with mock.patch.object(m, "_fetch_text", return_value=html):
            artifacts = m.list_package_artifacts(
                "https://nightly.repo.amd.com/rocm/whl-next/rocm/",
                "10.3.0a20261027",
            )
        self.assertEqual(
            artifacts,
            [
                "rocm-10.3.0a20261027.tar.gz",
                "rocm_sdk_core-10.3.0a20261027-py3-none-linux_x86_64.whl",
            ],
        )

    def test_copy_packages_uploads_core_artifacts(self):
        def fake_collect(*, index_url, package_names, version, dest_dir, dry_run=False):
            del index_url, dry_run
            path = dest_dir / f"rocm-{version}.tar.gz"
            path.write_bytes(b"payload")
            self.assertEqual(package_names, ["rocm"])
            return [path]

        with (
            mock.patch.object(
                m,
                "list_index_packages",
                return_value=["rocm", "torch"],
            ),
            mock.patch.object(
                m,
                "collect_artifacts_for_packages",
                side_effect=fake_collect,
            ),
            mock.patch.object(m, "upload_product_artifacts", return_value=1) as mock_upload,
        ):
            count = m.copy_packages(
                rocm_version="10.3.0a20261027",
                source_stream="nightly",
                dest_stream="dev",
                products=["core"],
                dry_run=True,
            )

        self.assertEqual(count, 1)
        mock_upload.assert_called_once()
        self.assertEqual(mock_upload.call_args.kwargs["product"], "core")
        self.assertEqual(mock_upload.call_args.kwargs["dest_stream"], "dev")
        self.assertTrue(mock_upload.call_args.kwargs["dry_run"])

    def test_copy_packages_rejects_same_stream(self):
        with self.assertRaises(ValueError):
            m.copy_packages(
                rocm_version="10.3.0a20261027",
                source_stream="dev",
                dest_stream="dev",
                products=["core"],
                dry_run=True,
            )


if __name__ == "__main__":
    unittest.main()
