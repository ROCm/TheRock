#!/usr/bin/env python
"""Unit tests for publish_pytorch_to_release_bucket.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))

from github_actions.publish_pytorch_to_release_bucket import main


class TestPublishPytorchToReleaseBucket(unittest.TestCase):
    """Tests for the main() CLI entry point."""

    def setUp(self):
        # Real directory so the script's existence check passes; the
        # upload itself is mocked so no S3 contact happens.
        self._tmp = tempfile.TemporaryDirectory()
        self.source_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    @mock.patch("github_actions.publish_pytorch_to_release_bucket.gha_set_output")
    def test_dev_uploads_to_v4_whl_in_dev_python(self, mock_set_output, mock_upload):
        mock_upload.return_value = 3
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--dry-run",
            ]
        )

        self.assertEqual(mock_upload.call_count, 1)
        call_args = mock_upload.call_args
        source, dest = call_args.args
        self.assertEqual(source, self.source_dir)
        self.assertEqual(dest.bucket, "therock-dev-python")
        self.assertEqual(dest.relative_path, "v4/whl")
        self.assertEqual(call_args.kwargs.get("include"), ["*.whl"])
        mock_set_output.assert_called_once_with(
            {"package_index_url": "https://rocm.devreleases.amd.com/whl-multi-arch/"}
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_nightly_selects_nightly_bucket(self, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "nightly",
                "--dry-run",
            ]
        )

        _source, dest = mock_upload.call_args.args
        self.assertEqual(dest.bucket, "therock-nightly-python")
        self.assertEqual(dest.relative_path, "v4/whl")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_prerelease_selects_prerelease_bucket(self, mock_upload):
        mock_upload.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "prerelease",
                "--dry-run",
            ]
        )

        _source, dest = mock_upload.call_args.args
        self.assertEqual(dest.bucket, "therock-prerelease-python")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_directory")
    def test_raises_when_no_wheels_uploaded(self, mock_upload):
        mock_upload.return_value = 0
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    def test_raises_when_source_dir_missing(self):
        missing = self.source_dir / "does-not-exist"
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(missing),
                    "--release-type",
                    "dev",
                    "--dry-run",
                ]
            )

    def test_invalid_release_type_rejected(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "release",
                    "--dry-run",
                ]
            )

    # -----------------------------------------------------------------------
    # Structured product-local publishing (--structured)
    # -----------------------------------------------------------------------

    def _touch(self, name: str) -> None:
        (self.source_dir / name).write_bytes(b"")

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_files")
    @mock.patch("github_actions.publish_pytorch_to_release_bucket.gha_set_output")
    def test_structured_places_wheels_in_package_dirs(
        self, mock_set_output, mock_upload_files
    ):
        self._touch("torch-2.10.0-cp312-cp312-linux_x86_64.whl")
        self._touch("amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl")
        # Non-wheel artifacts must be ignored by the planner.
        self._touch("index.html")
        mock_upload_files.return_value = 2
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--structured",
                "--dry-run",
            ]
        )

        self.assertEqual(mock_upload_files.call_count, 1)
        (uploads,) = mock_upload_files.call_args.args
        dest_by_name = {source.name: dest.relative_path for source, dest in uploads}
        self.assertNotIn("index.html", dest_by_name)
        self.assertEqual(
            dest_by_name["torch-2.10.0-cp312-cp312-linux_x86_64.whl"],
            "v5/rocm/pytorch/whl/torch/torch-2.10.0-cp312-cp312-linux_x86_64.whl",
        )
        self.assertEqual(
            dest_by_name["amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl"],
            "v5/rocm/pytorch/whl/amd-torch-device-gfx942/"
            "amd_torch_device_gfx942-2.10.0-py3-none-linux_x86_64.whl",
        )
        for _source, dest in uploads:
            self.assertEqual(dest.bucket, "therock-dev-python")
        # The index-URL output is unchanged under structured publishing.
        mock_set_output.assert_called_once_with(
            {"package_index_url": "https://rocm.devreleases.amd.com/whl-multi-arch/"}
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_files")
    def test_structured_whl_next(self, mock_upload_files):
        self._touch("torch-2.10.0-cp312-cp312-linux_x86_64.whl")
        mock_upload_files.return_value = 1
        main(
            [
                "--source-dir",
                os.fspath(self.source_dir),
                "--release-type",
                "dev",
                "--structured",
                "--python-index",
                "whl-next",
                "--dry-run",
            ]
        )

        (uploads,) = mock_upload_files.call_args.args
        _source, dest = uploads[0]
        self.assertEqual(
            dest.relative_path,
            "v5/rocm/pytorch/whl-next/torch/"
            "torch-2.10.0-cp312-cp312-linux_x86_64.whl",
        )

    @mock.patch("_therock_utils.storage_backend.S3StorageBackend.upload_files")
    def test_structured_raises_when_no_wheels(self, mock_upload_files):
        # Empty source dir: the planner yields nothing and the script fails
        # fast without calling the backend.
        with self.assertRaises(FileNotFoundError):
            main(
                [
                    "--source-dir",
                    os.fspath(self.source_dir),
                    "--release-type",
                    "dev",
                    "--structured",
                    "--dry-run",
                ]
            )
        mock_upload_files.assert_not_called()


if __name__ == "__main__":
    unittest.main()
