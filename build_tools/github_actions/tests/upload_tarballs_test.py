# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for upload_tarballs.py.

Tests verify that tarball URLs are constructed from the workflow output
destination fields and that multiarch tarballs continue to be exported
correctly even if the filename format changes.
"""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import build_tools.github_actions.upload_tarballs as mod


class TestUploadTarballsMain(unittest.TestCase):
    @patch("build_tools.github_actions.upload_tarballs.gha_set_output")
    @patch("build_tools.github_actions.upload_tarballs.create_storage_backend")
    @patch("build_tools.github_actions.upload_tarballs.WorkflowOutputRoot")
    def test_main_exports_multiarch_url(
        self,
        mock_workflow_output_root,
        mock_create_storage_backend,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            multiarch_tarball = (
                tarballs_dir / "therock-dist-linux-multiarch-7.13.0.tar.gz"
            )
            multiarch_tarball.write_text("x")

            dest = types.SimpleNamespace(
                bucket="therock-dev-artifacts",
                relative_path="25834210506-linux/tarballs",
                s3_uri="s3://therock-dev-artifacts/25834210506-linux/tarballs",
            )

            mock_workflow_output_root.from_workflow_run.return_value.tarballs.return_value = (
                dest
            )
            mock_create_storage_backend.return_value.upload_directory.return_value = 1

            rc = mod.main(
                [
                    "--input-tarballs-dir",
                    str(tarballs_dir),
                    "--run-id",
                    "25834210506",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                ]
            )

            self.assertEqual(rc, 0)
            mock_gha_set_output.assert_called_once()

            payload = mock_gha_set_output.call_args.args[0]
            urls = json.loads(payload["tarball_urls"])

            self.assertEqual(
                urls["multiarch"],
                "https://therock-dev-artifacts.s3.amazonaws.com/"
                "25834210506-linux/tarballs/therock-dist-linux-multiarch-7.13.0.tar.gz",
            )

    @patch("build_tools.github_actions.upload_tarballs.gha_set_output")
    @patch("build_tools.github_actions.upload_tarballs.create_storage_backend")
    @patch("build_tools.github_actions.upload_tarballs.WorkflowOutputRoot")
    def test_main_treats_tarball_without_family_as_multiarch(
        self,
        mock_workflow_output_root,
        mock_create_storage_backend,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            future_multiarch_tarball = tarballs_dir / "therock-dist-linux-7.13.0.tar.gz"
            future_multiarch_tarball.write_text("x")

            dest = types.SimpleNamespace(
                bucket="therock-dev-artifacts",
                relative_path="25834210506-linux/tarballs",
                s3_uri="s3://therock-dev-artifacts/25834210506-linux/tarballs",
            )

            mock_workflow_output_root.from_workflow_run.return_value.tarballs.return_value = (
                dest
            )
            mock_create_storage_backend.return_value.upload_directory.return_value = 1

            rc = mod.main(
                [
                    "--input-tarballs-dir",
                    str(tarballs_dir),
                    "--run-id",
                    "25834210506",
                    "--platform",
                    "linux",
                    "--release-type",
                    "dev",
                ]
            )

            self.assertEqual(rc, 0)
            mock_gha_set_output.assert_called_once()

            payload = mock_gha_set_output.call_args.args[0]
            urls = json.loads(payload["tarball_urls"])

            self.assertEqual(
                urls["multiarch"],
                "https://therock-dev-artifacts.s3.amazonaws.com/"
                "25834210506-linux/tarballs/therock-dist-linux-7.13.0.tar.gz",
            )


if __name__ == "__main__":
    unittest.main()
