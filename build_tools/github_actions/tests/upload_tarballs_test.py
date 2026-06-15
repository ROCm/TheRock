# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for upload_tarballs.py.

Tests verify that tarball URLs are constructed from the workflow output
destination fields and that multiarch tarballs continue to be exported
correctly even if the filename format changes.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add build_tools to path so _therock_utils is importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
# Add github_actions to path so upload_tarballs and github_actions_api are importable.
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import upload_tarballs as mod


class TestUploadTarballsRun(unittest.TestCase):
    @patch("upload_tarballs.gha_set_output")
    def test_run_exports_multiarch_url(
        self,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            multiarch_tarball = (
                tarballs_dir / "therock-dist-linux-multiarch-7.13.0.tar.gz"
            )
            multiarch_tarball.write_text("x")

            staging_dir = tarballs_dir / "staging"
            staging_dir.mkdir()

            rc = mod.run(
                input_tarballs_dir=tarballs_dir,
                run_id="25834210506",
                platform="linux",
                release_type="dev",
                output_dir=staging_dir,
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

    @patch("upload_tarballs.gha_set_output")
    def test_run_treats_tarball_without_family_as_multiarch(
        self,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            future_multiarch_tarball = tarballs_dir / "therock-dist-linux-7.13.0.tar.gz"
            future_multiarch_tarball.write_text("x")

            staging_dir = tarballs_dir / "staging"
            staging_dir.mkdir()
            rc = mod.run(
                input_tarballs_dir=tarballs_dir,
                run_id="25834210506",
                platform="linux",
                release_type="dev",
                output_dir=staging_dir,
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
