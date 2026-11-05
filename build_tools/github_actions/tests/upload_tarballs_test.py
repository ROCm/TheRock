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
    def test_run_exports_multiarch_url_with_other_tarballs_present(
        self,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)

            # This is the default "multiarch" tarball that will have its link
            # posted to github outputs.
            base_multiarch_tarball = (
                tarballs_dir
                / "therock-dist-linux-multiarch-7.14.0.dev0+13caf791.tar.gz"
            )
            base_multiarch_tarball.write_text("x")

            # A base per-family tarball (URL not posted).
            base_family_tarball = (
                tarballs_dir
                / "therock-dist-linux-gfx94X-dcgpu-7.14.0.dev0+13caf791.tar.gz"
            )
            base_family_tarball.write_text("x")

            # A test multiarch tarball (URL not posted).
            test_multiarch_tarball = (
                tarballs_dir
                / "therock-dist-linux-multiarch-tests-7.14.0.dev0+13caf791.tar.gz"
            )
            test_multiarch_tarball.write_text("x")

            # A test per-family tarball (URL not posted).
            test_family_tarball = (
                tarballs_dir
                / "therock-dist-linux-gfx94X-dcgpu-tests-7.14.0.dev0+13caf791.tar.gz"
            )
            test_family_tarball.write_text("x")

            staging_dir = tarballs_dir / "staging"
            staging_dir.mkdir()
            rc = mod.run(
                input_tarballs_dir=tarballs_dir,
                run_id="27993936036",
                platform="linux",
                release_type="dev",
                output_dir=staging_dir,
            )
            self.assertEqual(rc, 0)

            # Should have set a single output - just the base multiarch tarball.
            mock_gha_set_output.assert_called_once()
            payload = mock_gha_set_output.call_args.args[0]
            urls = json.loads(payload["tarball_urls"])
            self.assertEqual(
                urls["multiarch"],
                "https://therock-dev-artifacts.s3.amazonaws.com/"
                "27993936036-linux/tarballs/"
                "therock-dist-linux-multiarch-7.14.0.dev0%2B13caf791.tar.gz",
            )

    @patch("upload_tarballs.gha_set_output")
    def test_run_rejects_base_family_urls_without_multiarch(
        self,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            # A base per-family tarball that should *not* be treated as
            # "multiarch", this is an error.
            gfx94x_tarball = (
                tarballs_dir / "therock-dist-linux-gfx94X-dcgpu-7.13.0.tar.gz"
            )
            gfx94x_tarball.write_text("x")

            staging_dir = tarballs_dir / "staging"
            staging_dir.mkdir()
            with self.assertRaisesRegex(ValueError, "No multiarch tarball URL"):
                mod.run(
                    input_tarballs_dir=tarballs_dir,
                    run_id="25834210506",
                    platform="linux",
                    release_type="dev",
                    output_dir=staging_dir,
                )

            mock_gha_set_output.assert_not_called()

    @patch("upload_tarballs.gha_set_output")
    def test_run_exports_per_family_urls(
        self,
        mock_gha_set_output,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            version = "7.14.0.dev0+13caf791"

            for name in (
                f"therock-dist-linux-multiarch-{version}.tar.gz",
                f"therock-dist-linux-gfx94X-dcgpu-{version}.tar.gz",
                f"therock-dist-linux-gfx950-dcgpu-{version}.tar.gz",
                # Test companions are not distros and must not be exported.
                f"therock-dist-linux-gfx950-dcgpu-tests-{version}.tar.gz",
                f"therock-dist-linux-multiarch-tests-{version}.tar.gz",
            ):
                (tarballs_dir / name).write_text("x")

            staging_dir = tarballs_dir / "staging"
            staging_dir.mkdir()
            rc = mod.run(
                input_tarballs_dir=tarballs_dir,
                run_id="27993936036",
                platform="linux",
                release_type="dev",
                output_dir=staging_dir,
            )
            self.assertEqual(rc, 0)

            payload = mock_gha_set_output.call_args.args[0]
            urls = json.loads(payload["tarball_urls"])

            self.assertEqual(
                sorted(urls),
                [
                    "gfx94X-dcgpu",
                    "gfx94X_dcgpu",
                    "gfx950-dcgpu",
                    "gfx950_dcgpu",
                    "multiarch",
                ],
            )
            # A family name contains "-", so it must survive version stripping
            # intact rather than being split on the last dash.
            expected = (
                "https://therock-dev-artifacts.s3.amazonaws.com/"
                "27993936036-linux/tarballs/"
                "therock-dist-linux-gfx950-dcgpu-7.14.0.dev0%2B13caf791.tar.gz"
            )
            self.assertEqual(urls["gfx950-dcgpu"], expected)
            # The underscore alias is what a GitHub expression can reach with
            # dot access, and must point at the same tarball.
            self.assertEqual(urls["gfx950_dcgpu"], expected)

    @patch("upload_tarballs.gha_set_output")
    def test_run_exports_every_family_name_shape(
        self,
        mock_gha_set_output,
    ) -> None:
        """Covers the family name shapes amdgpu_family_matrix.py actually uses.

        A bare name (gfx1151), a "-dcgpu" / "-all" suffixed one, and an
        uppercase-infix one each have to come back as the key, with only the
        version stripped off the end.
        """
        families = [
            "gfx908",
            "gfx1151",
            "gfx94X-dcgpu",
            "gfx950-dcgpu",
            "gfx110X-all",
            "gfx1250-strict",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tarballs_dir = Path(tmpdir)
            # A plain release version, distinct in shape from the dev version
            # used above, so stripping is covered against both.
            version = "7.13.0"

            (
                tarballs_dir / f"therock-dist-linux-multiarch-{version}.tar.gz"
            ).write_text("x")
            for family in families:
                (
                    tarballs_dir / f"therock-dist-linux-{family}-{version}.tar.gz"
                ).write_text("x")
                # Each family's test companion must stay out of the output.
                (
                    tarballs_dir / f"therock-dist-linux-{family}-tests-{version}.tar.gz"
                ).write_text("x")

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

            payload = mock_gha_set_output.call_args.args[0]
            urls = json.loads(payload["tarball_urls"])

            aliases = [f.replace("-", "_") for f in families]
            self.assertEqual(
                sorted(urls), sorted(set(families + aliases + ["multiarch"]))
            )
            for family in families:
                # A dash-free family is its own alias, so both lookups land on
                # the single key emitted for it.
                self.assertEqual(urls[family], urls[family.replace("-", "_")])
                self.assertEqual(
                    urls[family],
                    "https://therock-dev-artifacts.s3.amazonaws.com/"
                    "25834210506-linux/tarballs/"
                    f"therock-dist-linux-{family}-{version}.tar.gz",
                )


if __name__ == "__main__":
    unittest.main()
