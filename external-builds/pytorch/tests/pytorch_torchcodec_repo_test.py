# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for torchcodec ref selection and build checks."""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import build_prod_wheels
import pytorch_torchcodec_repo as repo


class SelectHashtagTest(unittest.TestCase):
    def test_default_tag_for_supported_torch(self):
        for torch_version in ("2.11", "2.13.0a0", "2.14.0", "2.15.0a0", "3.0.0"):
            with self.subTest(torch_version=torch_version):
                self.assertEqual(
                    repo.select_hashtag(torch_version), repo.DEFAULT_STABLE_TAG
                )

    def test_no_torch_version_uses_default_tag(self):
        self.assertEqual(repo.select_hashtag(None), repo.DEFAULT_STABLE_TAG)

    def test_torch_below_minimum_without_override_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "torch 2.10"):
            repo.select_hashtag("2.10.0")

    def test_override_wins_and_allows_older_torch(self):
        overrides = {"2.10": "v0.10.0", "2.13": "v0.15.0"}
        with mock.patch.object(repo, "TORCH_VERSION_TO_TAG", overrides):
            self.assertEqual(repo.select_hashtag("2.10.1"), "v0.10.0")
            self.assertEqual(repo.select_hashtag("2.13.0a0"), "v0.15.0")
            self.assertEqual(repo.select_hashtag("2.14.0"), repo.DEFAULT_STABLE_TAG)

    def test_unparsable_torch_version(self):
        with self.assertRaises(ValueError):
            repo.select_hashtag("nightly")

    def test_read_torch_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            torch_dir = Path(temp_dir)
            self.assertIsNone(repo.read_torch_version(torch_dir))
            (torch_dir / "version.txt").write_text("2.13.0a0\n")
            self.assertEqual(repo.read_torch_version(torch_dir), "2.13.0a0")


class BuildArgsTest(unittest.TestCase):
    def make_args(self, **overrides) -> argparse.Namespace:
        args = argparse.Namespace(
            build_triton=False,
            build_pytorch_audio=False,
            build_pytorch_vision=False,
            build_pytorch_torchcodec=None,
            build_apex=False,
            triton_dir=None,
            pytorch_audio_dir=None,
            pytorch_vision_dir=None,
            pytorch_torchcodec_dir=None,
            apex_dir=None,
            pytorch_dir=None,
            enable_pytorch_flash_attention=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_enabled_by_checkout_dir_on_linux(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_args(pytorch_torchcodec_dir=Path(temp_dir))
            with mock.patch.object(build_prod_wheels, "is_windows", False):
                build_prod_wheels.validate_build_args(argparse.ArgumentParser(), args)
            self.assertTrue(args.build_pytorch_torchcodec)

    def test_checkout_dir_is_ignored_on_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_args(pytorch_torchcodec_dir=Path(temp_dir))
            with mock.patch.object(build_prod_wheels, "is_windows", True):
                build_prod_wheels.validate_build_args(argparse.ArgumentParser(), args)
            self.assertFalse(args.build_pytorch_torchcodec)

    def test_explicit_build_on_windows_is_an_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_args(
                pytorch_torchcodec_dir=Path(temp_dir), build_pytorch_torchcodec=True
            )
            with mock.patch.object(build_prod_wheels, "is_windows", True):
                with self.assertRaises(SystemExit):
                    build_prod_wheels.validate_build_args(
                        argparse.ArgumentParser(), args
                    )

    def test_build_requires_source_dir(self):
        args = self.make_args(build_pytorch_torchcodec=True)
        with mock.patch.object(build_prod_wheels, "is_windows", False):
            with self.assertRaises(SystemExit):
                build_prod_wheels.validate_build_args(argparse.ArgumentParser(), args)


class BuildStepTest(unittest.TestCase):
    def test_newer_python_is_skipped_without_building(self):
        args = argparse.Namespace()
        with mock.patch.object(build_prod_wheels.sys, "version_info", (3, 15, 0)):
            with mock.patch.object(build_prod_wheels, "run_command") as run_command:
                with mock.patch.object(
                    build_prod_wheels, "check_torchcodec_build_requirements"
                ) as check:
                    build_prod_wheels.do_build_pytorch_torchcodec(
                        args, Path("unused"), {}
                    )
        run_command.assert_not_called()
        check.assert_not_called()

    def test_missing_scikit_build_core_is_an_error(self):
        with mock.patch.object(
            build_prod_wheels.importlib.metadata,
            "version",
            side_effect=build_prod_wheels.importlib.metadata.PackageNotFoundError,
        ):
            with self.assertRaisesRegex(RuntimeError, "scikit-build-core"):
                build_prod_wheels.check_torchcodec_build_requirements()

    def test_old_scikit_build_core_is_an_error(self):
        with mock.patch.object(
            build_prod_wheels.importlib.metadata, "version", return_value="0.9.0"
        ):
            with self.assertRaisesRegex(RuntimeError, "0.9.0"):
                build_prod_wheels.check_torchcodec_build_requirements()


if __name__ == "__main__":
    unittest.main()
