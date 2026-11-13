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

    def test_enabled_by_checkout_dir_on_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_args(pytorch_torchcodec_dir=Path(temp_dir))
            with mock.patch.object(build_prod_wheels, "is_windows", True):
                build_prod_wheels.validate_build_args(argparse.ArgumentParser(), args)
            self.assertTrue(args.build_pytorch_torchcodec)

    def test_explicit_build_on_windows_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_args(
                pytorch_torchcodec_dir=Path(temp_dir), build_pytorch_torchcodec=True
            )
            with mock.patch.object(build_prod_wheels, "is_windows", True):
                build_prod_wheels.validate_build_args(argparse.ArgumentParser(), args)
            self.assertTrue(args.build_pytorch_torchcodec)

    def test_no_checkout_dir_does_not_enable_the_build(self):
        args = self.make_args()
        with mock.patch.object(build_prod_wheels, "is_windows", True):
            build_prod_wheels.validate_build_args(argparse.ArgumentParser(), args)
        self.assertFalse(args.build_pytorch_torchcodec)

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

    def test_windows_toolchain_is_set_up_only_on_windows(self):
        args = argparse.Namespace(
            version_suffix="",
            release_type="dev",
            pytorch_git_ref="nightly",
            clean=False,
        )
        for is_windows in (True, False):
            with self.subTest(is_windows=is_windows):
                with mock.patch.multiple(
                    build_prod_wheels,
                    is_windows=is_windows,
                    check_torchcodec_build_requirements=mock.DEFAULT,
                    setup_torchcodec_windows_toolchain=mock.DEFAULT,
                    compute_build_version=mock.DEFAULT,
                    remove_dir_if_exists=mock.DEFAULT,
                    run_command=mock.DEFAULT,
                    find_built_wheel=mock.DEFAULT,
                    copy_to_output=mock.DEFAULT,
                ) as mocks:
                    mocks["compute_build_version"].return_value = "0.17.0"
                    build_prod_wheels.do_build_pytorch_torchcodec(
                        args, Path("unused"), {}
                    )
                setup = mocks["setup_torchcodec_windows_toolchain"]
                self.assertEqual(setup.call_count, 1 if is_windows else 0)


class WindowsToolchainTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bin_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_env(self) -> dict[str, str]:
        return {
            "PATH": os.pathsep.join([self.bin_dir, "unrelated"]),
            "CC": "clang-cl.exe",
            "CXX": "clang-cl.exe",
        }

    def add_cl(self):
        cl = Path(self.bin_dir) / ("cl.exe" if os.name == "nt" else "cl")
        cl.write_text("")
        cl.chmod(0o755)

    def test_missing_msvc_environment_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "MSVC environment"):
            build_prod_wheels.setup_torchcodec_windows_toolchain(self.make_env())

    def test_clang_cl_is_the_default_compiler(self):
        self.add_cl()
        env = self.make_env()
        with mock.patch.dict(os.environ):
            os.environ.pop(build_prod_wheels.TORCHCODEC_WINDOWS_COMPILER_VAR, None)
            build_prod_wheels.setup_torchcodec_windows_toolchain(env)
        self.assertEqual(env["CC"], "clang-cl.exe")
        self.assertEqual(env["CXX"], "clang-cl.exe")
        self.assertEqual(env["DISTUTILS_USE_SDK"], "1")

    def test_msvc_removes_the_common_compiler(self):
        self.add_cl()
        env = self.make_env()
        with mock.patch.dict(
            os.environ, {build_prod_wheels.TORCHCODEC_WINDOWS_COMPILER_VAR: "msvc"}
        ):
            build_prod_wheels.setup_torchcodec_windows_toolchain(env)
        self.assertNotIn("CC", env)
        self.assertNotIn("CXX", env)
        self.assertEqual(env["DISTUTILS_USE_SDK"], "1")

    def test_unknown_compiler_is_an_error(self):
        self.add_cl()
        with mock.patch.dict(
            os.environ, {build_prod_wheels.TORCHCODEC_WINDOWS_COMPILER_VAR: "gcc"}
        ):
            with self.assertRaisesRegex(RuntimeError, "gcc"):
                build_prod_wheels.setup_torchcodec_windows_toolchain(self.make_env())


if __name__ == "__main__":
    unittest.main()
