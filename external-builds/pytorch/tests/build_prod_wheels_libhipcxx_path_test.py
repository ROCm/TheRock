# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import build_prod_wheels
from build_prod_wheels import add_libhipcxx_include_path


class AddLibhipcxxIncludePathTest(unittest.TestCase):
    def _run(self, *, make_dir: bool, env_cpath=None, os_cpath=None):
        with tempfile.TemporaryDirectory() as tmp:
            rocm = Path(tmp)
            hipccl = rocm / "include" / "hipccl"
            if make_dir:
                hipccl.mkdir(parents=True)
            env = {"ROCM_PATH": os.fspath(rocm)}
            if env_cpath is not None:
                env["CPATH"] = env_cpath
            os_env = {} if os_cpath is None else {"CPATH": os_cpath}
            with patch.dict(os.environ, os_env, clear=True):
                add_libhipcxx_include_path(env)
            return env, os.fspath(hipccl)

    def test_adds_hipccl_include_dir(self):
        env, hipccl = self._run(make_dir=True)
        self.assertEqual(env["CPATH"], hipccl)

    def test_missing_directory_leaves_env_untouched(self):
        env, _ = self._run(make_dir=False)
        self.assertNotIn("CPATH", env)

    def test_prepends_to_cpath_in_env(self):
        env, hipccl = self._run(make_dir=True, env_cpath="/a")
        self.assertEqual(env["CPATH"], os.pathsep.join([hipccl, "/a"]))

    def test_keeps_cpath_inherited_from_process_environment(self):
        env, hipccl = self._run(make_dir=True, os_cpath="/b")
        self.assertEqual(env["CPATH"], os.pathsep.join([hipccl, "/b"]))


class TorchvisionBuildEnvTest(unittest.TestCase):
    def test_torchvision_build_gets_libhipcxx_on_cpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            rocm = Path(tmp) / "rocm"
            (rocm / "include" / "hipccl").mkdir(parents=True)
            args = Namespace(
                version_suffix="",
                release_type="dev",
                pytorch_git_ref="nightly",
                pytorch_build_number="1",
                clean=False,
            )
            env = {"ROCM_PATH": os.fspath(rocm)}
            with patch.object(
                build_prod_wheels, "compute_build_version", return_value="1.0"
            ), patch.object(
                build_prod_wheels, "get_installed_package_version", return_value="2"
            ), patch.object(
                build_prod_wheels, "remove_dir_if_exists"
            ), patch.object(
                build_prod_wheels, "find_built_wheel"
            ), patch.object(
                build_prod_wheels, "copy_to_output"
            ), patch.object(
                build_prod_wheels, "run_command"
            ) as run_command:
                build_prod_wheels.do_build_pytorch_vision(args, Path(tmp), env)
        run_env = run_command.call_args.kwargs["env"]
        self.assertIn(os.fspath(rocm / "include" / "hipccl"), run_env["CPATH"])


if __name__ == "__main__":
    unittest.main()
