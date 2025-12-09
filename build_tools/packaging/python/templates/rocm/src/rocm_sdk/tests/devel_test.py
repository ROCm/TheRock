"""Test of the library trees."""

"""Installation package tests for the core package."""

import importlib
from pathlib import Path
import platform
import subprocess
import sys
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk


class ROCmDevelTest(unittest.TestCase):
    def testInstallationLayout(self):
        """The `rocm_sdk` and devel module must be siblings on disk."""
        sdk_path = Path(rocm_sdk.__file__)
        self.assertEqual(
            sdk_path.name,
            "__init__.py",
            msg="Expected `rocm_sdk` module to be a non-namespace package",
        )
        import rocm_sdk_devel

        devel_path = Path(rocm_sdk_devel.__file__)
        self.assertEqual(
            devel_path.name,
            "__init__.py",
            msg=f"Expected `rocm_sdk_devel` module to be a non-namespace package",
        )
        self.assertEqual(
            sdk_path.parent.parent,
            devel_path.parent.parent,
            msg="Paths are not siblings",
        )

    def testCLIPathBin(self):
        output = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "path", "--bin"], capture=True
            )
            .decode()
            .strip()
        )
        path = Path(output)
        self.assertTrue(path.exists(), msg=f"Expected bin path {path} to exist")

    def testCLIPathCMake(self):
        output = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "path", "--cmake"],
                capture=True,
            )
            .decode()
            .strip()
        )
        path = Path(output)
        self.assertTrue(path.exists(), msg=f"Expected cmake path {path} to exist")
        hip_file = path / "hip" / "hip-config.cmake"
        self.assertTrue(
            hip_file.exists(), msg=f"Expected hip config to exist {hip_file}"
        )

    def testCLIPathRoot(self):
        output = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "path", "--root"], capture=True
            )
            .decode()
            .strip()
        )
        path = Path(output)
        self.assertTrue(path.exists(), msg=f"Expected root path {path} to exist")
        bin_path = path / "bin"
        self.assertTrue(bin_path.exists(), msg=f"Expected bin path {bin_path} to exist")

    @unittest.skipIf(
        platform.system() == "Windows", "root LLVM symlink only exists on Linux"
    )
    def testRootLLVMSymlinkExists(self):
        # We had a bug where the root llvm/ symlink, which is for backwards compat,
        # was not materialized. Verify it is.
        output = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "path", "--root"], capture=True
            )
            .decode()
            .strip()
        )
        path = Path(output) / "llvm" / "bin" / "clang++"
        self.assertTrue(path.exists(), msg=f"Expected {path} to exist")

    def testSharedLibrariesLoad(self):
        # Make sure the devel package is expanded.
        _ = (
            utils.exec(
                [sys.executable, "-P", "-m", "rocm_sdk", "path", "--root"], capture=True
            )
            .decode()
            .strip()
        )

        # Ensure that the platform package exists now.
        mod_name = di.ALL_PACKAGES["devel"].get_py_package_name(
            target_family=di.determine_target_family()
        )
        mod = importlib.import_module(mod_name)
        utils.assert_is_physical_package(mod)
        so_paths = utils.get_module_shared_libraries(mod)

        self.assertTrue(
            so_paths, msg="Expected core package to contain shared libraries"
        )

        for so_path in so_paths:
            if "amd_smi" in str(so_path) or "goamdsmi" in str(so_path):
                # TODO: Library preloads for amdsmi need to be implement.
                # Though this is not needed for the amd-smi client.
                self.skipTest("Skipping amdsmi test")
            if "clang_rt" in str(so_path):
                # clang_rt and sanitizer libraries are not all intended to be
                # loadable arbitrarily.
                continue
            if "lib/roctracer" in str(so_path) or "share/roctracer" in str(so_path):
                # Internal roctracer libraries are meant to be pre-loaded
                # explicitly and cannot necessarily be loaded standalone.
                continue
            if "lib/rocprofiler-sdk/" in str(
                so_path
            ) or "libexec/rocprofiler-sdk/" in str(so_path):
                # Internal rocprofiler-sdk libraries are meant to be pre-loaded
                # explicitly and cannot necessarily be loaded standalone.
                continue
            with self.subTest(msg="Check shared library loads", so_path=so_path):
                # Load each in an isolated process because not all libraries in the tree
                # are designed to load into the same process (i.e. LLVM runtime libs,
                # etc).
                command = "import ctypes; import sys; ctypes.CDLL(sys.argv[1])"
                subprocess.check_call(
                    [sys.executable, "-P", "-c", command, str(so_path)]
                )
