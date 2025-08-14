"""Installation package tests for the core package."""

import importlib
import locale
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk

utils.assert_is_physical_package(rocm_sdk)

core_mod_name = di.ALL_PACKAGES["core"].get_py_package_name()
core_mod = importlib.import_module(core_mod_name)
utils.assert_is_physical_package(core_mod)

so_paths = utils.get_module_shared_libraries(core_mod)
is_windows = platform.system() == "Windows"

LINUX_CONSOLE_SCRIPT_TESTS = [
    # These currently only have unprefixed names (e.g. 'clang') on Windows.
    # These tools are only available on Linux.
    ("rocm_agent_enumerator", [], "", True),
    ("rocminfo", [], "", True),
    ("rocm-smi", [], "Management", True),
]

CONSOLE_SCRIPT_TESTS = [
    ("amdclang", ["--help"], "clang LLVM compiler", True),
    ("amdclang++", ["--help"], "clang LLVM compiler", True),
    ("amdclang-cpp", ["--help"], "clang LLVM compiler", True),
    ("amdclang-cl", ["-help"], "clang LLVM compiler", True),
    ("amdflang", ["--help"], "clang LLVM compiler", True),
    ("amdlld", ["-flavor", "ld.lld", "--help"], "USAGE:", True),
    ("hipcc", ["--help"], "clang LLVM compiler", True),
    ("hipconfig", [], "HIP version:", True),
] + (LINUX_CONSOLE_SCRIPT_TESTS if not is_windows else [])


class ROCmCoreTest(unittest.TestCase):
    def testInstallationLayout(self):
        """The `rocm_sdk` and core module must be siblings on disk."""
        sdk_path = Path(rocm_sdk.__file__)
        self.assertEqual(
            sdk_path.name,
            "__init__.py",
            msg="Expected `rocm_sdk` module to be a non-namespace package",
        )
        core_path = Path(core_mod.__file__)
        self.assertEqual(
            core_path.name,
            "__init__.py",
            msg="Expected core module to be a non-namespace package",
        )
        self.assertEqual(
            sdk_path.parent.parent,
            core_path.parent.parent,
            msg="Paths are not siblings",
        )

    def testSharedLibrariesLoad(self):
        self.assertTrue(
            so_paths, msg="Expected core package to contain shared libraries"
        )

        for so_path in so_paths:
            if "clang_rt" in so_path.name:
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

    def testConsoleScripts(self):
        for script_name, cl, expected_text, required in CONSOLE_SCRIPT_TESTS:
            script_path = utils.find_console_script(script_name)
            if not required and script_path is None:
                continue
            with self.subTest(msg=f"Check console-script {script_name}"):
                self.assertIsNotNone(
                    script_path,
                    msg=f"Console script {script_path} does not exist",
                )
                encoding = locale.getpreferredencoding()
                output_text = subprocess.check_output(
                    [script_path] + cl,
                    stderr=subprocess.STDOUT,
                ).decode(encoding)
                if expected_text not in output_text:
                    self.fail(
                        f"Expected '{expected_text}' in console-script {script_name} outuput:\n"
                        f"{output_text}"
                    )

    def testPreloadLibraries(self):
        target_family = di.determine_target_family()

        for lib_entry in di.ALL_LIBRARIES.values():
            # Only test for packages we have installed.
            if lib_entry.package.has_py_package(target_family):
                with self.subTest(
                    msg="Check rocm_sdk.preload_libraries",
                    shortname=lib_entry.shortname,
                ):
                    rocm_sdk.preload_libraries(lib_entry.shortname)
