# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installation package tests for media libraries (rocdecode, rocjpeg)."""

import importlib
from pathlib import Path
import subprocess
import sys
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk

utils.assert_is_physical_package(rocm_sdk)

core_mod_name = di.ALL_PACKAGES["core"].get_py_package_name()
core_mod = importlib.import_module(core_mod_name)
utils.assert_is_physical_package(core_mod)

MEDIA_LIBRARIES = ["rocdecode", "rocjpeg"]
CORE_PATH = Path(core_mod.__file__).parent


def _find_media_so_files(lib_name: str) -> list[Path]:
    """Returns all .so files for a media library in the core package."""
    return list(CORE_PATH.glob(f"**/lib{lib_name}.so*"))


class ROCmMediaTest(unittest.TestCase):
    def testMediaSharedLibrariesExist(self):
        """Media library .so files must be present in the core package."""
        for lib_name in MEDIA_LIBRARIES:
            with self.subTest(msg=f"Check {lib_name} shared library exists"):
                self.assertTrue(
                    _find_media_so_files(lib_name),
                    msg=f"Expected lib{lib_name}.so* in core package at {CORE_PATH}",
                )

    def testMediaSharedLibrariesLoad(self):
        """Each media library .so must be loadable via ctypes in an isolated process."""
        for lib_name in MEDIA_LIBRARIES:
            for so_path in _find_media_so_files(lib_name):
                if so_path.suffix == ".so" or ".so." in so_path.name:
                    with self.subTest(msg=f"Check {lib_name} loads", so_path=so_path):
                        command = "import ctypes; import sys; ctypes.CDLL(sys.argv[1])"
                        subprocess.check_call(
                            [sys.executable, "-c", command, str(so_path)]
                        )

    def testMediaPreloadLibraries(self):
        """preload_libraries must succeed for each media library."""
        target_family = di.determine_target_family()
        for lib_name in MEDIA_LIBRARIES:
            lib_entry = di.ALL_LIBRARIES.get(lib_name)
            if lib_entry is None:
                continue
            if not lib_entry.package.has_py_package(target_family):
                continue
            with self.subTest(
                msg=f"Check rocm_sdk.preload_libraries('{lib_name}')",
            ):
                rocm_sdk.preload_libraries(lib_name)
