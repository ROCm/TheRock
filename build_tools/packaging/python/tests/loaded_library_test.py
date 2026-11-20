# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the SDK package's loaded-library checks."""

import base64
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import platform
import tempfile
import unittest
from unittest.mock import patch

UTILS_PATH = Path(__file__).parent.parent / "templates/rocm/src/rocm_sdk/tests/utils.py"
spec = importlib.util.spec_from_file_location("sdk_test_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class LoadedLibraryTest(unittest.TestCase):
    def test_loader_path(self):
        name = "kernel32.dll" if platform.system() == "Windows" else "libdl.so.2"
        library = ctypes.CDLL(name)
        actual = utils.get_loaded_library_path(library)
        self.assertTrue(actual.is_absolute())
        self.assertTrue(actual.is_file())
        # Use the independently resolved file to reopen by its absolute path.
        self.assertEqual(
            utils.get_loaded_library_path(ctypes.CDLL(str(actual))), actual
        )

    def test_replaced_bytes_and_conflicting_path(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "package" / "runtime.dll"
            expected.parent.mkdir()
            original = b"version=1; approved runtime"
            expected.write_bytes(original)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(original).digest())
                .decode()
                .rstrip("=")
            )
            with patch.object(utils, "get_loaded_library_path", return_value=expected):
                utils.assert_loaded_library(None, expected, digest)
                expected.write_bytes(b"version=1; modified runtime")
                with self.assertRaisesRegex(AssertionError, "SHA-256 differs"):
                    utils.assert_loaded_library(None, expected, digest)
            conflict = Path(directory) / "package-old" / "runtime.dll"
            conflict.parent.mkdir()
            conflict.write_bytes(original)
            with patch.object(utils, "get_loaded_library_path", return_value=conflict):
                with self.assertRaisesRegex(
                    AssertionError, "Unexpected loaded library"
                ):
                    utils.assert_loaded_library(None, expected, digest)

    def test_missing_record_fails(self):
        with patch.object(utils.importlib.metadata, "files", return_value=None):
            with self.assertRaisesRegex(AssertionError, "not in.*RECORD"):
                utils.get_record_sha256("empty-package", Path(__file__))


if __name__ == "__main__":
    unittest.main()
