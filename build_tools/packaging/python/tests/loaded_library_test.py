# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the SDK package's loaded-library checks."""

import base64
import _ctypes
import csv
import ctypes
import hashlib
import importlib.util
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

UTILS_PATH = Path(__file__).parent.parent / "templates/rocm/src/rocm_sdk/tests/utils.py"
spec = importlib.util.spec_from_file_location("sdk_test_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class LoadedLibraryTest(unittest.TestCase):
    def test_record_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "fixture" / "runtime.dll"
            library.parent.mkdir()
            library.write_bytes(b"fixture runtime")
            info = root / "loaded_library_fixture-1.0.dist-info"
            info.mkdir()
            (info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: loaded-library-fixture\nVersion: 1.0\n",
                encoding="utf-8",
            )
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(library.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            with patch.object(sys, "path", [str(root), *sys.path]):
                with self.assertRaisesRegex(AssertionError, "not in.*RECORD"):
                    utils.get_record_sha256(["loaded-library-fixture"], library)
                for recorded_hash in (f"sha256={digest}", "", "sha512=unused"):
                    with self.subTest(recorded_hash=recorded_hash):
                        with (info / "RECORD").open(
                            "w", newline="", encoding="utf-8"
                        ) as stream:
                            csv.writer(stream).writerows(
                                [
                                    ["fixture/unrelated.dll", f"sha256={digest}", "15"],
                                    [
                                        "fixture/runtime.dll",
                                        recorded_hash,
                                        str(library.stat().st_size),
                                    ],
                                ]
                            )
                        if recorded_hash.startswith("sha256="):
                            self.assertEqual(
                                utils.get_record_sha256(
                                    ["loaded-library-fixture"], library
                                ),
                                digest,
                            )
                            if not utils.is_windows:
                                alias = root / "alias.so"
                                alias.symlink_to(library)
                                self.assertEqual(
                                    utils.get_record_sha256(
                                        ["loaded-library-fixture"], alias
                                    ),
                                    digest,
                                )
                            with self.assertRaisesRegex(
                                AssertionError, "not in.*RECORD"
                            ):
                                utils.get_record_sha256(
                                    ["loaded-library-fixture"], Path(__file__)
                                )
                        else:
                            with self.assertRaisesRegex(
                                AssertionError, "Missing SHA-256"
                            ):
                                utils.get_record_sha256(
                                    ["loaded-library-fixture"], library
                                )

    def test_device_wheel_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "_rocm_fixture_libraries" / "bin" / "device.dll"
            library.parent.mkdir(parents=True)
            library.write_bytes(b"device wheel payload")
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(library.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            for name in ("rocm_fixture_libraries", "rocm_fixture_device"):
                info = root / f"{name}-1.0.dist-info"
                info.mkdir()
                (info / "METADATA").write_text(
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
                    encoding="utf-8",
                )
                (info / "top_level.txt").write_text(
                    "_rocm_fixture_libraries\n", encoding="utf-8"
                )
                payload = "bin/device.dll" if name.endswith("device") else "__init__.py"
                (info / "RECORD").write_text(
                    f"_rocm_fixture_libraries/{payload},sha256={digest},20\n",
                    encoding="utf-8",
                )
            with patch.object(sys, "path", [str(root), *sys.path]):
                owners = utils.importlib.metadata.packages_distributions()[
                    "_rocm_fixture_libraries"
                ]
                self.assertEqual(
                    set(owners), {"rocm_fixture_libraries", "rocm_fixture_device"}
                )
                with self.assertRaisesRegex(AssertionError, "not in.*RECORD"):
                    utils.get_record_sha256(["rocm_fixture_libraries"], library)
                self.assertEqual(utils.get_record_sha256(owners, library), digest)

    def test_real_loaded_library_in_subprocess(self):
        # Reuse a compiled CPython extension as a loadable fixture. Each child
        # exits before cleanup, so Windows never has to delete a loaded DLL.
        command = """
import ctypes
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location("sdk_test_utils", sys.argv[1])
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
library = ctypes.CDLL(sys.argv[2])
utils.assert_loaded_library(library, Path(sys.argv[3]), sys.argv[4])
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "package" / Path(_ctypes.__file__).name
            conflict = root / "package-old" / expected.name
            for target in (expected, conflict):
                target.parent.mkdir()
                shutil.copyfile(_ctypes.__file__, target)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(expected.read_bytes()).digest())
                .decode()
                .rstrip("=")
            )
            for loaded_path, failure in (
                (expected, None),
                (conflict, "Unexpected loaded library"),
            ):
                with self.subTest(loaded_path=loaded_path):
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            command,
                            str(UTILS_PATH.resolve()),
                            str(loaded_path),
                            str(expected),
                            digest,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if failure is None:
                        self.assertEqual(result.returncode, 0, result.stderr)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(failure, result.stderr)
            # Trailing bytes preserve the loadable binary and its version
            # information while changing its digest.
            with expected.open("ab") as stream:
                stream.write(b"modified after installation")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    command,
                    str(UTILS_PATH.resolve()),
                    str(expected),
                    str(expected),
                    digest,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 differs", result.stderr)

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


if __name__ == "__main__":
    unittest.main()
