# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the torchvision codec pre-build helpers."""

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(
    0, os.fspath(Path(__file__).parent.parent.parent / "external-builds" / "pytorch")
)

from build_prod_wheels import (  # noqa: E402
    _assert_vision_image_codecs,
    _insert_conda_bin_on_path,
)


class InsertCondaBinOnPathTest(unittest.TestCase):
    def test_conda_follows_build_python_and_is_not_duplicated(self):
        python_bin = str(Path(sys.executable).resolve().parent)
        prefix = Path("/opt/conda")
        path = _insert_conda_bin_on_path(
            os.pathsep.join([python_bin, "/usr/bin"]), prefix
        )
        path = _insert_conda_bin_on_path(path, prefix)
        parts = path.split(os.pathsep)
        self.assertEqual(parts, [python_bin, str(prefix / "bin"), "/usr/bin"])

    def test_conda_does_not_precede_an_earlier_rocm_bin(self):
        python_bin = str(Path(sys.executable).resolve().parent)
        prefix = Path("/opt/conda")
        path = _insert_conda_bin_on_path(
            os.pathsep.join(["/opt/rocm/bin", python_bin, "/usr/bin"]), prefix
        )
        parts = path.split(os.pathsep)
        self.assertEqual(
            parts, ["/opt/rocm/bin", python_bin, str(prefix / "bin"), "/usr/bin"]
        )


class VisionImageCodecCheckTest(unittest.TestCase):
    def _wheel(self, payload: bytes) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        wheel = Path(directory.name) / "torchvision-0.29.0-py3-none-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("torchvision/image_stable.so", payload)
        return wheel

    def test_rejects_stub_codec_strings(self):
        wheel = self._wheel(b"decode_png: torchvision not compiled with libPNG support")
        with self.assertRaises(RuntimeError):
            _assert_vision_image_codecs(wheel)

    def test_accepts_extension_without_stub_strings(self):
        wheel = self._wheel(b"\x7fELFimage")
        _assert_vision_image_codecs(wheel)

    def test_rejects_wheel_without_image_extension(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        wheel = Path(directory.name) / "torchvision-0.29.0-py3-none-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("torchvision/__init__.py", b"")
        with self.assertRaises(RuntimeError):
            _assert_vision_image_codecs(wheel)
