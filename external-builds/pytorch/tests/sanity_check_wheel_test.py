# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the torchvision image codec check in sanity_check_wheel.py."""

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sanity_check_wheel import missing_image_codecs

NO_PNG = b"decode_png: torchvision not compiled with libPNG support"
NO_JPEG = b"decode_jpeg: torchvision not compiled with libjpeg support"


def _wheel(directory: Path, files: dict[str, bytes]) -> Path:
    wheel = directory / "torchvision-0.0.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return wheel


class MissingImageCodecsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reports_missing_codecs(self):
        wheel = _wheel(
            self.dir, {"torchvision/image_stable.so": b"\0" + NO_PNG + b"\0" + NO_JPEG}
        )
        self.assertEqual(missing_image_codecs(wheel), ["libjpeg", "libpng"])

    def test_codecs_present(self):
        wheel = _wheel(self.dir, {"torchvision/image_stable.so": b"\0png_read\0"})
        self.assertEqual(missing_image_codecs(wheel), [])

    def test_ignores_other_libraries(self):
        wheel = _wheel(self.dir, {"torchvision/_C.so": NO_PNG})
        self.assertEqual(missing_image_codecs(wheel), [])

    def test_no_image_extension(self):
        wheel = _wheel(self.dir, {"torchvision/__init__.py": b""})
        self.assertEqual(missing_image_codecs(wheel), [])


if __name__ == "__main__":
    unittest.main()
