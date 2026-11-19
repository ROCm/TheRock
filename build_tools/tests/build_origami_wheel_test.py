# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise the wheel-builder lifecycle with a real local PEP 517 backend."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_origami_wheel import build_wheel


BACKEND = """import json
from pathlib import Path
import zipfile

def get_requires_for_build_wheel(config_settings=None):
    return []

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    if FAILURE:
        raise RuntimeError("intentional backend failure")
    filename = "fixture-1.0-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / filename, "w") as wheel:
        wheel.writestr("fixture/settings.json", json.dumps(config_settings))
        wheel.writestr("fixture-1.0.dist-info/METADATA", "Metadata-Version: 2.1\\nName: fixture\\nVersion: 1.0\\n")
        wheel.writestr("fixture-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n")
        wheel.writestr("fixture-1.0.dist-info/RECORD", "")
    return filename
"""


class BuildOrigamiWheelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source with spaces"
        self.source.mkdir()
        self.rocm = self.root / "installed ROCm"
        self.rocm.mkdir()
        self.output = self.root / "wheels"
        (self.source / "pyproject.toml").write_text(
            '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n'
        )
        (self.source / "backend.py").write_text(BACKEND.replace("FAILURE", "False"))
        self.original_files = {p.name: p.read_bytes() for p in self.source.iterdir()}

    def assert_source_unchanged(self):
        self.assertEqual(
            {
                str(p.relative_to(self.source)): p.read_bytes()
                for p in self.source.rglob("*")
                if p.is_file()
            },
            self.original_files,
        )

    def test_installed_prefix_and_external_build_directory(self):
        build_wheel(self.source, self.rocm, self.output, isolation=False)
        with zipfile.ZipFile(next(self.output.glob("*.whl"))) as wheel:
            settings = json.loads(wheel.read("fixture/settings.json"))
        self.assertEqual(settings["cmake.define.ORIGAMI_BUILD_FROM_SOURCE"], "OFF")
        self.assertEqual(
            settings["cmake.define.CMAKE_PREFIX_PATH"], str(self.rocm.resolve())
        )
        build_directory = Path(settings["build-dir"])
        self.assertNotIn(self.source, build_directory.parents)
        self.assertFalse(build_directory.parent.exists())
        self.assert_source_unchanged()

    def test_default_isolated_frontend_build(self):
        build_wheel(self.source, self.rocm, self.output)
        self.assertEqual(len(list(self.output.glob("*.whl"))), 1)
        self.assert_source_unchanged()

    def test_backend_failure_propagates(self):
        (self.source / "backend.py").write_text(BACKEND.replace("FAILURE", "True"))
        with self.assertRaises(subprocess.CalledProcessError):
            build_wheel(self.source, self.rocm, self.output, isolation=False)
        self.assertFalse(list(self.output.glob("*.whl")))


if __name__ == "__main__":
    unittest.main()
