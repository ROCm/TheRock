# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.cmake_amdgpu_targets import (
    AmdgpuTargetInfo,
    build_family_to_targets,
    parse_amdgpu_targets_cmake,
)


class ParseAmdgpuTargetsCmakeTest(unittest.TestCase):
    def _parse(self, cmake_text: str) -> list[AmdgpuTargetInfo]:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cmake", delete=False) as f:
            f.write(textwrap.dedent(cmake_text))
            tmp_path = Path(f.name)
        try:
            return parse_amdgpu_targets_cmake(tmp_path)
        finally:
            tmp_path.unlink()

    def test_single_line_no_exclude(self):
        infos = self._parse(
            """\
            therock_add_amdgpu_target(gfx942 "MI300A/MI300X CDNA" FAMILY dcgpu-all gfx94X-all gfx94X-dcgpu)
            """
        )
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].gfx_target, "gfx942")
        self.assertEqual(infos[0].product_name, "MI300A/MI300X CDNA")
        self.assertEqual(infos[0].families, ["dcgpu-all", "gfx94X-all", "gfx94X-dcgpu"])

    def test_multiline_with_exclude(self):
        infos = self._parse(
            """\
            therock_add_amdgpu_target(gfx900 "Vega 10 / MI25" FAMILY dgpu-all gfx900-dgpu
              EXCLUDE_TARGET_PROJECTS
                hipBLASLt # https://github.com/ROCm/TheRock/issues/1062
                hipSPARSELt
            )
            """
        )
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].gfx_target, "gfx900")
        self.assertEqual(infos[0].families, ["dgpu-all", "gfx900-dgpu"])

    def test_no_family(self):
        # Targets without an explicit FAMILY still parse correctly.
        infos = self._parse(
            """\
            therock_add_amdgpu_target(gfx9999 "Hypothetical" EXCLUDE_TARGET_PROJECTS someLib)
            """
        )
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].gfx_target, "gfx9999")
        self.assertEqual(infos[0].families, [])

    def test_multiple_targets(self):
        infos = self._parse(
            """\
            therock_add_amdgpu_target(gfx1100 "RX 7900 XTX" FAMILY dgpu-all gfx110X-all gfx110X-dgpu)
            therock_add_amdgpu_target(gfx1101 "RX 7800 XT" FAMILY dgpu-all gfx110X-all gfx110X-dgpu)
            """
        )
        self.assertEqual(len(infos), 2)
        self.assertEqual(infos[0].gfx_target, "gfx1100")
        self.assertEqual(infos[1].gfx_target, "gfx1101")
        for info in infos:
            self.assertIn("gfx110X-all", info.families)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            parse_amdgpu_targets_cmake(Path("/nonexistent/path.cmake"))


class BuildFamilyToTargetsTest(unittest.TestCase):
    def test_self_family(self):
        infos = [AmdgpuTargetInfo("gfx942", "MI300X", [])]
        mapping = build_family_to_targets(infos)
        self.assertEqual(mapping["gfx942"], ["gfx942"])

    def test_explicit_families(self):
        infos = [
            AmdgpuTargetInfo("gfx1100", "RX 7900 XTX", ["dgpu-all", "gfx110X-all"]),
            AmdgpuTargetInfo("gfx1101", "RX 7800 XT", ["dgpu-all", "gfx110X-all"]),
        ]
        mapping = build_family_to_targets(infos)
        self.assertIn("gfx1100", mapping["gfx110X-all"])
        self.assertIn("gfx1101", mapping["gfx110X-all"])
        self.assertEqual(mapping["gfx1100"], ["gfx1100"])
        self.assertEqual(mapping["gfx1101"], ["gfx1101"])

    def test_no_duplicates(self):
        # A target appearing in two calls with the same family must not be duplicated.
        infos = [
            AmdgpuTargetInfo("gfx942", "MI300X", ["dcgpu-all", "gfx94X-dcgpu"]),
        ]
        mapping = build_family_to_targets(infos)
        self.assertEqual(mapping["gfx942"].count("gfx942"), 1)


if __name__ == "__main__":
    unittest.main()
