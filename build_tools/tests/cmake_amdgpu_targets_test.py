# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils.cmake_amdgpu_targets import (
    AmdgpuTargetInfo,
    amdgpu_family_map,
    build_family_to_targets,
    parse_amdgpu_targets_cmake,
)


class ParseAmdgpuTargetsCmakeTest(unittest.TestCase):
    def _parse(self, cmake_text: str) -> list[AmdgpuTargetInfo]:
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


ROOT = Path(__file__).resolve().parents[2]


class CmakeTargetSelectionTest(unittest.TestCase):
    def test_registration(self):
        infos = parse_amdgpu_targets_cmake(ROOT / "cmake/therock_amdgpu_targets.cmake")
        strict = next(info for info in infos if info.gfx_target == "gfx1250-strict")
        self.assertEqual(strict.families, ["dcgpu-all", "gfx125X-all", "gfx125X-dcgpu"])
        families = amdgpu_family_map()
        self.assertEqual(families["gfx1250"], ["gfx1250"])
        self.assertEqual(families["gfx1250-strict"], ["gfx1250-strict"])
        for family in ("gfx125X-all", "gfx125X-dcgpu", "dcgpu-all"):
            self.assertIn("gfx1250-strict", families[family])

    def test_cmake_selection_and_defaults(self):
        cases = [
            ("set(THEROCK_AMDGPU_FAMILIES gfx1250)", "gfx1250", "gfx1250", "all"),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx125X-all)",
                "gfx1250-strict;gfx1250",
                "gfx1250-strict;gfx1250",
                "all",
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx125X-dcgpu)\nset(THEROCK_TEST_AMDGPU_FAMILIES gfx125X-dcgpu)",
                "gfx1250-strict;gfx1250",
                "gfx1250-strict;gfx1250",
                "gfx1250-strict;gfx1250",
            ),
            (
                "set(THEROCK_AMDGPU_TARGETS gfx1250)",
                "gfx1250",
                "THEROCK_DIST_AMDGPU_TARGETS-NOTFOUND",
                "all",
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250-strict)\nset(THEROCK_TEST_AMDGPU_FAMILIES gfx1250-strict)",
                "gfx1250-strict",
                "gfx1250-strict",
                "gfx1250-strict",
            ),
            (
                "set(THEROCK_AMDGPU_TARGETS gfx1250-strict)\nset(THEROCK_DIST_AMDGPU_TARGETS gfx1250-strict)\nset(THEROCK_TEST_AMDGPU_TARGETS gfx1250-strict)",
                "gfx1250-strict",
                "gfx1250-strict",
                "gfx1250-strict",
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250-strict)",
                "gfx1250-strict",
                "gfx1250-strict",
                "all",
            ),
            (
                "set(THEROCK_AMDGPU_TARGETS gfx1250-strict)",
                "gfx1250-strict",
                "THEROCK_DIST_AMDGPU_TARGETS-NOTFOUND",
                "all",
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250 gfx1250-strict)\nset(THEROCK_AMDGPU_DIST_BUNDLE_NAME combined)",
                "gfx1250;gfx1250-strict",
                "gfx1250;gfx1250-strict",
                "all",
            ),
            (
                "set(THEROCK_DIST_AMDGPU_FAMILIES gfx1250-strict)",
                "THEROCK_AMDGPU_TARGETS-NOTFOUND",
                "gfx1250-strict",
                "all",
            ),
            (
                "set(THEROCK_DIST_AMDGPU_TARGETS gfx1250 gfx1250-strict)",
                "THEROCK_AMDGPU_TARGETS-NOTFOUND",
                "gfx1250;gfx1250-strict",
                "all",
            ),
            (
                "set(THEROCK_DIST_AMDGPU_FAMILIES gfx125X-all)",
                "THEROCK_AMDGPU_TARGETS-NOTFOUND",
                "gfx1250-strict;gfx1250",
                "all",
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250-strict)\nset(THEROCK_TEST_AMDGPU_TARGETS gfx1250)",
                "gfx1250-strict",
                "gfx1250-strict",
                "gfx1250",
            ),
            (
                "set(THEROCK_AMDGPU_FAMILIES gfx1250-strict)\nset(THEROCK_TEST_AMDGPU_FAMILIES gfx125X-all)",
                "gfx1250-strict",
                "gfx1250-strict",
                "gfx1250-strict;gfx1250",
            ),
        ]
        for settings, build, dist, tests in cases:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmp:
                script = Path(tmp) / "probe.cmake"
                script.write_text(
                    f"""cmake_minimum_required(VERSION 3.25)
include("{ROOT.as_posix()}/cmake/therock_amdgpu_targets.cmake")
{settings}
therock_validate_amdgpu_targets()
if(NOT THEROCK_AMDGPU_TARGETS STREQUAL "{build}")
  message(FATAL_ERROR "Wrong build targets: ${{THEROCK_AMDGPU_TARGETS}}")
endif()
if(NOT THEROCK_DIST_AMDGPU_TARGETS STREQUAL "{dist}")
  message(FATAL_ERROR "Wrong dist targets: ${{THEROCK_DIST_AMDGPU_TARGETS}}")
endif()
"""
                )
                with script.open("a") as f:
                    if tests == "all":
                        f.write(
                            "get_property(expected GLOBAL PROPERTY THEROCK_AMDGPU_TARGETS)\n"
                        )
                    else:
                        f.write(f'set(expected "{tests}")\n')
                    f.write(
                        'if(NOT THEROCK_TEST_AMDGPU_TARGETS STREQUAL expected)\nmessage(FATAL_ERROR "Wrong test targets: ${THEROCK_TEST_AMDGPU_TARGETS}; expected: ${expected}")\nendif()\n'
                    )
                subprocess.run(
                    ["cmake", "-P", str(script)],
                    check=True,
                    capture_output=True,
                    text=True,
                )


if __name__ == "__main__":
    unittest.main()
