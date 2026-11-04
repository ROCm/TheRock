# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
import expand_amdgpu_families
from _therock_utils.cmake_amdgpu_targets import expand_families


_SAMPLE_MAP: dict[str, list[str]] = {
    "gfx942": ["gfx942"],
    "gfx1100": ["gfx1100"],
    "gfx1101": ["gfx1101"],
    "gfx94X-dcgpu": ["gfx942"],
    "gfx110X-all": ["gfx1100", "gfx1101"],
    "dcgpu-all": ["gfx942"],
}


class ExpandFamiliesTest(unittest.TestCase):
    def test_single_family_expands(self):
        self.assertEqual(expand_families(["gfx94X-dcgpu"], _SAMPLE_MAP), ["gfx942"])

    def test_multiple_families_preserve_order(self):
        self.assertEqual(
            expand_families(["gfx110X-all", "gfx94X-dcgpu"], _SAMPLE_MAP),
            ["gfx1100", "gfx1101", "gfx942"],
        )

    def test_overlap_deduplicates_keeping_first_occurrence(self):
        # Both dcgpu-all and gfx94X-dcgpu expand to gfx942; the second occurrence
        # is dropped.
        self.assertEqual(
            expand_families(["dcgpu-all", "gfx94X-dcgpu"], _SAMPLE_MAP),
            ["gfx942"],
        )

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(expand_families([], _SAMPLE_MAP), [])

    def test_unknown_family_raises_by_default(self):
        with self.assertRaisesRegex(ValueError, "Unknown AMD GPU families"):
            expand_families(["gfx94X-dcgpu", "gfxNOPE"], _SAMPLE_MAP)

    def test_unknown_family_silently_skipped_when_not_strict(self):
        self.assertEqual(
            expand_families(
                ["gfx94X-dcgpu", "gfxNOPE", "gfx110X-all"],
                _SAMPLE_MAP,
                strict=False,
            ),
            ["gfx942", "gfx1100", "gfx1101"],
        )


class ExcludeImplicitTargetsTest(unittest.TestCase):
    def _expand_and_exclude(
        self,
        requested_families: list[str],
        implicit_targets_to_exclude: set[str],
    ) -> list[str]:
        return expand_amdgpu_families.expand_requested_families(
            requested_families,
            _SAMPLE_MAP,
            implicit_targets_to_exclude,
        )

    def test_exclusion_preserves_other_targets_and_explicit_requests(self):
        cases = (
            (["gfx110X-all"], ["gfx1100"]),
            (["gfx110X-all", "gfx1101"], ["gfx1100", "gfx1101"]),
            (["gfx1101", "gfx110X-all"], ["gfx1101", "gfx1100"]),
            (["gfx110X-all", "gfx1100"], ["gfx1100"]),
        )
        for requested_families, expected in cases:
            with self.subTest(requested_families=requested_families):
                self.assertEqual(
                    self._expand_and_exclude(requested_families, {"gfx1101"}),
                    expected,
                )

    def test_multiple_exclusions_are_independent(self):
        self.assertEqual(
            self._expand_and_exclude(
                ["gfx94X-dcgpu", "gfx110X-all", "gfx1101"],
                {"gfx942", "gfx1101"},
            ),
            ["gfx1100", "gfx1101"],
        )

    def test_empty_exclusions_preserve_expansion(self):
        self.assertEqual(
            self._expand_and_exclude(["gfx94X-dcgpu", "gfx110X-all"], set()),
            ["gfx942", "gfx1100", "gfx1101"],
        )

    def test_unregistered_exclusion_does_not_change_validation(self):
        family_map = {
            "gfx1100": ["gfx1100"],
            "gfx110X-all": ["gfx1100"],
        }
        self.assertEqual(
            expand_amdgpu_families.expand_requested_families(
                ["gfx110X-all"],
                family_map,
                {"gfx1102"},
            ),
            ["gfx1100"],
        )
        with self.assertRaisesRegex(ValueError, "gfx1102"):
            expand_amdgpu_families.expand_requested_families(
                ["gfx1102"],
                family_map,
                {"gfx1102"},
            )


class ExpandAmdgpuFamiliesMainTest(unittest.TestCase):
    """End-to-end test of the script's main() against the real CMake file."""

    def _run_and_capture(self, *argv: str) -> str:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = expand_amdgpu_families.main(list(argv))
        self.assertEqual(rc, 0)
        return buf.getvalue().strip()

    def test_main_expands_single_known_family(self):
        # gfx94X-dcgpu should always contain gfx942.
        out = self._run_and_capture("--amdgpu-families", "gfx94X-dcgpu")
        self.assertIn("gfx942", out.split(","))

    def test_main_empty_input_prints_empty(self):
        for policy_args in ((), ("--exclude-implicit-targets", "gfx1101")):
            with self.subTest(policy_args=policy_args):
                out = self._run_and_capture(
                    "--amdgpu-families",
                    "",
                    *policy_args,
                )
                self.assertEqual(out, "")

    def test_main_unknown_family_raises(self):
        with self.assertRaises(ValueError):
            expand_amdgpu_families.main(["--amdgpu-families", "gfxNOPE"])

    def test_main_device_extras_mode(self):
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "github-output"
            with mock.patch.dict(
                os.environ,
                {"GITHUB_OUTPUT": os.fspath(output_path)},
            ):
                out = self._run_and_capture(
                    "--amdgpu-families",
                    "gfx94X-dcgpu",
                    "--output-mode",
                    "device-extras",
                )

            # gha_set_output logs extra lines; the result is the first line.
            first_line = out.split("\n")[0]
            self.assertIn("device-gfx942", first_line.split(","))
            for token in first_line.split(","):
                self.assertTrue(token.startswith("device-"), token)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "device_extras=device-gfx942\n",
            )

    def test_main_targets_mode_is_default(self):
        out = self._run_and_capture("--amdgpu-families", "gfx94X-dcgpu")
        self.assertIn("gfx942", out.split(","))
        self.assertNotIn("device-", out)

    def test_main_applies_exclusions_and_retains_explicit_targets(self):
        cases = (
            (" ; gfx110X-all ; gfx110X-all ;; ", False),
            (" gfx110X-all ; gfx1101 ;; ", True),
        )
        for families, retain_excluded_target in cases:
            with self.subTest(families=families):
                targets = self._run_and_capture(
                    "--amdgpu-families",
                    families,
                    "--exclude-implicit-targets",
                    " ; gfx1101 ; gfx1101 ;; ",
                ).split(",")
                self.assertIn("gfx1100", targets)
                self.assertEqual("gfx1101" in targets, retain_excluded_target)
                self.assertEqual(len(targets), len(set(targets)))

    def test_shared_owner_device_extras(self):
        for families in ("gfx1250", "gfx1250-strict", "gfx1250;gfx1250-strict"):
            with self.subTest(families=families):
                out = self._run_and_capture(
                    "--amdgpu-families", families, "--output-mode", "device-extras"
                )
                self.assertEqual(out.splitlines()[0], "device-gfx1250")
                self.assertNotIn("gfx1250-strict", out)

    def test_targets_mode_preserves_distinct_targets(self):
        self.assertEqual(
            self._run_and_capture("--amdgpu-families", "gfx1250;gfx1250-strict"),
            "gfx1250,gfx1250-strict",
        )


if __name__ == "__main__":
    unittest.main()
