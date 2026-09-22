"""
Unit tests for external-builds/jax/skip_tests/create_skip_tests.py
"""

import os
import sys
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
JAX_DIR = THIS_DIR.parents[2] / "external-builds" / "jax"

sys.path.insert(0, os.fspath(JAX_DIR))

from skip_tests import create_skip_tests

# The versions whose convolution tests are skipped on gfx94X, and one that has
# no data file at all.
FILTERED_VERSION = "0.10.2"
UNFILTERED_VERSION = "0.11.0"

GFX94_FAMILY = "gfx94X-dcgpu"


class DataFilesTest(unittest.TestCase):
    def test_generic_file_always_applies(self):
        files = create_skip_tests.data_files("")

        self.assertEqual([f.name for f in files], ["generic.py"])

    def test_version_file_is_added_when_it_exists(self):
        files = create_skip_tests.data_files(FILTERED_VERSION)

        self.assertEqual(
            [f.name for f in files], ["generic.py", f"jax_{FILTERED_VERSION}.py"]
        )

    def test_unknown_version_contributes_nothing(self):
        # Adding a JAX version to CI must not require a data file here.
        files = create_skip_tests.data_files("9.9.9")

        self.assertEqual([f.name for f in files], ["generic.py"])

    def test_all_loads_every_version_file(self):
        files = create_skip_tests.data_files("all")

        self.assertEqual(files[0].name, "generic.py")
        self.assertGreater(len(files), 1)
        for path in files[1:]:
            self.assertTrue(path.name.startswith("jax_"))


class KeywordExpressionTest(unittest.TestCase):
    def test_no_entries_gives_no_expression(self):
        self.assertEqual(
            create_skip_tests.keyword_expression(UNFILTERED_VERSION, GFX94_FAMILY), ""
        )

    def test_family_selects_the_section(self):
        expression = create_skip_tests.keyword_expression(
            FILTERED_VERSION, GFX94_FAMILY
        )

        self.assertIn("not conv", expression)

    def test_other_family_is_unaffected(self):
        self.assertEqual(
            create_skip_tests.keyword_expression(FILTERED_VERSION, "gfx110X-all"), ""
        )

    def test_arch_matches_the_family_section(self):
        # A section named gfx94 also covers a caller naming the arch directly.
        expression = create_skip_tests.keyword_expression(FILTERED_VERSION, "gfx942")

        self.assertIn("not conv", expression)

    def test_missing_family_matches_nothing(self):
        self.assertEqual(create_skip_tests.keyword_expression(FILTERED_VERSION, ""), "")

    def test_unless_keeps_tests_the_deny_would_catch(self):
        expression = create_skip_tests.keyword_expression(
            FILTERED_VERSION, GFX94_FAMILY
        )

        # -k matches substrings, so conv also catches convert and conversion.
        self.assertIn("((not conv) or convert or conversion)", expression)

    def test_entries_are_joined_with_and(self):
        expression = create_skip_tests.keyword_expression(
            FILTERED_VERSION, GFX94_FAMILY
        )

        self.assertIn("and not sumpool", expression)
        self.assertIn("and not polymul", expression)

    def test_debug_selects_only_the_skipped_tests(self):
        expression = create_skip_tests.keyword_expression(
            FILTERED_VERSION, GFX94_FAMILY, debug=True
        )

        self.assertNotIn("not ", expression)
        self.assertIn("conv or ", expression)
        self.assertIn("polymul", expression)


class DataFileContentsTest(unittest.TestCase):
    def test_every_entry_has_a_deny_keyword(self):
        for path in create_skip_tests.data_files("all"):
            for section, entries in create_skip_tests._load_data_file(path).items():
                for entry in entries.get("keywords", []):
                    with self.subTest(file=path.name, section=section, entry=entry):
                        self.assertIn("deny", entry)
                        self.assertIsInstance(entry["deny"], str)
                        self.assertIsInstance(entry.get("unless", []), list)


class MainTest(unittest.TestCase):
    def test_prints_the_expression(self):
        # Smoke test of the CLI used to inspect a configuration locally.
        self.assertEqual(
            create_skip_tests.main(
                ["--jax-version", UNFILTERED_VERSION, "--amdgpu-family", GFX94_FAMILY]
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
