#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for therock_build_flags.py."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from therock_build_flags import (
    BuildFlagError,
    FlagDeclaration,
    build_flags_suffix,
    flag_cmake_args,
    format_build_flags,
    load_flag_registry,
    parse_build_flags,
)

# A miniature registry, so these tests do not depend on which flags happen to
# be declared in FLAGS.cmake today.
REGISTRY = {
    "BOOL_DEFAULT_OFF": FlagDeclaration(
        name="BOOL_DEFAULT_OFF", type="BOOL", default_value="OFF"
    ),
    "BOOL_DEFAULT_ON": FlagDeclaration(
        name="BOOL_DEFAULT_ON", type="BOOL", default_value="ON"
    ),
    "INT_FREE": FlagDeclaration(name="INT_FREE", type="INTEGER", default_value="0"),
    "INT_CONSTRAINED": FlagDeclaration(
        name="INT_CONSTRAINED",
        type="INTEGER",
        default_value="-17",
        valid_values=("-17", "5"),
    ),
}


class LoadFlagRegistryTest(unittest.TestCase):
    def test_parses_the_real_flags_cmake(self):
        registry = load_flag_registry()
        self.assertTrue(registry)
        for name, declaration in registry.items():
            self.assertEqual(name, declaration.name)
            self.assertIn(declaration.type, ("BOOL", "INTEGER"))
            self.assertEqual(
                declaration.cmake_variable, f"THEROCK_FLAG_{declaration.name}"
            )

    def test_parses_declaration_fields(self):
        registry = load_flag_registry()
        # Canary flags exist specifically to be stable test subjects.
        integer_flag = registry["ROCM_BUILD_FLAGS_CANARY_INTEGER_NEGATIVE"]
        self.assertEqual(integer_flag.type, "INTEGER")
        self.assertEqual(integer_flag.default_value, "-17")
        self.assertEqual(integer_flag.valid_values, ("-17",))
        self.assertEqual(registry["ROCM_BUILD_FLAGS_CANARY_BOOL_TRUE"].type, "BOOL")

    def test_missing_registry_raises(self):
        with self.assertRaises(BuildFlagError):
            load_flag_registry(Path("does/not/exist/FLAGS.cmake"))


class ParseBuildFlagsTest(unittest.TestCase):
    def parse(self, raw):
        return parse_build_flags(raw, REGISTRY)

    def test_empty_input(self):
        self.assertEqual(self.parse(""), {})
        self.assertEqual(self.parse("   "), {})
        self.assertEqual(parse_build_flags(None, REGISTRY), {})

    def test_name_value_pairs(self):
        self.assertEqual(
            self.parse("BOOL_DEFAULT_OFF=ON,INT_CONSTRAINED=5"),
            {"BOOL_DEFAULT_OFF": "ON", "INT_CONSTRAINED": "5"},
        )

    def test_bare_name_means_on(self):
        self.assertEqual(self.parse("BOOL_DEFAULT_OFF"), {"BOOL_DEFAULT_OFF": "ON"})

    def test_bare_name_rejected_for_integer(self):
        with self.assertRaisesRegex(BuildFlagError, "requires an explicit value"):
            self.parse("INT_FREE")

    def test_default_on_flag_can_be_forced_off(self):
        self.assertEqual(self.parse("BOOL_DEFAULT_ON=OFF"), {"BOOL_DEFAULT_ON": "OFF"})

    def test_bool_aliases_are_normalized(self):
        for raw, expected in [
            ("BOOL_DEFAULT_OFF=on", "ON"),
            ("BOOL_DEFAULT_OFF=true", "ON"),
            ("BOOL_DEFAULT_OFF=1", "ON"),
            ("BOOL_DEFAULT_ON=off", "OFF"),
            ("BOOL_DEFAULT_ON=no", "OFF"),
            ("BOOL_DEFAULT_ON=0", "OFF"),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(list(self.parse(raw).values()), [expected])

    def test_whitespace_and_empty_entries_tolerated(self):
        self.assertEqual(
            self.parse(" BOOL_DEFAULT_OFF = ON , , INT_FREE=3 "),
            {"BOOL_DEFAULT_OFF": "ON", "INT_FREE": "3"},
        )

    def test_unknown_flag_fails(self):
        with self.assertRaisesRegex(BuildFlagError, "Unknown build flag 'NOPE'"):
            self.parse("NOPE=ON")

    def test_unknown_flag_is_case_sensitive(self):
        with self.assertRaises(BuildFlagError):
            self.parse("bool_default_off=ON")

    def test_bad_bool_value_fails(self):
        with self.assertRaisesRegex(BuildFlagError, "Invalid value 'maybe'"):
            self.parse("BOOL_DEFAULT_OFF=maybe")

    def test_bad_integer_value_fails(self):
        with self.assertRaisesRegex(BuildFlagError, "Expected an integer"):
            self.parse("INT_FREE=abc")

    def test_integer_outside_valid_values_fails(self):
        with self.assertRaisesRegex(BuildFlagError, "is not permitted"):
            self.parse("INT_CONSTRAINED=7")

    def test_negative_integer_in_valid_values(self):
        self.assertEqual(self.parse("INT_CONSTRAINED=-17"), {"INT_CONSTRAINED": "-17"})

    def test_duplicate_flag_fails(self):
        with self.assertRaisesRegex(BuildFlagError, "more than once"):
            self.parse("BOOL_DEFAULT_OFF=ON,BOOL_DEFAULT_OFF=OFF")

    def test_missing_name_fails(self):
        with self.assertRaisesRegex(BuildFlagError, "Malformed build flag entry"):
            self.parse("=ON")


class RenderingTest(unittest.TestCase):
    FLAGS = {"BOOL_DEFAULT_ON": "OFF", "INT_CONSTRAINED": "5"}

    def test_format_is_sorted_and_canonical(self):
        self.assertEqual(
            format_build_flags({"INT_CONSTRAINED": "5", "BOOL_DEFAULT_ON": "OFF"}),
            "BOOL_DEFAULT_ON=OFF,INT_CONSTRAINED=5",
        )

    def test_cmake_args(self):
        self.assertEqual(
            flag_cmake_args(self.FLAGS),
            [
                "-DTHEROCK_FLAG_BOOL_DEFAULT_ON=OFF",
                "-DTHEROCK_FLAG_INT_CONSTRAINED=5",
            ],
        )

    def test_no_flags_render_nothing(self):
        self.assertEqual(flag_cmake_args({}), [])
        self.assertEqual(format_build_flags({}), "")
        self.assertEqual(build_flags_suffix({}), "")

    def test_suffix_is_stable_and_order_independent(self):
        suffix = build_flags_suffix(self.FLAGS)
        self.assertTrue(suffix.startswith("flags-"))
        self.assertEqual(len(suffix), len("flags-") + 8)
        self.assertEqual(
            suffix,
            build_flags_suffix({"INT_CONSTRAINED": "5", "BOOL_DEFAULT_ON": "OFF"}),
        )

    def test_suffix_differs_per_value(self):
        self.assertNotEqual(
            build_flags_suffix({"BOOL_DEFAULT_ON": "OFF"}),
            build_flags_suffix({"BOOL_DEFAULT_ON": "ON"}),
        )


if __name__ == "__main__":
    unittest.main()
