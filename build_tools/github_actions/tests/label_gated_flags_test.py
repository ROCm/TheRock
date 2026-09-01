#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for label_gated_flags.py"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from label_gated_flags import (
    FLAG_RE,
    LABEL_GATED_FLAGS,
    collect_flags,
    parse_labels_from_event,
    render_summary,
    validate_label_gated_flags,
)

# A stand-in for the shipped (empty) map, so the tests exercise real behavior
# without depending on whatever entries happen to be shipped.
FAKE_MAP: dict[str, dict[str, list[str]]] = {
    "rocm-libraries": {
        "ci:alpha": ["-DTHEROCK_FLAG_ALPHA=ON"],
        "ci:beta": [
            "-DTHEROCK_FLAG_BETA=ON",
            "-DTHEROCK_FLAG_GAMMA=OFF",
        ],
    },
    "rocm-systems": {
        "ci:delta": ["-DTHEROCK_FLAG_DELTA=ON"],
    },
}


def _write_event(payload) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(payload, f)
        return f.name


def _pr_event(*label_names: str) -> str:
    return _write_event(
        {"pull_request": {"labels": [{"name": name} for name in label_names]}}
    )


class ShippedMapTest(unittest.TestCase):
    """The map that actually ships must always be well-formed."""

    def test_shipped_map_is_valid(self):
        validate_label_gated_flags()

    def test_shipped_map_is_empty(self):
        # Not a correctness requirement, but a reviewer tripwire: adding an
        # entry changes what CI builds and should be a deliberate act.
        self.assertEqual(LABEL_GATED_FLAGS, {})


class ValidationTest(unittest.TestCase):
    def test_empty_map_is_valid(self):
        validate_label_gated_flags({})

    def test_fake_map_is_valid(self):
        validate_label_gated_flags(FAKE_MAP)

    def test_rejects_non_dict_map(self):
        with self.assertRaises(ValueError):
            validate_label_gated_flags(["ci:alpha"])

    def test_rejects_empty_repo_key(self):
        with self.assertRaises(ValueError):
            validate_label_gated_flags({"": {"ci:alpha": ["-DTHEROCK_FLAG_A=ON"]}})

    def test_rejects_non_dict_repo_value(self):
        with self.assertRaises(ValueError):
            validate_label_gated_flags({"rocm-libraries": ["-DTHEROCK_FLAG_A=ON"]})

    def test_rejects_empty_label_key(self):
        with self.assertRaises(ValueError):
            validate_label_gated_flags(
                {"rocm-libraries": {"": ["-DTHEROCK_FLAG_A=ON"]}}
            )

    def test_rejects_bare_string_flags(self):
        # A bare string would iterate character by character.
        with self.assertRaises(ValueError):
            validate_label_gated_flags(
                {"rocm-libraries": {"ci:alpha": "-DTHEROCK_FLAG_A=ON"}}
            )

    def test_rejects_empty_flag_list(self):
        with self.assertRaises(ValueError):
            validate_label_gated_flags({"rocm-libraries": {"ci:alpha": []}})

    def test_rejects_unprefixed_flag(self):
        # `therock_declare_flag` adds the THEROCK_FLAG_ prefix, so setting the
        # unprefixed name at top level is a silent no-op.
        with self.assertRaises(ValueError):
            validate_label_gated_flags(
                {"rocm-libraries": {"ci:alpha": ["-DMIOPEN_ENABLE_WRAPPER=ON"]}}
            )

    def test_rejects_clobbered_namespace(self):
        # TheRock generates its own THEROCK_ENABLE_* options after these are
        # spliced in, and cmake takes the last -D.
        with self.assertRaises(ValueError):
            validate_label_gated_flags(
                {"rocm-libraries": {"ci:alpha": ["-DTHEROCK_ENABLE_BLAS=ON"]}}
            )

    def test_rejects_non_boolean_value(self):
        for value in ["1", "TRUE", "on", "YES", ""]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_label_gated_flags(
                        {"rocm-libraries": {"ci:alpha": [f"-DTHEROCK_FLAG_A={value}"]}}
                    )

    def test_rejects_lowercase_flag_name(self):
        with self.assertRaises(ValueError):
            validate_label_gated_flags(
                {"rocm-libraries": {"ci:alpha": ["-DTHEROCK_FLAG_alpha=ON"]}}
            )

    def test_rejects_shell_metacharacters(self):
        # The whole safety argument for splicing these onto a command line.
        for injected in [
            "-DTHEROCK_FLAG_A=ON; rm -rf /",
            "-DTHEROCK_FLAG_A=ON && echo hi",
            "-DTHEROCK_FLAG_A=ON | tee x",
            "-DTHEROCK_FLAG_A=$(id)",
            "-DTHEROCK_FLAG_A=`id`",
            "-DTHEROCK_FLAG_A=ON'",
            '-DTHEROCK_FLAG_A=ON"',
            "-DTHEROCK_FLAG_A=ON\nrm -rf /",
            "-DTHEROCK_FLAG_A=ON --trace",
        ]:
            with self.subTest(injected=injected):
                with self.assertRaises(ValueError):
                    validate_label_gated_flags(
                        {"rocm-libraries": {"ci:alpha": [injected]}}
                    )

    def test_flag_re_accepts_canonical_forms(self):
        self.assertTrue(FLAG_RE.fullmatch("-DTHEROCK_FLAG_A=ON"))
        self.assertTrue(FLAG_RE.fullmatch("-DTHEROCK_FLAG_A_B2=OFF"))


class ParseLabelsTest(unittest.TestCase):
    def test_reads_label_names(self):
        path = _pr_event("ci:alpha", "documentation")
        self.assertEqual(
            parse_labels_from_event(path, "pull_request"), ["ci:alpha", "documentation"]
        )

    def test_non_pull_request_event_reads_nothing(self):
        path = _pr_event("ci:alpha")
        for event_name in ["workflow_dispatch", "push", "schedule", "release", ""]:
            with self.subTest(event_name=event_name):
                self.assertEqual(parse_labels_from_event(path, event_name), [])

    def test_missing_path(self):
        self.assertEqual(parse_labels_from_event(None, "pull_request"), [])
        self.assertEqual(parse_labels_from_event("", "pull_request"), [])

    def test_nonexistent_file(self):
        missing = str(Path(tempfile.gettempdir()) / "no-such-event-payload-12345.json")
        self.assertEqual(parse_labels_from_event(missing, "pull_request"), [])

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{not json")
            path = f.name
        self.assertEqual(parse_labels_from_event(path, "pull_request"), [])

    def test_tolerates_junk_payload_shapes(self):
        for payload in [
            [],
            {},
            {"pull_request": None},
            {"pull_request": []},
            {"pull_request": {}},
            {"pull_request": {"labels": None}},
            {"pull_request": {"labels": "ci:alpha"}},
        ]:
            with self.subTest(payload=payload):
                path = _write_event(payload)
                self.assertEqual(parse_labels_from_event(path, "pull_request"), [])

    def test_skips_malformed_label_entries(self):
        path = _write_event(
            {
                "pull_request": {
                    "labels": [
                        "ci:alpha",
                        {"color": "ff0000"},
                        {"name": ""},
                        {"name": 42},
                        {"name": "ci:beta"},
                    ]
                }
            }
        )
        self.assertEqual(parse_labels_from_event(path, "pull_request"), ["ci:beta"])


class CollectFlagsTest(unittest.TestCase):
    def test_no_labels(self):
        self.assertEqual(collect_flags([], "rocm-libraries", FAKE_MAP), ([], []))

    def test_unmapped_labels_match_nothing(self):
        self.assertEqual(
            collect_flags(["documentation", "ci:skip"], "rocm-libraries", FAKE_MAP),
            ([], []),
        )

    def test_single_label(self):
        self.assertEqual(
            collect_flags(["ci:alpha"], "rocm-libraries", FAKE_MAP),
            (["ci:alpha"], ["-DTHEROCK_FLAG_ALPHA=ON"]),
        )

    def test_label_with_multiple_flags(self):
        self.assertEqual(
            collect_flags(["ci:beta"], "rocm-libraries", FAKE_MAP),
            (["ci:beta"], ["-DTHEROCK_FLAG_BETA=ON", "-DTHEROCK_FLAG_GAMMA=OFF"]),
        )

    def test_ordering_follows_the_map_not_the_payload(self):
        forward = collect_flags(["ci:alpha", "ci:beta"], "rocm-libraries", FAKE_MAP)
        reverse = collect_flags(["ci:beta", "ci:alpha"], "rocm-libraries", FAKE_MAP)
        self.assertEqual(forward, reverse)
        self.assertEqual(
            forward,
            (
                ["ci:alpha", "ci:beta"],
                [
                    "-DTHEROCK_FLAG_ALPHA=ON",
                    "-DTHEROCK_FLAG_BETA=ON",
                    "-DTHEROCK_FLAG_GAMMA=OFF",
                ],
            ),
        )

    def test_labels_are_matched_as_exact_strings(self):
        # Not a regex, not a prefix.
        weird = {"rocm-libraries": {"weird.label+name(1)": ["-DTHEROCK_FLAG_W=ON"]}}
        self.assertEqual(
            collect_flags(["weird.label+name(1)"], "rocm-libraries", weird),
            (["weird.label+name(1)"], ["-DTHEROCK_FLAG_W=ON"]),
        )
        self.assertEqual(
            collect_flags(["weirdxlabel+name(1)"], "rocm-libraries", weird), ([], [])
        )

    def test_scoped_to_repository(self):
        # A rocm-systems label must not pick up a rocm-libraries entry.
        self.assertEqual(
            collect_flags(["ci:alpha"], "rocm-systems", FAKE_MAP), ([], [])
        )
        self.assertEqual(
            collect_flags(["ci:delta"], "rocm-systems", FAKE_MAP),
            (["ci:delta"], ["-DTHEROCK_FLAG_DELTA=ON"]),
        )
        self.assertEqual(
            collect_flags(["ci:delta"], "rocm-libraries", FAKE_MAP), ([], [])
        )

    def test_unknown_or_empty_repository(self):
        self.assertEqual(collect_flags(["ci:alpha"], "rocgdb", FAKE_MAP), ([], []))
        self.assertEqual(collect_flags(["ci:alpha"], "", FAKE_MAP), ([], []))

    def test_duplicate_agreeing_flags_appear_once(self):
        mapping = {
            "rocm-libraries": {
                "ci:one": ["-DTHEROCK_FLAG_A=ON"],
                "ci:two": ["-DTHEROCK_FLAG_A=ON", "-DTHEROCK_FLAG_B=ON"],
            }
        }
        self.assertEqual(
            collect_flags(["ci:one", "ci:two"], "rocm-libraries", mapping),
            (["ci:one", "ci:two"], ["-DTHEROCK_FLAG_A=ON", "-DTHEROCK_FLAG_B=ON"]),
        )

    def test_conflicting_values_raise(self):
        mapping = {
            "rocm-libraries": {
                "ci:on": ["-DTHEROCK_FLAG_A=ON"],
                "ci:off": ["-DTHEROCK_FLAG_A=OFF"],
            }
        }
        with self.assertRaisesRegex(ValueError, "conflicting values"):
            collect_flags(["ci:on", "ci:off"], "rocm-libraries", mapping)

    def test_defaults_to_the_shipped_map(self):
        self.assertEqual(collect_flags(["ci:alpha"], "rocm-libraries"), ([], []))


class RenderSummaryTest(unittest.TestCase):
    def test_non_pull_request_event(self):
        summary = render_summary(
            event_name="workflow_dispatch",
            repo_name="rocm-libraries",
            labels=[],
            matched=[],
            flags=[],
        )
        self.assertIn("not `pull_request`", summary)
        self.assertIn("default configuration", summary)

    def test_no_external_repo(self):
        summary = render_summary(
            event_name="pull_request", repo_name="", labels=[], matched=[], flags=[]
        )
        self.assertIn("No external repository", summary)

    def test_flags_off(self):
        summary = render_summary(
            event_name="pull_request",
            repo_name="rocm-libraries",
            labels=["documentation"],
            matched=[],
            flags=[],
        )
        self.assertIn("`documentation`", summary)
        self.assertIn("**Matched gating labels:** _none_", summary)
        self.assertIn("**Appended cmake options:** _none_", summary)
        self.assertNotIn("IMPORTANT", summary)

    def test_flags_on_names_exactly_what_it_acted_on(self):
        summary = render_summary(
            event_name="pull_request",
            repo_name="rocm-libraries",
            labels=["ci:alpha", "documentation"],
            matched=["ci:alpha"],
            flags=["-DTHEROCK_FLAG_ALPHA=ON"],
        )
        self.assertIn("**Matched gating labels:** `ci:alpha`", summary)
        self.assertIn("`-DTHEROCK_FLAG_ALPHA=ON`", summary)
        self.assertIn("IMPORTANT", summary)
        self.assertIn("Stage reuse is forced off", summary)


if __name__ == "__main__":
    unittest.main()
