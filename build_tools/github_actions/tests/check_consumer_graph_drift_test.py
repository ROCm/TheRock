#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for check_consumer_graph_drift.py"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from check_consumer_graph_drift import check, normalize, write


class TestNormalize(unittest.TestCase):
    def _write(self, obj) -> Path:
        path = Path(tempfile.mkdtemp()) / "graph.json"
        path.write_text(json.dumps(obj))
        return path

    def test_normalize_is_order_insensitive(self) -> None:
        # Same edges, different key/consumer order -> identical normalized text.
        a = self._write({"b": {"consumers": ["z", "a"]}, "a": {"consumers": []}})
        b = self._write({"a": {"consumers": []}, "b": {"consumers": ["a", "z"]}})
        self.assertEqual(normalize(a), normalize(b))

    def test_normalize_has_trailing_newline_and_indent(self) -> None:
        text = normalize(self._write({"a": {"consumers": ["b"]}}))
        self.assertTrue(text.endswith("\n"))
        self.assertIn('  "a"', text)  # 2-space indent

    def test_normalize_tolerates_missing_consumers_key(self) -> None:
        # A node with no "consumers" key normalizes to an empty list, not a crash.
        text = normalize(self._write({"a": {}}))
        self.assertIn('"consumers": []', text)


class TestCheck(unittest.TestCase):
    def _write(self, obj) -> Path:
        path = Path(tempfile.mkdtemp()) / "graph.json"
        path.write_text(json.dumps(obj))
        return path

    def test_check_passes_when_equal_modulo_order(self) -> None:
        committed = self._write({"a": {"consumers": ["b", "c"]}})
        emitted = self._write({"a": {"consumers": ["c", "b"]}})
        self.assertEqual(check(committed, emitted), 0)

    def test_check_fails_on_edge_change(self) -> None:
        committed = self._write({"a": {"consumers": ["b"]}})
        emitted = self._write({"a": {"consumers": ["b", "c"]}})
        self.assertEqual(check(committed, emitted), 1)


class TestWrite(unittest.TestCase):
    def test_write_makes_check_pass(self) -> None:
        root = Path(tempfile.mkdtemp())
        committed = root / "committed.json"
        emitted = root / "emitted.json"
        # Stale committed copy, differently-ordered emitted graph.
        committed.write_text(json.dumps({"a": {"consumers": ["stale"]}}))
        emitted.write_text(json.dumps({"a": {"consumers": ["c", "b"]}}))

        self.assertEqual(write(committed, emitted), 0)
        # After write, committed matches emitted and check passes.
        self.assertEqual(check(committed, emitted), 0)
        # And it was written in normalized form (sorted, trailing newline).
        self.assertEqual(committed.read_text(), normalize(emitted))


if __name__ == "__main__":
    unittest.main()
