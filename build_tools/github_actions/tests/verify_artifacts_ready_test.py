#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for verify_artifacts_ready.py."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import verify_artifacts_ready as v


class TestDecide(unittest.TestCase):
    """Producer selection: prebuilt_prefix non-empty -> copy_prebuilt."""

    def test_empty_prefix_selects_build_source(self):
        producer, result = v.decide(
            prebuilt_prefix="",
            build_source_result="success",
            copy_prebuilt_result="failure",
        )
        self.assertEqual(producer, "build_source")
        self.assertEqual(result, "success")

    def test_non_empty_prefix_selects_copy_prebuilt(self):
        producer, result = v.decide(
            prebuilt_prefix="rerun-2026-04-30",
            build_source_result="failure",
            copy_prebuilt_result="success",
        )
        self.assertEqual(producer, "copy_prebuilt")
        self.assertEqual(result, "success")


class TestMain(unittest.TestCase):
    """End-to-end exit-code behavior."""

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = v.main(argv)
        return code, buf.getvalue()

    def test_source_mode_success_exits_zero(self):
        code, out = self._run(
            [
                "--prebuilt-prefix",
                "",
                "--build-source-result",
                "success",
                "--copy-prebuilt-result",
                "skipped",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("build_source", out)

    def test_source_mode_failure_exits_one(self):
        code, out = self._run(
            [
                "--prebuilt-prefix",
                "",
                "--build-source-result",
                "failure",
                "--copy-prebuilt-result",
                "success",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("build_source", out)
        self.assertIn("failure", out)

    def test_source_mode_ignores_copy_result(self):
        # Copy result should not affect source mode.
        code, _ = self._run(
            [
                "--prebuilt-prefix",
                "",
                "--build-source-result",
                "success",
                "--copy-prebuilt-result",
                "failure",
            ]
        )
        self.assertEqual(code, 0)

    def test_prebuilt_mode_success_exits_zero(self):
        code, out = self._run(
            [
                "--prebuilt-prefix",
                "rerun-foo",
                "--build-source-result",
                "skipped",
                "--copy-prebuilt-result",
                "success",
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("copy_prebuilt", out)

    def test_prebuilt_mode_failure_exits_one(self):
        code, out = self._run(
            [
                "--prebuilt-prefix",
                "rerun-foo",
                "--build-source-result",
                "success",
                "--copy-prebuilt-result",
                "failure",
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("copy_prebuilt", out)
        self.assertIn("failure", out)

    def test_prebuilt_mode_ignores_source_result(self):
        # Source result should not affect prebuilt mode.
        code, _ = self._run(
            [
                "--prebuilt-prefix",
                "rerun-foo",
                "--build-source-result",
                "failure",
                "--copy-prebuilt-result",
                "success",
            ]
        )
        self.assertEqual(code, 0)

    def test_cancelled_active_producer_exits_one(self):
        code, _ = self._run(
            [
                "--prebuilt-prefix",
                "",
                "--build-source-result",
                "cancelled",
                "--copy-prebuilt-result",
                "success",
            ]
        )
        self.assertEqual(code, 1)

    def test_skipped_active_producer_exits_one(self):
        # In source-build mode, a `skipped` source job is a failure: nothing
        # produced the artifacts.
        code, _ = self._run(
            [
                "--prebuilt-prefix",
                "",
                "--build-source-result",
                "skipped",
                "--copy-prebuilt-result",
                "success",
            ]
        )
        self.assertEqual(code, 1)

    def test_whitespace_prefix_is_treated_as_empty(self):
        # Defensive: leading/trailing whitespace in inputs shouldn't flip mode.
        code, _ = self._run(
            [
                "--prebuilt-prefix",
                "   ",
                "--build-source-result",
                "success",
                "--copy-prebuilt-result",
                "failure",
            ]
        )
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
