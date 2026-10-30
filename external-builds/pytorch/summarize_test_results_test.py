#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import tempfile
import unittest
from pathlib import Path

import summarize_test_results as summary


class SummarizeTestResultsTest(unittest.TestCase):
    def test_failure_preserves_and_formats_full_error(self):
        xml = """\
<testsuite>
  <testcase classname="tests.TestExample" name="test_failure">
    <failure message="AssertionError: values differ">Traceback line 1
Traceback line 2
expected &lt; actual | difference</failure>
  </testcase>
</testsuite>
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "report.xml"
            xml_path.write_text(xml)
            failures = summary.parse_junit_xml(xml_path)

        self.assertEqual(len(failures), 1)
        message = failures[0]["message"]
        self.assertEqual(
            message,
            "AssertionError: values differ\n\n"
            "Traceback line 1\n"
            "Traceback line 2\n"
            "expected < actual | difference",
        )
        self.assertEqual(
            summary.format_error_message(message),
            "<details><summary>view</summary><pre>"
            "AssertionError: values differ&#10;&#10;"
            "Traceback line 1&#10;"
            "Traceback line 2&#10;"
            "expected &lt; actual &#124; difference"
            "</pre></details>",
        )


if __name__ == "__main__":
    unittest.main()
