# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from report_failed_tests import (
    TestResult,
    parse_junit_xml,
    parse_gtest_json,
    generate_metrics_output,
)


class TestReportFailedTests(unittest.TestCase):
    def test_parse_junit_xml_with_failures(self):
        xml_content = """<?xml version="1.0"?>
<testsuite name="Suite" tests="2" failures="1">
  <testcase name="pass" classname="Suite"/>
  <testcase name="fail" classname="Suite">
    <failure message="failed"/>
  </testcase>
</testsuite>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "ctest-comp-shard1.xml"
            xml_file.write_text(xml_content)
            result = parse_junit_xml(xml_file)

            self.assertEqual(result.component, "comp")
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(result.status, "failure")
            self.assertIn("Suite.fail", result.failed_tests)

    def test_parse_gtest_json_with_failures(self):
        json_content = {
            "tests": 2,
            "failures": 1,
            "time": "1.0s",
            "testsuites": [
                {
                    "name": "Suite",
                    "testsuite": [
                        {"name": "pass"},
                        {"name": "fail", "failures": [{"failure": "error"}]},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "gtest-comp-shard1.json"
            json_file.write_text(json.dumps(json_content))
            result = parse_gtest_json(json_file)

            self.assertEqual(result.component, "comp")
            self.assertEqual(result.failed_count, 1)
            self.assertIn("Suite.fail", result.failed_tests)

    def test_generate_metrics_output(self):
        results = [
            TestResult(
                component="rocblas",
                failed_tests=["Test.Fail"],
                exit_code=1,
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        self.assertIn("metadata", output)
        self.assertIn("metrics", output)
        self.assertEqual(output["metrics"][0]["status"], "failure")


if __name__ == "__main__":
    unittest.main()
