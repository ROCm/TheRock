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
    parse_stdout_log,
    create_fallback_result,
    find_and_parse_results,
    generate_metrics_output,
)


class TestParseJunitXml(unittest.TestCase):
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

    def test_parse_junit_xml_with_timeout(self):
        """Test that timeout failures are properly detected."""
        xml_content = """<?xml version="1.0"?>
<testsuite name="rocblas-test_quick_suite" tests="1" failures="1" time="600.25">
  <testcase name="rocblas-test_quick_suite" classname="rocblas-test_quick_suite" time="600.25">
    <failure type="Timeout">Test timeout after 600.25 seconds</failure>
  </testcase>
</testsuite>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "ctest-rocblas-shard1.xml"
            xml_file.write_text(xml_content)
            result = parse_junit_xml(xml_file)

            self.assertEqual(result.component, "rocblas")
            self.assertEqual(result.timeout_count, 1)
            self.assertEqual(result.failure_reason, "timeout")
            self.assertTrue(
                any("Timeout" in t for t in result.failed_tests),
                f"Expected timeout marker in {result.failed_tests}",
            )

    def test_parse_junit_xml_testsuites_wrapper(self):
        """Test parsing with <testsuites> as root element."""
        xml_content = """<?xml version="1.0"?>
<testsuites tests="3" failures="1" time="10.5">
  <testsuite name="Suite1" tests="2" failures="0">
    <testcase name="pass1" classname="Suite1"/>
    <testcase name="pass2" classname="Suite1"/>
  </testsuite>
  <testsuite name="Suite2" tests="1" failures="1">
    <testcase name="fail1" classname="Suite2">
      <failure message="assertion failed"/>
    </testcase>
  </testsuite>
</testsuites>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            xml_file = Path(tmpdir) / "ctest-multi-shard1.xml"
            xml_file.write_text(xml_content)
            result = parse_junit_xml(xml_file)

            self.assertEqual(result.component, "multi")
            self.assertEqual(result.total_tests, 3)
            self.assertEqual(result.failed_count, 1)
            self.assertEqual(result.duration_seconds, 10.5)


class TestParseGtestJson(unittest.TestCase):
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

    def test_parse_gtest_json_all_pass(self):
        json_content = {
            "tests": 3,
            "failures": 0,
            "disabled": 1,
            "time": "2.5s",
            "testsuites": [
                {
                    "name": "Suite",
                    "testsuite": [
                        {"name": "test1"},
                        {"name": "test2"},
                        {"name": "test3"},
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "gtest-passing-shard1.json"
            json_file.write_text(json.dumps(json_content))
            result = parse_gtest_json(json_file)

            self.assertEqual(result.status, "success")
            self.assertEqual(result.total_tests, 3)
            self.assertEqual(result.skipped_count, 1)
            self.assertEqual(result.duration_seconds, 2.5)


class TestParseStdoutLog(unittest.TestCase):
    def test_parse_gtest_failures_from_stdout(self):
        """Test parsing GTest failures from stdout."""
        log_content = """
[==========] Running 100 tests from 10 test suites.
[----------] 10 tests from SuiteA
[ RUN      ] SuiteA.Test1
[       OK ] SuiteA.Test1 (5 ms)
[ RUN      ] SuiteA.Test2
[  FAILED  ] SuiteA.Test2 (10 ms)
[----------] 10 tests from SuiteA (100 ms total)
[==========] 100 tests from 10 test suites ran. (1000 ms total)
[  PASSED  ] 99 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] SuiteA.Test2
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "mycomponent")

            self.assertEqual(result.component, "mycomponent")
            self.assertEqual(result.total_tests, 100)
            self.assertEqual(result.passed_tests, 99)
            self.assertIn("SuiteA.Test2", result.failed_tests)

    def test_parse_ctest_timeout_from_stdout(self):
        """Test parsing CTest timeout from stdout."""
        log_content = """
Test project /path/to/build
    Start 1: rocblas-test_quick_suite
1/1 Test #1: rocblas-test_quick_suite .........***Timeout 600.25 sec
0% tests passed, 1 tests failed out of 1
The following tests FAILED:
          1 - rocblas-test_quick_suite (Timeout)
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "rocblas")

            self.assertEqual(result.timeout_count, 1)
            self.assertEqual(result.failure_reason, "timeout")
            self.assertTrue(
                any("Timeout" in t for t in result.failed_tests),
                f"Expected timeout marker in {result.failed_tests}",
            )


class TestFallbackResult(unittest.TestCase):
    def test_create_fallback_result_timeout(self):
        result = create_fallback_result("rocblas", 124, "timeout")

        self.assertEqual(result.component, "rocblas")
        self.assertEqual(result.status, "failure")
        self.assertEqual(result.failure_reason, "timeout")
        self.assertEqual(result.timeout_count, 1)
        self.assertTrue(any("Timeout" in t for t in result.failed_tests))

    def test_create_fallback_result_sigterm(self):
        result = create_fallback_result("rocblas", 143, "timeout")

        self.assertEqual(result.exit_code, 143)
        self.assertEqual(result.failure_reason, "timeout")

    def test_create_fallback_result_success(self):
        result = create_fallback_result("rocblas", 0)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.failed_count, 0)


class TestFindAndParseResults(unittest.TestCase):
    def test_fallback_to_stdout_log(self):
        """Test that stdout log is used when no structured output exists."""
        log_content = """
[==========] 10 tests from 1 test suite ran.
[  PASSED  ] 9 tests.
[  FAILED  ] 1 test:
[  FAILED  ] Suite.FailingTest
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()

            log_file = Path(tmpdir) / "stdout.log"
            log_file.write_text(log_content)

            results = find_and_parse_results(
                results_dir, stdout_log=log_file, step_name="mytest"
            )

            self.assertEqual(len(results), 1)
            self.assertIn("Suite.FailingTest", results[0].failed_tests)

    def test_fallback_with_exit_code(self):
        """Test fallback result when no files and no stdout log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()

            results = find_and_parse_results(
                results_dir, step_name="rocblas", fallback_exit_code=143
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "failure")
            self.assertEqual(results[0].failure_reason, "timeout")


class TestGenerateMetricsOutput(unittest.TestCase):
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

    def test_generate_metrics_with_timeout(self):
        results = [
            TestResult(
                component="rocblas",
                failed_tests=["rocblas-test (Timeout)"],
                timeout_count=1,
                exit_code=1,
                status="failure",
                failure_reason="timeout",
                total_tests=1,
                passed_tests=0,
                failed_count=1,
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        metric = output["metrics"][0]
        self.assertEqual(metric["timeout_count"], 1)
        self.assertEqual(metric["failure_reason"], "timeout")
        self.assertEqual(metric["total_tests"], 1)


if __name__ == "__main__":
    unittest.main()
