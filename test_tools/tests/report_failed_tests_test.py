# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from report_failed_tests import (
    FailedTest,
    ParsedTestOutput,
    TestFramework,
    TestResult,
    TestStatus,
    create_fallback_result,
    find_and_parse_results,
    generate_metrics_output,
    parse_stdout_log,
    parse_test_output,
)


# =============================================================================
# Test Output Parsing Tests
# =============================================================================


class TestGTestParser:
    """Tests for GTest output parsing."""

    def test_parse_passed_test(self):
        content = """
[ RUN      ] TestSuite.PassingTest
[       OK ] TestSuite.PassingTest (10 ms)
"""
        result = parse_test_output(content)
        passed = [t for t in result.tests if t.status == TestStatus.PASSED]
        assert len(passed) == 1
        assert passed[0].name == "TestSuite.PassingTest"
        assert passed[0].framework == TestFramework.GTEST

    def test_parse_failed_test(self):
        content = """
[ RUN      ] TestSuite.FailingTest
[  FAILED  ] TestSuite.FailingTest (25 ms)
"""
        result = parse_test_output(content)
        failed = [t for t in result.tests if t.status == TestStatus.FAILED]
        assert len(failed) == 1
        assert failed[0].name == "TestSuite.FailingTest"

    def test_parse_multiple_tests(self):
        content = """
[ RUN      ] Suite.Test1
[       OK ] Suite.Test1 (5 ms)
[ RUN      ] Suite.Test2
[  FAILED  ] Suite.Test2 (10 ms)
[ RUN      ] Suite.Test3
[       OK ] Suite.Test3 (3 ms)
"""
        result = parse_test_output(content)
        assert len(result.tests) == 3

    def test_parse_interrupted_test(self):
        content = """
[ RUN      ] Suite.InterruptedTest
Some output...
Process killed by signal
"""
        result = parse_test_output(content)
        running = [t for t in result.tests if t.status == TestStatus.RUNNING]
        assert len(running) == 1
        assert running[0].name == "Suite.InterruptedTest"


class TestCTestParser:
    """Tests for CTest output parsing."""

    def test_parse_passed_test(self):
        content = """
      Start  1: my_test
1/1 Test  #1: my_test ......................   Passed    1.23 sec
"""
        result = parse_test_output(content)
        passed = [t for t in result.tests if t.status == TestStatus.PASSED]
        assert len(passed) == 1
        assert passed[0].name == "my_test"
        assert passed[0].framework == TestFramework.CTEST

    def test_parse_timeout_test(self):
        content = """
      Start  1: slow_test
1/1 Test  #1: slow_test ....................***Timeout  600.00 sec
"""
        result = parse_test_output(content)
        timeouts = [t for t in result.tests if t.status == TestStatus.TIMEOUT]
        assert len(timeouts) == 1
        assert timeouts[0].name == "slow_test"


class TestCombinedParsing:
    """Tests for combined ctest + gtest parsing."""

    def test_nested_gtest_in_ctest(self):
        """Test that both inner gtest failures AND outer ctest failure are reported."""
        content = """
      Start  1: unit_tests
1/1 Test  #1: unit_tests ...................***Failed    5.00 sec

[ RUN      ] Math.Addition
[       OK ] Math.Addition (1 ms)
[ RUN      ] Math.Division
[  FAILED  ] Math.Division (2 ms)

The following tests FAILED:
      1 - unit_tests (Failed)
"""
        result = parse_test_output(content)

        failed = [t for t in result.tests if t.status == TestStatus.FAILED]
        assert len(failed) == 2

        gtest_failed = [t for t in failed if t.framework == TestFramework.GTEST]
        assert len(gtest_failed) == 1
        assert gtest_failed[0].name == "Math.Division"

        ctest_failed = [t for t in failed if t.framework == TestFramework.CTEST]
        assert len(ctest_failed) == 1
        assert ctest_failed[0].name == "unit_tests"

    def test_ctest_timeout_preserved(self):
        """Test that ctest timeout is preserved even with gtest output."""
        content = """
      Start  1: slow_tests
1/1 Test  #1: slow_tests ...................***Timeout  600.00 sec

[ RUN      ] Slow.Test1
[       OK ] Slow.Test1 (100 ms)
[ RUN      ] Slow.Test2
"""
        result = parse_test_output(content)

        timeouts = [t for t in result.tests if t.status == TestStatus.TIMEOUT]
        assert len(timeouts) == 1
        assert timeouts[0].name == "slow_tests"

        running = [t for t in result.tests if t.status == TestStatus.RUNNING]
        assert len(running) == 1
        assert running[0].name == "Slow.Test2"


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestParseStdoutLog:
    def test_parse_gtest_failures_from_stdout(self):
        log_content = """
[ RUN      ] SuiteA.Test1
[       OK ] SuiteA.Test1 (5 ms)
[ RUN      ] SuiteA.Test2
[  FAILED  ] SuiteA.Test2 (10 ms)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "mycomponent")

            assert result.component == "mycomponent"
            assert len(result.failed_tests) == 1
            assert result.failed_tests[0].name == "SuiteA.Test2"
            assert result.failed_tests[0].is_outer is False

    def test_parse_ctest_timeout_from_stdout(self):
        log_content = """
1/1 Test #1: rocblas-test .........***Timeout 600.25 sec

The following tests FAILED:
          1 - rocblas-test (Timeout)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "rocblas")

            timeouts = [t for t in result.failed_tests if t.status == "timeout"]
            assert len(timeouts) >= 1
            assert timeouts[0].is_outer is True

    def test_parse_nested_gtest_in_ctest(self):
        log_content = """
      Start  1: unit_tests
1/1 Test  #1: unit_tests ...................***Failed    5.00 sec

[ RUN      ] Suite.Test1
[       OK ] Suite.Test1 (1 ms)
[ RUN      ] Suite.Test2
[  FAILED  ] Suite.Test2 (2 ms)

The following tests FAILED:
      1 - unit_tests (Failed)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "unit-tests")

            assert result.status == "failure"

            inner_tests = [t for t in result.failed_tests if not t.is_outer]
            outer_tests = [t for t in result.failed_tests if t.is_outer]

            assert len(inner_tests) == 1
            assert inner_tests[0].name == "Suite.Test2"

            assert len(outer_tests) == 1
            assert outer_tests[0].name == "unit_tests"

    def test_parse_empty_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "empty.log"
            log_file.write_text("")
            result = parse_stdout_log(log_file, "empty-tests")

            assert result.status == "success"
            assert len(result.failed_tests) == 0


class TestFallbackResult:
    def test_create_fallback_result_timeout(self):
        result = create_fallback_result("rocblas", 124, "timeout")

        assert result.status == "failure"
        assert result.failure_reason == "timeout"
        assert result.failed_tests[0].is_outer is True

    def test_create_fallback_result_success(self):
        result = create_fallback_result("rocblas", 0)

        assert result.status == "success"
        assert len(result.failed_tests) == 0


class TestFindAndParseResults:
    def test_parse_stdout_log(self):
        log_content = """
[  FAILED  ] Suite.FailingTest
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stdout.log"
            log_file.write_text(log_content)

            results = find_and_parse_results(stdout_log=log_file, step_name="mytest")

            assert len(results) == 1

    def test_fallback_with_exit_code(self):
        results = find_and_parse_results(
            stdout_log=Path("/nonexistent/path.log"),
            step_name="rocblas",
            fallback_exit_code=143,
        )

        assert len(results) == 1
        assert results[0].status == "failure"
        assert results[0].failure_reason == "timeout"


class TestGenerateMetricsOutput:
    def test_generate_metrics_inner_tests(self):
        results = [
            TestResult(
                component="rocblas",
                failed_tests=[
                    FailedTest(name="Test.Fail1", status="failure", is_outer=False),
                ],
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        assert len(output["metrics"]) == 1
        assert output["metrics"][0]["sub_step_name"] == "Test.Fail1"
        assert "step_name" not in output["metrics"][0]

    def test_generate_metrics_outer_tests(self):
        results = [
            TestResult(
                component="rocblas",
                failed_tests=[
                    FailedTest(name="unit_tests", status="failure", is_outer=True),
                ],
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        assert len(output["metrics"]) == 1
        assert output["metrics"][0]["step_name"] == "unit_tests"
        assert "sub_step_name" not in output["metrics"][0]

    def test_generate_metrics_mixed(self):
        results = [
            TestResult(
                component="test",
                failed_tests=[
                    FailedTest(name="Suite.Inner", status="failure", is_outer=False),
                    FailedTest(name="outer_test", status="failure", is_outer=True),
                ],
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="test")

        assert len(output["metrics"]) == 2
        assert output["metrics"][0]["sub_step_name"] == "Suite.Inner"
        assert output["metrics"][1]["step_name"] == "outer_test"

    def test_generate_metrics_no_failures(self):
        results = [TestResult(component="rocblas", status="success")]
        output = generate_metrics_output(results, step_name="rocblas")

        assert len(output["metrics"]) == 0
