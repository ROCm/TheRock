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
    TestResult,
    create_fallback_result,
    find_and_parse_results,
    generate_metrics_output,
    parse_stdout_log,
)


class TestParseStdoutLog:
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

            assert result.component == "mycomponent"
            assert len(result.failed_tests) == 1
            assert result.failed_tests[0].name == "SuiteA.Test2"
            assert result.failed_tests[0].is_outer is False  # gtest is inner

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

            assert result.timeout_count >= 1
            timeouts = [t for t in result.failed_tests if t.status == "timeout"]
            assert len(timeouts) >= 1
            assert timeouts[0].is_outer is True  # ctest is outer

    def test_parse_nested_gtest_in_ctest(self):
        """Test parsing inner gtest failures AND outer ctest failure."""
        log_content = """
      Start  1: unit_tests
1/1 Test  #1: unit_tests ...................***Failed    5.00 sec

[ RUN      ] Suite.Test1
[       OK ] Suite.Test1 (1 ms)
[ RUN      ] Suite.Test2
[  FAILED  ] Suite.Test2 (2 ms)

[  FAILED  ] 1 test, listed below:
[  FAILED  ] Suite.Test2

The following tests FAILED:
      1 - unit_tests (Failed)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "unit-tests")

            assert result.status == "failure"

            # Should have both inner gtest failure AND outer ctest failure
            inner_tests = [t for t in result.failed_tests if not t.is_outer]
            outer_tests = [t for t in result.failed_tests if t.is_outer]

            assert len(inner_tests) == 1
            assert inner_tests[0].name == "Suite.Test2"

            assert len(outer_tests) == 1
            assert outer_tests[0].name == "unit_tests"

    def test_parse_empty_log(self):
        """Test parsing empty log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "empty.log"
            log_file.write_text("")
            result = parse_stdout_log(log_file, "empty-tests")

            assert result.status == "success"
            assert result.failed_count == 0
            assert len(result.failed_tests) == 0

    def test_parse_interrupted_test(self):
        """Test detecting interrupted tests."""
        log_content = """
[ RUN      ] Suite.LongTest
Some output...
Process killed by signal
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(log_content)
            result = parse_stdout_log(log_file, "interrupted-tests")

            assert result.status == "failure"
            interrupted = [
                t for t in result.failed_tests if t.failure_reason == "interrupted"
            ]
            assert len(interrupted) == 1


class TestFallbackResult:
    def test_create_fallback_result_timeout(self):
        result = create_fallback_result("rocblas", 124, "timeout")

        assert result.component == "rocblas"
        assert result.status == "failure"
        assert result.failure_reason == "timeout"
        assert result.timeout_count == 1
        assert len(result.failed_tests) == 1
        assert result.failed_tests[0].status == "timeout"
        assert result.failed_tests[0].is_outer is True

    def test_create_fallback_result_sigterm(self):
        result = create_fallback_result("rocblas", 143, "timeout")

        assert result.exit_code == 143
        assert result.failure_reason == "timeout"
        assert result.failed_tests[0].is_outer is True

    def test_create_fallback_result_success(self):
        result = create_fallback_result("rocblas", 0)

        assert result.status == "success"
        assert result.failed_count == 0
        assert len(result.failed_tests) == 0


class TestFindAndParseResults:
    def test_parse_stdout_log(self):
        """Test that stdout log is parsed."""
        log_content = """
[==========] 10 tests from 1 test suite ran.
[  PASSED  ] 9 tests.
[  FAILED  ] 1 test:
[  FAILED  ] Suite.FailingTest
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "stdout.log"
            log_file.write_text(log_content)

            results = find_and_parse_results(stdout_log=log_file, step_name="mytest")

            assert len(results) == 1
            assert any(t.name == "Suite.FailingTest" for t in results[0].failed_tests)

    def test_fallback_with_exit_code(self):
        """Test fallback result when no stdout log."""
        results = find_and_parse_results(
            stdout_log=Path("/nonexistent/path.log"),
            step_name="rocblas",
            fallback_exit_code=143,
        )

        assert len(results) == 1
        assert results[0].status == "failure"
        assert results[0].failure_reason == "timeout"
        assert results[0].failed_tests[0].is_outer is True

    def test_no_fallback_on_success_exit(self):
        """Test no fallback when exit code is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "empty.log"
            log_file.write_text("")

            results = find_and_parse_results(
                stdout_log=log_file,
                step_name="rocblas",
                fallback_exit_code=0,
            )

            # No results because log is empty and exit code is 0
            assert len(results) == 0


class TestGenerateMetricsOutput:
    def test_generate_metrics_inner_tests(self):
        """Test that inner (gtest) tests use sub_step_name."""
        results = [
            TestResult(
                component="rocblas",
                failed_tests=[
                    FailedTest(name="Test.Fail1", status="failure", is_outer=False),
                    FailedTest(name="Test.Fail2", status="failure", is_outer=False),
                ],
                exit_code=1,
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        assert "metadata" in output
        assert "metrics" in output
        assert len(output["metrics"]) == 2

        # Inner tests should use sub_step_name
        assert output["metrics"][0]["sub_step_name"] == "Test.Fail1"
        assert "step_name" not in output["metrics"][0]
        assert output["metrics"][1]["sub_step_name"] == "Test.Fail2"
        assert "step_name" not in output["metrics"][1]

    def test_generate_metrics_outer_tests(self):
        """Test that outer (ctest) tests use step_name."""
        results = [
            TestResult(
                component="rocblas",
                failed_tests=[
                    FailedTest(name="unit_tests", status="failure", is_outer=True),
                ],
                exit_code=1,
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        assert len(output["metrics"]) == 1
        # Outer tests should use step_name
        assert output["metrics"][0]["step_name"] == "unit_tests"
        assert "sub_step_name" not in output["metrics"][0]

    def test_generate_metrics_mixed_inner_outer(self):
        """Test mixed inner (gtest) and outer (ctest) failures."""
        results = [
            TestResult(
                component="hipblaslt",
                failed_tests=[
                    FailedTest(
                        name="Suite.InnerFail", status="failure", is_outer=False
                    ),
                    FailedTest(name="unit_tests", status="failure", is_outer=True),
                ],
                exit_code=1,
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="hipblaslt")

        assert len(output["metrics"]) == 2

        # Inner test uses sub_step_name
        inner_metric = output["metrics"][0]
        assert inner_metric["sub_step_name"] == "Suite.InnerFail"
        assert "step_name" not in inner_metric

        # Outer test uses step_name
        outer_metric = output["metrics"][1]
        assert outer_metric["step_name"] == "unit_tests"
        assert "sub_step_name" not in outer_metric

    def test_generate_metrics_with_timeout(self):
        """Test that timeout tests have status='timeout' and failure_reason."""
        results = [
            TestResult(
                component="rocblas",
                failed_tests=[
                    FailedTest(
                        name="rocblas-test",
                        status="timeout",
                        is_outer=True,
                        failure_reason="timeout",
                    ),
                ],
                timeout_count=1,
                exit_code=1,
                status="failure",
                failure_reason="timeout",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        assert len(output["metrics"]) == 1
        metric = output["metrics"][0]
        assert metric["step_name"] == "rocblas-test"
        assert metric["status"] == "timeout"
        assert metric["failure_reason"] == "timeout"

    def test_generate_metrics_no_failures(self):
        """Test that no metrics are generated when there are no failures."""
        results = [
            TestResult(
                component="rocblas",
                failed_tests=[],
                exit_code=0,
                status="success",
            )
        ]
        output = generate_metrics_output(results, step_name="rocblas")

        assert len(output["metrics"]) == 0

    def test_generate_metrics_interrupted(self):
        """Test that interrupted tests have failure_reason='interrupted'."""
        results = [
            TestResult(
                component="test",
                failed_tests=[
                    FailedTest(
                        name="Suite.Test",
                        status="failure",
                        is_outer=False,
                        failure_reason="interrupted",
                    ),
                ],
                exit_code=1,
                status="failure",
            )
        ]
        output = generate_metrics_output(results, step_name="test")

        assert len(output["metrics"]) == 1
        metric = output["metrics"][0]
        assert metric["sub_step_name"] == "Suite.Test"
        assert metric["failure_reason"] == "interrupted"
