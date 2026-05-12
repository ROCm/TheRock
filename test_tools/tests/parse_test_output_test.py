#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for parse_test_output module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from parse_test_output import (
    ParsedTestOutput,
    TestCase,
    TestFramework,
    TestStatus,
    detect_frameworks,
    get_failed_test_names,
    parse_ctest_output,
    parse_gtest_output,
    parse_test_output,
)


class TestGTestParser:
    """Tests for GTest output parsing."""

    def test_parse_passed_test(self):
        content = """
[ RUN      ] TestSuite.PassingTest
[       OK ] TestSuite.PassingTest (10 ms)
"""
        tests = parse_gtest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "TestSuite.PassingTest"
        assert tests[0].status == TestStatus.PASSED
        assert tests[0].duration_ms == 10.0
        assert tests[0].framework == TestFramework.GTEST

    def test_parse_failed_test(self):
        content = """
[ RUN      ] TestSuite.FailingTest
/path/to/test.cpp:42: Failure
Expected: true
  Actual: false
[  FAILED  ] TestSuite.FailingTest (25 ms)
"""
        tests = parse_gtest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "TestSuite.FailingTest"
        assert tests[0].status == TestStatus.FAILED
        assert tests[0].duration_ms == 25.0

    def test_parse_skipped_test(self):
        content = """
[ RUN      ] TestSuite.SkippedTest
[  SKIPPED ] TestSuite.SkippedTest (0 ms)
"""
        tests = parse_gtest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "TestSuite.SkippedTest"
        assert tests[0].status == TestStatus.SKIPPED

    def test_parse_multiple_tests(self):
        content = """
[ RUN      ] Suite.Test1
[       OK ] Suite.Test1 (5 ms)
[ RUN      ] Suite.Test2
[  FAILED  ] Suite.Test2 (10 ms)
[ RUN      ] Suite.Test3
[       OK ] Suite.Test3 (3 ms)
"""
        tests = parse_gtest_output(content)
        assert len(tests) == 3

        names = {t.name: t for t in tests}
        assert names["Suite.Test1"].status == TestStatus.PASSED
        assert names["Suite.Test2"].status == TestStatus.FAILED
        assert names["Suite.Test3"].status == TestStatus.PASSED

    def test_parse_failure_summary(self):
        """Test parsing failed tests from the summary section."""
        content = """
[==========] 10 tests from 2 test suites ran. (1000 ms total)
[  PASSED  ] 8 tests.
[  FAILED  ] 2 tests, listed below:
[  FAILED  ] Suite.FailedTest1
[  FAILED  ] Suite.FailedTest2

 2 FAILED TESTS
"""
        tests = parse_gtest_output(content)
        failed = [t for t in tests if t.status == TestStatus.FAILED]
        assert len(failed) == 2
        names = {t.name for t in failed}
        assert "Suite.FailedTest1" in names
        assert "Suite.FailedTest2" in names

    def test_parse_interrupted_test(self):
        """Test detecting tests that started but never completed."""
        content = """
[ RUN      ] Suite.InterruptedTest
Some output...
Process killed by signal
"""
        tests = parse_gtest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "Suite.InterruptedTest"
        assert tests[0].status == TestStatus.RUNNING

    def test_ignore_summary_counts(self):
        """Test that summary counts are not treated as test names."""
        content = """
[  PASSED  ] 42 tests.
[  FAILED  ] 3 tests, listed below:
"""
        tests = parse_gtest_output(content)
        # Should not create tests named "42" or "3"
        for test in tests:
            assert not test.name.isdigit()


class TestCTestParser:
    """Tests for CTest output parsing."""

    def test_parse_passed_test(self):
        content = """
      Start  1: my_test
1/1 Test  #1: my_test ......................   Passed    1.23 sec
"""
        tests = parse_ctest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "my_test"
        assert tests[0].status == TestStatus.PASSED
        assert tests[0].duration_ms == pytest.approx(1230.0)
        assert tests[0].framework == TestFramework.CTEST

    def test_parse_failed_test(self):
        content = """
      Start  1: failing_test
1/1 Test  #1: failing_test .................***Failed    2.50 sec
"""
        tests = parse_ctest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "failing_test"
        assert tests[0].status == TestStatus.FAILED

    def test_parse_timeout_test(self):
        content = """
      Start  1: slow_test
1/1 Test  #1: slow_test ....................***Timeout  600.00 sec
"""
        tests = parse_ctest_output(content)
        assert len(tests) == 1
        assert tests[0].name == "slow_test"
        assert tests[0].status == TestStatus.TIMEOUT
        assert tests[0].duration_ms == pytest.approx(600000.0)

    def test_parse_multiple_tests(self):
        content = """
      Start  1: test1
1/3 Test  #1: test1 ........................   Passed    0.50 sec
      Start  2: test2
2/3 Test  #2: test2 ........................***Failed    1.00 sec
      Start  3: test3
3/3 Test  #3: test3 ........................***Timeout  300.00 sec

66% tests passed, 2 tests failed out of 3

The following tests FAILED:
      2 - test2 (Failed)
      3 - test3 (Timeout)
"""
        tests = parse_ctest_output(content)
        assert len(tests) == 3

        names = {t.name: t for t in tests}
        assert names["test1"].status == TestStatus.PASSED
        assert names["test2"].status == TestStatus.FAILED
        assert names["test3"].status == TestStatus.TIMEOUT

    def test_parse_failure_summary_only(self):
        """Test parsing from summary when execution lines are missing."""
        content = """
The following tests FAILED:
      5 - complex_test (Failed)
     12 - another_test (Timeout)
"""
        tests = parse_ctest_output(content)
        assert len(tests) == 2

        names = {t.name: t for t in tests}
        assert names["complex_test"].status == TestStatus.FAILED
        assert names["another_test"].status == TestStatus.TIMEOUT


class TestFrameworkDetection:
    """Tests for framework detection."""

    def test_detect_gtest(self):
        content = "[ RUN      ] TestSuite.TestName"
        frameworks = detect_frameworks(content)
        assert TestFramework.GTEST in frameworks

    def test_detect_ctest(self):
        content = "1/1 Test  #1: my_test ....   Passed"
        frameworks = detect_frameworks(content)
        assert TestFramework.CTEST in frameworks

    def test_detect_both(self):
        content = """
1/1 Test  #1: gtest_wrapper ....   Passed
[ RUN      ] Suite.InnerTest
[       OK ] Suite.InnerTest (10 ms)
"""
        frameworks = detect_frameworks(content)
        assert TestFramework.GTEST in frameworks
        assert TestFramework.CTEST in frameworks


class TestCombinedParsing:
    """Tests for combined ctest + gtest parsing."""

    def test_nested_gtest_in_ctest(self):
        """Test that both inner gtest failures AND outer ctest failure are reported."""
        content = """
      Start  1: unit_tests
1/1 Test  #1: unit_tests ...................***Failed    5.00 sec

Output from unit_tests:
[ RUN      ] Math.Addition
[       OK ] Math.Addition (1 ms)
[ RUN      ] Math.Division
/path/test.cpp:10: Failure
Division by zero!
[  FAILED  ] Math.Division (2 ms)
[ RUN      ] Math.Subtraction
[       OK ] Math.Subtraction (1 ms)

[==========] 3 tests from 1 test suite ran. (4 ms total)
[  PASSED  ] 2 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] Math.Division

 1 FAILED TEST

The following tests FAILED:
      1 - unit_tests (Failed)
"""
        result = parse_test_output(content)

        # Should have both inner gtest failure AND outer ctest failure
        failed = [t for t in result.tests if t.status == TestStatus.FAILED]
        assert len(failed) == 2

        # Inner gtest failure
        gtest_failed = [t for t in failed if t.framework == TestFramework.GTEST]
        assert len(gtest_failed) == 1
        assert gtest_failed[0].name == "Math.Division"
        assert gtest_failed[0].parent_test == "unit_tests"

        # Outer ctest failure
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

        # Should have both the timeout and the interrupted test
        timeouts = [t for t in result.tests if t.status == TestStatus.TIMEOUT]
        assert len(timeouts) == 1
        assert timeouts[0].name == "slow_tests"
        assert timeouts[0].framework == TestFramework.CTEST

        running = [t for t in result.tests if t.status == TestStatus.RUNNING]
        assert len(running) == 1
        assert running[0].name == "Slow.Test2"

    def test_ctest_failure_without_gtest(self):
        """Test ctest failure when there's no inner gtest output."""
        content = """
      Start  1: script_test
1/1 Test  #1: script_test ..................***Failed    1.00 sec

Some random output that's not gtest format

The following tests FAILED:
      1 - script_test (Failed)
"""
        result = parse_test_output(content)

        # Should report the ctest failure
        failed = [t for t in result.tests if t.status == TestStatus.FAILED]
        assert len(failed) == 1
        assert failed[0].name == "script_test"
        assert failed[0].framework == TestFramework.CTEST


class TestGetFailedTestNames:
    """Tests for the convenience function."""

    def test_simple_failures(self):
        content = """
[ RUN      ] Suite.Test1
[  FAILED  ] Suite.Test1 (10 ms)
[ RUN      ] Suite.Test2
[       OK ] Suite.Test2 (5 ms)
"""
        names = get_failed_test_names(content)
        assert names == ["Suite.Test1"]

    def test_timeout_annotation(self):
        content = """
1/1 Test  #1: slow_test ....................***Timeout  600.00 sec
"""
        names = get_failed_test_names(content)
        assert names == ["slow_test (Timeout)"]

    def test_interrupted_annotation(self):
        content = """
[ RUN      ] Suite.Interrupted
Process killed
"""
        names = get_failed_test_names(content)
        assert names == ["Suite.Interrupted (Interrupted)"]


class TestParsedTestOutputSummary:
    """Tests for summary calculations."""

    def test_summary_counts(self):
        content = """
[ RUN      ] Suite.Pass1
[       OK ] Suite.Pass1 (1 ms)
[ RUN      ] Suite.Pass2
[       OK ] Suite.Pass2 (1 ms)
[ RUN      ] Suite.Fail1
[  FAILED  ] Suite.Fail1 (1 ms)
[ RUN      ] Suite.Skip1
[  SKIPPED ] Suite.Skip1 (0 ms)
"""
        result = parse_test_output(content)

        assert result.total_tests == 4
        assert result.passed_count == 2
        assert result.failed_count == 1
        assert result.skipped_count == 1
        assert result.has_failures is True

    def test_no_failures(self):
        content = """
[ RUN      ] Suite.Test1
[       OK ] Suite.Test1 (1 ms)
"""
        result = parse_test_output(content)
        assert result.has_failures is False
