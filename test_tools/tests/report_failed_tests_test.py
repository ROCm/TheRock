# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from report_failed_tests import (
    TestFramework,
    TestStatus,
    parse_test_output,
    parse_stdout_log,
    generate_metrics_output,
    TestResult,
    FailedTest,
)


class TestParseTestOutput:
    def test_gtest_failures(self):
        content = """
[ RUN      ] Suite.Test1
[       OK ] Suite.Test1 (5 ms)
[ RUN      ] Suite.Test2
[  FAILED  ] Suite.Test2 (10 ms)
"""
        result = parse_test_output(content)
        assert TestFramework.GTEST in result.frameworks_detected
        failed = [t for t in result.tests if t.status == TestStatus.FAILED]
        assert len(failed) == 1
        assert failed[0].name == "Suite.Test2"

    def test_ctest_timeout(self):
        content = """
1/1 Test  #1: slow_test ....................***Timeout  600.00 sec
"""
        result = parse_test_output(content)
        assert TestFramework.CTEST in result.frameworks_detected
        timeouts = [t for t in result.tests if t.status == TestStatus.TIMEOUT]
        assert len(timeouts) == 1

    def test_nested_gtest_in_ctest(self):
        content = """
      Start  1: unit_tests
1/1 Test  #1: unit_tests ...................***Failed    5.00 sec

[ RUN      ] Math.Add
[       OK ] Math.Add (1 ms)
[ RUN      ] Math.Div
[  FAILED  ] Math.Div (2 ms)

The following tests FAILED:
      1 - unit_tests (Failed)
"""
        result = parse_test_output(content)
        assert TestFramework.GTEST in result.frameworks_detected
        assert TestFramework.CTEST in result.frameworks_detected

        gtest_failed = [
            t
            for t in result.tests
            if t.framework == TestFramework.GTEST and t.status == TestStatus.FAILED
        ]
        ctest_failed = [
            t
            for t in result.tests
            if t.framework == TestFramework.CTEST and t.status == TestStatus.FAILED
        ]

        assert len(gtest_failed) == 1
        assert gtest_failed[0].name == "Math.Div"
        assert len(ctest_failed) == 1
        assert ctest_failed[0].name == "unit_tests"

    def test_interrupted_test(self):
        content = """
[ RUN      ] Suite.LongTest
Process killed
"""
        result = parse_test_output(content)
        running = [t for t in result.tests if t.status == TestStatus.RUNNING]
        assert len(running) == 1

    def test_gtest_with_getparam(self):
        content = """
[ RUN      ] quick/check_matrix_hyb.util/f32_r_50_50_0b_general_L_sorted_rand_auto_0
[  FAILED  ] quick/check_matrix_hyb.util/f32_r_50_50_0b_general_L_sorted_rand_auto_0, where GetParam() = { function: "check_matrix_hyb" }
"""
        result = parse_test_output(content)
        failed = [t for t in result.tests if t.status == TestStatus.FAILED]
        assert len(failed) == 1
        assert (
            failed[0].name
            == "quick/check_matrix_hyb.util/f32_r_50_50_0b_general_L_sorted_rand_auto_0"
        )
        running = [t for t in result.tests if t.status == TestStatus.RUNNING]
        assert len(running) == 0


class TestParseStdoutLog:
    def test_parses_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text("[ RUN      ] S.T\n[  FAILED  ] S.T (1 ms)")
            result = parse_stdout_log(log_file, "comp")
            assert result.status == "failure"
            assert len(result.failed_tests) == 1

    def test_inner_outer_distinction(self):
        content = """
1/1 Test  #1: wrapper ...................***Failed    1.00 sec
[ RUN      ] Inner.Test
[  FAILED  ] Inner.Test (1 ms)
The following tests FAILED:
      1 - wrapper (Failed)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            log_file.write_text(content)
            result = parse_stdout_log(log_file, "comp")
            inner = [t for t in result.failed_tests if not t.is_outer]
            outer = [t for t in result.failed_tests if t.is_outer]
            assert len(inner) == 1
            assert len(outer) == 1


class TestGenerateMetrics:
    def test_inner_uses_sub_step_name(self):
        results = [
            TestResult(
                component="t",
                failed_tests=[
                    FailedTest(name="Inner.T", status="failure", is_outer=False)
                ],
                status="failure",
            )
        ]
        output = generate_metrics_output(results, "t")
        assert "sub_step_name" in output["metrics"][0]
        assert "step_name" not in output["metrics"][0]

    def test_outer_uses_step_name(self):
        results = [
            TestResult(
                component="t",
                failed_tests=[
                    FailedTest(name="outer", status="failure", is_outer=True)
                ],
                status="failure",
            )
        ]
        output = generate_metrics_output(results, "t")
        assert "step_name" in output["metrics"][0]
        assert "sub_step_name" not in output["metrics"][0]
