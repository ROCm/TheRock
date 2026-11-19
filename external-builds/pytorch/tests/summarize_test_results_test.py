# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for test counting and completeness in summarize_test_results.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

import summarize_test_results as summarize

SUCCESS_OUTPUT = "test_nn 1/1 was successful\ntest_cuda 1/1 was successful\n"


def junit_xml(*cases: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<testsuites><testsuite name="pytest">'
        + "".join(cases)
        + "</testsuite></testsuites>\n"
    )


def case(name: str, classname: str = "test_nn.TestNN", body: str = "") -> str:
    return (
        f'<testcase classname="{classname}" name="{name}" time="0.1">{body}</testcase>'
    )


FAILURE = '<failure message="boom">trace</failure>'
ERROR = '<error message="setup broke">trace</error>'
SKIP = '<skipped message="not supported on ROCm" />'
SKIP_FAST = (
    '<skipped message="test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST" />'
)
RERUN = '<rerun message="first attempt failed" />'


class ReportSet:
    """Builds a test-reports directory shaped like run_test.py output."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def add_report(self, module: str, xml: str, attempt: str = "a") -> None:
        module_dir = self.root / "python-pytest" / module
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / f"{module}-{attempt}.xml").write_text(xml)

    def add_module_log(self, module: str) -> None:
        (self.root / f"{module}_1.1_0123456789abcdef_.log").write_text("")

    def record_run(
        self, name: str, output: str = SUCCESS_OUTPUT, *, completed: bool = True
    ) -> None:
        (self.root / summarize.STARTED_MARKER.format(name)).write_text("")
        if completed:
            (self.root / summarize.RUNNER_OUTPUT.format(name)).write_text(output)
            (self.root / summarize.COMPLETION_MARKER.format(name)).write_text("")


class TestCaseBase(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp = Path(self._temp_dir.name)
        self.reports = ReportSet(self.temp / "test-reports")

    def tearDown(self):
        self._temp_dir.cleanup()

    def complete_report_set(self) -> None:
        self.reports.add_report(
            "test_nn",
            junit_xml(case("test_a"), case("test_b", body=SKIP)),
        )
        self.reports.add_module_log("test_nn")
        self.reports.record_run("core")


class CollectOutcomesTest(TestCaseBase):
    def test_retry_pass_is_flaky_not_failed(self):
        # run_test.py reruns a failed module, writing a second report.
        self.reports.add_report(
            "test_nn", junit_xml(case("test_a", body=FAILURE)), attempt="1"
        )
        self.reports.add_report("test_nn", junit_xml(case("test_a")), attempt="2")

        outcomes = summarize.collect_outcomes(self.reports.root)

        outcome = outcomes[("test_nn", "TestNN", "test_a")]
        self.assertEqual(outcome["status"], "PASSED")
        self.assertTrue(outcome["flaky"])

    def test_rerun_records_mark_a_pass_flaky(self):
        self.reports.add_report("test_nn", junit_xml(case("test_a", body=RERUN)))

        outcome = summarize.collect_outcomes(self.reports.root)[
            ("test_nn", "TestNN", "test_a")
        ]

        self.assertEqual(outcome["status"], "PASSED")
        self.assertTrue(outcome["flaky"])

    def test_failure_on_every_attempt_stays_failed(self):
        self.reports.add_report(
            "test_nn", junit_xml(case("test_a", body=FAILURE)), attempt="1"
        )
        self.reports.add_report(
            "test_nn", junit_xml(case("test_a", body=ERROR)), attempt="2"
        )

        outcome = summarize.collect_outcomes(self.reports.root)[
            ("test_nn", "TestNN", "test_a")
        ]

        self.assertEqual(outcome["status"], "FAILED")
        self.assertFalse(outcome["flaky"])

    def test_count_outcomes_excludes_skips_from_run(self):
        self.reports.add_report(
            "test_nn",
            junit_xml(
                case("test_pass"),
                case("test_fail", body=FAILURE),
                case("test_error", body=ERROR),
                case("test_skip", body=SKIP),
            ),
        )

        counts = summarize.count_outcomes(summarize.collect_outcomes(self.reports.root))

        self.assertEqual(
            counts,
            {
                "PASSED": 1,
                "FAILED": 1,
                "ERROR": 1,
                "SKIPPED": 1,
                "NOT_SELECTED": 0,
                "FLAKY": 0,
                "RUN": 3,
            },
        )

    def test_skip_fast_skips_are_not_selected(self):
        self.reports.add_report(
            "test_linalg",
            junit_xml(
                case("test_slow", "test_linalg.TestLinalg"),
                case("test_fast", "test_linalg.TestLinalg", SKIP_FAST),
                case("test_rocm", "test_linalg.TestLinalg", SKIP),
            ),
        )

        counts = summarize.count_outcomes(summarize.collect_outcomes(self.reports.root))

        self.assertEqual(
            (counts["RUN"], counts["SKIPPED"], counts["NOT_SELECTED"]), (1, 1, 1)
        )


class CheckReportSetTest(TestCaseBase):
    def test_complete_report_set(self):
        self.complete_report_set()

        self.assertEqual(summarize.check_report_set(self.reports.root), [])

    def test_missing_directory(self):
        self.assertEqual(
            summarize.check_report_set(self.temp / "missing"), ["no test reports"]
        )

    def test_no_started_marker(self):
        self.reports.add_report("test_nn", junit_xml(case("test_a")))

        self.assertIn(
            "no test runner started", summarize.check_report_set(self.reports.root)
        )

    def test_runner_that_did_not_exit(self):
        # A step timeout kills the runner before it writes its completion
        # marker or output.
        self.complete_report_set()
        self.reports.record_run("transformers", completed=False)

        problems = summarize.check_report_set(self.reports.root)

        self.assertIn("the transformers test runner did not exit", problems)
        self.assertIn("the transformers test runner output is missing", problems)

    def test_module_killed_by_signal(self):
        self.complete_report_set()
        self.reports.record_run(
            "core", "test_cuda failed!\nReceived signal: SIGSEGV\n" + SUCCESS_OUTPUT
        )

        self.assertIn(
            "a test module was killed by a signal",
            summarize.check_report_set(self.reports.root),
        )

    def test_unparseable_report(self):
        self.complete_report_set()
        self.reports.add_report("test_cuda", "<testsuites><testsuite>")

        self.assertIn(
            "unreadable report test_cuda-a.xml",
            summarize.check_report_set(self.reports.root),
        )

    def test_non_junit_xml_is_unreadable(self):
        self.complete_report_set()
        self.reports.add_report("test_cuda", "<coverage />")

        self.assertIn(
            "unreadable report test_cuda-a.xml",
            summarize.check_report_set(self.reports.root),
        )

    def test_launched_module_without_report_that_failed(self):
        # pytest writes JUnit at session end, so a module that fails to import
        # leaves only its run_test.py log behind.
        self.complete_report_set()
        self.reports.add_module_log("test_linalg")
        self.reports.record_run("core", SUCCESS_OUTPUT + "test_linalg 1/1 failed!\n")

        self.assertIn(
            "no report from module(s) that did not succeed: test_linalg",
            summarize.check_report_set(self.reports.root),
        )

    def test_launched_module_without_report_or_outcome_line(self):
        self.complete_report_set()
        self.reports.add_module_log("test_linalg")

        self.assertIn(
            "no report from module(s) that did not succeed: test_linalg",
            summarize.check_report_set(self.reports.root),
        )

    def test_successful_module_without_report_is_complete(self):
        # Modules that skip on this platform exit 0 without writing JUnit.
        self.complete_report_set()
        self.reports.add_module_log("test_cuda")

        self.assertEqual(summarize.check_report_set(self.reports.root), [])

    def test_nested_module_outcome_lines_match_dotted_report_names(self):
        self.complete_report_set()
        self.reports.add_module_log("inductor.test_smoke")
        self.reports.record_run(
            "core", SUCCESS_OUTPUT + "inductor/test_smoke 1/1 was successful\n"
        )

        self.assertEqual(summarize.check_report_set(self.reports.root), [])


class WriteCountOutputsTest(TestCaseBase):
    def read_outputs(self, output_file: Path) -> dict[str, str]:
        return dict(line.split("=", 1) for line in output_file.read_text().splitlines())

    def test_complete_counts(self):
        self.complete_report_set()
        self.reports.add_report(
            "test_cuda",
            junit_xml(
                case("test_x", "test_cuda.TestCuda", FAILURE),
                case("test_y", "test_cuda.TestCuda", ERROR),
            ),
        )
        output_file = self.temp / "github_output"

        summarize.write_count_outputs(self.reports.root, str(output_file))

        self.assertEqual(
            self.read_outputs(output_file),
            {
                "run_test_count": "3",
                "passed_test_count": "1",
                "failed_test_count": "2",
                "skipped_test_count": "1",
                "not_selected_test_count": "0",
                "results_complete": "true",
            },
        )

    def test_incomplete_report_set_still_writes_counts(self):
        self.complete_report_set()
        self.reports.record_run("transformers", completed=False)
        output_file = self.temp / "github_output"

        summarize.write_count_outputs(self.reports.root, str(output_file))

        outputs = self.read_outputs(output_file)
        self.assertEqual(outputs["run_test_count"], "1")
        self.assertEqual(outputs["results_complete"], "false")

    def test_no_reports_writes_only_incomplete(self):
        output_file = self.temp / "github_output"

        summarize.write_count_outputs(self.temp / "missing", str(output_file))

        self.assertEqual(self.read_outputs(output_file), {"results_complete": "false"})


class StandardSummaryTest(TestCaseBase):
    def add_labelled_set(self, name: str, metadata: dict) -> ReportSet:
        reports = ReportSet(self.temp / "artifacts" / name)
        reports.add_report(
            "test_nn",
            junit_xml(case("test_a"), case("test_b", body=FAILURE)),
        )
        reports.add_module_log("test_nn")
        reports.record_run("core")
        (reports.root / summarize.REPORT_METADATA).write_text(json.dumps(metadata))
        return reports

    def test_find_report_sets_by_metadata_and_name(self):
        self.add_labelled_set("anything", {"os": "Linux", "amdgpu_family": "gfx94X"})
        unlabelled = ReportSet(self.temp / "artifacts" / "test-reports-Windows-gfx120X")
        unlabelled.add_report("test_nn", junit_xml(case("test_a")))

        sets = summarize.find_report_sets(self.temp / "artifacts")

        self.assertEqual(
            [(os_name, family) for os_name, family, _ in sets],
            [("Linux", "gfx94X"), ("Windows", "gfx120X")],
        )

    def test_single_artifact_extracted_without_name_directory(self):
        # download-artifact drops the artifact-name directory when one
        # artifact matches, so metadata is the only label.
        self.complete_report_set()
        (self.reports.root / summarize.REPORT_METADATA).write_text(
            json.dumps({"os": "Linux", "amdgpu_family": "gfx94X"})
        )

        sets = summarize.find_report_sets(self.reports.root)

        self.assertEqual(sets, [("Linux", "gfx94X", self.reports.root)])

    def test_summary_table_and_incomplete_warning(self):
        self.add_labelled_set("linux", {"os": "Linux", "amdgpu_family": "gfx94X"})
        windows = self.add_labelled_set(
            "windows", {"os": "Windows", "amdgpu_family": "gfx120X"}
        )
        windows.record_run("transformers", completed=False)
        summary_file = self.temp / "summary.md"

        failed = summarize.write_standard_summary(
            self.temp / "artifacts", str(summary_file)
        )

        summary = summary_file.read_text()
        self.assertEqual(failed, 2)
        self.assertIn("| Linux | gfx94X | 2 | 1 | 1 | 0 | 0 | 0 |", summary)
        self.assertIn("| **Total** | | **4** | **2** | **2** |", summary)
        self.assertIn(
            "Incomplete results for Windows | gfx120X: the transformers test "
            "runner did not exit",
            summary,
        )
        self.assertNotIn("Incomplete results for Linux", summary)

    def test_pipe_in_failure_message_is_escaped(self):
        reports = self.add_labelled_set(
            "linux", {"os": "Linux", "amdgpu_family": "gfx94X"}
        )
        reports.add_report(
            "test_cuda",
            junit_xml(
                case("test_z", "test_cuda.TestCuda", '<failure message="a|b" />')
            ),
        )
        summary_file = self.temp / "summary.md"

        summarize.write_standard_summary(self.temp / "artifacts", str(summary_file))

        self.assertIn("| a\\|b |", summary_file.read_text())


class CombinedSummaryTest(TestCaseBase):
    def add_shard(self, shard: int, *, completed: bool = True) -> ReportSet:
        reports = ReportSet(self.temp / "artifacts" / f"test-reports-default-{shard}-2")
        reports.add_report(
            "test_nn",
            junit_xml(case(f"test_fail_{shard}", body=FAILURE), case("test_ok")),
        )
        reports.add_module_log("test_nn")
        reports.record_run("full", completed=completed)
        return reports

    def write(self, expected: int | None) -> dict:
        return summarize.write_combined_summary(
            self.temp / "artifacts",
            "gfx94X",
            str(self.temp / "summary.md"),
            expected_report_count=expected,
            github_output=str(self.temp / "github_output"),
        )

    def test_all_expected_shards_complete(self):
        self.add_shard(1)
        self.add_shard(2)

        outputs = self.write(expected=2)

        self.assertEqual(
            outputs,
            {
                "failed_test_count": 2,
                "observed_report_count": 2,
                "expected_report_count": 2,
                "results_complete": True,
                "test_counts": (
                    '{"default":{"run":4,"passed":2,"failed":2,"error":0,'
                    '"skipped":0,"not_selected":0,"flaky":0}}'
                ),
            },
        )
        self.assertIn(
            "results_complete=true",
            (self.temp / "github_output").read_text().splitlines(),
        )

    def test_counts_are_reported_per_config(self):
        self.add_shard(1)
        slow = ReportSet(self.temp / "artifacts" / "test-reports-slow-1-1")
        slow.add_report(
            "test_linalg",
            junit_xml(
                case("test_slow", "test_linalg.TestLinalg"),
                case("test_fast", "test_linalg.TestLinalg", SKIP_FAST),
            ),
        )
        slow.add_module_log("test_linalg")
        slow.record_run("full", "test_linalg 1/1 was successful\n")

        outputs = self.write(expected=2)

        counts = json.loads(outputs["test_counts"])
        self.assertEqual(counts["default"]["run"], 2)
        self.assertEqual((counts["slow"]["run"], counts["slow"]["skipped"]), (1, 0))
        self.assertEqual(counts["slow"]["not_selected"], 1)
        self.assertIn(
            "| slow | 1 | 1 | 0 | 0 | 0 | 1 | 0 |",
            (self.temp / "summary.md").read_text(),
        )

    def test_retry_pass_is_not_a_failure(self):
        reports = self.add_shard(1)
        reports.add_report("test_nn", junit_xml(case("test_fail_1")), attempt="retry")

        self.assertEqual(self.write(expected=1)["failed_test_count"], 0)

    def test_missing_shard_is_incomplete(self):
        self.add_shard(1)

        outputs = self.write(expected=2)

        self.assertEqual(outputs["observed_report_count"], 1)
        self.assertFalse(outputs["results_complete"])

    def test_incomplete_shard_is_incomplete(self):
        self.add_shard(1)
        self.add_shard(2, completed=False)

        outputs = self.write(expected=2)

        self.assertFalse(outputs["results_complete"])
        self.assertIn(
            "Incomplete shard default 2/2: the full test runner did not exit",
            (self.temp / "summary.md").read_text(),
        )

    def test_unknown_expected_count_is_incomplete(self):
        self.add_shard(1)

        self.assertFalse(self.write(expected=None)["results_complete"])

    def test_missing_artifacts_directory_writes_outputs(self):
        outputs = summarize.write_combined_summary(
            self.temp / "missing",
            "gfx94X",
            "",
            expected_report_count=2,
            github_output=str(self.temp / "github_output"),
        )

        self.assertEqual(outputs["observed_report_count"], 0)
        self.assertFalse(outputs["results_complete"])

    def test_count_expected_reports_from_matrix(self):
        matrix = {"include": [{"shard": 1}, {"shard": 2}, {"shard": 3}]}

        self.assertEqual(summarize.count_expected_reports(matrix), 3)
        self.assertIsNone(summarize.count_expected_reports(None))


if __name__ == "__main__":
    unittest.main()
