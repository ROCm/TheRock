# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import summarize_component_test_repetition as summary


class ComponentRepetitionSummaryTest(unittest.TestCase):
    def setUp(self):
        self.sample = "repeat-linux-gfx120x-1-rocblas"
        self.execution = dict(
            sample_id=self.sample,
            platform="linux",
            family="gfx120X-all",
            component="rocblas",
            iteration=1,
            shard=1,
            total_shards=2,
            runner_label="triage-runner",
            expect_failure=False,
        )
        self.manifest = dict(
            expected=[self.execution], test_type="comprehensive", repetitions=1
        )

    def job(self, test="success", conclusion="success", total=2, **overrides):
        return (
            dict(
                id=123,
                html_url="https://example/job/123",
                name=f"{self.sample} / Test rocblas (shard 1/{total}) (gfx120X-all)",
                conclusion=conclusion,
                runner_name="physical-triage-host",
                labels=["self-hosted", "triage-runner"],
                steps=[] if test is None else [dict(name="Test", conclusion=test)],
            )
            | overrides
        )

    def report(self, jobs):
        return summary.summarize_jobs(self.manifest, jobs)

    def test_final_outcomes_include_advisory_failures_and_nonexecution(self):
        self.execution["expect_failure"] = True
        for test, conclusion, outcome in (
            ("success", "success", "passed"),
            ("failure", "success", "test_failure"),
            (None, "failure", "infrastructure_failure"),
            ("skipped", "failure", "infrastructure_failure"),
            ("skipped", "success", "test_not_run"),
            (None, "cancelled", "test_not_run"),
            ("cancelled", "cancelled", "test_incomplete"),
            ("success", "failure", "post_test_failure"),
            ("timed_out", "failure", "test_failure"),
        ):
            with self.subTest(test=test, conclusion=conclusion):
                report = self.report([self.job(test, conclusion)])
                self.assertEqual(report["counts"], {outcome: 1})
                self.assertTrue(report["results"][0]["expect_failure"])
        report = self.report([self.job()])
        self.assertFalse(report["incomplete"])
        self.assertEqual(
            report["results"][0]["jobs"][0]["runner_name"], "physical-triage-host"
        )

    def test_missing_duplicate_and_mismatched_jobs_make_report_incomplete(self):
        for jobs, outcome in (
            ([], "missing"),
            ([self.job(), self.job(id=456)], "duplicate"),
            ([self.job(total=3)], "shard_mismatch"),
            ([self.job(labels=["production-runner"])], "runner_mismatch"),
        ):
            with self.subTest(outcome=outcome):
                report = self.report(jobs)
                self.assertEqual(report["counts"], {outcome: 1})
                self.assertTrue(report["incomplete"])
        for name in (
            self.job()["name"].replace("gfx120x-1", "gfx120x-2"),
            f"{self.sample} / unexpected child",
        ):
            with self.subTest(name=name):
                report = self.report([self.job(), self.job(id=456, name=name)])
                self.assertEqual(len(report["unexpected_jobs"]), 1)
                self.assertTrue(report["incomplete"])

    def test_disabled_target_placeholders_do_not_make_report_incomplete(self):
        placeholders = [
            self.job(
                test=None,
                conclusion="skipped",
                id=456 + index,
                name=f"repeat-{target}-${{{{ matrix.iteration }}}}"
                "-${{ matrix.component.job_name }}",
            )
            for index, target in enumerate(
                (
                    "linux-gfx110x",
                    "linux-gfx1151",
                    "windows-gfx110x",
                    "windows-gfx120x",
                    "windows-gfx1151",
                )
            )
        ]
        report = self.report([self.job(), *placeholders])
        self.assertEqual(report["counts"], {"passed": 1})
        self.assertEqual(report["unexpected_jobs"], [])
        self.assertFalse(report["incomplete"])

        report = self.report(placeholders)
        self.assertEqual(report["counts"], {"missing": 1})
        self.assertTrue(report["incomplete"])

    def test_placeholder_filter_preserves_unexpected_executions(self):
        placeholder = self.job(
            test=None,
            conclusion="skipped",
            id=456,
            name="repeat-windows-gfx120x-${{ matrix.iteration }}"
            "-${{ matrix.component.job_name }}",
        )
        for overrides in (
            {"conclusion": "cancelled"},
            {"conclusion": "failure"},
            {"steps": [dict(name="Setup", conclusion="success")]},
            {"name": placeholder["name"].replace("windows", "linux")},
            {"name": "repeat-windows-gfx120x-1-rocblas"},
            {"name": placeholder["name"].replace("gfx120x", "gfx950")},
            {"name": placeholder["name"] + " / unexpected child"},
        ):
            with self.subTest(overrides=overrides):
                unexpected = placeholder | overrides
                report = self.report([self.job(), unexpected])
                self.assertEqual(report["unexpected_jobs"], [unexpected])
                self.assertTrue(report["incomplete"])

        report = self.report([self.job(test="skipped", conclusion="skipped")])
        self.assertEqual(report["counts"], {"test_not_run": 1})
        self.assertTrue(report["incomplete"])

    def test_iterations_shards_and_report_context(self):
        self.manifest["expected"] += [
            dict(self.execution, shard=2),
            dict(
                self.execution, iteration=2, sample_id=self.sample.replace("-1-", "-2-")
            ),
        ]
        self.manifest["test_type"] = "standard"
        report = self.report([self.job()])
        self.assertEqual(report["counts"], {"passed": 1, "missing": 2})
        report["run_attempt"] = 1
        markdown = summary.render_report(report)
        for text in (
            "Tier: standard.",
            "internal retries",
            "individual tests and initial retry attempts are not measured",
        ):
            self.assertIn(text, markdown)

    def test_nested_target_workflow_names_match_expected_samples(self):
        job = self.job(name=f"linux gfx120X-all / {self.job()['name']}")
        report = self.report([job])
        self.assertEqual(report["counts"], {"passed": 1})
        self.assertEqual(report["unexpected_jobs"], [])
        self.assertFalse(report["incomplete"])

        unexpected = self.job(
            id=456,
            name=job["name"].replace("gfx120x-1", "gfx120x-2"),
        )
        report = self.report([job, unexpected])
        self.assertEqual(report["unexpected_jobs"], [unexpected])
        self.assertTrue(report["incomplete"])

    def test_report_explains_automatically_omitted_platforms(self):
        self.manifest["omitted_platforms"] = ["windows"]
        report = self.report([self.job()])
        report["run_attempt"] = 1
        self.assertEqual(report["omitted_platforms"], ["windows"])
        self.assertFalse(report["incomplete"])
        self.assertIn(
            "Platforms omitted because the selected tests have no configured tests there: windows.",
            summary.render_report(report),
        )

    def test_pagination_reads_all_pages_of_current_attempt(self):
        pages = [{"jobs": [{"id": p * 100 + i} for i in range(100)]} for p in range(11)]
        pages.append({"jobs": [{"id": 1100}]})
        with patch.object(summary, "gha_send_request", side_effect=pages) as api:
            jobs = summary.fetch_attempt_jobs("ROCm/TheRock", "123", 2)
        self.assertEqual((len(jobs), api.call_count), (1101, 12))
        self.assertIn("/attempts/2/jobs?per_page=100&page=12", api.call_args.args[0])

    def test_cli_errors_write_incomplete_reports_and_step_summary(self):
        for missing in (False, True):
            with self.subTest(
                missing=missing
            ), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                output = Path(directory)
                manifest = output / "manifest.json"
                if not missing:
                    manifest.write_text(json.dumps(self.manifest))
                error = None if missing else RuntimeError("API unavailable")
                stack.enter_context(
                    patch.object(
                        summary,
                        "fetch_attempt_jobs",
                        return_value=[],
                        side_effect=error,
                    )
                )
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        GITHUB_STEP_SUMMARY=str(output / "step.md"),
                        CONFIGURATION_RESULT="failure" if missing else "success",
                    )
                )
                argv = [
                    "summary",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    directory,
                    "--run-id",
                    "123",
                ]
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch("builtins.print"))
                self.assertEqual(summary.main(), 1)
                report = json.loads((output / "report.json").read_text())
                self.assertTrue(report["incomplete"])
                self.assertTrue(report["errors"])
                self.assertTrue((output / "step.md").is_file())
                if error:
                    self.assertEqual(report["counts"], {"missing": 1})
                    self.assertIn("API unavailable", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
