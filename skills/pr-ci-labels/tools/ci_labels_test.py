# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ci_labels.py (the pr-ci-labels skill's label-timeline report).

Run locally:
    python -m pytest skills/pr-ci-labels/tools/ci_labels_test.py
    # or
    python -m unittest skills/pr-ci-labels/tools/ci_labels_test.py -v
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

# Make `ci_labels` importable regardless of the working directory.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import ci_labels as cl  # noqa: E402

SKIP_PATTERNS = ["*.md", "docs/*"]
SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


def _label(event, label, when):
    return {"event": event, "label": label, "created_at": when}


def _ci_run(sha, when, conclusion="success", tested=True):
    """A CI run; `tested` is the job evidence (True built or tested, False not, None unknown)."""
    return {
        "id": 1,
        "name": "TheRock Multi-Arch CI",
        "head_sha": sha,
        "created_at": when,
        "status": "completed",
        "conclusion": conclusion,
        "url": "https://example.invalid/run",
        "tested": tested,
    }


def _commits(*shas):
    return [{"oid": s} for s in shas]


class CiLabelHelpersTest(unittest.TestCase):
    def test_labels_are_replayed_up_to_the_run(self):
        events = [
            _label("labeled", "ci:gfx950", "2026-11-01T10:00:00Z"),
            _label("labeled", "ci:skip", "2026-11-01T12:00:00Z"),
            _label("unlabeled", "ci:gfx950", "2026-11-01T13:00:00Z"),
        ]
        self.assertEqual(cl.labels_at(events, "2026-11-01T11:00:00Z"), ["ci:gfx950"])
        self.assertEqual(
            cl.labels_at(events, "2026-11-01T12:30:00Z"), ["ci:gfx950", "ci:skip"]
        )
        self.assertEqual(cl.labels_at(events, None), ["ci:skip"])

    def test_narrowing_labels(self):
        for label in (
            "ci:skip",
            "ci:no-downstream-test",
            "ci:cpu-test-only",
            "ci:exactly",
            "ci:platform:linux",
            "test:hipblaslt",
            "test_filter:quick",
        ):
            self.assertTrue(cl.is_narrowing_label(label), label)
        for label in ("ci:gfx950", "ci:run-all-archs", "test_filter:comprehensive"):
            self.assertFalse(cl.is_narrowing_label(label), label)

    def test_skip_patterns_come_from_the_common_list(self):
        text = '[skip_ci]\ncommon = ["*.md", "docs/*"]\nlinux = ["linux-only/*"]\n'
        self.assertEqual(cl.parse_skip_patterns(text), ["*.md", "docs/*"])

    def test_star_crosses_directories_like_ci_does(self):
        self.assertTrue(cl.is_ci_skippable("projects/foo/README.md", SKIP_PATTERNS))
        self.assertFalse(cl.is_ci_skippable("projects/foo/src/a.cpp", SKIP_PATTERNS))

    def test_heuristic_without_patterns_does_not_treat_cmakelists_as_docs(self):
        self.assertFalse(cl.is_ci_skippable("CMakeLists.txt", None))
        self.assertTrue(cl.is_ci_skippable("docs/guide.rst", None))

    def test_ci_workflow_filter_skips_asan_and_nightly(self):
        rx = cl.CI_WORKFLOWS
        self.assertTrue(cl.matches_any("Multi-Arch CI", rx))
        self.assertTrue(cl.matches_any("TheRock Multi-Arch CI", rx))
        self.assertFalse(cl.matches_any("TheRock Multi-Arch CI ASAN", rx))
        self.assertFalse(cl.matches_any("TheRock Multi-Arch Nightly CI", rx))


class CiLabelAnalysisTest(unittest.TestCase):
    """Pre-merge 'CI not bypassed since review' (reference.md)."""

    def test_head_commit_passed_is_go(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "go")

    def test_docs_commit_plus_ci_skip_after_a_green_run_is_go(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [_label("labeled", "ci:skip", "2026-11-01T12:00:00Z")],
            _commits(SHA_A, SHA_B),
            {SHA_B: ["docs/guide.md"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "go")
        self.assertEqual(report["current_labels"], ["ci:skip"])

    def test_code_commit_after_ci_skip_is_hold(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [_label("labeled", "ci:skip", "2026-11-01T12:00:00Z")],
            _commits(SHA_A, SHA_B),
            {SHA_B: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "hold")
        self.assertIn("ci:skip", report["reason"])

    def test_label_already_used_by_the_passing_run_is_not_new(self):
        # The run that passed already ran under ci:no-downstream-test; the review judged
        # that set, so a later code commit is uncovered but not a bypass.
        events = [_label("labeled", "ci:no-downstream-test", "2026-11-01T09:00:00Z")]
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            events,
            _commits(SHA_A, SHA_B),
            {SHA_B: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "caution")

    def test_label_added_after_the_last_run_is_reported(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [_label("labeled", "ci:gfx950", "2026-11-01T12:00:00Z")],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "caution")
        self.assertIn("ci:gfx950", report["reason"])
        self.assertEqual(
            [e["label"] for e in report["label_changes_since"]], ["ci:gfx950"]
        )

    def test_non_ci_label_added_after_the_run_is_ignored(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [_label("labeled", "project: hipblaslt", "2026-11-01T12:00:00Z")],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "go")
        self.assertEqual(report["current_labels"], [])
        self.assertEqual(report["label_changes_since"], [])

    def test_unknown_commit_files_are_treated_as_code(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [_label("labeled", "ci:skip", "2026-11-01T12:00:00Z")],
            _commits(SHA_A, SHA_B),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "hold")

    def test_newer_failed_run_does_not_hide_the_last_pass(self):
        report = cl.analyze_ci_labels(
            [
                _ci_run(SHA_A, "2026-11-01T10:00:00Z"),
                _ci_run(SHA_B, "2026-11-01T12:00:00Z", conclusion="failure"),
            ],
            [],
            _commits(SHA_A, SHA_B),
            {SHA_B: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["last_passing_run"]["head_sha"], SHA_A)
        self.assertEqual(report["verdict"], "caution")

    def test_no_passing_run_is_caution(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z", conclusion="failure")],
            [],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "caution")
        self.assertIsNone(report["last_passing_run"])

    def test_rebased_history_is_caution_not_go(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_C, "2026-11-01T10:00:00Z")],
            [],
            _commits(SHA_A, SHA_B),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "caution")
        self.assertIn("not in the PR's current history", report["reason"])

    def test_skip_only_pass_after_ci_skip_is_not_a_baseline(self):
        # Pass on A, add ci:skip, push code commit B: B's run "passes" without building or
        # testing anything, so A stays the baseline and B's code is uncovered.
        runs = [
            {**_ci_run(SHA_A, "2026-11-01T10:00:00Z"), "tested": True},
            _ci_run(SHA_B, "2026-11-01T12:00:00Z", tested=False),
        ]
        report = cl.analyze_ci_labels(
            runs,
            [_label("labeled", "ci:skip", "2026-11-01T11:00:00Z")],
            _commits(SHA_A, SHA_B),
            {SHA_B: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["last_passing_run"]["head_sha"], SHA_A)
        self.assertEqual(report["verdict"], "hold")

    def test_pass_that_built_and_tested_nothing_is_not_a_baseline(self):
        runs = [
            {**_ci_run(SHA_A, "2026-11-01T10:00:00Z"), "tested": True},
            {**_ci_run(SHA_B, "2026-11-01T12:00:00Z"), "tested": False},
        ]
        report = cl.analyze_ci_labels(
            runs, [], _commits(SHA_A, SHA_B), {SHA_B: ["src/scan.cpp"]}, SKIP_PATTERNS
        )
        self.assertEqual(report["last_passing_run"]["head_sha"], SHA_A)
        self.assertEqual(report["verdict"], "caution")

    def test_docs_only_pr_with_no_tested_run_is_go(self):
        runs = [{**_ci_run(SHA_A, "2026-11-01T10:00:00Z"), "tested": False}]
        report = cl.analyze_ci_labels(
            runs, [], _commits(SHA_A), {SHA_A: ["docs/guide.md"]}, SKIP_PATTERNS
        )
        self.assertIsNone(report["last_passing_run"])
        self.assertEqual(report["verdict"], "go")

    def test_code_with_no_tested_run_and_ci_skip_is_hold(self):
        runs = [_ci_run(SHA_A, "2026-11-01T10:00:00Z", tested=False)]
        report = cl.analyze_ci_labels(
            runs,
            [_label("labeled", "ci:skip", "2026-11-01T09:00:00Z")],
            _commits(SHA_A),
            {SHA_A: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "hold")

    def test_pass_with_unreadable_jobs_is_not_coverage(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z", tested=None)],
            [],
            _commits(SHA_A),
            {SHA_A: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertIsNone(report["last_passing_run"])
        self.assertEqual(report["verdict"], "caution")
        self.assertIn("could not be checked", report["reason"])

    def test_unreadable_newer_pass_falls_back_to_older_verified_pass(self):
        report = cl.analyze_ci_labels(
            [
                _ci_run(SHA_A, "2026-11-01T10:00:00Z"),
                _ci_run(SHA_B, "2026-11-01T12:00:00Z", tested=None),
            ],
            [],
            _commits(SHA_A, SHA_B),
            {SHA_B: ["src/scan.cpp"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["last_passing_run"]["head_sha"], SHA_A)
        self.assertEqual(report["verdict"], "caution")
        self.assertIn("could not be checked", report["reason"])

    def test_docs_commit_does_not_clear_an_unused_coverage_label(self):
        # Pass on A, add ci:gfx950, push docs-only B: gfx950 has still never run.
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            [_label("labeled", "ci:gfx950", "2026-11-01T11:00:00Z")],
            _commits(SHA_A, SHA_B),
            {SHA_B: ["docs/guide.md"]},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "caution")
        self.assertIn("ci:gfx950", report["reason"])

    def test_unknown_label_history_keeps_current_labels_and_holds(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            None,
            _commits(SHA_A, SHA_B),
            {SHA_B: ["src/scan.cpp"]},
            SKIP_PATTERNS,
            current_labels=["ci:skip", "project: hipblaslt"],
        )
        self.assertFalse(report["label_history_known"])
        self.assertEqual(report["current_labels"], ["ci:skip"])
        self.assertEqual(report["verdict"], "hold")
        self.assertIn("label history could not be read", report["reason"])

    def test_unknown_label_history_never_gives_go_from_a_run(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z")],
            None,
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
            current_labels=[],
        )
        self.assertEqual(report["verdict"], "caution")
        self.assertTrue(report["runs"][0]["labels_unknown"])

    def test_unknown_label_history_still_allows_a_fully_skipped_pr(self):
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z", tested=False)],
            None,
            _commits(SHA_A),
            {SHA_A: ["docs/guide.md"]},
            SKIP_PATTERNS,
            current_labels=[],
        )
        self.assertEqual(report["verdict"], "go")

    def test_coverage_label_near_run_creation_downgrades_go(self):
        report = cl.analyze_ci_labels(
            [{**_ci_run(SHA_A, "2026-11-01T10:00:30Z"), "tested": True}],
            [_label("labeled", "ci:gfx950", "2026-11-01T10:00:00Z")],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["runs"][0]["uncertain_labels"], ["ci:gfx950"])
        self.assertEqual(report["verdict"], "caution")
        self.assertIn("may not have used", report["reason"])

    def test_narrowing_label_near_run_creation_keeps_go(self):
        # If the run missed a narrowing label it tested more, not less.
        report = cl.analyze_ci_labels(
            [{**_ci_run(SHA_A, "2026-11-01T10:00:30Z"), "tested": True}],
            [_label("labeled", "ci:no-downstream-test", "2026-11-01T10:00:00Z")],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["verdict"], "go")

    def test_rerun_uses_labels_present_when_it_started(self):
        # Created at 10:00, re-run at 14:00 after ci:gfx950 was added at 12:00. Setup reads
        # labels when the attempt starts, so the re-run used ci:gfx950.
        run = {
            **_ci_run(SHA_A, "2026-11-01T10:00:00Z"),
            "labels_read_at": "2026-11-01T14:00:00Z",
        }
        report = cl.analyze_ci_labels(
            [run],
            [_label("labeled", "ci:gfx950", "2026-11-01T12:00:00Z")],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["runs"][0]["labels"], ["ci:gfx950"])
        self.assertEqual(report["label_changes_since"], [])
        self.assertEqual(report["verdict"], "go")

    def test_job_evidence_counts_even_if_replayed_labels_say_ci_skip(self):
        # Replayed labels are an estimate; a run with a successful build job built.
        report = cl.analyze_ci_labels(
            [_ci_run(SHA_A, "2026-11-01T10:00:00Z", tested=True)],
            [_label("labeled", "ci:skip", "2026-11-01T09:00:00Z")],
            _commits(SHA_A),
            {},
            SKIP_PATTERNS,
        )
        self.assertEqual(report["last_passing_run"]["head_sha"], SHA_A)


class GhApiPagesTest(unittest.TestCase):
    def test_failed_page_is_none_not_empty(self):
        with mock.patch.object(cl, "gh_api_json", return_value=None):
            self.assertIsNone(cl.gh_api_pages("repos/o/r/issues/5/events"))

    def test_empty_list_is_a_real_empty_result(self):
        with mock.patch.object(cl, "gh_api_json", return_value=[]):
            self.assertEqual(cl.gh_api_pages("repos/o/r/issues/5/events"), [])


class CiLabelFetchTest(unittest.TestCase):
    """The gh wiring, with gh replaced by canned API responses."""

    def test_unreadable_label_history_uses_the_prs_current_labels(self):
        sha_run = (
            f"repos/o/r/actions/runs?head_sha={SHA_A}&event=pull_request&per_page=100"
        )
        api = {
            sha_run: {
                "workflow_runs": [
                    {
                        "id": 7,
                        "name": "TheRock Multi-Arch CI",
                        "head_sha": SHA_A,
                        "created_at": "2026-11-01T10:00:00Z",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://example.invalid/7",
                    }
                ]
            },
            "repos/o/r/actions/runs/7/jobs?filter=latest&per_page=30&page=1": _jobs(
                ("Linux::release / Build Multi-Arch Stages / math-libs", "success")
            ),
        }
        data = {
            "number": 5,
            "url": "https://github.com/o/r/pull/5",
            "baseRefName": "develop",
            "commits": _commits(SHA_A),
            "labels": [{"name": "ci:skip"}],
        }
        with mock.patch.object(
            cl, "gh_api_json", side_effect=api.get
        ), mock.patch.object(cl, "gh_api_pages", return_value=None), mock.patch.object(
            cl, "load_skip_patterns", return_value=(SKIP_PATTERNS, "skip-ci patterns")
        ):
            report = cl.fetch_ci_label_report(data, None, None)
        self.assertEqual(report["current_labels"], ["ci:skip"])
        self.assertFalse(report["label_history_known"])
        self.assertEqual(report["verdict"], "caution")

    def test_report_from_mocked_gh(self):
        api = {
            f"repos/o/r/actions/runs?head_sha={SHA_A}&event=pull_request&per_page=100": {
                "workflow_runs": [
                    {
                        "id": 7,
                        "name": "TheRock Multi-Arch CI",
                        "head_sha": SHA_A,
                        # Created under ci:skip, re-run after the label was removed. Setup
                        # reads labels when the attempt starts, so the re-run built.
                        "created_at": "2026-11-01T10:00:00Z",
                        "run_started_at": "2026-11-01T14:00:00Z",
                        "status": "completed",
                        "conclusion": "success",
                        "html_url": "https://example.invalid/7",
                    },
                    {
                        "id": 8,
                        "name": "TheRock Multi-Arch CI ASAN",
                        "head_sha": SHA_A,
                        "created_at": "2026-11-01T10:00:00Z",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://example.invalid/8",
                    },
                ]
            },
            f"repos/o/r/actions/runs?head_sha={SHA_B}&event=pull_request&per_page=100": {
                "workflow_runs": []
            },
            "repos/o/r/actions/runs/7/jobs?filter=latest&per_page=30&page=1": {
                "jobs": [
                    {"name": "setup / setup", "conclusion": "success"},
                    {
                        "name": "Linux::release / Build Multi-Arch Stages / math-libs",
                        "conclusion": "success",
                    },
                ]
            },
            f"repos/o/r/commits/{SHA_B}?per_page=100&page=1": {
                "files": [{"filename": "docs/guide.md"}]
            },
        }
        events = [
            {
                "event": "labeled",
                "label": {"name": "ci:skip"},
                "created_at": "2026-11-01T09:00:00Z",
            },
            {
                "event": "unlabeled",
                "label": {"name": "ci:skip"},
                "created_at": "2026-11-01T12:00:00Z",
            },
            {"event": "assigned", "created_at": "2026-11-01T12:05:00Z"},
        ]
        data = {
            "number": 5,
            "url": "https://github.com/o/r/pull/5",
            "baseRefName": "develop",
            "commits": _commits(SHA_A, SHA_B),
        }
        with mock.patch.object(
            cl, "gh_api_json", side_effect=api.get
        ), mock.patch.object(
            cl, "gh_api_pages", return_value=events
        ) as pages, mock.patch.object(
            cl, "load_skip_patterns", return_value=(SKIP_PATTERNS, "skip-ci patterns")
        ), mock.patch.object(
            cl, "resolve_repo", return_value="wrong/checkout"
        ):
            # No --repo: the repository must come from the PR URL, not the checkout.
            report = cl.fetch_ci_label_report(data, None, None)
        pages.assert_called_once_with("repos/o/r/issues/5/events")
        self.assertEqual([r["id"] for r in report["runs"]], [7])
        self.assertEqual(report["runs"][0]["labels"], [])
        self.assertTrue(report["runs"][0]["tested"])
        self.assertEqual(report["verdict"], "go")
        self.assertEqual(report["skip_method"], "skip-ci patterns")

    def test_repo_from_pr_url(self):
        self.assertEqual(
            cl.repo_from_pr_url("https://github.com/ROCm/rocm-libraries/pull/15973"),
            "ROCm/rocm-libraries",
        )
        self.assertIsNone(cl.repo_from_pr_url("15973"))


def _jobs(*pairs):
    return {"jobs": [{"name": n, "conclusion": c} for n, c in pairs]}


class RunBuiltOrTestedTest(unittest.TestCase):
    def _call(self, pages):
        responses = {
            f"repos/o/r/actions/runs/1/jobs?filter=latest&per_page=30&page={i}": p
            for i, p in enumerate(pages, start=1)
        }
        with mock.patch.object(cl, "gh_api_json", side_effect=responses.get):
            return cl.run_built_or_tested("o/r", 1)

    def test_skip_only_run_built_nothing(self):
        page = _jobs(
            ("setup / setup", "success"),
            ("Linux::skip", "skipped"),
            ("Windows::skip", "skipped"),
            ("CI Summary", "success"),
        )
        self.assertIs(self._call([page]), False)

    def test_build_job_on_a_later_page_counts(self):
        filler = _jobs(*[("setup / setup", "success")] * 30)
        page2 = _jobs(("Linux::release / Test gfx94X-dcgpu / hipblaslt", "success"))
        self.assertIs(self._call([filler, page2]), True)

    def test_skipped_build_job_does_not_count(self):
        page = _jobs(
            ("Linux::release / Build Multi-Arch Stages / math-libs", "skipped")
        )
        self.assertIs(self._call([page]), False)

    def test_unreadable_jobs_are_unknown(self):
        self.assertIsNone(self._call([None]))


class FetchCommitFilesTest(unittest.TestCase):
    def _call(self, pages):
        responses = {
            f"repos/o/r/commits/{SHA_B}?per_page=100&page={i}": p
            for i, p in enumerate(pages, start=1)
        }
        with mock.patch.object(cl, "gh_api_json", side_effect=responses.get):
            return cl.fetch_commit_files("o/r", SHA_B)

    def test_reads_every_page(self):
        docs = {"files": [{"filename": f"docs/{i}.md"} for i in range(100)]}
        code = {"files": [{"filename": "src/scan.cpp"}]}
        files = self._call([docs, code])
        self.assertEqual(len(files), 101)
        self.assertIn("src/scan.cpp", files)

    def test_failed_page_is_unknown(self):
        docs = {"files": [{"filename": f"docs/{i}.md"} for i in range(100)]}
        self.assertIsNone(self._call([docs, None]))

    def test_api_file_limit_is_unknown(self):
        full = {"files": [{"filename": f"docs/{i}.md"} for i in range(100)]}
        with mock.patch.object(cl, "COMMIT_FILES_LIMIT", 200):
            self.assertIsNone(self._call([full, full, full]))


if __name__ == "__main__":
    unittest.main()
