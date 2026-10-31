# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Add repo root to PYTHONPATH
sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
sys.path.insert(0, os.fspath(Path(__file__).parents[2]))

import patch_coverage_report
from patch_coverage_report import (
    CoverageIndex,
    DiffLine,
    FileResult,
    Hunk,
    PatchCoverage,
    Repository,
    annotations,
    changed_files,
    changed_lines,
    line_maps,
    load_reports,
    map_line,
    measure,
    parse_diff,
    parse_lcov,
    render_html,
    render_markdown,
    report_project,
)

WORKSPACE = "/home/runner/work/rocm-libraries/rocm-libraries/external-rocm-libraries"


def _lcov(records: dict[str, dict[int, int]]) -> str:
    lines = []
    for path, hits in records.items():
        lines += [f"SF:{path}"]
        lines += [f"DA:{number},{count}" for number, count in hits.items()]
        lines += ["end_of_record"]
    return "\n".join(lines) + "\n"


class ParseDiffTest(unittest.TestCase):
    def test_hunks_and_paths(self):
        diff = "\n".join(
            [
                "diff --git a/src/a.cpp b/src/a.cpp",
                "index 1111111..2222222 100644",
                "--- a/src/a.cpp",
                "+++ b/src/a.cpp",
                "@@ -5,2 +5,3 @@ void f()",
                "-old",
                "-old",
                "+new",
                "+new",
                "+new",
                "@@ -20 +21 @@",
                "--- this removed line starts with dashes",
                "+++ this added line starts with pluses",
                "@@ -30,0 +31,2 @@",
                "+x",
                "+y",
                "diff --git a/added.txt b/added.txt",
                "new file mode 100644",
                "--- /dev/null",
                "+++ b/added.txt",
                "@@ -0,0 +1 @@",
                "+hello",
            ]
        )
        files = parse_diff(diff)
        self.assertEqual(len(files), 2)
        self.assertEqual((files[0].old_path, files[0].new_path), ("src/a.cpp",) * 2)
        self.assertEqual(
            files[0].hunks,
            [Hunk(5, 2, 5, 3), Hunk(20, 1, 21, 1), Hunk(30, 0, 31, 2)],
        )
        self.assertEqual(files[0].added, [5, 6, 7, 21, 31, 32])
        self.assertEqual((files[1].old_path, files[1].new_path), (None, "added.txt"))
        self.assertEqual(files[1].added, [1])

    def test_added_lines_with_context(self):
        diff = "\n".join(
            [
                "diff --git a/a.cpp b/a.cpp",
                "--- a/a.cpp",
                "+++ b/a.cpp",
                "@@ -1,4 +1,5 @@ int f()",
                " a",
                "-b",
                "+B",
                "+C",
                " d",
                "@@ -10,3 +11,3 @@",
                " x",
                "-y",
                "+Y",
                " z",
                "\\ No newline at end of file",
            ]
        )
        file_diff = parse_diff(diff)[0]
        self.assertEqual(file_diff.added, [2, 3, 12])
        self.assertEqual(
            file_diff.lines,
            [
                DiffLine("@@", "@@ -1,4 +1,5 @@ int f()"),
                DiffLine(" ", "a", 1, 1),
                DiffLine("-", "b", old=2),
                DiffLine("+", "B", new=2),
                DiffLine("+", "C", new=3),
                DiffLine(" ", "d", 3, 4),
                DiffLine("@@", "@@ -10,3 +11,3 @@"),
                DiffLine(" ", "x", 10, 11),
                DiffLine("-", "y", old=11),
                DiffLine("+", "Y", new=12),
                DiffLine(" ", "z", 12, 13),
            ],
        )

    def test_rename_and_unusual_names(self):
        diff = "\n".join(
            [
                "diff --git a/old.cpp b/new.cpp",
                "similarity index 100%",
                "rename from old.cpp",
                "rename to new.cpp",
                'diff --git "a/caf\\303\\251.cpp" "b/caf\\303\\251.cpp"',
                '--- "a/caf\\303\\251.cpp"',
                '+++ "b/caf\\303\\251.cpp"',
                "@@ -1 +1 @@",
                "-a",
                "+b",
                "diff --git a/with space.cpp b/with space.cpp",
                "--- a/with space.cpp\t",
                "+++ b/with space.cpp\t",
                "@@ -1 +1 @@",
                "-a",
                "+b",
            ]
        )
        files = parse_diff(diff)
        self.assertEqual((files[0].old_path, files[0].new_path), ("old.cpp", "new.cpp"))
        self.assertEqual(files[0].hunks, [])
        self.assertEqual(files[1].new_path, "café.cpp")
        self.assertEqual(files[2].new_path, "with space.cpp")

    def test_rejects_unknown_hunk_header(self):
        with self.assertRaises(ValueError):
            parse_diff("diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ nonsense @@\n")


class MapLineTest(unittest.TestCase):
    def test_no_hunks_is_identity(self):
        self.assertEqual(map_line(7, []), 7)

    def test_modification(self):
        hunks = [Hunk(5, 2, 5, 3)]
        self.assertEqual(map_line(4, hunks), 4)
        self.assertIsNone(map_line(5, hunks))
        self.assertIsNone(map_line(6, hunks))
        self.assertEqual(map_line(7, hunks), 8)

    def test_insertion_goes_after_its_line(self):
        hunks = [Hunk(10, 0, 11, 3)]
        self.assertEqual(map_line(10, hunks), 10)
        self.assertEqual(map_line(11, hunks), 14)

    def test_insertion_at_start(self):
        self.assertEqual(map_line(1, [Hunk(0, 0, 1, 2)]), 3)

    def test_deletion(self):
        hunks = [Hunk(3, 2, 2, 0)]
        self.assertEqual(map_line(2, hunks), 2)
        self.assertIsNone(map_line(4, hunks))
        self.assertEqual(map_line(5, hunks), 3)

    def test_shifts_accumulate(self):
        hunks = [Hunk(0, 0, 1, 2), Hunk(10, 1, 12, 3), Hunk(20, 4, 24, 1)]
        self.assertEqual(map_line(9, hunks), 11)
        self.assertEqual(map_line(15, hunks), 19)
        self.assertEqual(map_line(30, hunks), 31)


class ReportTest(unittest.TestCase):
    def test_parse_lcov_keeps_highest_count(self):
        text = "TN:\nSF:/src/a.cpp\nDA:1,0\nDA:2,3\nDA:1,4\nDA:bad\nend_of_record\n"
        self.assertEqual(parse_lcov(text), {"/src/a.cpp": {1: 4, 2: 3}})

    def test_report_project(self):
        self.assertEqual(
            report_project("coverage-report-hiprand-gfx94X-dcgpu"), "hiprand"
        )
        self.assertEqual(
            report_project("coverage-report-rocprofiler-sdk-gfx950-dcgpu"),
            "rocprofiler-sdk",
        )
        self.assertEqual(report_project("coverage-report-rocrand"), "rocrand")

    def test_load_reports_merges_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            for family, hits in (
                ("gfx94X-dcgpu", {1: 0, 2: 1}),
                ("gfx950-dcgpu", {1: 2}),
            ):
                report = Path(tmp) / f"coverage-report-rocrand-{family}"
                report.mkdir()
                (report / "coverage.info").write_text(_lcov({"/src/a.cpp": hits}))
            (Path(tmp) / "coverage-profraw-rocrand-gfx94X-dcgpu-1").mkdir()
            self.assertEqual(
                load_reports(Path(tmp)), {"rocrand": {"/src/a.cpp": {1: 2, 2: 1}}}
            )

    def test_load_reports_from_one_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "coverage-report-hiprand-gfx94X-dcgpu"
            report.mkdir()
            (report / "coverage.info").write_text(_lcov({"/src/h.cpp": {3: 1}}))
            self.assertEqual(load_reports(report), {"hiprand": {"/src/h.cpp": {3: 1}}})

    def test_index_matches_whole_path_components(self):
        index = CoverageIndex(
            {
                "rocrand": {f"{WORKSPACE}/projects/rocrand/src/a.cpp": {1: 1}},
                "hiprand": {f"{WORKSPACE}/xprojects/rocrand/src/a.cpp": {1: 1}},
            }
        )
        self.assertEqual(
            index.lookup("projects/rocrand/src/a.cpp"), {"rocrand": {1: 1}}
        )
        self.assertEqual(index.lookup("rocrand/src/b.cpp"), {})


class MeasureTest(unittest.TestCase):
    def test_classifies_changed_lines(self):
        index = CoverageIndex(
            {
                "rocrand": {
                    f"{WORKSPACE}/src/a.cpp": {10: 3, 12: 0, 20: 0, 30: 0},
                    f"{WORKSPACE}/src/gone.cpp": {1: 1},
                },
                "hiprand": {f"{WORKSPACE}/src/a.cpp": {12: 1, 20: 0}},
            }
        )
        changes = {
            # Head lines 10-13 map one down: the built commit inserts a line
            # at the top, and changes head line 29.
            "src/a.cpp": [9, 11, 13, 19, 29],
            "src/gone.cpp": [1],
            "tests/a_test.cpp": [1, 2],
        }
        maps = {
            "src/a.cpp": [Hunk(0, 0, 1, 1), Hunk(29, 1, 30, 1)],
            "src/gone.cpp": None,
        }
        coverage = measure(changes, maps, index)

        a, gone = coverage.files
        self.assertEqual(a.path, "src/a.cpp")
        self.assertEqual(a.measured_by, ["hiprand", "rocrand"])
        # Built 10 ran in rocrand, built 12 in hiprand only, built 14 is not
        # code, built 20 ran in neither.
        self.assertEqual(a.covered, [9, 11])
        self.assertEqual(a.missed, [19])
        self.assertEqual(a.unmapped, [29])
        self.assertEqual(a.by_project, {"rocrand": (1, 2), "hiprand": (1, 1)})
        self.assertEqual(gone.unmapped, [1])
        self.assertEqual(coverage.unmeasured, ["tests/a_test.cpp"])
        self.assertEqual((coverage.covered, coverage.missed), (2, 1))
        self.assertEqual(coverage.by_project(), {"hiprand": (1, 1), "rocrand": (1, 2)})


class OutputTest(unittest.TestCase):
    DETAILS = dict(
        repository="ROCm/rocm-libraries",
        pr_number=42,
        merge_base="a" * 40,
        head="b" * 40,
        built="c" * 40,
    )

    def _coverage(self):
        return PatchCoverage(
            files=[
                FileResult(
                    path="src/a,b.cpp",
                    measured_by=["rocrand"],
                    covered=[1],
                    missed=[3, 4, 9],
                    by_project={"rocrand": (1, 3)},
                ),
                FileResult(path="src/comments.cpp", measured_by=["rocrand"]),
            ],
            unmeasured=["tests/a_test.cpp"],
        )

    def test_markdown(self):
        text = render_markdown(
            self._coverage(),
            projects=["rocrand"],
            sources={"src/a,b.cpp": [f"line {n}" for n in range(1, 10)]},
            **self.DETAILS,
        )
        self.assertIn("**1 of 4 changed lines covered (25.00%).**", text)
        self.assertIn(
            "[ROCm/rocm-libraries#42](https://github.com/ROCm/rocm-libraries/pull/42) "
            "changes 3 file(s)",
            text,
        )
        self.assertIn("| `src/a,b.cpp` | 1 | 3 | 25.00% | rocrand |", text)
        self.assertIn("| `src/comments.cpp` | 0 | 0 | - | rocrand |", text)
        self.assertIn("`src/a,b.cpp` lines 3-4, 9", text)
        self.assertIn("3 | line 3\n4 | line 4\n9 | line 9", text)
        self.assertIn("- `tests/a_test.cpp`", text)
        # One project, so no per-project table.
        self.assertNotIn("| Project |", text)

    def test_markdown_without_reports_or_code(self):
        empty = PatchCoverage(files=[], unmeasured=["README.md"])
        self.assertIn(
            "No coverage report was found",
            render_markdown(empty, projects=[], sources={}, **self.DETAILS),
        )
        self.assertIn(
            "None of the changed lines is code",
            render_markdown(empty, projects=["hiprand"], sources={}, **self.DETAILS),
        )

    def test_html(self):
        diffs = parse_diff(
            "\n".join(
                [
                    "diff --git a/src/a.cpp b/src/a.cpp",
                    "--- a/src/a.cpp",
                    "+++ b/src/a.cpp",
                    "@@ -1,3 +1,5 @@",
                    " int f() {",
                    "-  return 0;",
                    "+  if (x < 1)",
                    "+    return 1;",
                    "+  // done",
                    " }",
                    "diff --git a/tests/a_test.cpp b/tests/a_test.cpp",
                    "--- a/tests/a_test.cpp",
                    "+++ b/tests/a_test.cpp",
                    "@@ -1 +1,2 @@",
                    " TEST(A, B) {}",
                    "+TEST(A, C) {}",
                ]
            )
        )
        index = CoverageIndex(
            {"rocrand": {f"{WORKSPACE}/src/a.cpp": {1: 7, 2: 7, 3: 0, 5: 7}}}
        )
        coverage = measure(changed_lines(diffs), {}, index)
        details = dict(
            coverage=coverage,
            projects=["rocrand"],
            diffs=diffs,
            maps={},
            index=index,
            **self.DETAILS,
        )
        tests = {
            "rocrand": {
                "Suite.t/10": {"src/a.cpp": [2]},
                "Suite.t/2": {"src/a.cpp": [2]},
                "Suite.t/1": {"src/a.cpp": [2]},
                "Other.idle": {"src/a.cpp": []},
            }
        }
        text = render_html(tests=tests, **details)

        self.assertIn(
            '<p class="headline">1 of 2 changed lines covered (50.00%).</p>', text
        )
        self.assertIn("+  if (x &lt; 1)", text)
        for css in ("covered", "missed", "notcode", "removed"):
            self.assertIn(f'<tr class="{css}">', text)
        self.assertIn('<span title="rocrand: 7">7</span>', text)
        # Line 3 is code no test ran; line 4 is a comment, so has no count.
        self.assertIn('<span title="rocrand: 0">0</span>', text)
        self.assertIn(
            'href="https://github.com/ROCm/rocm-libraries/blob/'
            f'{"b" * 40}/src/a.cpp#L2"',
            text,
        )
        self.assertIn('<span title="Suite: 3">3</span>', text)
        self.assertIn("3 test(s) in 1 suite(s) ran at least one changed line.", text)
        self.assertNotIn("Other.idle", text)
        positions = [text.index(f"<code>Suite.t/{n}</code>") for n in (1, 2, 10)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("1 changed file(s) no report measures", text)
        self.assertIn("<code>tests/a_test.cpp</code>", text)

        text = render_html(**details)
        self.assertNotIn("Tests that ran the changed lines", text)
        self.assertNotIn("<td>Tests</td>", text)

    def test_annotations(self):
        commands = annotations(self._coverage(), limit=10)
        self.assertEqual(
            commands[0],
            "::warning file=src/a%2Cb.cpp,line=3,endLine=4,title=Not covered by "
            "tests::These changed lines ran in none of the tests of rocrand.",
        )
        self.assertTrue(commands[1].startswith("::warning file=src/a%2Cb.cpp,line=9,"))
        self.assertEqual(len(annotations(self._coverage(), limit=1)), 1)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", os.fspath(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.com",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.com",
        ),
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class GitTest(unittest.TestCase):
    """Runs against a local upstream holding a pull request and its merge."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        upstream = self.tmp / "upstream"
        upstream.mkdir()
        _git(upstream, "init", "--quiet")
        _git(upstream, "symbolic-ref", "HEAD", "refs/heads/main")
        _git(upstream, "config", "uploadpack.allowFilter", "true")
        _git(upstream, "config", "uploadpack.allowAnySHA1InWant", "true")
        original = [f"line{n}" for n in range(1, 11)]
        self.merge_base = _commit(
            upstream, {"src/a.cpp": "\n".join(original) + "\n"}, "base"
        )

        _git(upstream, "checkout", "--quiet", "-b", "pr")
        pr = list(original)
        pr[4] = "changed5"
        pr[8:8] = ["new8a", "new8b"]
        self.head = _commit(
            upstream,
            {
                "src/a.cpp": "\n".join(pr) + "\n",
                "src/c.cpp": "c1\nc2\nc3\n",
                "tests/t.cpp": "t1\n",
            },
            "pr",
        )

        _git(upstream, "checkout", "--quiet", "main")
        _commit(
            upstream,
            {"src/a.cpp": "\n".join(["top1", "top2", "top3", *original]) + "\n"},
            "base moves on",
        )
        _git(upstream, "merge", "--quiet", "--no-ff", "--no-edit", "-m", "merge", "pr")
        self.built = _git(upstream, "rev-parse", "HEAD")
        self.url = upstream.resolve().as_uri()

    def tearDown(self):
        self._tmp.cleanup()

    def test_changed_lines_and_mapping(self):
        repo = Repository(self.tmp / "scratch", self.url)
        repo.fetch([self.merge_base, self.head, self.built])
        changes = changed_lines(changed_files(repo, self.merge_base, self.head))
        self.assertEqual(
            changes,
            {"src/a.cpp": [5, 9, 10], "src/c.cpp": [1, 2, 3], "tests/t.cpp": [1]},
        )
        maps = line_maps(repo, self.head, self.built, list(changes))
        self.assertEqual(maps, {"src/a.cpp": [Hunk(0, 0, 1, 3)]})
        self.assertEqual(
            [map_line(n, maps["src/a.cpp"]) for n in (5, 9, 10)], [8, 12, 13]
        )

    def test_main(self):
        reports = self.tmp / "reports" / "coverage-report-rocrand-gfx94X-dcgpu"
        reports.mkdir(parents=True)
        (reports / "coverage.info").write_text(
            _lcov(
                {
                    f"{WORKSPACE}/src/a.cpp": {8: 5, 12: 0},
                    f"{WORKSPACE}/src/c.cpp": {1: 1, 2: 0},
                }
            )
        )
        summary = self.tmp / "summary.md"
        json_output = self.tmp / "out" / "patch_coverage.json"
        html_output = self.tmp / "out" / "patch_coverage.html"
        tests_json = self.tmp / "tests.json"
        tests_json.write_text(json.dumps({"rocrand": {"S.t": {"src/a.cpp": [5]}}}))
        real_repository = patch_coverage_report.Repository

        def local_repository(path, url, token=None):
            return real_repository(path, self.url, token)

        with mock.patch.object(
            patch_coverage_report, "Repository", local_repository
        ), mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": os.fspath(summary)}):
            exit_code = patch_coverage_report.main(
                [
                    "--repository=ROCm/rocm-libraries",
                    "--pr-number=42",
                    f"--base-sha={self.built}",
                    f"--head-sha={self.head}",
                    f"--built-sha={self.built}",
                    f"--merge-base-sha={self.merge_base}",
                    f"--reports-dir={self.tmp / 'reports'}",
                    f"--json-output={json_output}",
                    f"--html-output={html_output}",
                    f"--tests-json={tests_json}",
                ]
            )
        self.assertEqual(exit_code, 0)
        report = json.loads(json_output.read_text())
        self.assertEqual(
            report["totals"], {"covered": 2, "not_covered": 2, "percent": 50.0}
        )
        self.assertEqual(
            [
                (f["path"], f["covered_lines"], f["not_covered_lines"])
                for f in report["files"]
            ],
            [("src/a.cpp", [5], [9]), ("src/c.cpp", [1], [2])],
        )
        self.assertEqual(report["unmeasured_files"], ["tests/t.cpp"])
        text = summary.read_text()
        self.assertIn("**2 of 4 changed lines covered (50.00%).**", text)
        self.assertIn("\n9 | new8a\n", text)
        self.assertIn("\n2 | c2\n", text)
        page = html_output.read_text()
        self.assertIn("2 of 4 changed lines covered (50.00%).", page)
        # Head line 5 is built line 8, which ran 5 times.
        self.assertIn(
            '<span title="rocrand: 5">5</span></td><td class="n">'
            '<span title="S: 1">1</span>',
            page,
        )
        self.assertIn("+changed5", page)


if __name__ == "__main__":
    unittest.main()
