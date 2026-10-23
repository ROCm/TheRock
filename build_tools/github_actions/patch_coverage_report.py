# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reports how much of a pull request's change its tests run: patch coverage.

multi_arch_ci_coverage_pr.yml runs this once every project's coverage report is
in. It joins two inputs:

- The lines the pull request adds, as its "Files changed" tab shows them: the
  diff from the merge base to the head. The coverage build compiled the merge
  commit, where the base branch's own changes since may have moved them, so
  each line is mapped onto that commit. A clean merge keeps every line the pull
  request changed; any the built commit has differently are listed apart.
- Each project's lcov report (`coverage.info` in the `coverage-report-*`
  artifacts), whose line hit counts say which lines are code and which of
  those ran.

A changed line counts when some report has a hit count for it, and is covered
when any project's tests ran it. Changed lines no report counts are not code as
far as coverage goes: comments, declarations, or code no measured binary
contains. A changed file no report mentions is not measured, which includes the
tests themselves.

The commits come from a blobless depth-1 fetch, so even for rocm-libraries the
cost is three trees and the blobs of the changed files.

The results go to the job summary, a JSON report, an HTML page showing each
measured file's diff with the hit count of every line, and warning annotations
on uncovered lines, numbered as in the pull request head so they land on its diff.
Low coverage is reported, not failed: the job fails only when it cannot compute
the numbers.
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from html import escape
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Sequence
from urllib.parse import quote

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))

from github_actions_api import (
    gha_append_step_summary,
    gha_resolve_git_ref,
    gha_send_request,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

REPORT_ARTIFACT_PREFIX = "coverage-report-"
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# Paths per `git diff` invocation, comfortably inside command line limits.
DIFF_PATHS_PER_CALL = 200
# Bounds on the job summary; the JSON report has everything.
MAX_SOURCE_LINES_PER_FILE = 40
MAX_SOURCE_LINES = 200
MAX_RANGES_LISTED = 20
MAX_UNMEASURED_FILES_LISTED = 50
DEFAULT_MAX_ANNOTATIONS = 10

# Hit count per line number of one source file.
LineHits = dict[int, int]


@dataclass(frozen=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int


@dataclass(frozen=True)
class DiffLine:
    """A line of a hunk: its header ("@@"), context (" "), added or removed."""

    kind: str
    text: str
    old: Optional[int] = None
    new: Optional[int] = None


@dataclass
class FileDiff:
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    hunks: list[Hunk] = field(default_factory=list)
    # New-side numbers of the lines the hunks add.
    added: list[int] = field(default_factory=list)
    lines: list[DiffLine] = field(default_factory=list)


@dataclass
class FileResult:
    """The changed lines of one file some report measures, in head numbering."""

    path: str
    measured_by: list[str]
    covered: list[int] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)
    unmapped: list[int] = field(default_factory=list)
    # Per project: changed lines its tests ran, and ones they did not.
    by_project: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass
class PatchCoverage:
    files: list[FileResult]
    unmeasured: list[str]

    @property
    def covered(self) -> int:
        return sum(len(f.covered) for f in self.files)

    @property
    def missed(self) -> int:
        return sum(len(f.missed) for f in self.files)

    def by_project(self) -> dict[str, tuple[int, int]]:
        totals: dict[str, tuple[int, int]] = {}
        for result in self.files:
            for project, (covered, missed) in result.by_project.items():
                total_covered, total_missed = totals.get(project, (0, 0))
                totals[project] = (total_covered + covered, total_missed + missed)
        return dict(sorted(totals.items()))


def _unquote_path(raw: str) -> str:
    """Undoes git's quoting of unusual paths, e.g. "caf\\303\\251.cpp".

    With core.quotepath on, git escapes every byte outside printable ASCII, so
    a quoted path is ASCII holding C escapes of its UTF-8 bytes.
    """
    if len(raw) < 2 or not (raw.startswith('"') and raw.endswith('"')):
        return raw
    escaped = raw[1:-1].encode("ascii", "backslashreplace").decode("unicode_escape")
    return escaped.encode("latin-1").decode("utf-8", "replace")


def _diff_path(raw: str, prefix: str) -> Optional[str]:
    # git ends the name with a tab when it holds a space.
    path = _unquote_path(raw.rstrip("\t"))
    if path == "/dev/null":
        return None
    return path.removeprefix(prefix)


def parse_diff(text: str) -> list[FileDiff]:
    """The files, hunks, and added lines of a `git diff` listing."""
    files: list[FileDiff] = []
    current: Optional[FileDiff] = None
    old_line = new_line = 0
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = FileDiff()
            files.append(current)
        elif current is None:
            continue
        elif line.startswith("@@"):
            match = HUNK_HEADER.match(line)
            if not match:
                raise ValueError(f"Unrecognized hunk header: {line}")
            old_start, old_count, new_start, new_count = match.groups()
            current.hunks.append(
                Hunk(
                    old_start=int(old_start),
                    old_count=1 if old_count is None else int(old_count),
                    new_start=int(new_start),
                    new_count=1 if new_count is None else int(new_count),
                )
            )
            current.lines.append(DiffLine("@@", line))
            old_line, new_line = int(old_start), int(new_start)
        elif current.hunks:
            # Hunk content, which can itself start with "---" or "+++". Removed
            # lines and "\ No newline at end of file" leave the new side alone.
            if line.startswith("+"):
                current.added.append(new_line)
                current.lines.append(DiffLine("+", line[1:], new=new_line))
                new_line += 1
            elif line.startswith(" "):
                current.lines.append(DiffLine(" ", line[1:], old_line, new_line))
                old_line += 1
                new_line += 1
            elif line.startswith("-"):
                current.lines.append(DiffLine("-", line[1:], old=old_line))
                old_line += 1
        elif line.startswith("--- "):
            current.old_path = _diff_path(line[4:], "a/")
        elif line.startswith("+++ "):
            current.new_path = _diff_path(line[4:], "b/")
        elif line.startswith("rename from "):
            current.old_path = _unquote_path(line.removeprefix("rename from "))
        elif line.startswith("rename to "):
            current.new_path = _unquote_path(line.removeprefix("rename to "))
    return files


def map_line(line: int, hunks: Sequence[Hunk]) -> Optional[int]:
    """Where an old-side line lands on the new side; None if the hunks change it.

    The hunks must be in order, as git lists them. One that only inserts
    (old_count 0) does so after line old_start.
    """
    shift = 0
    for hunk in hunks:
        if hunk.old_count:
            if line < hunk.old_start:
                break
            if line < hunk.old_start + hunk.old_count:
                return None
        elif line <= hunk.old_start:
            break
        shift += hunk.new_count - hunk.old_count
    return line + shift


class Repository:
    """A scratch repository holding only the commits being compared."""

    def __init__(self, path: Path, url: str, token: Optional[str] = None):
        self.path = path
        self.url = url
        self._env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        if token:
            # Through the environment, so the token is on no command line.
            # The on-demand blob fetches of later commands need it too.
            credentials = base64.b64encode(f"x-access-token:{token}".encode())
            self._env.update(
                GIT_CONFIG_COUNT="1",
                GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {credentials.decode()}",
            )

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", os.fspath(self.path), *args],
            capture_output=True,
            text=True,
            env=self._env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {args[0]} failed ({result.returncode}): {result.stderr.strip()}"
            )
        return result.stdout

    def fetch(self, shas: Sequence[str]) -> None:
        if not (self.path / ".git").exists():
            self.path.mkdir(parents=True, exist_ok=True)
            self.git("init", "--quiet")
            self.git("remote", "add", "origin", self.url)
            # Makes origin the source of the blobs the fetch leaves out, which
            # diff and show then fetch as they need them.
            self.git("config", "remote.origin.promisor", "true")
            self.git("config", "remote.origin.partialclonefilter", "blob:none")
        self.git(
            "fetch",
            "--quiet",
            "--no-tags",
            "--filter=blob:none",
            "--depth=1",
            "origin",
            *dict.fromkeys(shas),
        )

    def diff(self, old: str, new: str, *options: str, paths: Sequence[str] = ()):
        # Pins what user configuration could change, as GitHub's diff has it.
        return self.git(
            "-c",
            "core.quotepath=on",
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=all",
            "--diff-algorithm=myers",
            "--indent-heuristic",
            *options,
            old,
            new,
            "--",
            *paths,
        )

    def show_lines(self, sha: str, path: str) -> list[str]:
        return self.git("show", f"{sha}:{path}").splitlines()


def changed_files(repo: Repository, merge_base: str, head: str) -> list[FileDiff]:
    """The files the pull request adds lines to, as its diff shows them.

    With GitHub's three lines of context: the amount changes how git aligns
    some edits, so fewer would mark different lines than the pull request shows.
    """
    text = repo.diff(
        merge_base, head, "--unified=3", "--find-renames", "--diff-filter=AMR"
    )
    return [f for f in parse_diff(text) if f.new_path and f.added]


def changed_lines(files: Sequence[FileDiff]) -> dict[str, list[int]]:
    """Per file, the head line numbers the pull request adds."""
    return {f.new_path: f.added for f in files}


# Per changed file, the hunks taking head line numbers to the built commit's,
# or None when the built commit no longer has the file.
LineMaps = dict[str, Optional[list[Hunk]]]


def line_maps(
    repo: Repository, head: str, built: str, paths: Sequence[str]
) -> LineMaps:
    """How the built commit changed the given files of the head.

    Files it has unchanged are left out, so the usual case reads no blobs.
    """
    if head == built:
        return {}
    wanted = set(paths)
    statuses = {}
    for line in repo.diff(head, built, "--name-status", "--no-renames").splitlines():
        status, _, path = line.partition("\t")
        path = _unquote_path(path)
        if path in wanted:
            statuses[path] = status
    maps: LineMaps = {path: None for path, s in statuses.items() if s == "D"}
    modified = sorted(path for path, s in statuses.items() if s != "D")
    for i in range(0, len(modified), DIFF_PATHS_PER_CALL):
        batch = modified[i : i + DIFF_PATHS_PER_CALL]
        text = repo.diff(head, built, "--unified=0", "--no-renames", paths=batch)
        for file_diff in parse_diff(text):
            if file_diff.old_path:
                maps[file_diff.old_path] = file_diff.hunks
    return maps


def find_merge_base(repository: str, base: str, head: str) -> str:
    """The commit a pull request's diff starts from."""
    url = (
        f"https://api.github.com/repos/{repository}/compare/{base}...{head}?per_page=1"
    )
    return gha_send_request(url)["merge_base_commit"]["sha"]


def parse_lcov(text: str) -> dict[str, LineHits]:
    """Line hit counts per source file of an lcov report."""
    files: dict[str, LineHits] = {}
    current: Optional[LineHits] = None
    for line in text.splitlines():
        if line.startswith("SF:"):
            current = files.setdefault(line[3:].strip(), {})
        elif line.startswith("DA:") and current is not None:
            fields = line[3:].split(",")
            try:
                number, hits = int(fields[0]), int(fields[1])
            except (IndexError, ValueError):
                continue
            current[number] = max(current.get(number, 0), hits)
        elif line == "end_of_record":
            current = None
    return files


def _merge_hits(into: LineHits, hits: Mapping[int, int]) -> None:
    for number, count in hits.items():
        into[number] = max(into.get(number, 0), count)


def report_project(artifact_name: str) -> str:
    """The project of a coverage-report-<project>-<family> artifact.

    Project keys can hold '-' (rocprofiler-sdk); GPU families start with gfx.
    """
    name = artifact_name.removeprefix(REPORT_ARTIFACT_PREFIX)
    match = re.match(r"(.+?)-gfx", name)
    return match.group(1) if match else name


def load_reports(reports_dir: Path) -> dict[str, dict[str, LineHits]]:
    """Per project, the line hit counts of its coverage-report-* artifacts.

    reports_dir holds one artifact per subdirectory, or is one artifact. A
    project tested on several GPU families merges into one entry, so a line
    counts as run if it ran on any of them.
    """
    reports: dict[str, dict[str, LineHits]] = {}
    found = [*reports_dir.glob("coverage.info"), *reports_dir.glob("*/coverage.info")]
    for lcov in sorted(found):
        project = report_project(lcov.parent.resolve().name)
        files = reports.setdefault(project, {})
        for path, hits in parse_lcov(lcov.read_text(errors="replace")).items():
            _merge_hits(files.setdefault(path, {}), hits)
        logging.info("Loaded %s for %s", lcov, project)
    return reports


class CoverageIndex:
    """Finds a repository file's line hit counts in each project's report.

    Reports record absolute paths under wherever the build had the sources,
    so a file matches on whole trailing path components.
    """

    def __init__(self, reports: Mapping[str, Mapping[str, LineHits]]):
        self._by_name: dict[str, list[tuple[str, str, LineHits]]] = defaultdict(list)
        for project, files in reports.items():
            for path, hits in files.items():
                self._by_name[PurePosixPath(path).name].append((path, project, hits))

    def lookup(self, path: str) -> dict[str, LineHits]:
        found: dict[str, LineHits] = {}
        for report_path, project, hits in self._by_name.get(
            PurePosixPath(path).name, ()
        ):
            if report_path == path or report_path.endswith("/" + path):
                _merge_hits(found.setdefault(project, {}), hits)
        return found


def measure(
    changes: Mapping[str, Sequence[int]], maps: LineMaps, index: CoverageIndex
) -> PatchCoverage:
    files: list[FileResult] = []
    unmeasured: list[str] = []
    for path in sorted(changes):
        hits_by_project = index.lookup(path)
        if not hits_by_project:
            unmeasured.append(path)
            continue
        result = FileResult(path=path, measured_by=sorted(hits_by_project))
        counts = {project: [0, 0] for project in hits_by_project}
        hunks = maps.get(path, [])
        for line in changes[path]:
            built_line = None if hunks is None else map_line(line, hunks)
            if built_line is None:
                result.unmapped.append(line)
                continue
            ran_in = {
                project: hits[built_line] > 0
                for project, hits in hits_by_project.items()
                if built_line in hits
            }
            for project, ran in ran_in.items():
                counts[project][0 if ran else 1] += 1
            if any(ran_in.values()):
                result.covered.append(line)
            elif ran_in:
                result.missed.append(line)
        result.by_project = {
            project: (covered, missed)
            for project, (covered, missed) in counts.items()
            if covered or missed
        }
        files.append(result)
    return PatchCoverage(files=files, unmeasured=unmeasured)


def percent(covered: int, missed: int) -> Optional[float]:
    total = covered + missed
    return round(100.0 * covered / total, 2) if total else None


def format_percent(covered: int, missed: int) -> str:
    value = percent(covered, missed)
    return "-" if value is None else f"{value:.2f}%"


def line_ranges(lines: Sequence[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for line in sorted(lines):
        if ranges and line == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], line)
        else:
            ranges.append((line, line))
    return ranges


def format_ranges(lines: Sequence[int], limit: Optional[int] = None) -> str:
    ranges = line_ranges(lines)
    text = ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges[:limit]
    )
    return text + ", ..." if limit is not None and len(ranges) > limit else text


def _escape_property(value: str) -> str:
    for char, escape in (("%", "%25"), ("\r", "%0D"), ("\n", "%0A")):
        value = value.replace(char, escape)
    return value.replace(":", "%3A").replace(",", "%2C")


def annotations(coverage: PatchCoverage, limit: int) -> list[str]:
    """Workflow commands marking uncovered changed lines, at most `limit`."""
    commands = []
    for result in coverage.files:
        for start, end in line_ranges(result.missed):
            what = "This changed line" if start == end else "These changed lines"
            commands.append(
                f"::warning file={_escape_property(result.path)},line={start},"
                f"endLine={end},title=Not covered by tests::{what} ran in none "
                f"of the tests of {', '.join(result.measured_by)}."
            )
    return commands[:limit]


def headline(coverage: PatchCoverage, projects: Sequence[str]) -> str:
    covered, missed = coverage.covered, coverage.missed
    if not projects:
        return (
            "No coverage report was found, so there is nothing to measure the "
            "change against."
        )
    if covered + missed:
        return (
            f"{covered} of {covered + missed} changed lines covered "
            f"({format_percent(covered, missed)})."
        )
    return "None of the changed lines is code the coverage reports measure."


def render_markdown(
    coverage: PatchCoverage,
    *,
    repository: str,
    pr_number: Optional[int],
    merge_base: str,
    head: str,
    built: str,
    projects: Sequence[str],
    sources: Mapping[str, Sequence[str]],
) -> str:
    summary = headline(coverage, projects)
    pr = (
        f"[{repository}#{pr_number}](https://github.com/{repository}/pull/{pr_number})"
        if pr_number
        else f"The pull request to {repository}"
    )
    changed = len(coverage.files) + len(coverage.unmeasured)
    lines = [
        "## Patch coverage",
        "",
        f"**{summary}**" if projects else summary,
        "",
        f"{pr} changes {changed} file(s) (`{merge_base[:10]}...{head[:10]}`). "
        f"Coverage is from the build of `{built[:10]}` and the reports of "
        f"{', '.join(projects) or 'no project'}.",
        "",
    ]

    if coverage.files:
        lines += [
            "| File | Covered | Not covered | Patch coverage | Measured by |",
            "| --- | --: | --: | --: | --- |",
        ]
        for result in coverage.files:
            c, m = len(result.covered), len(result.missed)
            lines.append(
                f"| `{result.path}` | {c} | {m} | {format_percent(c, m)} "
                f"| {', '.join(result.measured_by)} |"
            )
        lines.append("")

    by_project = coverage.by_project()
    if len(by_project) > 1:
        lines += [
            "| Project | Covered | Not covered | Patch coverage |",
            "| --- | --: | --: | --: |",
            *(
                f"| {project} | {c} | {m} | {format_percent(c, m)} |"
                for project, (c, m) in by_project.items()
            ),
            "",
        ]

    budget = MAX_SOURCE_LINES
    listings = []
    for result in coverage.files:
        picked = result.missed[: min(MAX_SOURCE_LINES_PER_FILE, budget)]
        if not picked:
            continue
        budget -= len(picked)
        text = sources.get(result.path, ())
        width = len(str(picked[-1]))
        listings += [
            f"`{result.path}` lines {format_ranges(result.missed, MAX_RANGES_LISTED)}",
            "",
            "```text",
            *(
                f"{line:>{width}} | {text[line - 1] if line <= len(text) else ''}"
                for line in picked
            ),
            *(
                [f"... {len(result.missed) - len(picked)} more"]
                if len(result.missed) > len(picked)
                else []
            ),
            "```",
            "",
        ]
    if listings:
        lines += [
            "<details><summary>Changed lines no test ran</summary>",
            "",
            *listings,
            "</details>",
            "",
        ]

    unmapped = [f for f in coverage.files if f.unmapped]
    if unmapped:
        where = "; ".join(
            f"`{f.path}` {format_ranges(f.unmapped, MAX_RANGES_LISTED)}"
            for f in unmapped
        )
        lines += [
            "These changed lines differ in the built commit, so their coverage "
            f"says nothing about this change: {where}.",
            "",
        ]

    if coverage.unmeasured:
        listed = coverage.unmeasured[:MAX_UNMEASURED_FILES_LISTED]
        more = len(coverage.unmeasured) - len(listed)
        lines += [
            f"<details><summary>{len(coverage.unmeasured)} changed file(s) no "
            "report measures</summary>",
            "",
            "Tests, build files, documentation, and the code of projects this "
            "run did not measure.",
            "",
            *(f"- `{path}`" for path in listed),
            *([f"- ... {more} more"] if more else []),
            "",
            "</details>",
            "",
        ]
    return "\n".join(lines)


# Per project, test, and file: the changed lines, in head numbering, the test ran.
TestLines = Mapping[str, Mapping[str, Mapping[str, Sequence[int]]]]

HTML_STYLE = """
body { font: 14px/1.5 system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
  color: #1f2328; max-width: 1400px; margin: 2em auto; padding: 0 1em; }
h1 { font-size: 1.6em; margin-bottom: 0.2em; }
h2 { font-size: 1.25em; margin-top: 2em; padding-bottom: 0.3em;
  border-bottom: 1px solid #d1d9e0; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
code, table.diff { font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
.headline { font-size: 1.2em; font-weight: 600; }
table.data { border-collapse: collapse; margin: 0.5em 0; }
table.data th, table.data td { border: 1px solid #d1d9e0; padding: 0.25em 0.6em;
  text-align: left; vertical-align: top; }
table.data th { background: #f6f8fa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.bar { display: inline-block; width: 8em; height: 0.6em; margin-right: 0.5em;
  vertical-align: middle; background: #ffcecb; border-radius: 3px; overflow: hidden; }
.bar span { display: block; height: 100%; background: #2da44e; }
.legend span { display: inline-block; margin-right: 0.4em; padding: 0 0.5em;
  border: 1px solid #d1d9e0; border-radius: 3px; }
details { margin: 0.6em 0; }
details > summary { cursor: pointer; }
details.file { border: 1px solid #d1d9e0; border-radius: 6px; }
details.file > summary { padding: 0.5em 0.8em; background: #f6f8fa; }
.scroll { overflow-x: auto; }
table.diff { border-collapse: collapse; width: 100%; }
table.diff td { padding: 0 0.6em; white-space: pre; vertical-align: top; }
table.diff td.n { width: 1%; text-align: right; color: #59636e; user-select: none; }
table.diff tr.hunk td { padding: 0.2em 0.6em; background: #ddf4ff; color: #59636e; }
.removed { background: #f6f8fa; color: #818b98; }
.covered { background: #dafbe1; }
.missed { background: #ffebe9; }
.notcode { background: #eef1f5; }
.unmapped { background: #fff8c5; }
"""

LINE_STATES = {
    "covered": "Covered: a test ran it",
    "missed": "Not covered: no test ran it",
    "notcode": "Not code: no report counts it",
    "unmapped": "Not measured: the built commit has it differently",
    "removed": "Removed",
}


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def render_html(
    coverage: PatchCoverage,
    *,
    repository: str,
    pr_number: Optional[int],
    merge_base: str,
    head: str,
    built: str,
    projects: Sequence[str],
    diffs: Sequence[FileDiff],
    maps: LineMaps,
    index: CoverageIndex,
    tests: Optional[TestLines] = None,
) -> str:
    """A self-contained page with every measured file's diff and line hit counts.

    Given `tests`, it also lists the tests that ran each changed line.
    """
    url = f"https://github.com/{repository}"
    diff_of = {f.new_path: f for f in diffs}

    def link(href: str, text: str) -> str:
        return f'<a href="{escape(href)}">{text}</a>'

    def source(path: str, line: Optional[int] = None) -> str:
        return f"{url}/blob/{head}/{quote(path)}" + (f"#L{line}" if line else "")

    def commit(sha: str) -> str:
        return link(f"{url}/commit/{sha}", f"<code>{sha[:10]}</code>")

    def bar(covered: int, missed: int) -> str:
        value = percent(covered, missed)
        if value is None:
            return "-"
        return (
            f'<span class="bar"><span style="width:{value}%"></span></span>{value:.2f}%'
        )

    def cell(value: object, css: str = "num") -> str:
        return f'<td class="{css}">{value}</td>'

    # Per file and changed line, the tests that ran it.
    ran_by: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for by_test in (tests or {}).values():
        for test, files in by_test.items():
            for path, lines in files.items():
                for line in lines:
                    ran_by[path][line].append(test)

    title = f"Patch coverage: {repository}" + (f"#{pr_number}" if pr_number else "")
    pr = (
        link(f"{url}/pull/{pr_number}", escape(f"{repository}#{pr_number}"))
        if pr_number
        else escape(f"The pull request to {repository}")
    )
    page = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{escape(title)}</title>",
        f"<style>{HTML_STYLE}</style>",
        "</head><body>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="headline">{escape(headline(coverage, projects))}</p>',
        f"<p>{pr} changes {len(coverage.files) + len(coverage.unmeasured)} "
        f"file(s) ({commit(merge_base)}...{commit(head)}). Coverage is from the "
        f"build of {commit(built)} and the reports of "
        f"{escape(', '.join(projects) or 'no project')}.</p>",
    ]

    show_unmapped = any(f.unmapped for f in coverage.files)
    show_tests = tests is not None
    if coverage.files:
        page += [
            "<h2>Files</h2>",
            '<table class="data"><tr><th>File</th><th>Changed lines</th>'
            "<th>Covered</th><th>Not covered</th><th>Not code</th>"
            + ("<th>Not measured</th>" if show_unmapped else "")
            + "<th>Patch coverage</th><th>Measured by</th></tr>",
        ]
        for i, result in enumerate(coverage.files):
            c, m, u = len(result.covered), len(result.missed), len(result.unmapped)
            changed = len(diff_of[result.path].added) if result.path in diff_of else 0
            page.append(
                f"<tr><td>{link(f'#file-{i}', f'<code>{escape(result.path)}</code>')}"
                f"</td>{cell(changed)}{cell(c)}{cell(m)}{cell(max(changed - c - m - u, 0))}"
                + (cell(u) if show_unmapped else "")
                + f"{cell(bar(c, m), '')}{cell(escape(', '.join(result.measured_by)), '')}</tr>"
            )
        page.append("</table>")

    by_project = coverage.by_project()
    if len(by_project) > 1:
        page.append(
            '<table class="data"><tr><th>Project</th><th>Covered</th>'
            "<th>Not covered</th><th>Patch coverage</th></tr>"
        )
        page += [
            f"<tr>{cell(escape(project), '')}{cell(c)}{cell(m)}{cell(bar(c, m), '')}</tr>"
            for project, (c, m) in by_project.items()
        ]
        page.append("</table>")

    if coverage.files:
        page += [
            "<h2>Changed lines</h2>",
            "<p>Each measured file's diff, numbered as in the pull request, with how "
            "many times the tests ran each line of the built commit"
            + (" and how many tests ran each changed line" if show_tests else "")
            + '.</p><p class="legend">'
            + "".join(
                f'<span class="{css}">{text}</span>'
                for css, text in LINE_STATES.items()
            )
            + "</p>",
        ]
    columns = 6 if show_tests else 5
    for i, result in enumerate(coverage.files):
        file_diff = diff_of.get(result.path)
        found = index.lookup(result.path)
        hunks = maps.get(result.path, [])
        state = {line: "covered" for line in result.covered}
        state.update((line, "missed") for line in result.missed)
        state.update((line, "unmapped") for line in result.unmapped)
        rows = []
        for line in file_diff.lines if file_diff else ():
            if line.kind == "@@":
                rows.append(
                    f'<tr class="hunk"><td colspan="{columns}">{escape(line.text)}</td></tr>'
                )
                continue
            css, hits, tested = "", "", ""
            if line.kind == "-":
                css = "removed"
            else:
                if line.kind == "+":
                    css = state.get(line.new, "notcode")
                built_line = None if hunks is None else map_line(line.new, hunks)
                counts = {
                    project: line_hits[built_line]
                    for project, line_hits in found.items()
                    if built_line in line_hits
                }
                if counts:
                    detail = ", ".join(f"{p}: {n:,}" for p, n in sorted(counts.items()))
                    hits = f'<span title="{escape(detail)}">{max(counts.values()):,}</span>'
                names = (
                    ran_by[result.path].get(line.new, ()) if line.kind == "+" else ()
                )
                if names:
                    suites = defaultdict(int)
                    for name in names:
                        suites[name.split(".", 1)[0]] += 1
                    detail = ", ".join(f"{s}: {n}" for s, n in sorted(suites.items()))
                    tested = f'<span title="{escape(detail)}">{len(names):,}</span>'
            new = link(source(result.path, line.new), str(line.new)) if line.new else ""
            rows.append(
                f'<tr class="{css}">{cell(line.old or "", "n")}{cell(new, "n")}'
                f'{cell(hits, "n")}'
                + (cell(tested, "n") if show_tests else "")
                + f'{cell(escape(line.kind + line.text), "")}</tr>'
            )
        c, m = len(result.covered), len(result.missed)
        header = (
            '<tr class="hunk"><td>Old</td><td>New</td><td>Hits</td>'
            + ("<td>Tests</td>" if show_tests else "")
            + "<td></td></tr>"
        )
        page.append(
            f'<details class="file" id="file-{i}"{" open" if c or m else ""}>'
            f"<summary><code>{escape(result.path)}</code>: {c} covered, {m} not "
            f"covered ({link(source(result.path), 'file at head')})</summary>"
            f'<div class="scroll"><table class="diff">{header}{"".join(rows)}'
            "</table></div></details>"
        )

    if coverage.unmeasured:
        page += [
            f"<h2>{len(coverage.unmeasured)} changed file(s) no report measures</h2>",
            "<p>Tests, build files, documentation, and the code of projects this run "
            "did not measure.</p><ul>",
            *(
                f"<li>{link(source(path), f'<code>{escape(path)}</code>')}</li>"
                for path in coverage.unmeasured
            ),
            "</ul>",
        ]

    if show_tests:
        by_suite: dict[str, dict[str, Mapping[str, Sequence[int]]]] = defaultdict(dict)
        for by_test in tests.values():
            for test, files in by_test.items():
                if any(files.values()):
                    by_suite[test.split(".", 1)[0]][test] = files

        def ran(files: Iterable[Mapping[str, Sequence[int]]]) -> str:
            union: dict[str, set[int]] = defaultdict(set)
            for each in files:
                for path, lines in each.items():
                    union[path].update(lines)
            return "; ".join(
                f'<span title="{escape(path)}">{escape(PurePosixPath(path).name)}</span> '
                f"{format_ranges(sorted(lines))}"
                for path, lines in sorted(union.items())
                if lines
            )

        count = sum(map(len, by_suite.values()))
        page += [
            "<h2>Tests that ran the changed lines</h2>",
            f"<p>{count:,} test(s) in {len(by_suite)} suite(s) ran at least one "
            "changed line.</p>",
            '<table class="data"><tr><th>Suite</th><th>Tests</th>'
            "<th>Changed lines they ran</th></tr>",
            *(
                f"<tr>{cell(f'<code>{escape(suite)}</code>', '')}"
                f"{cell(len(suite_tests))}{cell(ran(suite_tests.values()), '')}</tr>"
                for suite, suite_tests in sorted(by_suite.items())
            ),
            "</table>",
        ]
        for suite, suite_tests in sorted(by_suite.items()):
            page += [
                f"<details><summary><code>{escape(suite)}</code>: "
                f"{len(suite_tests):,} test(s)</summary>",
                '<table class="data"><tr><th>Test</th><th>Changed lines it ran</th></tr>',
                *(
                    f"<tr>{cell(f'<code>{escape(test)}</code>', '')}"
                    f"{cell(ran([suite_tests[test]]), '')}</tr>"
                    for test in sorted(suite_tests, key=_natural_key)
                ),
                "</table></details>",
            ]
    page.append("</body></html>")
    return "\n".join(page) + "\n"


def to_json(
    coverage: PatchCoverage,
    *,
    repository: str,
    pr_number: Optional[int],
    merge_base: str,
    head: str,
    built: str,
    projects: Sequence[str],
) -> dict:
    def counts(covered: int, missed: int) -> dict:
        return {
            "covered": covered,
            "not_covered": missed,
            "percent": percent(covered, missed),
        }

    return {
        "repository": repository,
        "pull_request": pr_number,
        "merge_base_sha": merge_base,
        "head_sha": head,
        "built_sha": built,
        "projects": list(projects),
        "totals": counts(coverage.covered, coverage.missed),
        "by_project": {
            project: counts(c, m) for project, (c, m) in coverage.by_project().items()
        },
        "files": [
            {
                "path": result.path,
                "measured_by": result.measured_by,
                "covered_lines": result.covered,
                "not_covered_lines": result.missed,
                "unmapped_lines": result.unmapped,
            }
            for result in coverage.files
        ],
        "unmeasured_files": coverage.unmeasured,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        required=True,
        help="owner/name of the repository the pull request is in",
    )
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument(
        "--base-sha", required=True, help="Tip of the pull request's base branch"
    )
    parser.add_argument("--head-sha", required=True, help="Pull request head commit")
    parser.add_argument(
        "--built-sha",
        required=True,
        help="Commit the coverage build compiled, normally the merge commit. "
        "A branch or tag is resolved through the GitHub API",
    )
    parser.add_argument(
        "--merge-base-sha",
        default=None,
        help="Merge base of the base and head commits (default: from the "
        "GitHub API, as the pull request's diff has it)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory holding the downloaded coverage-report-<project>-<family> "
        "artifacts, one per subdirectory, or one such artifact",
    )
    parser.add_argument(
        "--git-dir",
        type=Path,
        default=None,
        help="Directory for the fetched commits (default: a temporary one)",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Page with every measured file's diff and line hit counts",
    )
    parser.add_argument(
        "--tests-json",
        type=Path,
        default=None,
        help="Adds to the HTML page the tests that ran each changed line, read as "
        '{"<project>": {"<test>": {"<file>": [<head line>, ...]}}}',
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=DEFAULT_MAX_ANNOTATIONS,
        help="Most uncovered line ranges to annotate on the pull request",
    )
    args = parser.parse_args(argv)
    if args.tests_json and not args.html_output:
        parser.error("--tests-json needs --html-output")

    reports = load_reports(args.reports_dir)
    merge_base = args.merge_base_sha or find_merge_base(
        args.repository, args.base_sha, args.head_sha
    )
    built = args.built_sha
    if not re.fullmatch(r"[0-9a-f]{40}", built):
        built = gha_resolve_git_ref(args.repository, built)
    logging.info("Diffing %s...%s, built %s", merge_base, args.head_sha, built)
    with tempfile.TemporaryDirectory(prefix="patch-coverage-") as scratch:
        repo = Repository(
            args.git_dir or Path(scratch),
            f"https://github.com/{args.repository}.git",
            token=os.environ.get("GITHUB_TOKEN"),
        )
        repo.fetch([merge_base, args.head_sha, built])
        diffs = changed_files(repo, merge_base, args.head_sha)
        changes = changed_lines(diffs)
        maps = line_maps(repo, args.head_sha, built, list(changes))
        index = CoverageIndex(reports)
        coverage = measure(changes, maps, index)
        sources = {
            f.path: repo.show_lines(args.head_sha, f.path)
            for f in coverage.files
            if f.missed
        }

    details = dict(
        repository=args.repository,
        pr_number=args.pr_number,
        merge_base=merge_base,
        head=args.head_sha,
        built=built,
        projects=sorted(reports),
    )
    markdown = render_markdown(coverage, sources=sources, **details)
    gha_append_step_summary(markdown)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown + "\n")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(to_json(coverage, **details), indent=2) + "\n"
        )
    if args.html_output:
        tests = json.loads(args.tests_json.read_text()) if args.tests_json else None
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(
            render_html(
                coverage, diffs=diffs, maps=maps, index=index, tests=tests, **details
            )
        )
    for command in annotations(coverage, args.max_annotations):
        print(command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
