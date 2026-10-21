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

The results go to the job summary, a JSON report, and warning annotations on
uncovered lines, numbered as in the pull request head so they land on its diff.
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
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional, Sequence

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


@dataclass
class FileDiff:
    old_path: Optional[str] = None
    new_path: Optional[str] = None
    hunks: list[Hunk] = field(default_factory=list)
    # New-side numbers of the lines the hunks add.
    added: list[int] = field(default_factory=list)


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
    new_line = 0
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
            new_line = int(new_start)
        elif current.hunks:
            # Hunk content, which can itself start with "---" or "+++". Removed
            # lines and "\ No newline at end of file" leave the new side alone.
            if line.startswith("+"):
                current.added.append(new_line)
                new_line += 1
            elif line.startswith(" "):
                new_line += 1
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


def changed_lines(repo: Repository, merge_base: str, head: str) -> dict[str, list[int]]:
    """Per file, the head line numbers the pull request adds.

    With GitHub's three lines of context: the amount changes how git aligns
    some edits, so fewer would mark different lines than the pull request shows.
    """
    text = repo.diff(
        merge_base, head, "--unified=3", "--find-renames", "--diff-filter=AMR"
    )
    return {
        file_diff.new_path: file_diff.added
        for file_diff in parse_diff(text)
        if file_diff.new_path and file_diff.added
    }


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

    A project tested on several GPU families merges into one entry, so a line
    counts as run if it ran on any of them.
    """
    reports: dict[str, dict[str, LineHits]] = {}
    for lcov in sorted(reports_dir.glob(f"{REPORT_ARTIFACT_PREFIX}*/coverage.info")):
        project = report_project(lcov.parent.name)
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
    covered, missed = coverage.covered, coverage.missed
    if not projects:
        headline = (
            "No coverage report was produced, so there is nothing to measure "
            "the change against. The coverage jobs say why."
        )
    elif covered + missed:
        headline = (
            f"**{covered} of {covered + missed} changed lines covered "
            f"({format_percent(covered, missed)}).**"
        )
    else:
        headline = "**None of the changed lines is code the coverage reports measure.**"
    pr = (
        f"[{repository}#{pr_number}](https://github.com/{repository}/pull/{pr_number})"
        if pr_number
        else f"The pull request to {repository}"
    )
    changed = len(coverage.files) + len(coverage.unmeasured)
    lines = [
        "## Patch coverage",
        "",
        headline,
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
        help="Directory holding the downloaded coverage-report-* artifacts",
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
        "--max-annotations",
        type=int,
        default=DEFAULT_MAX_ANNOTATIONS,
        help="Most uncovered line ranges to annotate on the pull request",
    )
    args = parser.parse_args(argv)

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
        changes = changed_lines(repo, merge_base, args.head_sha)
        maps = line_maps(repo, args.head_sha, built, list(changes))
        coverage = measure(changes, maps, CoverageIndex(reports))
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
    for command in annotations(coverage, args.max_annotations):
        print(command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
