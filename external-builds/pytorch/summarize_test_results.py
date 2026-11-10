#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse JUnit XML test reports and write a GitHub Actions step summary.

Two modes of operation:

1. **Single shard** (default) — scans one test-reports directory and writes a
   per-shard summary.  Called from each matrix job.

2. **Combined** (``--combined-dir``) — scans a directory that contains
   multiple downloaded artifact directories named
   ``test-reports-<config>-<shard>-<num_shards>``.  Produces one unified
   table with a *Config* and *Shard* column so all failures are visible in
   one place.  Called from a post-matrix summary job.  Also writes
   ``failed_test_count``, ``observed_report_count``, ``expected_report_count``,
   and ``results_complete`` to ``$GITHUB_OUTPUT``.

3. **Standard** (``--standard-dir``) — scans artifact directories named
   ``test-reports-<os>-<amdgpu_family>`` from test_pytorch_wheels.yml and
   reports tests run, passed, failed, errored, skipped, and flaky per
   platform and family, then lists failures and skip reasons.

4. **Counts** (``--counts-output``) — writes ``run_test_count``,
   ``passed_test_count``, ``failed_test_count``, ``skipped_test_count``, and
   ``results_complete`` for one test-reports directory to ``$GITHUB_OUTPUT``
   so the job's Quartz notification carries them.

Every mode counts a test's final outcome: a test that passed on a retry is
flaky, not failed. ``results_complete`` uses `check_report_set`, so a timed-out
runner or a module killed by a signal never certifies a partial count.

Usage:
    # Per-shard (inside matrix job):
    python summarize_test_results.py

    # Combined (summary job after all shards):
    python summarize_test_results.py --combined-dir artifacts/

    # Standard suite across platforms and families:
    python summarize_test_results.py --standard-dir artifacts/

    # Test counts as step outputs (inside the test job):
    python summarize_test_results.py --counts-output
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


# Child elements pytest-rerunfailures and flaky plugins attach to a testcase
# that failed at least once before its final outcome.
_RERUN_TAGS = ("rerun", "rerunFailure", "rerunError", "flakyFailure", "flakyError")


def parse_junit_testcases(xml_path: Path, *, strict: bool = False) -> list[dict]:
    """Extract every test case from a JUnit XML file with its outcome.

    Status is one of PASSED, FAILED, ERROR, or SKIPPED. ``rerun`` is True when
    the case carries rerun/flaky records from an earlier attempt. With
    ``strict``, an unparseable or non-JUnit file raises instead of reading as
    empty.
    """
    results = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        if strict:
            raise
        return results

    root = tree.getroot()

    testsuites = root.findall(".//testsuite")
    if root.tag == "testsuite":
        testsuites = [root]
    elif root.tag == "testsuites":
        testsuites = root.findall("testsuite")
    if not testsuites and strict:
        raise ValueError(f"Not a JUnit report: {xml_path}")

    for suite in testsuites:
        for testcase in suite.findall("testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            time_s = testcase.get("time", "")

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None:
                status = "FAILED"
                message = (failure.get("message") or "")[:200]
            elif error is not None:
                status = "ERROR"
                message = (error.get("message") or "")[:200]
            elif skipped is not None:
                status = "SKIPPED"
                message = (skipped.get("message") or skipped.text or "")[:200]
            else:
                status = "PASSED"
                message = ""

            results.append(
                {
                    "file": classname,
                    "class": (
                        classname.rsplit(".", 1)[-1] if "." in classname else classname
                    ),
                    "test": name,
                    "status": status,
                    "message": message,
                    "time": time_s,
                    "rerun": any(testcase.find(t) is not None for t in _RERUN_TAGS),
                }
            )

    return results


def parse_junit_xml(xml_path: Path) -> list[dict]:
    """Extract failed/errored test cases from a JUnit XML file."""
    return [
        case
        for case in parse_junit_testcases(xml_path)
        if case["status"] in ("FAILED", "ERROR")
    ]


def collect_results(reports_dir: Path) -> list[dict]:
    """Walk a single test-reports directory and collect all failures."""
    all_failures = []
    for xml_file in sorted(reports_dir.rglob("*.xml")):
        failures = parse_junit_xml(xml_file)
        for f in failures:
            f["report_file"] = xml_file.relative_to(reports_dir).as_posix()
        all_failures.extend(failures)
    return all_failures


def derive_test_file(report_path: str) -> str:
    """Derive the PyTorch test file name from the report path.

    e.g. 'python-pytest/distributions.test_distributions/...'
         -> 'distributions/test_distributions'
    """
    parts = report_path.split("/")
    if len(parts) >= 2:
        test_dir_name = parts[1] if parts[0].startswith("python") else parts[0]
        return test_dir_name.replace(".", "/")
    return report_path


# ---------------------------------------------------------------------------
# Report completeness (shared by every mode that publishes counts)
# ---------------------------------------------------------------------------

# The test step writes both files after the runner exits, so a timed-out or
# killed step leaves no marker.
COMPLETION_MARKER = "therock-test-complete.txt"
RUNNER_OUTPUT = "therock-runner-output.log"
# run_test.py renames each launched module's log to <module>_<shard>_<hex>_.log
# once its process exits, including when it dies by signal.
_MODULE_LOG_RE = re.compile(r"(?P<module>.+)_\d+\.\d+_[0-9a-f]{16}_\.log")
# run_test.py's failure message for a module killed by a signal.
_SIGNAL_RE = re.compile(r"Received signal: SIG\w+")
# run_test.py's per-module outcome lines, e.g. "inductor/test_smoke 1/1 failed!".
_MODULE_RESULT_RE = re.compile(
    r"^(?P<test>\S+) \d+/\d+ (?P<result>failed!|was successful)", re.MULTILINE
)


def check_report_set(reports_dir: Path) -> list[str]:
    """Returns why a test-reports directory is incomplete; empty if complete.

    Complete means the runner exited, its output reports no module killed by
    a signal, every JUnit file parses, and every module the runner launched
    either wrote a report or was reported successful without one (modules
    that skip on this platform exit 0 with no tests). Pytest writes a module's
    JUnit file at session end, so a crashed or unimportable module otherwise
    vanishes from the counts.
    """
    if not reports_dir.is_dir():
        return ["no test reports"]
    problems = []
    if not (reports_dir / COMPLETION_MARKER).is_file():
        problems.append("the test runner did not exit")
    runner_output = reports_dir / RUNNER_OUTPUT
    output_text = ""
    if not runner_output.is_file():
        problems.append("the test runner output is missing")
    else:
        output_text = runner_output.read_text(errors="replace")
        if _SIGNAL_RE.search(output_text):
            problems.append("a test module was killed by a signal")
    succeeded, failed = set(), set()
    for m in _MODULE_RESULT_RE.finditer(output_text):
        module = m.group("test").replace("/", ".").replace("\\", ".")
        (failed if m.group("result") == "failed!" else succeeded).add(module)

    xml_files = sorted(reports_dir.rglob("*.xml"))
    if not xml_files:
        problems.append("no JUnit reports")
    for xml_file in xml_files:
        try:
            parse_junit_testcases(xml_file, strict=True)
        except (ET.ParseError, ValueError):
            problems.append(f"unreadable report {xml_file.name}")

    reported = {xml_file.parent.name for xml_file in xml_files}
    launched = {
        m.group("module")
        for log in reports_dir.glob("*_.log")
        if (m := _MODULE_LOG_RE.fullmatch(log.name))
    }
    # Fail closed: a module missing a report counts unless the runner said it
    # succeeded (and never that it failed).
    missing = sorted(
        module
        for module in launched - reported
        if module not in succeeded or module in failed
    )
    if missing:
        problems.append(
            f"no report from module(s) that did not succeed: {', '.join(missing)}"
        )
    return problems


# ---------------------------------------------------------------------------
# Single-shard summary (called from each matrix job)
# ---------------------------------------------------------------------------


def write_shard_summary(
    failures: list[dict],
    test_config: str,
    shard: str,
    num_shards: str,
    amdgpu_family: str,
    summary_file: str,
) -> None:
    """Write a per-shard markdown summary."""
    lines = []

    shard_label = f"shard {shard}/{num_shards}" if shard and num_shards else ""
    heading_parts = [p for p in [test_config, shard_label, amdgpu_family] if p]
    heading_suffix = " — " + " | ".join(heading_parts) if heading_parts else ""

    if not failures:
        lines.append(f"### All tests passed{heading_suffix} :white_check_mark:")
        lines.append("")
    else:
        seen = set()
        rows = []
        for f in failures:
            test_file = derive_test_file(f["report_file"])
            key = (test_file, f["class"], f["test"])
            if key in seen:
                continue
            seen.add(key)
            msg = f["message"].replace("|", "\\|").replace("\n", " ")[:100]
            rows.append(
                f"| {test_file} | {f['class']} | {f['test']} "
                f"| {f['status']} | {msg} |"
            )

        lines.append(f"### {len(seen)} failed{heading_suffix}")
        lines.append("")
        lines.append("| Test File | Test Class | Test Name | Status | Error |")
        lines.append("|-----------|-----------|-----------|--------|-------|")
        lines.extend(rows)
        lines.append("")

    _emit(lines, summary_file)


# ---------------------------------------------------------------------------
# Combined summary (called from the post-matrix summary job)
# ---------------------------------------------------------------------------

_ARTIFACT_RE = re.compile(r"test-reports-(?P<config>.+)-(?P<shard>\d+)-(?P<num>\d+)")


def _build_report_header(
    torch_version: str,
    python_version: str,
    amdgpu_family: str,
    package_index_url: str,
    pytorch_git_ref: str,
) -> list[str]:
    """Build the PyTorch Test Report header lines."""
    pytorch_repo_org = "pytorch" if pytorch_git_ref == "nightly" else "ROCm"
    pytorch_web_url = f"https://github.com/{pytorch_repo_org}/pytorch"
    index_url = f"{package_index_url}/{amdgpu_family}/"

    lines = [
        "## PyTorch Test Report",
        "",
        f"* Torch version: `{torch_version}`",
        f"* Python version: `{python_version}`",
        f"* GPU family: `{amdgpu_family}`",
        f"* Package index: {index_url}",
        f"* PyTorch source code: {pytorch_web_url}/tree/{pytorch_git_ref}",
        "",
        "To reproduce, see [Running/testing PyTorch]"
        "(https://github.com/ROCm/TheRock/tree/main/external-builds/pytorch"
        "#runningtesting-pytorch) and setup with:",
        "",
        "```bash",
        "# Fetch pytorch source files, including tests:",
        f"git clone --branch {pytorch_git_ref} {pytorch_web_url}.git",
        "",
        "# Install torch and test requirements",
        f"pip install --index-url={index_url} torch=={torch_version}",
        "pip install -r pytorch/.ci/docker/requirements-ci.txt",
        "```",
        "",
    ]
    return lines


def write_combined_summary(
    combined_dir: Path,
    amdgpu_family: str,
    summary_file: str,
    expected_report_count: int | None = None,
    github_output: str = "",
    torch_version: str = "",
    python_version: str = "",
    package_index_url: str = "",
    pytorch_git_ref: str = "",
) -> dict[str, object]:
    """Scan all shard artifact directories and write one combined table.

    ``failed_test_count`` counts tests whose final outcome failed or errored;
    a test that passed on a retry is not a failure. ``results_complete`` is
    true only when every expected shard reported and each is complete (see
    `check_report_set`).
    """
    lines: list[str] = []

    if torch_version and amdgpu_family:
        lines.extend(
            _build_report_header(
                torch_version,
                python_version,
                amdgpu_family,
                package_index_url,
                pytorch_git_ref,
            )
        )

    rows: list[tuple[str, int, str]] = []
    observed_report_count = 0
    incomplete: list[str] = []
    shard_dirs = sorted(combined_dir.iterdir()) if combined_dir.is_dir() else []

    for artifact_dir in shard_dirs:
        m = _ARTIFACT_RE.fullmatch(artifact_dir.name)
        if not artifact_dir.is_dir() or not m:
            continue
        observed_report_count += 1
        config = m.group("config")
        shard = m.group("shard")
        shard_label = f"{shard}/{m.group('num')}"

        problems = check_report_set(artifact_dir)
        if problems:
            incomplete.append(f"{config} {shard_label}: {'; '.join(problems)}")
        for (test_file, cls, test), outcome in sorted(
            collect_outcomes(artifact_dir).items()
        ):
            if outcome["status"] not in ("FAILED", "ERROR"):
                continue
            rows.append(
                (
                    config,
                    int(shard),
                    f"| {config} | {shard_label} | {test_file} | {cls} | {test} "
                    f"| {outcome['status']} | {_escape(outcome['message'])} |",
                )
            )

    results_complete = (
        expected_report_count is not None
        and expected_report_count > 0
        and observed_report_count == expected_report_count
        and not incomplete
    )
    expected_display = (
        str(expected_report_count) if expected_report_count is not None else "unknown"
    )
    completeness = "complete" if results_complete else "incomplete"
    lines.append(
        f"**Reports:** {observed_report_count} observed / {expected_display} "
        f"expected ({completeness})"
    )
    lines.append("")
    for reason in incomplete:
        lines.append(f"- Incomplete shard {reason}")
    if incomplete:
        lines.append("")

    if not rows and results_complete:
        lines.append(f"### All tests passed — {amdgpu_family} :white_check_mark:")
        lines.append("")
    elif not rows:
        lines.append(
            f"### No failed tests found in available reports — {amdgpu_family} "
            ":warning:"
        )
        lines.append("")
    else:
        rows.sort(key=lambda r: (r[0], r[1]))
        lines.append(f"### {len(rows)} failures across all shards — {amdgpu_family}")
        lines.append("")
        lines.append(
            "| Config | Shard | Test File | Test Class "
            "| Test Name | Status | Error |"
        )
        lines.append(
            "|--------|-------|-----------|-----------|" "-----------|--------|-------|"
        )
        lines.extend(row for _, _, row in rows)
        lines.append("")

    _emit(lines, summary_file)
    outputs: dict[str, object] = {
        "failed_test_count": len(rows),
        "observed_report_count": observed_report_count,
        "expected_report_count": expected_report_count or 0,
        "results_complete": results_complete,
    }
    _write_github_outputs(outputs, github_output)
    return outputs


# ---------------------------------------------------------------------------
# Standard summary (test_pytorch_wheels.yml across platforms and families)
# ---------------------------------------------------------------------------

_STANDARD_ARTIFACT_RE = re.compile(
    r"test-reports-(?P<os>Linux|Windows|macOS)-(?P<family>.+)"
)
# Written into each report set by test_pytorch_wheels.yml. download-artifact
# extracts a lone matching artifact straight into its path, dropping the
# artifact-name directory, so report sets identify themselves.
REPORT_METADATA = "therock-test-report.json"
# Keep the step summary well under GitHub's 1 MiB limit.
_MAX_FAILURE_ROWS = 200
_MAX_SKIP_REASONS = 50


def collect_outcomes(reports_dir: Path) -> dict[tuple[str, str, str], dict]:
    """Merges every attempt of each test into one final outcome.

    A test that passed on any attempt counts as passed; if another attempt
    failed (or the case has rerun records) it is also flagged flaky.
    """
    attempts: dict[tuple[str, str, str], list[dict]] = {}
    for xml_file in sorted(reports_dir.rglob("*.xml")):
        test_file = derive_test_file(xml_file.relative_to(reports_dir).as_posix())
        for case in parse_junit_testcases(xml_file):
            key = (test_file, case["class"], case["test"])
            attempts.setdefault(key, []).append(case)

    outcomes = {}
    for key, cases in attempts.items():
        statuses = {c["status"] for c in cases}
        for status in ("PASSED", "FAILED", "ERROR", "SKIPPED"):
            if status in statuses:
                break
        final = next(c for c in cases if c["status"] == status)
        flaky = status == "PASSED" and (
            bool(statuses & {"FAILED", "ERROR"}) or any(c["rerun"] for c in cases)
        )
        outcomes[key] = {**final, "flaky": flaky}
    return outcomes


def count_outcomes(outcomes: dict) -> dict[str, int]:
    counts = {"PASSED": 0, "FAILED": 0, "ERROR": 0, "SKIPPED": 0, "FLAKY": 0}
    for outcome in outcomes.values():
        counts[outcome["status"]] += 1
        counts["FLAKY"] += outcome["flaky"]
    counts["RUN"] = counts["PASSED"] + counts["FAILED"] + counts["ERROR"]
    return counts


def write_count_outputs(reports_dir: Path, output_file: str) -> dict[str, object]:
    """Writes test counts for one reports directory as step outputs.

    ``failed_test_count`` has the same meaning as in the combined mode: final
    outcome failed or errored. ``results_complete`` is always written; the
    counts are written only when test cases were found.
    """
    problems = check_report_set(reports_dir)
    for problem in problems:
        print(f"Incomplete test reports: {problem}")
    outcomes = collect_outcomes(reports_dir) if reports_dir.is_dir() else {}

    outputs: dict[str, object] = {}
    if outcomes:
        counts = count_outcomes(outcomes)
        outputs = {
            "run_test_count": counts["RUN"],
            "passed_test_count": counts["PASSED"],
            "failed_test_count": counts["FAILED"] + counts["ERROR"],
            "skipped_test_count": counts["SKIPPED"],
        }
    else:
        print(f"No test cases found under {reports_dir}; writing no counts")
    outputs["results_complete"] = bool(outcomes) and not problems

    for name, value in outputs.items():
        print(f"{name}={_output_value(value)}")
    _write_github_outputs(outputs, output_file)
    return outputs


def _escape(text: str, limit: int = 100) -> str:
    return text.replace("|", "\\|").replace("\n", " ")[:limit]


def find_report_sets(standard_dir: Path) -> list[tuple[str, str, Path]]:
    """Returns (os, amdgpu_family, reports_dir) for each downloaded report set.

    Report sets are found by their metadata file, so both download layouts
    work: one artifact extracted directly into `standard_dir`, or several in
    per-artifact subdirectories. Artifacts without metadata fall back to the
    ``test-reports-<os>-<family>`` directory name.
    """
    sets = []
    for metadata in sorted(standard_dir.rglob(REPORT_METADATA)):
        try:
            data = json.loads(metadata.read_text())
        except (OSError, ValueError):
            data = {}
        sets.append(
            (
                str(data.get("os") or "unknown"),
                str(data.get("amdgpu_family") or "unknown"),
                metadata.parent,
            )
        )
    labelled = {path for _, _, path in sets}
    for artifact_dir in sorted(standard_dir.iterdir()):
        m = _STANDARD_ARTIFACT_RE.fullmatch(artifact_dir.name)
        if artifact_dir.is_dir() and m and artifact_dir not in labelled:
            sets.append((m.group("os"), m.group("family"), artifact_dir))
    return sets


def write_standard_summary(standard_dir: Path, summary_file: str) -> int:
    """Writes per-platform/family counts, failures, and skip reasons.

    Returns the number of failed or errored tests across all reports.
    """
    groups = [
        (os_name, family, collect_outcomes(reports_dir))
        for os_name, family, reports_dir in find_report_sets(standard_dir)
    ]

    lines = ["## PyTorch Test Results", ""]
    if not groups:
        lines.append(
            "No PyTorch test reports were uploaded. Check the Test PyTorch "
            "jobs for install or setup failures."
        )
        _emit(lines, summary_file)
        return 0

    columns = ["RUN", "PASSED", "FAILED", "ERROR", "SKIPPED", "FLAKY"]
    lines.append(
        "| Platform | GPU family | Tests run | Passed | Failed | Errors "
        "| Skipped | Flaky |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    totals = dict.fromkeys(columns, 0)
    for os_name, family, outcomes in groups:
        counts = count_outcomes(outcomes)
        for col in columns:
            totals[col] += counts[col]
        cells = " | ".join(str(counts[col]) for col in columns)
        lines.append(f"| {os_name} | {family} | {cells} |")
    if len(groups) > 1:
        cells = " | ".join(f"**{totals[col]}**" for col in columns)
        lines.append(f"| **Total** | | {cells} |")
    lines.append("")

    for os_name, family, reports_dir in find_report_sets(standard_dir):
        problems = check_report_set(reports_dir)
        if problems:
            lines.append(
                f"> [!WARNING]\n> Incomplete results for {os_name} | {family}: "
                f"{'; '.join(problems)}. Counts may omit tests."
            )
            lines.append("")

    for os_name, family, outcomes in groups:
        label = f"{os_name} | {family}"
        failures = sorted(
            (k, v) for k, v in outcomes.items() if v["status"] in ("FAILED", "ERROR")
        )
        if failures:
            lines.append(f"### {len(failures)} failed — {label}")
            lines.append("")
            lines.append("| Test File | Test Class | Test Name | Status | Error |")
            lines.append("|-----------|-----------|-----------|--------|-------|")
            for (test_file, cls, test), f in failures[:_MAX_FAILURE_ROWS]:
                lines.append(
                    f"| {test_file} | {cls} | {test} | {f['status']} "
                    f"| {_escape(f['message'])} |"
                )
            if len(failures) > _MAX_FAILURE_ROWS:
                lines.append(
                    f"\n{len(failures) - _MAX_FAILURE_ROWS} more failures are in "
                    f"the test-reports-{os_name}-{family} artifact."
                )
            lines.append("")

        reasons: dict[str, list[str]] = {}
        for (test_file, cls, test), outcome in sorted(outcomes.items()):
            if outcome["status"] == "SKIPPED":
                reason = _escape(outcome["message"] or "(no reason given)", 150)
                reasons.setdefault(reason, []).append(f"{test_file}::{cls}::{test}")
        if reasons:
            skipped = sum(len(tests) for tests in reasons.values())
            ranked = sorted(reasons.items(), key=lambda r: (-len(r[1]), r[0]))
            lines.append(
                f"<details><summary>{skipped} skipped — {label} "
                f"({len(reasons)} reasons)</summary>"
            )
            lines.append("")
            lines.append("| Count | Reason | Example |")
            lines.append("|---:|---|---|")
            for reason, tests in ranked[:_MAX_SKIP_REASONS]:
                lines.append(f"| {len(tests)} | {reason} | `{_escape(tests[0])}` |")
            if len(ranked) > _MAX_SKIP_REASONS:
                lines.append(
                    f"\n{len(ranked) - _MAX_SKIP_REASONS} more skip reasons are in "
                    f"the test-reports-{os_name}-{family} artifact."
                )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    _emit(lines, summary_file)
    return totals["FAILED"] + totals["ERROR"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _output_value(value: object) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _write_github_outputs(outputs: dict[str, object], output_file: str) -> None:
    if not output_file:
        return
    with open(output_file, "a") as fh:
        for key, value in outputs.items():
            fh.write(f"{key}={_output_value(value)}\n")


def _emit(lines: list[str], summary_file: str) -> None:
    summary = "\n".join(lines) + "\n"
    if summary_file:
        with open(summary_file, "a") as fh:
            fh.write(summary)
    print(summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("external-builds/pytorch/pytorch/test/test-reports"),
        help="Path to a single shard's JUnit XML test-reports directory",
    )
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=None,
        help="Path to a directory of downloaded artifact dirs "
        "(test-reports-<config>-<shard>-<num>). Enables combined mode.",
    )
    parser.add_argument(
        "--standard-dir",
        type=Path,
        default=None,
        help="Path to a directory of downloaded artifact dirs "
        "(test-reports-<os>-<amdgpu_family>). Enables standard mode.",
    )
    parser.add_argument(
        "--counts-output",
        action="store_true",
        help="Write test counts for --reports-dir to $GITHUB_OUTPUT instead "
        "of a step summary.",
    )
    parser.add_argument(
        "--test-config",
        default=os.getenv("TEST_CONFIG", "unknown"),
    )
    parser.add_argument(
        "--amdgpu-family",
        default=os.getenv("AMDGPU_FAMILY", "unknown"),
    )
    parser.add_argument(
        "--torch-version",
        default=os.getenv("TORCH_VERSION", ""),
    )
    parser.add_argument(
        "--python-version",
        default=os.getenv("PYTHON_VERSION", ""),
    )
    parser.add_argument(
        "--package-index-url",
        default=os.getenv("PACKAGE_INDEX_URL", ""),
    )
    parser.add_argument(
        "--pytorch-git-ref",
        default=os.getenv("PYTORCH_GIT_REF", ""),
    )
    parser.add_argument(
        "--expected-report-count",
        type=int,
        default=None,
        help="Number of report artifacts expected from the prepared test matrix",
    )
    args = parser.parse_args()

    summary_file = os.getenv("GITHUB_STEP_SUMMARY", "")

    if args.counts_output:
        write_count_outputs(args.reports_dir, os.getenv("GITHUB_OUTPUT", ""))
        return 0

    if args.standard_dir:
        if not args.standard_dir.is_dir():
            print(f"Standard directory not found: {args.standard_dir}")
            return 0
        write_standard_summary(args.standard_dir, summary_file)
        return 0

    if args.combined_dir:
        # A missing directory still writes outputs (incomplete), so absent
        # artifacts are never read as zero failures.
        write_combined_summary(
            args.combined_dir,
            args.amdgpu_family,
            summary_file,
            expected_report_count=args.expected_report_count,
            github_output=os.getenv("GITHUB_OUTPUT", ""),
            torch_version=args.torch_version,
            python_version=args.python_version,
            package_index_url=args.package_index_url,
            pytorch_git_ref=args.pytorch_git_ref,
        )
        return 0

    if not args.reports_dir.is_dir():
        print(f"Reports directory not found: {args.reports_dir}")
        return 0

    shard = os.getenv("SHARD_NUMBER", "")
    num_shards = os.getenv("NUM_TEST_SHARDS", "")
    failures = collect_results(args.reports_dir)
    write_shard_summary(
        failures, args.test_config, shard, num_shards, args.amdgpu_family, summary_file
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
