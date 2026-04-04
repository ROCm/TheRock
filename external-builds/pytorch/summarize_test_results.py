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
   one place.  Called from a post-matrix summary job.

Usage:
    # Per-shard (inside matrix job):
    python summarize_test_results.py

    # Combined (summary job after all shards):
    python summarize_test_results.py --combined-dir artifacts/
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit_xml(xml_path: Path) -> list[dict]:
    """Extract failed/errored test cases from a JUnit XML file."""
    results = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return results

    root = tree.getroot()

    testsuites = root.findall(".//testsuite")
    if root.tag == "testsuite":
        testsuites = [root]
    elif root.tag == "testsuites":
        testsuites = root.findall("testsuite")

    for suite in testsuites:
        for testcase in suite.findall("testcase"):
            classname = testcase.get("classname", "")
            name = testcase.get("name", "")
            time_s = testcase.get("time", "")

            failure = testcase.find("failure")
            error = testcase.find("error")

            if failure is not None:
                status = "FAILED"
                message = (failure.get("message") or "")[:200]
            elif error is not None:
                status = "ERROR"
                message = (error.get("message") or "")[:200]
            else:
                continue

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
                }
            )

    return results


def collect_results(reports_dir: Path) -> list[dict]:
    """Walk a single test-reports directory and collect all failures."""
    all_failures = []
    for xml_file in sorted(reports_dir.rglob("*.xml")):
        failures = parse_junit_xml(xml_file)
        for f in failures:
            f["report_file"] = str(xml_file.relative_to(reports_dir))
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

_ARTIFACT_RE = re.compile(r"test-reports-(?P<config>[^-]+)-(?P<shard>\d+)-(?P<num>\d+)")


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
    torch_version: str = "",
    python_version: str = "",
    package_index_url: str = "",
    pytorch_git_ref: str = "",
) -> None:
    """Scan all artifact directories and write one combined table."""
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
    shard_dirs = sorted(combined_dir.iterdir())

    for artifact_dir in shard_dirs:
        if not artifact_dir.is_dir():
            continue
        m = _ARTIFACT_RE.match(artifact_dir.name)
        if not m:
            continue
        config = m.group("config")
        shard = m.group("shard")
        num = m.group("num")
        shard_label = f"{shard}/{num}"

        failures = collect_results(artifact_dir)
        seen = set()
        for f in failures:
            test_file = derive_test_file(f["report_file"])
            key = (config, shard_label, test_file, f["class"], f["test"])
            if key in seen:
                continue
            seen.add(key)
            msg = f["message"].replace("|", "\\|").replace("\n", " ")[:100]
            rows.append(
                (
                    config,
                    int(shard),
                    f"| {config} | {shard_label} | {test_file} "
                    f"| {f['class']} | {f['test']} | {f['status']} | {msg} |",
                )
            )

    if not rows:
        lines.append(f"### All tests passed — {amdgpu_family} :white_check_mark:")
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    args = parser.parse_args()

    summary_file = os.getenv("GITHUB_STEP_SUMMARY", "")

    if args.combined_dir:
        if not args.combined_dir.is_dir():
            print(f"Combined directory not found: {args.combined_dir}")
            return 0
        write_combined_summary(
            args.combined_dir,
            args.amdgpu_family,
            summary_file,
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
