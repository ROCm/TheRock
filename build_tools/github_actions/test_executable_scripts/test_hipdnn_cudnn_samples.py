#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
NVIDIA cudnn-frontend sample corpus compatibility test.

Drives the standalone CMake harness shipped inside the hipDNN artifact at
share/hipdnn/cudnn_samples. That harness acquires NVIDIA's unmodified
cudnn-frontend sample corpus and compiles it against hipDNN's cuDNN
compatibility shim.

This is not hipDNN's own sample suite; that one is test_hipdnn_samples.py.

The harness owns every policy decision: whether the shim is present at all,
which translation units are expected to compile, and which are expected to
fail. This driver only configures, builds, tests, and reports.
"""

import argparse
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

OUTPUT_ARTIFACTS_DIR = os.getenv("OUTPUT_ARTIFACTS_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

SHARD_INDEX = os.getenv("SHARD_INDEX", "1")
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", "1")

# Location of the harness project inside the installed artifact tree.
PAYLOAD_RELPATH = Path("share") / "hipdnn" / "cudnn_samples"

# These two are coupled and must stay together. The keep-going flag is Ninja's,
# not cmake's, so it travels after "--" and is only valid while the generator
# below is Ninja. Adding another generator means changing both: get it wrong and
# the build silently produces nothing while every sample reports as a compile
# failure.
CMAKE_GENERATOR = "Ninja"
BUILD_KEEP_GOING_ARGS = ["-k", "0"]

# The single test the harness registers when the compatibility shim is absent.
SKIP_TEST_NAME = "cudnn_samples_skipped"

# Per-translation-unit JSON sidecars, written by the harness.
REPORT_DIR_NAME = "cudnn_samples_report"

# The payload is installed unconditionally, so a missing directory means the
# artifact did not ship it -- a wiring bug, distinct from the shim being off.
EXIT_PAYLOAD_MISSING = 11
EXIT_FLAG_REQUIRED_BUT_OFF = 10

# Outcome tokens the harness writes into the sidecars. Any other value is
# surfaced as an anomaly instead of being folded into a count.
OUTCOME_COMPILED = "compiled"
OUTCOME_COMPILE_FAILED = "compile-failed"
OUTCOME_XFAIL_STILL_FAILING = "xfail-still-failing"
OUTCOME_XFAIL_NOW_COMPILES = "xfail-now-compiles"
OUTCOME_RAN_CLEAN = "ran-clean"
OUTCOME_RAN_NO_ASSERTIONS = "ran-no-assertions"
OUTCOME_RAN_WITH_ASSERTION_FAILURES = "ran-with-assertion-failures"
OUTCOME_CRASHED = "crashed"
OUTCOME_EXCLUDED = "excluded"

KNOWN_OUTCOMES = {
    OUTCOME_COMPILED,
    OUTCOME_COMPILE_FAILED,
    OUTCOME_XFAIL_STILL_FAILING,
    OUTCOME_XFAIL_NOW_COMPILES,
    OUTCOME_RAN_CLEAN,
    OUTCOME_RAN_NO_ASSERTIONS,
    OUTCOME_RAN_WITH_ASSERTION_FAILURES,
    OUTCOME_CRASHED,
    OUTCOME_EXCLUDED,
}

# A translation unit that ran necessarily compiled first. "compiled" is both an
# intermediate and a terminal token: the compile probe writes it, and a RUN-tier
# run probe then rewrites the sidecar to one of the "ran-*" values. An entry
# left at "compiled" simply never reached its run probe.
RAN_OUTCOMES = {
    OUTCOME_RAN_CLEAN,
    OUTCOME_RAN_NO_ASSERTIONS,
    OUTCOME_RAN_WITH_ASSERTION_FAILURES,
    OUTCOME_CRASHED,
}
COMPILED_OUTCOMES = RAN_OUTCOMES | {OUTCOME_COMPILED, OUTCOME_XFAIL_NOW_COMPILES}

logging.basicConfig(level=logging.INFO)


def get_parallelism() -> int:
    """CPU budget for both the build and the test run.

    KUBE_CPU_REQUEST is Kubernetes-injected and may be fractional ("8.0"), so
    truncate at the dot the way the workflow's ${KUBE_CPU_REQUEST%.*} does.
    """
    kube_cpu_request = os.getenv("KUBE_CPU_REQUEST")
    if kube_cpu_request:
        try:
            return max(1, int(kube_cpu_request.split(".")[0]))
        except ValueError:
            logging.warning(f"Ignoring unparsable KUBE_CPU_REQUEST={kube_cpu_request!r}")
    return max(1, os.cpu_count() or 1)


def get_default_build_dir() -> Path:
    """A deliberately short build root.

    The harness registers dozens of targets whose object and link paths nest
    deeply; rooting the build inside the workspace overruns Windows MAX_PATH.
    """
    temp_root = os.getenv("RUNNER_TEMP") or tempfile.gettempdir()
    return Path(temp_root) / "cdnns"


def build_environment(artifacts_path: Path) -> dict:
    """Environment for CMake, the compiler, and the sample executables."""
    environ_vars = os.environ.copy()
    environ_vars["HIP_PLATFORM"] = "amd"

    if platform.system() == "Windows":
        # Both bin and lib, not just the prefix root: hipdnn_backend.dll and the ROCm
        # runtime DLLs live in those subdirectories, and Windows resolves imports off
        # PATH. Missing them surfaces as exit 0xC0000135 (STATUS_DLL_NOT_FOUND) from
        # every sample, which the launcher correctly-but-unhelpfully reports as a crash.
        prefixes = [str(artifacts_path / "bin"), str(artifacts_path / "lib"),
                    str(artifacts_path)]
        existing = environ_vars.get("PATH", "")
        environ_vars["PATH"] = ";".join(prefixes + ([existing] if existing else []))
    else:
        existing = environ_vars.get("LD_LIBRARY_PATH", "")
        rocm_lib = str(artifacts_path / "lib")
        environ_vars["LD_LIBRARY_PATH"] = ":".join(
            [rocm_lib] + ([existing] if existing else [])
        )

    return environ_vars


def configure(
    source_dir: Path, build_dir: Path, artifacts_path: Path, environ_vars: dict
):
    """Configure the harness as an external consumer of the installed hipDNN.

    The harness decides skip-vs-run here, from its find_package query for the
    cudnn_compatibility component, and acquires the sample corpus itself.
    Neither is this driver's business.
    """
    is_windows = platform.system() == "Windows"
    compiler_ext = ".exe" if is_windows else ""

    configure_cmd = [
        "cmake",
        "-B",
        str(build_dir),
        "-S",
        str(source_dir),
        f"-G{CMAKE_GENERATOR}",
        f"-DCMAKE_PREFIX_PATH={artifacts_path}",
        f"-DCMAKE_CXX_COMPILER={artifacts_path}/lib/llvm/bin/clang++{compiler_ext}",
        f"-DCMAKE_C_COMPILER={artifacts_path}/lib/llvm/bin/clang{compiler_ext}",
    ]

    # Windows needs a resource compiler specified
    if is_windows:
        configure_cmd.append("-DCMAKE_RC_COMPILER=rc.exe")

    logging.info(f"++ Configure: {shlex.join(configure_cmd)}")
    subprocess.run(configure_cmd, check=True, cwd=THEROCK_DIR, env=environ_vars)


def build(build_dir: Path, jobs: int, environ_vars: dict):
    """Build every sample target, keeping going past failures.

    The build exit status is ignored on purpose: each translation unit's
    compilation is itself a ctest case, so a compile failure must be reported
    per-sample rather than aborting the driver. Keeping going past the first
    error is what stops one broken sample from hiding every other result.
    """
    # "-j" is a cmake option and must precede "--"; everything after it goes to
    # the generator's native tool. See BUILD_KEEP_GOING_ARGS.
    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "-j",
        str(jobs),
        "--",
        *BUILD_KEEP_GOING_ARGS,
    ]
    logging.info(f"++ Build: {shlex.join(build_cmd)}")
    result = subprocess.run(build_cmd, check=False, cwd=THEROCK_DIR, env=environ_vars)
    logging.info(
        f"++ Build exited with {result.returncode}; status ignored on purpose "
        "(compilation is reported as a ctest case, not as a driver failure)"
    )


def run_ctest(build_dir: Path, jobs: int, environ_vars: dict) -> int:
    """Run the harness test suite. Failures are reported, never raised."""
    test_cmd = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--parallel",
        str(jobs),
    ]
    logging.info(f"++ Test: {shlex.join(test_cmd)}")
    result = subprocess.run(test_cmd, check=False, cwd=THEROCK_DIR, env=environ_vars)
    logging.info(f"++ Test exited with {result.returncode}")
    return result.returncode


def list_registered_tests(build_dir: Path, environ_vars: dict) -> list:
    """Names of the tests the harness registered, or [] if unavailable."""
    list_cmd = ["ctest", "--test-dir", str(build_dir), "--show-only=json-v1"]
    result = subprocess.run(
        list_cmd,
        check=False,
        cwd=THEROCK_DIR,
        env=environ_vars,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.warning(f"Could not enumerate ctest tests: {result.stderr.strip()}")
        return []
    try:
        listing = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        logging.warning(f"Could not parse the ctest test listing: {exc}")
        return []
    return [test.get("name", "") for test in listing.get("tests", [])]


def is_skip_run(test_names: list) -> bool:
    """True when the harness registered only its shim-absent placeholder."""
    return test_names == [SKIP_TEST_NAME]


def report_skip() -> int:
    """Announce that nothing was tested, and decide whether that is fatal."""
    print(
        "::notice title=cuDNN samples skipped::"
        "hipDNN was built with HIPDNN_ENABLE_CUDNN_COMPATIBILITY off, so the "
        "cuDNN compatibility shim is not installed. No sample was compiled or "
        "run: this job tested the wiring and nothing else."
    )
    if os.getenv("HIPDNN_CUDNN_SAMPLES_REQUIRE_FLAG") == "1":
        print(
            "::error title=cuDNN samples required but skipped::"
            "HIPDNN_CUDNN_SAMPLES_REQUIRE_FLAG=1 demands a real run, but "
            "HIPDNN_ENABLE_CUDNN_COMPATIBILITY is off in this build."
        )
        return EXIT_FLAG_REQUIRED_BUT_OFF
    return 0


def normalize_outcome(outcome) -> str:
    """Fold sidecar outcome spellings onto the canonical hyphenated token."""
    return str(outcome).strip().lower().replace("_", "-").replace(" ", "-")


def load_sidecars(report_dir: Path):
    """Read the per-translation-unit sidecars.

    Returns (entries, anomalies). A missing or malformed sidecar is recorded as
    an anomaly; it never aborts the run, because the report is the one thing
    this driver must always produce.

    EXCLUDED entries are written at configure time and have no ctest case at
    all, so sidecars and tests do not correspond one-to-one.
    """
    entries = []
    anomalies = []

    if not report_dir.is_dir():
        anomalies.append(f"No sidecar directory at {report_dir}")
        return entries, anomalies

    sidecars = sorted(report_dir.glob("*.json"))
    if not sidecars:
        anomalies.append(f"No sidecars found in {report_dir}")

    for sidecar in sidecars:
        try:
            entry = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            anomalies.append(f"{sidecar.name}: unreadable ({exc})")
            continue
        if not isinstance(entry, dict):
            anomalies.append(f"{sidecar.name}: expected a JSON object")
            continue
        if not entry.get("tu"):
            anomalies.append(f"{sidecar.name}: missing 'tu'")
            entry["tu"] = sidecar.stem
        if normalize_outcome(entry.get("outcome", "")) not in KNOWN_OUTCOMES:
            anomalies.append(
                f"{entry['tu']}: unrecognized outcome {entry.get('outcome')!r}"
            )
        entries.append(entry)

    entries.sort(key=lambda entry: str(entry["tu"]))
    return entries, anomalies


def summarize(entries: list) -> dict:
    """The headline counts."""
    counts = {
        "compiled": 0,
        "ran": 0,
        "ran-clean": 0,
        "ran-no-assertions": 0,
        "ran-with-assertion-failures": 0,
        "xfail-still-failing": 0,
    }
    for entry in entries:
        outcome = normalize_outcome(entry.get("outcome", ""))
        if outcome in COMPILED_OUTCOMES:
            counts["compiled"] += 1
        if outcome in RAN_OUTCOMES:
            counts["ran"] += 1
        if outcome == OUTCOME_RAN_CLEAN:
            counts["ran-clean"] += 1
        if outcome == OUTCOME_RAN_NO_ASSERTIONS:
            counts["ran-no-assertions"] += 1
        if outcome == OUTCOME_RAN_WITH_ASSERTION_FAILURES:
            counts["ran-with-assertion-failures"] += 1
        if outcome == OUTCOME_XFAIL_STILL_FAILING:
            counts["xfail-still-failing"] += 1
    return counts


def _cell(value) -> str:
    """Render one markdown table cell, keeping the row's pipes intact."""
    return str(value).replace("|", "\\|")


def _detail(entry: dict) -> str:
    parts = []
    assertion_failures = entry.get("assertion_failures")
    if isinstance(assertion_failures, int):
        parts.append(str(assertion_failures))
    reason = entry.get("reason")
    if reason:
        parts.append(str(reason))
    return " / ".join(parts) if parts else "-"


def format_report(entries: list, anomalies: list, counts: dict) -> str:
    lines = ["## hipDNN cuDNN sample corpus", ""]
    lines.append(
        "Counts: " + ", ".join(f"{name} {value}" for name, value in counts.items())
    )
    lines.append("")

    # An XFAIL entry that started compiling means the manifest has rotted. It is
    # the report's most consequential row, so call it out above the table.
    rotted = [
        entry
        for entry in entries
        if normalize_outcome(entry.get("outcome", "")) == OUTCOME_XFAIL_NOW_COMPILES
    ]
    if rotted:
        lines.append(
            f"**Manifest rot: {len(rotted)} translation unit(s) marked "
            "XFAIL_COMPILE now compile. Retier them.**"
        )
        lines.append("")
        for entry in rotted:
            lines.append(f"- `{entry.get('tu', '?')}`")
        lines.append("")

    lines.append("| TU | Tier | Outcome | Assertion failures / reason |")
    lines.append("| --- | --- | --- | --- |")
    for entry in entries:
        outcome = entry.get("outcome", "?")
        if normalize_outcome(outcome) == OUTCOME_XFAIL_NOW_COMPILES:
            outcome = f"**{outcome}**"
        lines.append(
            f"| {_cell(entry.get('tu', '?'))} "
            f"| {_cell(entry.get('tier', '?'))} "
            f"| {_cell(outcome)} "
            f"| {_cell(_detail(entry))} |"
        )
    if not entries:
        lines.append("| _(no sidecars)_ | - | - | - |")

    if anomalies:
        lines.append("")
        lines.append("### Report anomalies")
        lines.append("")
        for anomaly in anomalies:
            lines.append(f"- {anomaly}")
    lines.append("")
    return "\n".join(lines)


def emit_report(build_dir: Path):
    """Aggregate the sidecars into the job summary and onto stdout."""
    entries, anomalies = load_sidecars(build_dir / REPORT_DIR_NAME)
    counts = summarize(entries)
    report = format_report(entries, anomalies, counts)

    # Always print: the step summary only exists under GitHub Actions, and a
    # local run needs to be readable too.
    print(report)

    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as summary_file:
            summary_file.write(report)
            summary_file.write("\n")

    for anomaly in anomalies:
        logging.warning(f"Report anomaly: {anomaly}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and run NVIDIA's cudnn-frontend samples against "
        "hipDNN's cuDNN compatibility shim"
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory path (will be created if doesn't exist). "
        "If not specified, uses a short directory under the runner temp dir. "
        "Keep any override short: deep paths overrun MAX_PATH on Windows.",
    )
    args = parser.parse_args()

    if not OUTPUT_ARTIFACTS_DIR:
        raise RuntimeError("OUTPUT_ARTIFACTS_DIR environment variable not set")

    logging.info(f"Using OUTPUT_ARTIFACTS_DIR: {OUTPUT_ARTIFACTS_DIR}")
    logging.info(f"Shard {SHARD_INDEX} of {TOTAL_SHARDS}")

    artifacts_path = Path(OUTPUT_ARTIFACTS_DIR).resolve()
    source_dir = artifacts_path / PAYLOAD_RELPATH
    if not source_dir.is_dir():
        logging.error(
            f"cuDNN sample harness not found at {source_dir}. The payload is "
            "installed unconditionally, so this means the hipDNN artifact did "
            "not ship it, not that the compatibility shim is disabled."
        )
        return EXIT_PAYLOAD_MISSING

    build_dir = (args.build_dir or get_default_build_dir()).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using build directory: {build_dir}")

    jobs = get_parallelism()
    logging.info(f"Using {jobs} parallel jobs")

    environ_vars = build_environment(artifacts_path)

    configure(source_dir, build_dir, artifacts_path, environ_vars)
    build(build_dir, jobs, environ_vars)
    ctest_returncode = run_ctest(build_dir, jobs, environ_vars)

    if is_skip_run(list_registered_tests(build_dir, environ_vars)):
        if ctest_returncode == 0:
            return report_skip()
        logging.error(
            f"The harness registered only {SKIP_TEST_NAME}, but ctest exited "
            f"with {ctest_returncode} instead of reporting it as skipped."
        )
        return ctest_returncode

    emit_report(build_dir)
    return ctest_returncode


if __name__ == "__main__":
    sys.exit(main())
