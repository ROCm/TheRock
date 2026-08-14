#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs the JAX test suite against installed JAX ROCm wheels.

The suite scripts in the ROCm/jax checkout stay the source of truth for how the
tests run: ci/run_pytest_rocm.sh for a whole run, and ci/run_pytest_rocm_multi.sh
for the multi-accelerator subset on its own. This adds only what is ours:

  * the ROCm runtime workarounds this repository needs;
  * the known-bad tests from skip_tests/, as a pytest -k expression;
  * an optional --test-list, described under "Selecting a subset" below;
  * two layers of retry, described under "Retries" below;
  * holding the run to the CPUs the pod was given, described under "Resources".

Nothing the suite decides is repeated here. Its environment is read back out of
ci/utilities/rocm_test_env.sh, which both scripts source, or out of the marked
section of the suite script in a checkout from before that file existed, so a
version that changes its allocator or its XLA flags, including behind a
conditional, needs no change in this file. Reading it is required rather than
best-effort: the retry pass decides the result, so running it under a different
environment than the suite would make that verdict meaningless. A checkout that
holds neither fails the run with an error naming both.

Retries
-------

--in-process-reruns is pytest-rerunfailures: it repeats a failed test in the same
worker, recovering a crashed worker or a one-off runtime error.

--fresh-process-retry re-runs the reported failures in a new process, and decides
the result. It is not interchangeable with the above: a sticky failure leaves the
process failing every later test the same way, which no rerun inside it can
clear. It only speaks for the suite when the suite's own pytest sessions ran to
the end, which their reported exit codes are what say.

--no-retries turns off both, running the suite exactly as the script does.

Selecting a subset
------------------

--test-list names a file of test paths to run instead of the whole suite, which
is how CI gets a PR-sized run out of the same suite the nightly runs. It is
applied by ignoring everything under tests/ the list does not name, rather than
by passing the list to pytest, so that the suite script still decides how the
tests run. -k would leave the modules left out imported anyway, and importing
them is most of what a short run has time for.

Resources
---------

A container shares the host's kernel, so nothing a process can query describes
the container: on the gfx942 runners a job holding 25 cores of an 8-way shared
box reads 192 cores and 2268 GiB, and sizes every thread pool from that. What
the pod actually got arrives in KUBE_CPU_REQUEST, and --cpus says the same thing
on a machine that has no CI runner to inject it.

The run is pinned to that many CPUs, which the suite and the retry pass inherit.
Measured on a gfx942 runner, one worker holds 1021 threads unpinned and 107
pinned to 25 cores with OpenBLAS capped, against 16 workers at a time.

Example usage:

    # As CI runs it.
    python run_jax_tests.py --jax-dir jax --jax-ref rocm-jaxlib-v0.11.0 \\
        --amdgpu-family gfx94X-dcgpu

    # The suite with both retry layers off.
    python run_jax_tests.py --jax-dir jax --no-retries

    # Print the environment and commands without running anything.
    python run_jax_tests.py --jax-dir jax --jax-version 0.10.2 --dry-run

    # Run only the tests the skip list would have skipped.
    python run_jax_tests.py --jax-dir jax --jax-version 0.10.2 --debug

    # The subset PR CI runs.
    python run_jax_tests.py --jax-dir jax \\
        --test-list external-builds/jax/test_selection/small_tests.txt
"""

import argparse
import dataclasses
import fnmatch
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = THIS_SCRIPT_DIR.parent.parent

sys.path.insert(0, os.fspath(THIS_SCRIPT_DIR))
sys.path.insert(0, os.fspath(THEROCK_DIR / "test_tools"))

from list_pytest_failed_tests import failed_in_report
from skip_tests.create_skip_tests import keyword_expression

JAX_REF_PREFIX = "rocm-jaxlib-v"

# Relative to the ROCm/jax checkout --jax-dir names, which is also the working
# directory the suite runs in, rather than to wherever this script was called
# from.
RELATIVE_SUITE_SCRIPT = Path("ci") / "run_pytest_rocm.sh"

# The multi-accelerator tests on their own, which is the only subset that needs
# more than one GPU. Absent from a jax ref older than the split.
RELATIVE_MULTI_SUITE_SCRIPT = Path("ci") / "run_pytest_rocm_multi.sh"

# Sourced by the suite script for the JAXCI_* defaults the section below reads,
# and safe to source here too: it only assigns variables.
RELATIVE_SUITE_ENV_FILE = Path("ci") / "envs" / "default.env"

# The tree the suite script collects, so a selection names paths inside it and
# nothing outside it is ever ignored.
RELATIVE_TESTS_DIR = Path("tests")

# What pytest collects by default, which JAX does not override. Everything else
# under tests/ is data the suite reads, so leave it alone.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

# One report per pytest invocation the suite script makes, matched by pattern so
# that a version splitting the suite differently needs no change here.
REPORT_GLOB = "logs/pytest_results*.json"

# The pytest exit codes of a session that ran to the end. Above them the session
# stopped early instead: 2 interrupted, 3 internal error, 4 usage error, 5
# nothing collected. Such a report lists only the tests that got that far.
PYTEST_EXIT_ALL_PASSED = 0
PYTEST_EXIT_TESTS_FAILED = 1

# What both suite scripts source for their environment: exports, echoes and
# device queries, and nothing that installs or tests. Evaluating it is how the
# per-version environment reaches the retry pass, which has to match the run it
# is checking. Sourcing a suite script itself would run the tests, and copying
# its values here would be a second source of truth.
RELATIVE_SUITE_ENV_SCRIPT = Path("ci") / "utilities" / "rocm_test_env.sh"

# Where that environment lived before it moved into the file above, as a section
# of the one suite script there was, with the installs above it and the test run
# below it.
ENV_SECTION_START = "# Set up the generic test environment variables"
ENV_SECTION_END = "# Run tests"

# The environment carries the suite's own scratch variables, lowercase by
# convention, next to the configuration worth keeping. The xdist worker count is
# for the parallel run and would contradict the serial retry.
SUITE_ENV_EXCLUDES = ["JAX_ENABLE_ROCM_XDIST"]

# Which suite script a run invokes, so that the subset needing one GPU and the
# subset needing several can go to different runners. "all" is the whole suite,
# which leaves out what the host has no GPUs for, and "multi" is the
# multi-accelerator tests alone, which a 1-GPU host would skip entirely.
TEST_SUBSET_ALL = "all"
TEST_SUBSET_MULTI = "multi"
SUITE_SCRIPTS = {
    TEST_SUBSET_ALL: RELATIVE_SUITE_SCRIPT,
    TEST_SUBSET_MULTI: RELATIVE_MULTI_SUITE_SCRIPT,
}

# ROCm/HIP runtime tuning that avoids a pytest slowdown and hang. Ours rather than
# the suite's, so it belongs here.
# TODO:(magaonka-amd) remove once the system teams' fixes land.
THEROCK_ENV = {
    "ROCPROFILER_QUEUE_INTERPOSITION": "0",
    "DEBUG_HIP_DYNAMIC_QUEUES": "0",
    "HSA_NO_SCRATCH_RECLAIM": "1",
}

# A test runner is often a single-device slice of a shared host, where these hide
# the other devices but not other tenants. Logged next to the device serials from
# print_driver_gpu_info.py, they tell a noisy neighbour apart from a fault.
GPU_VISIBILITY_ENV_VARS = [
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
]

# What the pod was given, injected by the CI runners. A container shares the
# host's kernel, so nothing the process can query is about the container: the
# gfx942 runners report 192 cores and 2268 GiB to a job holding 25 cores of an
# 8-way shared box, and the cgroups are quiet because CI sets CPU requests and
# no limits. These variables are the only account of the real allocation.
CPU_REQUEST_VAR = "KUBE_CPU_REQUEST"
MEMORY_REQUEST_VAR = "KUBE_MEMORY_REQUEST"

# The most pytest workers the suite starts in parallel, see the num_processes
# cap in the suite's environment. Each one loads its own copy of every library.
SUITE_MAX_WORKERS = 16

# Set by the plugin build. Installed wheels carry their own bitcode and linker, so
# a leftover value points them at paths that do not exist here.
UNSET_VARS = [
    "ROCM_ROOT",
    "HIP_DEVICE_LIB_PATH",
    "JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH",
    "JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH",
]


def jax_version_from_ref(jax_ref: str) -> str:
    """The version a rocm-jax ref names, e.g. rocm-jaxlib-v0.11.0 -> 0.11.0."""
    return jax_ref.removeprefix(JAX_REF_PREFIX) if jax_ref else ""


class TestListError(Exception):
    """The subset a --test-list names cannot be run."""


def read_test_list(path: Path) -> list[Path]:
    """The test paths one selection file names, relative to the JAX checkout."""
    lines = path.read_text().splitlines()
    stripped = (line.partition("#")[0].strip() for line in lines)
    return [Path(line) for line in stripped if line]


def _collects_tests(path: Path) -> bool:
    return path.is_dir() or any(
        fnmatch.fnmatch(path.name, pattern) for pattern in TEST_FILE_PATTERNS
    )


def ignore_arguments(jax_dir: Path, selected: list[Path]) -> list[str]:
    """--ignore for every test path under tests/ the selection leaves out.

    Ignoring the complement is what lets the suite script keep its own pytest
    invocation. Walking the checkout rather than the list also settles what the
    list cannot know: a version that has since renamed or dropped a test simply
    has nothing there to ignore.

    Raises:
        TestListError: if the selection names a path outside tests/, or if the
            checkout has none of the tests it names. Silently running the whole
            suite, or none of it, would both pass for a subset.
    """
    tests_dir = (jax_dir / RELATIVE_TESTS_DIR).resolve()
    wanted = set()
    for path in selected:
        # Resolved, because "tests/../benchmarks/x_test.py" is under tests/ by
        # spelling only, and a symlink leading out of it is not under it at all.
        outside = not (jax_dir / path).resolve().is_relative_to(tests_dir)
        if path.is_absolute() or outside:
            raise TestListError(f"{path} is not under {RELATIVE_TESTS_DIR}/")
        wanted.add(jax_dir / path)

    missing = sorted(path for path in wanted if not path.exists())
    if missing:
        log(
            f"::warning::{len(missing)} selected test(s) are not in this JAX"
            f" checkout and will not run: {', '.join(p.name for p in missing)}"
        )
    if len(missing) == len(wanted):
        raise TestListError(
            f"none of the {len(wanted)} selected tests exist under"
            f" {jax_dir / RELATIVE_TESTS_DIR}"
        )

    ignored: list[Path] = []
    directories = [jax_dir / RELATIVE_TESTS_DIR]
    while directories:
        for child in sorted(directories.pop().iterdir()):
            if child in wanted or not _collects_tests(child):
                continue
            if child.is_dir() and any(path.is_relative_to(child) for path in wanted):
                directories.append(child)
            else:
                ignored.append(child)

    log(f"=== Running {len(wanted) - len(missing)} of the suite's tests")
    return [f"--ignore={path.relative_to(jax_dir).as_posix()}" for path in ignored]


def pytest_addopts(
    jax_version: str,
    amdgpu_family: str,
    in_process_reruns: int,
    debug: bool = False,
    ignore: list[str] | None = None,
) -> str:
    """The PYTEST_ADDOPTS value for one configuration."""
    opts = list(ignore or [])
    if in_process_reruns:
        # Crashed workers are retried regardless of --only-rerun. Everything else
        # is left to the fresh-process pass.
        opts += [
            "--reruns",
            str(in_process_reruns),
            "--reruns-delay",
            "5",
            "--only-rerun",
            "INTERNAL",
        ]

    expression = keyword_expression(jax_version, amdgpu_family, debug)
    if expression:
        opts += ["-k", expression]

    return shlex.join(opts)


class SuiteEnvironmentError(Exception):
    """The environment the suite runs under could not be read."""


def env_section(script: str) -> str:
    """The part of the suite script that only computes the environment."""
    lines = script.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == ENV_SECTION_START]
    ends = [i for i, line in enumerate(lines) if line.strip() == ENV_SECTION_END]
    if not starts or not ends or ends[-1] <= starts[0]:
        return ""
    return "\n".join(lines[starts[0] + 1 : ends[-1]])


def _read_env_dump(path: Path) -> dict[str, str]:
    entries = path.read_bytes().decode(errors="replace").split("\0")
    pairs = (entry.partition("=") for entry in entries)
    return {name: value for name, separator, value in pairs if separator}


def suite_env_program(jax_dir: Path) -> tuple[str, Path]:
    """The shell that computes the suite's environment, and the file it came from.

    A checkout from before the environment moved out of the suite script keeps it
    in a marked section there, which is what gets evaluated instead. The caller
    reports failures against the file this picked, since either can be the one at
    fault.

    Raises:
        SuiteEnvironmentError: if the checkout holds neither.
    """
    if (jax_dir / RELATIVE_SUITE_ENV_SCRIPT).exists():
        program = f"source {shlex.quote(os.fspath(RELATIVE_SUITE_ENV_SCRIPT))}"
        return program, RELATIVE_SUITE_ENV_SCRIPT

    older = jax_dir / RELATIVE_SUITE_SCRIPT
    section = env_section(older.read_text()) if older.exists() else ""
    if not section:
        raise SuiteEnvironmentError(
            f"{jax_dir / RELATIVE_SUITE_ENV_SCRIPT} does not exist and"
            f" {RELATIVE_SUITE_SCRIPT} has no section between"
            f" '{ENV_SECTION_START}' and '{ENV_SECTION_END}', so there is nothing"
            " to read the suite's environment from. Update"
            " RELATIVE_SUITE_ENV_SCRIPT, or ENV_SECTION_START and"
            " ENV_SECTION_END, to match what the checkout has."
        )
    return section, RELATIVE_SUITE_SCRIPT


def suite_environment(jax_dir: Path, env: dict[str, str]) -> dict[str, str]:
    """What the suite exports before it runs anything.

    Evaluated in a shell that dumps its environment on either side, so the result
    is what the suite itself changed rather than a guess at which names look
    interesting.

    Raises:
        SuiteEnvironmentError: if it cannot be found or evaluated. This fails the
            run by design, because the retry pass decides the result and a wrong
            environment there would decide it wrongly.
    """
    program, source = suite_env_program(jax_dir)
    if not (jax_dir / RELATIVE_SUITE_ENV_FILE).exists():
        raise SuiteEnvironmentError(
            f"{RELATIVE_SUITE_ENV_FILE} is missing, and {source} reads its defaults."
            f" Is {jax_dir} a complete ROCm/jax checkout?"
        )

    with tempfile.TemporaryDirectory() as tmp:
        before = Path(tmp) / "before"
        after = Path(tmp) / "after"
        dump = "env -0 > {}".format
        # The program's own status, kept across the dump that follows it, which
        # would otherwise be the status of the shell and always zero.
        script = "\n".join(
            [
                f"source {shlex.quote(os.fspath(RELATIVE_SUITE_ENV_FILE))}",
                dump(shlex.quote(os.fspath(before))),
                program,
                "program_status=$?",
                dump(shlex.quote(os.fspath(after))),
                "exit ${program_status}",
            ]
        )
        proc = subprocess.run(
            ["bash", "-c", script],
            cwd=jax_dir,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if not (before.exists() and after.exists()):
            raise SuiteEnvironmentError(
                f"Evaluating the environment {source} sets produced none:"
                f" {proc.stderr.strip()}"
            )
        if proc.returncode != 0:
            # It exports and queries devices, so a failure means some of what it
            # exported was computed from a command that did not work.
            raise SuiteEnvironmentError(
                f"The environment {source} sets exited {proc.returncode}, so what"
                f" it exported is only partly what the suite will run under:"
                f" {proc.stderr.strip()}"
            )
        baseline, exported = _read_env_dump(before), _read_env_dump(after)

    return {
        name: value
        for name, value in exported.items()
        if baseline.get(name) != value
        and name.isupper()
        and name not in SUITE_ENV_EXCLUDES
    }


def available_cpus() -> int:
    """The CPUs this process may run on, which is the node's on a CI runner."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def requested_cpus(env: dict[str, str], available: int) -> int | None:
    """The cores CPU_REQUEST_VAR names, or None if it names none usable.

    Kubernetes writes a CPU request in cores or in millicores, and both
    spellings have been seen on these runners: 25, 7.5 and 25000m are all the
    same order of magnitude. A bare number larger than the machine has can only
    be millicores.
    """
    value = env.get(CPU_REQUEST_VAR, "").strip()
    try:
        cores = float(value.removesuffix("m")) / (1000 if value.endswith("m") else 1)
    except ValueError:
        return None
    if cores > available and cores / 1000 >= 1:
        cores /= 1000
    if cores < 1:
        # Half a core still gets one, but nothing at all means the value was
        # never a CPU request.
        return 1 if cores > 0 else None
    return min(int(cores), available)


def limit_to_cpus(cores: int) -> list[int]:
    """Pins this process, and everything it starts, to that many CPUs.

    The thread pools underneath JAX size themselves from the CPUs they can see
    rather than from what the pod was given. Measured on a gfx942 runner, one
    worker holds 1021 threads, 256 of them XLA's Eigen pool; pinned to the 25
    cores the pod has, it holds 179. Sixteen workers of the first kind is
    16,000 threads on a box that owes this job 25 cores.
    """
    chosen = sorted(os.sched_getaffinity(0))[:cores]
    os.sched_setaffinity(0, set(chosen))
    return chosen


def openblas_threads(cpus: int) -> int:
    """The OpenBLAS threads one worker gets out of a pod with that many cores.

    numpy and scipy each bundle one, and it is the only threaded library the
    suite pulls in that reads an environment variable. Pinning alone would
    leave every worker sizing its pool to the whole allocation.
    """
    return max(1, cpus // SUITE_MAX_WORKERS)


def log_pod_resources(env: dict[str, str]) -> None:
    """What the pod was given, next to what the process can see."""
    log("=== Resources")
    for name in (CPU_REQUEST_VAR, MEMORY_REQUEST_VAR):
        log(f"  {name}={env.get(name, 'unset')}")
    log(f"  cpus visible to this process={available_cpus()}")
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        total = meminfo.read_text().partition("\n")[0]
        log(f"  {total} (the node's, whatever this container was given)")


def size_to_pod(env: dict[str, str], override: int | None) -> int | None:
    """Holds the run to the pod's CPUs, and reports what it was given.

    Returns the cores to size thread pools from, or None when the allocation is
    unknown: then everything underneath goes on sizing itself from the node, as
    it did before any of this.
    """
    log_pod_resources(env)

    cpus = override or requested_cpus(env, available_cpus())
    if not cpus:
        log(
            f"::warning::{CPU_REQUEST_VAR} is unset or holds no usable core"
            " count, so the thread pools will size themselves from the whole"
            " node. Pass --cpus to say what this machine may use."
        )
        return None

    if not hasattr(os, "sched_setaffinity"):
        log(f"  {platform.system()} cannot pin CPUs, so only the pools are sized")
        return cpus

    pinned = limit_to_cpus(cpus)
    log(f"  pinned to {len(pinned)} cpus: {pinned[0]}-{pinned[-1]}")
    return cpus


def test_environment(
    suite_env: dict[str, str],
    jax_version: str,
    amdgpu_family: str,
    in_process_reruns: int,
    debug: bool = False,
    cpus: int | None = None,
    ignore: list[str] | None = None,
) -> dict[str, str]:
    """The environment variables the suite and the retry pass need."""
    env = {**suite_env, **THEROCK_ENV}
    if cpus:
        env["OPENBLAS_NUM_THREADS"] = str(openblas_threads(cpus))
    # Left alone when there is nothing to add, so that --no-retries does not wipe
    # options the caller set.
    addopts = pytest_addopts(
        jax_version, amdgpu_family, in_process_reruns, debug, ignore
    )
    if addopts:
        env["PYTEST_ADDOPTS"] = addopts
    return env


def preflight_command() -> list[str]:
    """Reports the devices JAX sees, in a process that exits before the suite.

    A stack that imports but finds no device fails here in one line instead of as
    thousands of test errors.
    """
    return [sys.executable, "-c", "import jax; print(jax.local_devices())"]


def retry_command(nodeids: list[str]) -> list[str]:
    """The fresh-process pytest command for the tests that failed."""
    return [sys.executable, "-m", "pytest", "-n", "0", "--tb=short", *nodeids]


def log(*args) -> None:
    print(*args)
    sys.stdout.flush()


def run_command(args: list[str], cwd: Path, env: dict[str, str]) -> int:
    log(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    return subprocess.run(args, cwd=cwd, env=env).returncode


@dataclasses.dataclass(frozen=True)
class SuiteReports:
    """What the JSON reports say about the pytest sessions the suite ran."""

    failed: list[str]
    # Whether those sessions covered the whole suite. Node ids alone cannot tell
    # a session that ran every test and failed some from one that stopped
    # partway: both leave a report full of results. Retrying the failures of the
    # second kind would pass while the tests it never reached stay unrun.
    complete: bool


def remove_reports(jax_dir: Path) -> None:
    """Drops reports from an earlier run, which this one would read as its own."""
    for report in sorted(jax_dir.glob(REPORT_GLOB)):
        log(f"Removing the report an earlier run left behind: {report}")
        report.unlink()


def collect_reports(jax_dir: Path) -> SuiteReports:
    """The failures the suite reported, and whether it got through the suite.

    The top-level exit code is what says the latter, so a report missing it, or
    one that cannot be read at all, counts against the run rather than being
    skipped over.
    """
    reports = sorted(jax_dir.glob(REPORT_GLOB))
    if not reports:
        log(f"{jax_dir / REPORT_GLOB}: no reports found")
        return SuiteReports([], False)

    failed: list[str] = []
    exitcodes: list[int] = []
    for report in reports:
        try:
            parsed = json.loads(report.read_text())
            exitcode = parsed["exitcode"]
        except (OSError, ValueError, KeyError) as e:
            log(f"::warning::{report}: no pytest exit code to read ({e!r})")
            return SuiteReports(failed, False)
        log(f"{report}: pytest exited {exitcode}")
        failed += failed_in_report(parsed)
        exitcodes.append(exitcode)

    finished = (PYTEST_EXIT_ALL_PASSED, PYTEST_EXIT_TESTS_FAILED)
    early = [code for code in exitcodes if code not in finished]
    if early:
        log(f"::warning::pytest exited {early}, so it stopped before the end")
        return SuiteReports(failed, False)
    if PYTEST_EXIT_TESTS_FAILED not in exitcodes:
        log(
            "::warning::no session reported a test failure, so the suite failed"
            " for a reason its reports do not hold"
        )
        return SuiteReports(failed, False)
    return SuiteReports(failed, True)


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_jax_tests.py")
    p.add_argument(
        "--jax-dir",
        type=Path,
        default=Path(os.getenv("JAX_DIR", "jax")),
        help="ROCm/jax checkout holding the test suite (default: jax)",
    )
    p.add_argument(
        "--jax-version",
        default=os.getenv("JAX_VERSION", ""),
        help="JAX version under test (e.g. 0.10.2); derived from --jax-ref if unset",
    )
    p.add_argument(
        "--jax-ref",
        default=os.getenv("JAX_REF", ""),
        help=f"rocm-jax ref under test (e.g. {JAX_REF_PREFIX}0.11.0)",
    )
    p.add_argument(
        "--amdgpu-family",
        default=os.getenv("AMDGPU_FAMILY", ""),
        help="GPU family under test (e.g. gfx94X-dcgpu), selects skip lists",
    )
    p.add_argument(
        "--test-subset",
        choices=sorted(SUITE_SCRIPTS),
        default=os.getenv("JAX_TEST_SUBSET", TEST_SUBSET_ALL),
        help="Accelerator subset to run; 'multi' needs more than one GPU",
    )
    p.add_argument(
        "--test-list",
        type=Path,
        default=(
            Path(os.environ["JAX_TEST_LIST"]) if os.getenv("JAX_TEST_LIST") else None
        ),
        help="File of tests/ paths, one per line, to run instead of the whole suite",
    )
    p.add_argument(
        "--retries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="--no-retries turns off both retry layers, running only the suite",
    )
    p.add_argument(
        "--in-process-reruns",
        type=int,
        default=int(os.getenv("JAX_NUM_PYTEST_RERUNS", "2")),
        help="pytest-rerunfailures attempts within the same process (0 disables)",
    )
    p.add_argument(
        "--fresh-process-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry reported failures in a new process, which decides the result",
    )
    p.add_argument(
        "--max-retry-tests",
        type=int,
        default=40,
        help="Skip the fresh-process retry above this many failures (0 disables)",
    )
    p.add_argument(
        "--cpus",
        type=int,
        default=None,
        help=f"Cores this run may use (default: what {CPU_REQUEST_VAR} says)",
    )
    p.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run only the tests the skip lists would skip",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the environment and commands without running them",
    )
    args = p.parse_args(argv)

    if not args.retries:
        args.in_process_reruns = 0
        args.fresh_process_retry = False

    return args


def main(argv: list[str]) -> int:
    args = cmd_arguments(argv)

    jax_version = args.jax_version or jax_version_from_ref(args.jax_ref)

    jax_dir = args.jax_dir.resolve()
    suite_script = SUITE_SCRIPTS[args.test_subset]
    suite = jax_dir / suite_script
    if not suite.exists():
        if args.test_subset != TEST_SUBSET_ALL:
            # Whether it would have run is worth knowing before a runner is
            # taken for it, so this holds for a dry run too.
            log(
                f"::error::{suite} not found, so this jax ref cannot run the"
                f" '{args.test_subset}' subset on its own. Use a ref that has"
                " that script, or --test-subset all."
            )
            return 1
        if not args.dry_run:
            log(f"::error::{suite} not found. Is --jax-dir a ROCm/jax checkout?")
            return 1

    env = dict(os.environ)
    for name in UNSET_VARS:
        env.pop(name, None)

    cpus = size_to_pod(env, args.cpus)

    try:
        suite_env = suite_environment(jax_dir, env) if suite.exists() else {}
        ignore = (
            ignore_arguments(jax_dir, read_test_list(args.test_list))
            if args.test_list
            else []
        )
    except (SuiteEnvironmentError, TestListError, OSError) as e:
        log(f"::error::{e}")
        return 1
    overrides = test_environment(
        suite_env,
        jax_version,
        args.amdgpu_family,
        args.in_process_reruns,
        args.debug,
        cpus,
        ignore,
    )
    env.update(overrides)

    log(f"=== Testing JAX {jax_version or '(unknown version)'} in {jax_dir}")
    log(f"  host={platform.node()}")
    log(f"  subset={args.test_subset}")
    for name in GPU_VISIBILITY_ENV_VARS:
        log(f"  {name}={env.get(name, 'unset')}")
    for name, value in sorted(overrides.items()):
        log(f"  {name}={value}")

    # The suite script fails without this directory.
    if not args.dry_run:
        (jax_dir / "dist").mkdir(parents=True, exist_ok=True)

    suite_command = ["bash", os.fspath(suite_script)]
    if args.dry_run:
        log(f"++ Would exec [{jax_dir}]$ {shlex.join(preflight_command())}")
        log(f"++ Would exec [{jax_dir}]$ {shlex.join(suite_command)}")
        return 0

    log("=== Devices JAX sees")
    returncode = run_command(preflight_command(), jax_dir, env)
    if returncode != 0:
        log(
            "::error::JAX cannot see its devices, so the suite would only"
            " report that. See the traceback above."
        )
        return returncode

    log("=== Running the test suite")
    # Only this run's reports may decide this run's result.
    remove_reports(jax_dir)
    returncode = run_command(suite_command, jax_dir, env)
    if returncode == 0:
        return 0

    if not args.fresh_process_retry:
        return returncode

    reports = collect_reports(jax_dir)
    if not reports.complete:
        log(
            "The suite did not get through its tests, so retrying the failures"
            " it did report would say nothing about the rest."
        )
        return returncode

    failed = reports.failed
    if not failed:
        log(
            "The suite failed without reporting a failed test, so there is"
            " nothing to retry."
        )
        return returncode

    # The retry pass is serial, and a configuration this broken is something
    # other than a few poisoned workers.
    if args.max_retry_tests and len(failed) > args.max_retry_tests:
        log(
            f"::error::{len(failed)} failures is more than --max-retry-tests"
            f" {args.max_retry_tests}, so nothing will be retried"
        )
        return returncode

    # This pass decides the result: a sticky fault only clears in a new process.
    # It runs under the suite's own environment, read above, so it takes the same
    # path as the run it is checking.
    log(f"=== Retrying {len(failed)} failure(s) in a fresh process")
    return run_command(retry_command(failed), jax_dir, env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
