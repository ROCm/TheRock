#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs ROCm/jax-lab end-to-end workloads against the installed JAX ROCm wheels.

jax-lab stays the source of truth for how a workload runs: its
targets/<target>/run.sh prepares the workload, runs it and writes its
result.json. This adds only what is ours:

  * the workloads to run, from a list in test_selection/ (see its README);
  * JAXLIB_VERSION=preinstalled, so jax-lab runs the wheels under test instead
    of installing its own, and a pip constraint file pinning them, so that a
    workload's own requirements cannot replace them either;
  * the GPU count jax-lab's CI gives each workload, since a workload spreads
    over every device it can see and its batch sizes are set for that count;
  * a timeout per workload, so that one hang does not cost the others their
    result.

Every workload runs, and the run fails if any of them did. Each result.json is
copied to --results-dir/<target>/<workload>/.

Example usage:

    python run_jax_e2e.py --jax-lab-dir jax-lab \\
        --workload-list external-builds/jax/test_selection/e2e_medium.txt \\
        --runner linux-gfx942-8gpu-ossci-rocm --rocm-version 10.2.0a20261016

    # Print each workload's command and environment without running it.
    python run_jax_e2e.py --jax-lab-dir jax-lab \\
        --workload-list external-builds/jax/test_selection/e2e_medium.txt \\
        --runner local --rocm-version 10.2.0a20261016 --dry-run
"""

import argparse
import importlib.metadata
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# What jax-lab's own CI gives each workload.
WORKLOAD_TIMEOUT_MINUTES = 60

# The wheels under test, by normalized distribution name. The plugin and PJRT
# names carry the ROCm major they were built for.
JAX_STACK_PATTERNS = {
    "jax": r"jax",
    "jaxlib": r"jaxlib",
    "plugin": r"jax-rocm\d+-plugin",
    "pjrt": r"jax-rocm\d+-pjrt",
}


@dataclass(frozen=True)
class Workload:
    target: str
    name: str
    gpus: int

    @property
    def label(self) -> str:
        return f"{self.target}/{self.name}"


def read_workload_list(path: Path) -> list[Workload]:
    """The workloads in a list file, one "<target> <workload> <gpus>" per line."""
    workloads = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        if len(fields) != 3 or not fields[2].isdigit() or int(fields[2]) < 1:
            raise ValueError(
                f"{path}:{number}: expected '<target> <workload> <gpus>', got {line!r}"
            )
        workloads.append(Workload(fields[0], fields[1], int(fields[2])))
    if not workloads:
        raise ValueError(f"{path} names no workloads")
    return workloads


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_jax_stack() -> dict[str, str]:
    """Name to version for each installed wheel under test.

    Raises if jax, jaxlib or the plugin is missing, since jax-lab would then
    run whatever it found, or nothing.
    """
    stack = {}
    for dist in importlib.metadata.distributions():
        name = _normalize(dist.metadata["Name"] or "")
        if any(re.fullmatch(p, name) for p in JAX_STACK_PATTERNS.values()):
            stack[name] = dist.version
    found = {
        role
        for role, pattern in JAX_STACK_PATTERNS.items()
        if any(re.fullmatch(pattern, name) for name in stack)
    }
    missing = sorted({"jax", "jaxlib", "plugin"} - found)
    if missing:
        raise RuntimeError(f"JAX wheels under test not installed: {missing}")
    return stack


def constraints(stack: dict[str, str]) -> str:
    return "".join(f"{name}=={version}\n" for name, version in sorted(stack.items()))


def visible_devices(env: dict[str, str], count: int) -> str:
    """HIP_VISIBLE_DEVICES for a workload of count GPUs.

    Takes the first count of the devices already visible, so that it stays
    inside whatever the runner's isolation allows.
    """
    current = env.get("HIP_VISIBLE_DEVICES", "")
    devices = current.split(",") if current else [str(i) for i in range(count)]
    if len(devices) < count:
        raise ValueError(
            f"needs {count} GPUs but HIP_VISIBLE_DEVICES={current!r} has {len(devices)}"
        )
    return ",".join(devices[:count])


def rocm_sdk_root() -> Path:
    out = subprocess.run(
        ["rocm-sdk", "path", "--root"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip())


def workload_env(
    base: dict[str, str],
    workload: Workload,
    *,
    runner: str,
    rocm_version: str,
    constraint_file: Path,
    sdk_root: Path,
) -> dict[str, str]:
    """The environment jax-lab's run.sh expects, as its own CI sets it."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    env = dict(base)
    env.update(
        {
            "JAXLIB_VERSION": "preinstalled",
            "PIP_CONSTRAINT": os.fspath(constraint_file),
            "PYTHON": sys.executable,
            "PYTHON_VERSION": python_version,
            "COMBO": f"{workload.gpus}gpu-{workload.name}-{workload.target}-py{python_version}",
            "RUNNER": runner,
            "ROCM_VERSION": rocm_version,
            "ROCM_TAG": f"therock-{rocm_version}",
            "HIP_VISIBLE_DEVICES": visible_devices(base, workload.gpus),
            # The whole log rather than jax-lab's two-line tail on failure.
            "JAX_LAB_VERBOSE": "1",
            "PATH": f"{sdk_root / 'bin'}:{base.get('PATH', '')}",
            "LD_LIBRARY_PATH": ":".join(
                p
                for p in [os.fspath(sdk_root / "lib"), base.get("LD_LIBRARY_PATH")]
                if p
            ),
        }
    )
    return env


def workload_command(workload: Workload) -> list[str]:
    return ["bash", f"targets/{workload.target}/run.sh", workload.name]


def run_workload(
    jax_lab_dir: Path, workload: Workload, env: dict[str, str], timeout_minutes: int
) -> str:
    """Runs one workload and returns "passed", "failed" or "timed out"."""
    proc = subprocess.Popen(
        workload_command(workload), cwd=jax_lab_dir, env=env, start_new_session=True
    )
    try:
        returncode = proc.wait(timeout=timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        # A hung rank holds the GPUs, so take the whole process group down.
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return "timed out"
    return "passed" if returncode == 0 else "failed"


def collect_result(jax_lab_dir: Path, workload: Workload, results_dir: Path) -> bool:
    result = (
        jax_lab_dir / "targets" / workload.target / "run_artifacts" / workload.name
    ) / "result.json"
    if not result.is_file():
        return False
    dest = results_dir / workload.target / workload.name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result, dest / "result.json")
    return True


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run_jax_e2e.py", description=__doc__)
    p.add_argument(
        "--jax-lab-dir", type=Path, required=True, help="ROCm/jax-lab checkout"
    )
    p.add_argument(
        "--workload-list",
        type=Path,
        required=True,
        help="File of '<target> <workload> <gpus>' lines, e.g. test_selection/e2e_medium.txt",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=Path("e2e-results"),
        help="Where each workload's result.json is copied (default: e2e-results)",
    )
    p.add_argument(
        "--runner", required=True, help="Runner label, recorded in each result.json"
    )
    p.add_argument(
        "--rocm-version", required=True, help="ROCm version the wheels run against"
    )
    p.add_argument(
        "--timeout-minutes",
        type=int,
        default=WORKLOAD_TIMEOUT_MINUTES,
        help=f"Budget for each workload (default: {WORKLOAD_TIMEOUT_MINUTES})",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the environment and commands without running them",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = cmd_arguments(argv)
    jax_lab_dir = args.jax_lab_dir.resolve()
    workloads = read_workload_list(args.workload_list)
    for workload in workloads:
        run_sh = jax_lab_dir / workload_command(workload)[1]
        if not run_sh.is_file():
            raise FileNotFoundError(f"{workload.label}: no {run_sh}")

    stack = installed_jax_stack()
    print(f"JAX wheels under test: {stack}")
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    constraint_file = results_dir / "constraints.txt"
    constraint_file.write_text(constraints(stack))
    sdk_root = rocm_sdk_root()

    outcomes = {}
    for workload in workloads:
        env = workload_env(
            dict(os.environ),
            workload,
            runner=args.runner,
            rocm_version=args.rocm_version,
            constraint_file=constraint_file,
            sdk_root=sdk_root,
        )
        command = workload_command(workload)
        if args.dry_run:
            print(f"++ Would exec in {jax_lab_dir}: {' '.join(command)}")
            for key in ["COMBO", "HIP_VISIBLE_DEVICES", "JAXLIB_VERSION", "ROCM_TAG"]:
                print(f"     {key}={env[key]}")
            continue

        print(f"::group::{workload.label} on {workload.gpus} GPUs", flush=True)
        outcome = run_workload(jax_lab_dir, workload, env, args.timeout_minutes)
        print("::endgroup::", flush=True)
        if (
            not collect_result(jax_lab_dir, workload, results_dir)
            and outcome == "passed"
        ):
            outcome = "failed"
            print(f"::error::{workload.label} passed but wrote no result.json")
        outcomes[workload.label] = outcome
        if installed_jax_stack() != stack:
            # Every later workload would test something else.
            raise RuntimeError(
                f"{workload.label} replaced the JAX wheels under test:"
                f" {stack} -> {installed_jax_stack()}"
            )

    for label, outcome in outcomes.items():
        print(f"{outcome:>9}  {label}")
    failed = [label for label, outcome in outcomes.items() if outcome != "passed"]
    if failed:
        print(f"::error::{len(failed)} of {len(outcomes)} workloads failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
