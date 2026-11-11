#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Retain debugger and inferior state after the bounded GPU startup test fails."""

import argparse
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time


def capture_startup(
    gdb,
    inferior,
    output,
    *,
    timeout=20,
    env=None,
    break_main=True,
    inferior_args=(),
    capture_timeout=45,
):
    output.mkdir(parents=True, exist_ok=True)
    events = queue.Queue()
    started = time.monotonic()
    process = subprocess.Popen(
        [
            str(gdb),
            "-nx",
            "-q",
            "--interpreter=mi2",
            "--args",
            str(inferior),
            *map(str, inferior_args),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
    )
    inferior_pid = None
    stopped = None
    interrupted = False
    forced_cleanup = False

    def read_output():
        with (output / "startup-mi.log").open("w") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                events.put(line)
        events.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def send(commands):
        if process.poll() is None:
            try:
                process.stdin.write(commands)
                process.stdin.flush()
            except BrokenPipeError:
                pass  # Preserve the output of an already exiting debugger.

    def collect(seconds, *, until_stop=False):
        nonlocal inferior_pid, stopped
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                line = events.get(timeout=min(0.1, max(0.001, end - time.monotonic())))
            except queue.Empty:
                if process.poll() is not None:
                    return
                continue
            if line is None:
                return
            if match := re.search(r'=thread-group-started,.*\bpid="(\d+)"', line):
                inferior_pid = int(match.group(1))
            if line.startswith("*stopped"):
                stopped = line.strip()
                if until_stop:
                    return

    try:
        send("1-gdb-set pagination off\n2-gdb-set confirm off\n3-gdb-set mi-async on\n")
        if break_main:
            send("4-break-insert main\n")
        send("5-exec-run\n")
        collect(timeout, until_stop=True)
        if stopped is None and process.poll() is None:
            interrupted = True
            send("6-exec-interrupt --all\n")
            collect(5, until_stop=True)
            if stopped is None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                collect(5, until_stop=True)
        send(
            '7-interpreter-exec console "info program"\n'
            '8-interpreter-exec console "info threads"\n'
            '9-interpreter-exec console "thread 1"\n'
            '10-interpreter-exec console "bt 30"\n'
            '11-interpreter-exec console "thread apply all bt 30"\n'
            '12-interpreter-exec console "info sharedlibrary"\n'
            "13-gdb-exit\n"
        )
        collect(capture_timeout)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            forced_cleanup = True
    finally:
        if process.poll() is None:
            forced_cleanup = True
            # Only kill the child belonging to this newly created debugger.
            # An exited/reused PID must never identify an unrelated process.
            if inferior_pid:
                try:
                    status = Path(f"/proc/{inferior_pid}/status").read_text()
                    parent = int(re.search(r"^PPid:\s+(\d+)", status, re.MULTILINE)[1])
                    executable = Path(f"/proc/{inferior_pid}/exe").resolve(strict=True)
                    if parent == process.pid and executable == inferior.resolve():
                        os.kill(inferior_pid, signal.SIGKILL)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        reader.join(timeout=5)
    result = {
        "gdb": str(gdb),
        "inferior": str(inferior),
        "inferior_arguments": list(inferior_args),
        "elapsed_seconds": time.monotonic() - started,
        "stop_event": stopped,
        "main_breakpoint_requested": break_main,
        "reached_main": (
            bool(stopped and 'reason="breakpoint-hit"' in stopped)
            if break_main
            else None
        ),
        "interrupt_attempted": interrupted,
        "forced_cleanup": forced_cleanup,
        "gdb_exit_code": process.returncode,
    }
    (output / "startup-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def gpu_test_command(artifacts):
    return [
        sys.executable,
        str(artifacts / "tests/rocgdb/test_rocgdb.py"),
        "--one-by-one",
        "--one-by-one-test-timeout",
        "300",
        "--max-failed-retries",
        "0",
        "--no-xfail",
        "--toolchain",
        "llvm",
        "--tests",
        "gdb.rocm/simple.exp",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_ARTIFACTS_DIR", "build")),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("test-diagnostics/rocgdb-gpu-startup")
    )
    parser.add_argument("--capture-inferior", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=20)
    args = parser.parse_args(argv)
    artifacts = args.artifacts_dir.resolve()
    gdb = artifacts / "bin/rocgdb"
    if args.capture_inferior:
        result = capture_startup(
            gdb,
            args.capture_inferior.resolve(),
            args.output,
            timeout=args.startup_timeout,
        )
        return int(not result["stop_event"] or result["forced_cleanup"])

    suite = artifacts / "tests/rocgdb/gdb/testsuite"
    result = subprocess.run(gpu_test_command(artifacts), check=False)
    if result.returncode:
        log = suite / "gdb.log"
        if log.is_file():
            matches = re.findall(
                r"^Starting program: (.+?)\s*$",
                log.read_text(errors="replace"),
                re.MULTILINE,
            )
            if matches:
                inferior = Path(matches[-1]).resolve()
                if inferior.is_file() and suite.resolve() in inferior.parents:
                    capture_startup(
                        gdb, inferior, args.output, timeout=args.startup_timeout
                    )
                else:
                    print(
                        "Refusing a diagnostic executable outside the installed test suite",
                        flush=True,
                    )
        # Diagnostic collection never converts the failed test to success.
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    sys.exit(main())
