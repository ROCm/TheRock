# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a configured shard Test command, retrying only failed attempts."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

from github_actions_api import gha_append_step_summary, gha_set_output


@dataclass
class Attempt:
    number: int
    exit_code: int
    duration_seconds: float
    timed_out: bool


def nonnegative_integer(value: str) -> int:
    try:
        number = float(value)
        if not number.is_integer() or number < 0:
            raise ValueError
        return int(number)
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("max_retries must be a nonnegative integer")


def stop_process_tree(process: subprocess.Popen) -> None:
    """Terminate descendants so another attempt cannot overlap timed-out tests."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if process.poll() is None:
                process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_attempt(
    script: Path, timeout_seconds: float, log_path: Path
) -> tuple[int, bool]:
    env = os.environ.copy()
    # A runner-level retry wrapper must not retry this helper or its child shell.
    env["RETRY_THIS_STEP"] = "false"
    env["RETRY_COUNT"] = "0"
    process = subprocess.Popen(
        ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", script.as_posix()],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        start_new_session=os.name != "nt",
    )

    def stream_log() -> None:
        with log_path.open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

    reader = threading.Thread(target=stream_log, daemon=True)
    reader.start()
    timed_out = False
    try:
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
    finally:
        stop_process_tree(process)
        reader.join(timeout=5)
        if not reader.is_alive():
            process.stdout.close()
    return (124 if timed_out else process.returncode), timed_out


def retry_command(
    script_text: str,
    *,
    max_retries: int,
    timeout_seconds: float,
    retry_delay_seconds: float,
    logs_dir: Path,
) -> list[Attempt]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    with tempfile.TemporaryDirectory(prefix="therock-test-retry-") as temp_dir:
        script = Path(temp_dir) / "test.sh"
        script.write_text(script_text, encoding="utf-8")
        for number in range(1, max_retries + 2):
            print(
                f"::group::Test command attempt {number}/{max_retries + 1}", flush=True
            )
            started = time.monotonic()
            try:
                exit_code, timed_out = run_attempt(
                    script, timeout_seconds, logs_dir / f"attempt-{number}.log"
                )
            finally:
                print("::endgroup::", flush=True)
            attempt = Attempt(number, exit_code, time.monotonic() - started, timed_out)
            attempts.append(attempt)
            (logs_dir / "attempts.json").write_text(
                json.dumps([asdict(item) for item in attempts], indent=2) + "\n",
                encoding="utf-8",
            )
            outcome = (
                "timed out" if timed_out else "passed" if exit_code == 0 else "failed"
            )
            gha_append_step_summary(
                f"- Test command attempt {number}/{max_retries + 1}: **{outcome}** "
                f"(exit {exit_code}, {attempt.duration_seconds:.1f}s).\n"
            )
            if exit_code == 0:
                break
            if number <= max_retries:
                print(
                    f"Test command failed; retrying in {retry_delay_seconds:g}s.",
                    flush=True,
                )
                time.sleep(retry_delay_seconds)
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-retries", required=True, type=nonnegative_integer)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    parser.add_argument("--timeout-minutes", type=float)
    parser.add_argument("--retry-delay-seconds", type=float, default=7)
    parser.add_argument("--logs-dir", type=Path, default=Path(".test-retry-logs"))
    args = parser.parse_args()
    if args.validate_only:
        return 0
    if args.timeout_minutes is None or not 0 < args.timeout_minutes < float("inf"):
        parser.error("--timeout-minutes must be finite and positive")
    if not 0 <= args.retry_delay_seconds < float("inf"):
        parser.error("--retry-delay-seconds must be finite and nonnegative")
    if args.configure_only:
        requested_minutes = (
            args.timeout_minutes * (args.max_retries + 1)
            + args.retry_delay_seconds * args.max_retries / 60
            + 1
        )
        # Reserve 75 minutes of the six-hour job limit for setup and cleanup.
        step_minutes = math.ceil(min(requested_minutes, 285))
        gha_set_output({"step_timeout_minutes": str(step_minutes)})
        if requested_minutes > step_minutes:
            print(
                "::warning::The Test step budget is capped at 285 minutes; "
                "the overall timeout may stop attempts before the retry limit is reached."
            )
        return 0
    script = os.environ.get("TEST_SCRIPT", "")
    if not script.strip():
        parser.error("TEST_SCRIPT must contain the configured component command")

    def cancel(signum: int, frame: object) -> None:
        # Cancellation is not a retryable failure; unwind process cleanup first.
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        attempts = retry_command(
            script,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_minutes * 60,
            retry_delay_seconds=args.retry_delay_seconds,
            logs_dir=args.logs_dir,
        )
    except KeyboardInterrupt:
        gha_append_step_summary("- Test command retry loop was cancelled.\n")
        return 130
    return 0 if attempts[-1].exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
