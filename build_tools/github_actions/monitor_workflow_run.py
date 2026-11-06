#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Monitor one Actions run and rerun failed jobs a bounded number of times."""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from collections.abc import Callable, Mapping
from typing import Protocol
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class MonitorError(RuntimeError):
    pass


class UncertainRerun(MonitorError):
    """A request may have reached GitHub; reconcile instead of repeating it."""


# GitHub asynchronously creates a new attempt after accepting a rerun request.
RERUN_REGISTRATION_SECONDS = 300
RERUN_POLL_SECONDS = 10


class RunAPI(Protocol):
    def request(self, method: str = "GET") -> dict[str, object]: ...

    def pause(self, seconds: float) -> None: ...


def rate_limit_delay(headers: Mapping[str, str], body: str) -> float | None:
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            return max(1, float(retry_after))
        except ValueError:
            try:
                return max(
                    1,
                    (
                        parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)
                    ).total_seconds(),
                )
            except (TypeError, ValueError):
                return 60
    if headers.get("X-RateLimit-Remaining") == "0":
        reset = headers.get("X-RateLimit-Reset")
        if reset:
            try:
                return max(1, float(reset) - time.time() + 5)
            except ValueError:
                return 60
    if "rate limit" in body.lower() or "secondary rate" in body.lower():
        return 60
    return None


class GitHub:
    def __init__(
        self, repository: str, run_id: str, token: str, deadline: float
    ) -> None:
        self.url = f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
        self.token = token
        self.deadline = deadline

    def pause(self, seconds: float) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise MonitorError(
                "Monitoring deadline reached; downstream may still be running"
            )
        time.sleep(min(seconds, remaining))
        if time.monotonic() >= self.deadline:
            raise MonitorError(
                "Monitoring deadline reached; downstream may still be running"
            )

    def request(self, method: str = "GET") -> dict[str, object]:
        backoff = 10
        while True:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise MonitorError("Monitoring deadline reached")
            request = Request(
                self.url + ("/rerun-failed-jobs" if method == "POST" else ""),
                method=method,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "therock-workflow-run-monitor",
                },
            )
            try:
                with urlopen(request, timeout=min(30, remaining)) as response:
                    payload = response.read()
                    result = json.loads(payload) if payload else {}
                    if not isinstance(result, dict):
                        raise MonitorError("GitHub returned an invalid response object")
                    return result
            except HTTPError as error:
                with error:
                    body = error.read().decode("utf-8", errors="replace")
                delay = (
                    rate_limit_delay(error.headers, body)
                    if error.code in (403, 429)
                    else None
                )
                if delay is not None or error.code == 429:
                    wait = max(delay or backoff, backoff)
                    print(
                        f"GitHub rate limited the request; waiting {wait:.0f}s",
                        flush=True,
                    )
                    self.pause(wait)
                    backoff = min(backoff * 2, 900)
                    continue
                if 500 <= error.code < 600:
                    if method == "POST":
                        raise UncertainRerun("Rerun returned a server error") from error
                    print(f"GitHub HTTP {error.code}; retrying status read", flush=True)
                    self.pause(backoff)
                    backoff = min(backoff * 2, 300)
                    continue
                raise MonitorError(f"GitHub HTTP {error.code}: {body[:500]}") from error
            except (URLError, TimeoutError, OSError) as error:
                if method == "POST":
                    raise UncertainRerun("Rerun response was lost") from error
                print("Network error; retrying status read", flush=True)
                self.pause(backoff)
                backoff = min(backoff * 2, 300)


def monitor(api: RunAPI, max_retries: int, poll_interval: int) -> int:
    retries = 0
    expected_attempt = None
    registration_deadline = None
    observed_attempt = None
    while True:
        run = api.request()
        attempt = run.get("run_attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise MonitorError("GitHub returned an invalid run_attempt")
        status = run["status"]
        conclusion = run.get("conclusion")

        if expected_attempt is not None:
            if attempt < expected_attempt:
                if time.monotonic() >= registration_deadline:
                    raise MonitorError(
                        "Rerun did not register within five minutes; stopping without another POST"
                    )
                api.pause(RERUN_POLL_SECONDS)
                continue
            if attempt != expected_attempt:
                raise MonitorError(
                    "An unexpected external rerun changed the attempt number"
                )
            expected_attempt = None
            registration_deadline = None
            observed_attempt = attempt
        elif observed_attempt is not None and attempt != observed_attempt:
            raise MonitorError("An external rerun changed the monitored attempt")
        else:
            observed_attempt = attempt

        print(
            f"Attempt {attempt}: {status}, conclusion={conclusion or 'pending'}",
            flush=True,
        )
        if status != "completed":
            api.pause(poll_interval)
            continue
        if conclusion == "success":
            print(f"Downstream succeeded after {retries} retries", flush=True)
            return 0
        if conclusion not in {"failure", "timed_out"}:
            raise MonitorError(f"Downstream conclusion is not retryable: {conclusion}")
        if retries >= max_retries:
            print(
                f"::error::Downstream failed; all {max_retries} retries exhausted",
                flush=True,
            )
            return 1

        print(f"Requesting failed-job retry {retries + 1}/{max_retries}", flush=True)
        try:
            api.request("POST")
        except UncertainRerun as error:
            print(f"{error}; checking whether GitHub registered the retry", flush=True)
        expected_attempt = attempt + 1
        registration_deadline = time.monotonic() + RERUN_REGISTRATION_SECONDS
        retries += 1


def bounded_integer(minimum: int, maximum: int) -> Callable[[str], int]:
    """Build an argparse validator for bounded integer arguments."""

    def parse(raw: str) -> int:
        if not re.fullmatch(r"[0-9]+", raw) or not minimum <= int(raw) <= maximum:
            raise argparse.ArgumentTypeError(
                f"must be an integer from {minimum} to {maximum}"
            )
        return int(raw)

    return parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository", required=True, help="Owner/repository of the downstream run"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        type=bounded_integer(1, 10**20),
        help="Downstream Actions run ID",
    )
    parser.add_argument(
        "--max-retries",
        type=bounded_integer(0, 100),
        default=2,
        help="Additional failed-job reruns (default: 2)",
    )
    parser.add_argument(
        "--poll-interval",
        type=bounded_integer(30, 900),
        default=60,
        help="Seconds between status reads (default: 60)",
    )
    parser.add_argument(
        "--timeout-minutes",
        type=bounded_integer(1, 350),
        default=350,
        help="Shared monitoring deadline across all attempts (default: 350 minutes)",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("--repository must be owner/repository")
    try:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise MonitorError("GITHUB_TOKEN or GH_TOKEN is required")
        print(
            f"Monitoring https://github.com/{args.repository}/actions/runs/{args.run_id}",
            flush=True,
        )
        api = GitHub(
            args.repository,
            str(args.run_id),
            token,
            time.monotonic() + args.timeout_minutes * 60,
        )
        return monitor(api, args.max_retries, args.poll_interval)
    except (MonitorError, ValueError, KeyError) as error:
        print(f"::error::{error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
