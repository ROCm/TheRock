# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Retry a component's complete test launch command using the workflow policy."""

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Test command to run in Bash")
    parser.add_argument(
        "--retry-count",
        type=int,
        default=os.environ.get("RETRY_COUNT", "1"),
        help="Number of retries after the initial attempt (default: RETRY_COUNT or 1)",
    )
    args = parser.parse_args()
    if args.retry_count < 0:
        parser.error("--retry-count must be non-negative")

    retry_count = args.retry_count
    if os.environ.get("RETRY_THIS_STEP", "true").lower() != "true":
        retry_count = 0
    step_name = os.environ.get("STEP_NAME") or "Test command"
    for attempt in range(1, retry_count + 2):
        print(f"{step_name}: attempt {attempt}/{retry_count + 1}", flush=True)
        # Workflows use Bash on Linux and Windows. Preserve environment
        # assignments, quoting and command chains, and stop on shell failures.
        result = subprocess.run(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-e",
                "-o",
                "pipefail",
                "-c",
                args.command,
            ],
            check=False,
        )
        if result.returncode == 0:
            return 0
        if result.returncode < 0 or result.returncode in (130, 143):
            # Do not retry interrupted or terminated commands.
            return 128 - result.returncode if result.returncode < 0 else result.returncode
        if attempt <= retry_count:
            print(
                f"{step_name}: exited with {result.returncode}; "
                "retrying immediately",
                file=sys.stderr,
                flush=True,
            )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
