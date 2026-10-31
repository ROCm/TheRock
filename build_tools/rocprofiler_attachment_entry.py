# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Execute the unchanged test wrapper, guarding its CTest entry before launch."""

import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "build_tools/github_actions/test_executable_scripts/test_rocprofiler_sdk.py"
)


def validate_dashboard(path: Path, expected_url: str) -> dict | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    urls = re.findall(
        r'^set\(CTEST_SUBMIT_URL "([^"\n]+)"\)$', data.decode(), re.MULTILINE
    )
    parsed = urllib.parse.urlsplit(expected_url)
    if (
        urls != [expected_url]
        or parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or not parsed.port
    ):
        raise ValueError(
            "generated CDash submission is not the exact owned loopback URL"
        )
    return dict(
        path=str(path),
        sha256=hashlib.sha256(data).hexdigest(),
        url=expected_url,
        loopback_only=True,
    )


def run_original(
    script: Path, args: list[str], dashboard: Path, url: str, receipt: Path
) -> None:
    original_run = subprocess.run
    original_argv, original_path = sys.argv[:], sys.path[:]

    def guarded_run(command, *positional, **keywords):
        if (
            isinstance(command, (list, tuple))
            and command
            and Path(command[0]).name == "ctest"
            and list(command) != ["ctest", "--version"]
        ):
            expected = [
                "ctest",
                "-S",
                str(dashboard),
                "--verbose",
                "--output-on-failure",
                "--no-tests=error",
            ]
            if list(command) != expected:
                raise ValueError("unexpected CTest dashboard entry")
            binding = validate_dashboard(dashboard, url)
            if binding is None:
                raise ValueError("CTest dashboard is absent before launch")
            binding.update(
                original_command=list(command),
                original_script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                before_ctest_launch=True,
            )
            receipt.write_text(json.dumps(binding, indent=2) + "\n")
        return original_run(command, *positional, **keywords)

    try:
        subprocess.run = guarded_run
        sys.argv = [str(script), *args]
        sys.path[0] = str(script.parent)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        subprocess.run = original_run
        sys.argv, sys.path = original_argv, original_path


def main() -> None:
    output = Path(os.environ["ATTACHMENT_DIAG_ROOT"]).resolve()
    if not output.is_relative_to(
        ROOT / "attachment-diagnostic"
    ) or not output.name.startswith("execution-"):
        raise ValueError("entry receipt is not in the owned diagnostic directory")
    run_original(
        SCRIPT,
        ["--enable-cdash"],
        ROOT / "build/share/rocprofiler-sdk/tests/build/dashboard.cmake",
        os.environ["THEROCK_CDASH_SUBMIT_URL"],
        output / "ctest-entry.json",
    )


if __name__ == "__main__":
    main()
