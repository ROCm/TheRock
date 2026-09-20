#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Conditionally wrap an allowlisted CI phase with resource_probe.py.

This is intentionally a small exec-only adapter: it does not inspect or emit
the child command, paths, or environment. The resource probe remains default
off, and a phase must be selected explicitly in the JSON phase list.
"""

import argparse
import json
import os
from pathlib import Path
import sys


PHASES = frozenset(
    {
        "fetch-inbound-artifacts",
        "fetch-sources",
        "configure",
        "build-stage",
        "build-tests",
        "dependency-install",
        "packaging-tests",
        "package-fetch",
        "package-build",
        "package-test",
        "package-repository-build",
        "package-upload",
        "publish-release-buckets",
        "artifact-push",
        "artifact-push-local-equivalent",
        "log-upload",
        "log-upload-local-equivalent",
    }
)


def parse_selected_phases(value: str) -> frozenset[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "selected phases must be a JSON array"
        ) from error
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise argparse.ArgumentTypeError("selected phases must be a JSON string array")
    unknown = sorted(set(parsed) - PHASES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown resource probe phases: {unknown}")
    return frozenset(parsed)


def parse_args(arguments: list[str]) -> argparse.Namespace:
    try:
        separator = arguments.index("--")
    except ValueError:
        separator = len(arguments)
    wrapper_args = arguments[:separator]
    command = arguments[separator + 1 :] if separator < len(arguments) else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--enabled", choices=("true", "false"), required=True)
    parser.add_argument("--selected-phases", type=parse_selected_phases, required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--interval-seconds", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument(
        "--detail-profile",
        choices=("basic", "process", "diagnostic"),
        default="basic",
    )
    parsed = parser.parse_args(wrapper_args)
    if not command:
        parser.error("missing command after --")
    parsed.command = command
    return parsed


def command_for(args: argparse.Namespace) -> list[str]:
    if args.enabled != "true" or args.phase not in args.selected_phases:
        return args.command
    probe = Path(__file__).with_name("resource_probe.py")
    return [
        sys.executable,
        str(probe),
        "--phase",
        args.phase,
        "--interval-seconds",
        str(args.interval_seconds),
        "--output-dir",
        str(args.output_dir),
        "--storage-path",
        str(args.storage_path),
        "--detail-profile",
        args.detail_profile,
        "--",
        *args.command,
    ]


def main(arguments: list[str]) -> None:
    args = parse_args(arguments)
    command = command_for(args)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main(sys.argv[1:])
