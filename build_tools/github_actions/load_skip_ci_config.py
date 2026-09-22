#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Load skip CI configuration from TOML file.

This script reads skip_ci_patterns from a TOML config file and outputs
them in a format suitable for GitHub Actions. External repos use this
to provide their own skip patterns to TheRock CI.

Usage:
    python load_skip_ci_config.py --config-path .github/skip-ci-config.toml

Config format (TOML):
    version = 1

    [skip_ci]
    common = ["*.md", "docs/*"]
    linux = ["projects/windows-only/*"]
    windows = ["projects/linux-only/*"]

Outputs (to $GITHUB_OUTPUT):
    skip_ci_patterns: JSON array of glob patterns that can skip CI
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Try tomllib (Python 3.11+), fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def load_skip_ci_config(config_path: str) -> List[str]:
    """Load skip CI patterns from TOML config file.

    Args:
        config_path: Path to the TOML config file.

    Returns:
        List of glob patterns that can skip CI.
    """
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return []

    if tomllib is None:
        print(
            "ERROR: tomllib (Python 3.11+) or tomli package required",
            file=sys.stderr,
        )
        return []

    with open(path, "rb") as f:
        config = tomllib.load(f)

    skip_ci = config.get("skip_ci", {})
    patterns: List[str] = []

    # Common patterns apply to all platforms
    common = skip_ci.get("common", [])
    if isinstance(common, list):
        patterns.extend(common)

    # Platform-specific patterns could be added based on RUNNER_OS
    # For now, just return common patterns (TheRock evaluates platform-specific)
    runner_os = os.environ.get("RUNNER_OS", "").lower()
    if runner_os == "linux":
        linux_patterns = skip_ci.get("linux", [])
        if isinstance(linux_patterns, list):
            patterns.extend(linux_patterns)
    elif runner_os == "windows":
        windows_patterns = skip_ci.get("windows", [])
        if isinstance(windows_patterns, list):
            patterns.extend(windows_patterns)

    return patterns


def set_github_output(key: str, value: str) -> None:
    """Write output to $GITHUB_OUTPUT or print if not in CI."""
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        print(f"{key}={value}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load skip CI configuration from TOML file",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Path to the skip CI config TOML file",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    patterns = load_skip_ci_config(args.config_path)
    print(f"Loaded {len(patterns)} skip CI patterns from {args.config_path}")

    # Output as JSON array
    set_github_output("skip_ci_patterns", json.dumps(patterns))

    return 0


if __name__ == "__main__":
    sys.exit(main())
