#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Loads CI runner configuration from therock-ci-config repository.

The config repo is checked out during workflow setup, and this module
reads the JSON config from that checkout. The checkout SHA is logged
for full traceability.

Testing: Run ci_config_loader_test.py for unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Default config path when checked out in workflows
DEFAULT_CONFIG_PATH = Path("ci-config")
CONFIG_FILENAME = "runner-config.json"


class ConfigError(Exception):
    """Raised when config loading or validation fails."""

    pass


def load_runner_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load runner configuration from the config repository checkout.

    Args:
        config_path: Path to the ci-config checkout directory.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigError: If config file is missing or invalid.
    """
    config_file = config_path / CONFIG_FILENAME

    if not config_file.exists():
        raise ConfigError(
            f"Config file not found: {config_file}. "
            f"Ensure therock-ci-config is checked out to {config_path}"
        )

    try:
        with open(config_file) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_file}: {e}")

    # Basic validation
    required_keys = ["version", "build_runners", "gpu_families"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ConfigError(f"Config missing required keys: {missing}")

    return config


def get_build_runners(config: dict[str, Any]) -> dict[str, Any]:
    """Extract build runner configuration.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Build runner config with platform -> variant -> labels mapping.
    """
    return config.get("build_runners", {})


def get_gpu_families(
    config: dict[str, Any],
    trigger_types: list[str],
) -> dict[str, Any]:
    """Get combined GPU family matrix for specified trigger types.

    Args:
        config: Loaded configuration dictionary.
        trigger_types: List of trigger types (presubmit, postsubmit, nightly).

    Returns:
        Combined family matrix with all families from specified triggers.
    """
    gpu_families = config.get("gpu_families", {})
    result: dict[str, Any] = {}

    for trigger_type in trigger_types:
        if trigger_type in gpu_families:
            for family_name, family_config in gpu_families[trigger_type].items():
                result[family_name] = family_config

    return result


def config_exists(config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Check if the config file exists at the given path.

    Args:
        config_path: Path to check for config.

    Returns:
        True if config file exists, False otherwise.
    """
    return (config_path / CONFIG_FILENAME).exists()


def log_config_version(config: dict[str, Any], config_path: Path) -> None:
    """Log config version and path for traceability.

    Args:
        config: Loaded configuration dictionary.
        config_path: Path where config was loaded from.
    """
    version = config.get("version", "unknown")
    print(f"CI Config loaded from: {config_path}")
    print(f"CI Config version: {version}")


if __name__ == "__main__":
    # Test loading config
    import sys

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = DEFAULT_CONFIG_PATH

    try:
        config = load_runner_config(path)
        log_config_version(config, path)
        print(f"Build runners: {list(get_build_runners(config).keys())}")
        print(f"GPU families (presubmit): {list(get_gpu_families(config, ['presubmit']).keys())}")
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
