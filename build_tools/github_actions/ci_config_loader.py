#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Versioned CI configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CURRENT_VERSION = "1"
DEFAULT_CONFIG_PATH = Path("ci-config")
CONFIG_FILENAME = "runner-config.json"


class ConfigError(Exception):
    pass


@dataclass
class ConfigV1:
    build_runners: dict[str, Any]
    gpu_families: dict[str, Any]
    _raw: dict[str, Any]

    def get_gpu_families(self, trigger_types: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for trigger_type in trigger_types:
            if trigger_type in self.gpu_families:
                for name, config in self.gpu_families[trigger_type].items():
                    result[name] = config
        return result


def load_config_v1(config_path: Path = DEFAULT_CONFIG_PATH) -> ConfigV1:
    config_file = config_path / CONFIG_FILENAME

    if not config_file.exists():
        raise ConfigError(f"Config not found: {config_file}")

    try:
        with open(config_file) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {config_file}: {e}")

    version = raw.get("version", "1")
    if version != CURRENT_VERSION:
        raise ConfigError(f"Config version mismatch: got {version}, expected {CURRENT_VERSION}")

    missing = [k for k in ("build_runners", "gpu_families") if k not in raw]
    if missing:
        raise ConfigError(f"Config missing required keys: {missing}")

    return ConfigV1(
        build_runners=raw["build_runners"],
        gpu_families=raw["gpu_families"],
        _raw=raw,
    )


def config_exists(config_path: Path = DEFAULT_CONFIG_PATH) -> bool:
    return (config_path / CONFIG_FILENAME).exists()


def load_runner_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return load_config_v1(config_path)._raw


def get_build_runners(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("build_runners", {})


def get_gpu_families(config: dict[str, Any], trigger_types: list[str]) -> dict[str, Any]:
    gpu_families = config.get("gpu_families", {})
    result: dict[str, Any] = {}
    for trigger_type in trigger_types:
        if trigger_type in gpu_families:
            for name, cfg in gpu_families[trigger_type].items():
                result[name] = cfg
    return result


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH

    try:
        config = load_config_v1(path)
        print(f"Loaded config v{CURRENT_VERSION} from: {path}")
        print(f"Build runners: {list(config.build_runners.keys())}")
        print(f"GPU families (presubmit): {list(config.get_gpu_families(['presubmit']).keys())}")
    except ConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
