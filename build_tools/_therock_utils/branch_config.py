# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Branch-local configuration parsing for TheRock."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from .build_topology import BuildTopology


@dataclass
class BranchArtifactGroupConfig:
    """Branch-local configuration for one artifact group."""

    source_sets: list[str] = field(default_factory=list)


@dataclass
class BranchConfig:
    """Parsed BRANCH_CONFIG.json contents."""

    flags: dict[str, str] = field(default_factory=dict)
    source_sets: list[str] = field(default_factory=list)
    artifact_groups: dict[str, BranchArtifactGroupConfig] = field(default_factory=dict)


def load_branch_config(
    config_path: Path,
    topology: BuildTopology | None = None,
    *,
    required: bool = False,
) -> BranchConfig:
    """Load and validate an optional BRANCH_CONFIG.json file."""
    if not config_path.exists():
        if required:
            raise FileNotFoundError(f"BRANCH_CONFIG.json not found at {config_path}")
        return BranchConfig()

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object")

    flags = _parse_flags(data.get("flags", {}), config_path)
    source_sets = _parse_source_set_list(data.get("source_sets", []), config_path)
    artifact_groups = _parse_artifact_groups(
        data.get("artifact_groups", {}), config_path
    )

    config = BranchConfig(
        flags=flags,
        source_sets=source_sets,
        artifact_groups=artifact_groups,
    )

    if topology is not None:
        _validate_config_references(config, topology, config_path)

    return config


def get_source_sets_for_artifact_groups(
    config: BranchConfig, artifact_group_names: list[str]
) -> list[str]:
    """Get branch-configured source sets for a set of artifact groups."""
    result: list[str] = []
    seen: set[str] = set()
    for artifact_group_name in artifact_group_names:
        group_config = config.artifact_groups.get(artifact_group_name)
        if not group_config:
            continue
        for source_set_name in group_config.source_sets:
            if source_set_name not in seen:
                seen.add(source_set_name)
                result.append(source_set_name)
    return result


def _parse_flags(raw_flags: Any, config_path: Path) -> dict[str, str]:
    if raw_flags is None:
        return {}
    if not isinstance(raw_flags, dict):
        raise ValueError(f"{config_path}: 'flags' must be a JSON object")

    flags: dict[str, str] = {}
    for flag_name, flag_value in raw_flags.items():
        if not isinstance(flag_name, str) or not _is_cmake_identifier(flag_name):
            raise ValueError(f"{config_path}: invalid flag name '{flag_name}'")
        if isinstance(flag_value, bool):
            flags[flag_name] = "ON" if flag_value else "OFF"
        elif isinstance(flag_value, str):
            flags[flag_name] = flag_value
        else:
            raise ValueError(
                f"{config_path}: flag '{flag_name}' value must be a string or boolean"
            )
    return flags


def _parse_source_set_list(raw_source_sets: Any, config_path: Path) -> list[str]:
    if raw_source_sets is None:
        return []
    if not isinstance(raw_source_sets, list) or not all(
        isinstance(name, str) for name in raw_source_sets
    ):
        raise ValueError(f"{config_path}: 'source_sets' must be a list of strings")
    return _dedupe(raw_source_sets)


def _parse_artifact_groups(
    raw_artifact_groups: Any, config_path: Path
) -> dict[str, BranchArtifactGroupConfig]:
    if raw_artifact_groups is None:
        return {}
    if not isinstance(raw_artifact_groups, dict):
        raise ValueError(f"{config_path}: 'artifact_groups' must be a JSON object")

    artifact_groups: dict[str, BranchArtifactGroupConfig] = {}
    for group_name, group_data in raw_artifact_groups.items():
        if not isinstance(group_name, str):
            raise ValueError(f"{config_path}: artifact group names must be strings")
        if not isinstance(group_data, dict):
            raise ValueError(
                f"{config_path}: artifact_groups.{group_name} must be a JSON object"
            )
        source_sets = _parse_source_set_list(
            group_data.get("source_sets", []), config_path
        )
        artifact_groups[group_name] = BranchArtifactGroupConfig(source_sets=source_sets)
    return artifact_groups


def _validate_config_references(
    config: BranchConfig, topology: BuildTopology, config_path: Path
) -> None:
    for source_set_name in config.source_sets:
        _validate_source_set_reference(source_set_name, topology, config_path)

    for group_name, group_config in config.artifact_groups.items():
        if group_name not in topology.artifact_groups:
            raise ValueError(
                f"{config_path}: artifact group '{group_name}' is not defined "
                "in BUILD_TOPOLOGY.toml"
            )
        for source_set_name in group_config.source_sets:
            _validate_source_set_reference(source_set_name, topology, config_path)


def _validate_source_set_reference(
    source_set_name: str, topology: BuildTopology, config_path: Path
) -> None:
    if source_set_name not in topology.source_sets:
        raise ValueError(
            f"{config_path}: source set '{source_set_name}' is not defined "
            "in BUILD_TOPOLOGY.toml"
        )


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _is_cmake_identifier(value: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value) is not None
