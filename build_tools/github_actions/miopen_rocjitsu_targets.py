# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Supported MIOpen emulator targets, shared by preparation and CI selection."""

import json
import re
from pathlib import Path

REGISTRY_PATH = Path(__file__).with_suffix(".json")
TEST_LABEL_GROUP = "miopen-rocjitsu"


def load_targets(path: Path = REGISTRY_PATH) -> dict[str, dict[str, str]]:
    targets = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(targets, dict) or not targets:
        raise ValueError(f"{path}: expected a non-empty target registry")
    for arch, target in targets.items():
        if not re.fullmatch(r"gfx[0-9]{2,3}[0-9a-f]", arch):
            raise ValueError(f"{path}: invalid emulator architecture {arch!r}")
        if not isinstance(target, dict):
            raise ValueError(f"{path}: {arch} must define family and config")
        family, config = target.get("family"), target.get("config")
        if not isinstance(family, str) or not re.fullmatch(r"gfx[A-Za-z0-9-]+", family):
            raise ValueError(f"{path}: {arch} has an invalid family")
        if not isinstance(config, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+\.json", config
        ):
            raise ValueError(f"{path}: {arch} must reference a profile basename")
    return targets


def component_names(targets: dict[str, dict[str, str]] | None = None) -> list[str]:
    if targets is None:
        targets = load_targets()
    return [f"miopen-rocjitsu-{arch}" for arch in targets]
