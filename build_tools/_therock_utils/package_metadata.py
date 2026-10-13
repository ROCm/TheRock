# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Dependency-free target ownership shared by build and installed consumers.

Groups only the supplied target names by package owner, preserving each target's
identity.
This module is also embedded verbatim into generated SDK distribution metadata.
"""

from collections.abc import Iterable
import re
from typing import TypedDict

OWNERSHIP_SCHEMA_VERSION = 1
# Targets normally map 1:1 to packages with the same name.
# Overrides allow multiple targets to share a package owner.
_OVERRIDE_PACKAGE_OWNERS = {"gfx1250-strict": "gfx1250"}
_OVERRIDE_TARGET_ARCHITECTURAL_FAMILIES = {
    "gfx1250": "gfx125X",
    "gfx1250-strict": "gfx125X",
}
_TARGET_FEATURES = re.compile(r"(?:[:-](?:xnack|sramecc)[+-])+$")


class OwnershipData(TypedDict):
    schema_version: int
    package_owners: dict[str, str]


def ownership_data() -> OwnershipData:
    """Return independent JSON-compatible data with an explicit schema version."""
    return {
        "schema_version": OWNERSHIP_SCHEMA_VERSION,
        "package_owners": dict(_OVERRIDE_PACKAGE_OWNERS),
    }


def canonical_target(target: str) -> str:
    """Remove recognized features without collapsing canonical variant names."""
    return _TARGET_FEATURES.sub("", target)


def package_owner(target: str) -> str:
    """Resolve a supplied target/family to its package owner."""
    target = canonical_target(target)
    owner = _OVERRIDE_PACKAGE_OWNERS.get(target)
    return target if owner is None else owner


def group_package_targets(targets: Iterable[str]) -> dict[str, list[str]]:
    """Group only supplied members, retaining their original names and order."""
    groups: dict[str, list[str]] = {}
    for target in targets:
        members = groups.setdefault(package_owner(target), [])
        if target not in members:
            members.append(target)
    return groups


def architectural_family(target: str) -> str | None:
    """Classify variants independently of enabled build-family membership."""
    return _OVERRIDE_TARGET_ARCHITECTURAL_FAMILIES.get(canonical_target(target))
