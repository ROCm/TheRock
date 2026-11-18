#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Expand a list of AMD GPU families to their constituent gfx targets.

Reads `cmake/therock_amdgpu_targets.cmake` (the authoritative source of
truth) via `_therock_utils.cmake_amdgpu_targets` and prints the union
of gfx targets for the requested families as a comma-separated string.

Used by CI workflows that need build-side gfx targets (e.g.
`--pytorch-rocm-arch`) from a family list. Mirrors the pattern used by
`artifact_manager.py --expand-family-to-targets`.

Callers may use --exclude-implicit-targets to remove selected targets that were
introduced only through family expansion. An exact target in the original
--amdgpu-families input remains selected.

Output modes:

* ``targets`` (default) — bare gfx targets, comma-separated:

      python expand_amdgpu_families.py --amdgpu-families "gfx94X-dcgpu;gfx120X-all"
      -> gfx942,gfx1200,gfx1201

* ``device-extras`` — pip device extras, comma-separated, and also
  written to ``$GITHUB_OUTPUT`` as ``device_extras=...``:

      python expand_amdgpu_families.py --amdgpu-families "gfx94X-dcgpu" \
        --output-mode=device-extras
      -> device-gfx942  (stdout)
      -> device_extras=device-gfx942  (GITHUB_OUTPUT)

Fails with a non-zero exit code and a clear message if any requested
family is not present in the CMake source — silent drops were the bug
this helper exists to prevent.
"""

import argparse
import sys
from pathlib import Path

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))
from _therock_utils.cmake_amdgpu_targets import (
    amdgpu_family_map,
    expand_families,
)

from _therock_utils.sdk_targets import group_package_targets
from github_actions.github_actions_api import gha_set_output


def parse_semicolon_list(value: str) -> list[str]:
    """Parse a semicolon-separated list, ignoring whitespace and empty items."""
    return [item.strip() for item in value.split(";") if item.strip()]


def exclude_implicit_targets(
    requested_families: list[str],
    expanded_targets: list[str],
    implicit_targets_to_exclude: set[str],
) -> list[str]:
    """Exclude policy targets unless they were exact tokens in the input."""
    directly_requested = set(requested_families)
    return [
        target
        for target in expanded_targets
        if target not in implicit_targets_to_exclude or target in directly_requested
    ]


def expand_requested_families(
    requested_families: list[str],
    family_map: dict[str, list[str]],
    implicit_targets_to_exclude: set[str],
) -> list[str]:
    """Expand requested families and remove targets selected only implicitly."""
    expanded_targets = expand_families(requested_families, family_map)
    return exclude_implicit_targets(
        requested_families,
        expanded_targets,
        implicit_targets_to_exclude,
    )


def format_device_extras(targets: list[str]) -> str:
    """Format concrete targets as their deduplicated pip device extras."""
    return ",".join(f"device-{owner}" for owner in group_package_targets(targets))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="expand_amdgpu_families.py")
    p.add_argument(
        "--amdgpu-families",
        required=True,
        type=str,
        help=(
            "Semicolon-separated list of AMD GPU families to expand "
            "(e.g. 'gfx94X-dcgpu;gfx120X-all')."
        ),
    )
    p.add_argument(
        "--exclude-implicit-targets",
        default="",
        type=str,
        help=(
            "Optional semicolon-separated targets to exclude when introduced "
            "only by family expansion. An exact target in --amdgpu-families "
            "is retained."
        ),
    )
    p.add_argument(
        "--output-mode",
        choices=["targets", "device-extras"],
        default="targets",
        help=(
            "'targets' prints bare gfx targets (default). "
            "'device-extras' prints pip device extras (device-gfxNNN) "
            "and writes them to GITHUB_OUTPUT as device_extras=..."
        ),
    )
    args = p.parse_args(argv)

    families = parse_semicolon_list(args.amdgpu_families)
    if not families:
        print("")
        return 0

    implicit_targets_to_exclude = set(
        parse_semicolon_list(args.exclude_implicit_targets)
    )
    targets = expand_requested_families(
        families,
        amdgpu_family_map(),
        implicit_targets_to_exclude,
    )

    if args.output_mode == "device-extras":
        result = format_device_extras(targets)
        print(result)
        gha_set_output({"device_extras": result})
    else:
        print(",".join(targets))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
