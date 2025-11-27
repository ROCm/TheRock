#!/usr/bin/env python
"""Configure the multi-arch CI pipeline.

This script parses workflow inputs and outputs configuration for the
multi-arch CI pipeline jobs.

Outputs (via GITHUB_OUTPUT):
    amdgpu_families_json: JSON array of GPU families for matrix builds
    enabled_stages: Comma-separated list of enabled stages
    artifact_group: Artifact group identifier
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from github_actions.github_actions_utils import gha_set_output


ALL_STAGES = [
    "foundation",
    "compiler-runtime",
    "math-libs",
    "comm-libs",
    "dctools-core",
    "profiler-apps",
]


def main(argv: list[str] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--amdgpu-families",
        type=str,
        required=True,
        help="Comma-separated list of GPU families (e.g., 'gfx94X-dcgpu,gfx110X-all')",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated list of stages to build (empty = all)",
    )
    parser.add_argument(
        "--build-variant",
        type=str,
        default="release",
        help="Build variant (e.g., 'release', 'asan')",
    )
    args = parser.parse_args(argv)

    # Parse families into JSON array
    families = [f.strip() for f in args.amdgpu_families.split(",") if f.strip()]
    families_json = json.dumps(families)

    # Determine enabled stages
    if args.stages:
        enabled_stages = args.stages
    else:
        enabled_stages = ",".join(ALL_STAGES)

    # Create artifact group name
    artifact_group = f"multi-arch-{args.build_variant}"

    # Output to GITHUB_OUTPUT
    gha_set_output(
        {
            "amdgpu_families_json": families_json,
            "enabled_stages": enabled_stages,
            "artifact_group": artifact_group,
        }
    )


if __name__ == "__main__":
    main(sys.argv[1:])
