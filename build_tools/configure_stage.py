#!/usr/bin/env python
"""Generate CMake configuration for building a specific stage.

This script uses BUILD_TOPOLOGY.toml to determine which features/artifacts
should be enabled for a specific build stage, and outputs the appropriate
CMake arguments.

Usage:
    # Generate CMake args for a stage
    python configure_stage.py \
        --stage math-libs \
        --amdgpu-families gfx94X-dcgpu \
        --output-cmake-args /tmp/stage_args.txt

    # Then use the generated args with CMake
    cmake -B build -S . $(cat /tmp/stage_args.txt) -GNinja

    # Or print to stdout for inspection
    python configure_stage.py --stage math-libs --print

The script generates flags like:
    -DTHEROCK_AMDGPU_FAMILIES=gfx94X-dcgpu
    -DTHEROCK_ENABLE_ALL=OFF
    -DTHEROCK_ENABLE_BLAS=ON
    -DTHEROCK_ENABLE_FFT=ON
    ...
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set

from _therock_utils.build_topology import BuildTopology
from github_actions.github_actions_utils import gha_set_output


def log(msg: str):
    """Print message and flush."""
    print(msg, file=sys.stderr, flush=True)


def get_topology() -> BuildTopology:
    """Load the BUILD_TOPOLOGY.toml from the repository root."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    topology_path = repo_root / "BUILD_TOPOLOGY.toml"
    if not topology_path.exists():
        raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found at {topology_path}")
    return BuildTopology(str(topology_path))


def get_stage_features(topology: BuildTopology, stage_name: str) -> Set[str]:
    """Get the set of feature names that should be enabled for a stage.

    This includes:
    1. Features for artifacts produced by this stage
    2. Features for artifacts that are inbound dependencies (needed but prebuilt)

    Note: The inbound dependencies will be marked as prebuilt via buildctl.py bootstrap,
    but CMake still needs their features enabled for dependency resolution.
    """
    if stage_name not in topology.build_stages:
        raise ValueError(f"Unknown stage: {stage_name}")

    # Get artifacts produced by this stage
    produced = topology.get_produced_artifacts(stage_name)

    # Get inbound artifacts (dependencies from previous stages)
    inbound = topology.get_inbound_artifacts(stage_name)

    # Combine: we need features for both produced and inbound artifacts
    all_artifacts = produced | inbound

    # Convert artifact names to feature names
    features = set()
    for artifact_name in all_artifacts:
        if artifact_name in topology.artifacts:
            artifact = topology.artifacts[artifact_name]
            feature_name = topology.get_artifact_feature_name(artifact)
            features.add(feature_name)

    return features


def generate_cmake_args(
    stage_name: str,
    amdgpu_families: str,
    topology: BuildTopology,
    include_comments: bool = False,
) -> List[str]:
    """Generate CMake arguments for building a specific stage.

    Args:
        stage_name: Name of the build stage
        amdgpu_families: Comma-separated GPU families
        topology: BuildTopology instance
        include_comments: Include comment lines explaining each flag

    Returns:
        List of CMake argument strings
    """
    args = []

    if include_comments:
        args.append(f"# CMake arguments for stage: {stage_name}")
        args.append("")

    # GPU families
    if amdgpu_families:
        args.append(f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}")

    # Disable all features by default, then enable only what we need
    if include_comments:
        args.append("")
        args.append("# Disable all features by default")
    args.append("-DTHEROCK_ENABLE_ALL=OFF")

    # Get features to enable for this stage
    features = get_stage_features(topology, stage_name)

    if include_comments:
        args.append("")
        args.append(f"# Enable features for stage '{stage_name}'")

    for feature in sorted(features):
        args.append(f"-DTHEROCK_ENABLE_{feature}=ON")

    return args


def main(argv: List[str] = None):
    parser = argparse.ArgumentParser(
        description="Generate CMake configuration for building a specific stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help="Build stage name (e.g., foundation, compiler-runtime, math-libs)",
    )
    parser.add_argument(
        "--amdgpu-families",
        type=str,
        default="",
        help="Comma-separated GPU families (e.g., gfx94X-dcgpu,gfx110X-all)",
    )
    parser.add_argument(
        "--output-cmake-args",
        type=Path,
        help="Output file for CMake arguments (one per line)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_args",
        help="Print CMake arguments to stdout",
    )
    parser.add_argument(
        "--comments",
        action="store_true",
        help="Include comments in output",
    )
    parser.add_argument(
        "--oneline",
        action="store_true",
        help="Output all arguments on a single line (for shell expansion)",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List available build stages and exit",
    )
    parser.add_argument(
        "--gha-output",
        action="store_true",
        help="Write cmake_args to GITHUB_OUTPUT (for GitHub Actions)",
    )

    args = parser.parse_args(argv)

    topology = get_topology()

    # List stages mode
    if args.list_stages:
        log("Available build stages:")
        for stage in topology.get_build_stages():
            log(f"  {stage.name} ({stage.type}): {stage.description}")
        return

    # Validate stage
    if args.stage not in topology.build_stages:
        available = ", ".join(s.name for s in topology.get_build_stages())
        parser.error(f"Unknown stage '{args.stage}'. Available stages: {available}")

    # Generate arguments
    cmake_args = generate_cmake_args(
        stage_name=args.stage,
        amdgpu_families=args.amdgpu_families,
        topology=topology,
        include_comments=args.comments and not args.oneline,
    )

    # Filter out comments if not requested
    if not args.comments:
        cmake_args = [a for a in cmake_args if not a.startswith("#") and a]

    # Output
    if args.oneline or args.gha_output:
        output = " ".join(cmake_args)
    else:
        output = "\n".join(cmake_args)

    if args.gha_output:
        gha_set_output({"cmake_args": output})
    elif args.output_cmake_args:
        args.output_cmake_args.write_text(output + "\n")
        log(f"Wrote CMake arguments to {args.output_cmake_args}")
    elif args.print_args:
        print(output)
    else:
        # Default: print to stdout
        print(output)


if __name__ == "__main__":
    main(sys.argv[1:])
