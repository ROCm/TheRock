#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate CMake configuration for building a specific stage or projects.

This script uses BUILD_TOPOLOGY.toml to determine which features/artifacts
should be enabled for a specific build stage or set of projects, and outputs
the appropriate CMake arguments.

Usage:
    # Generate CMake args for a stage
    python configure_stage.py \
        --stage math-libs \
        --amdgpu-families gfx94X-dcgpu \
        --output-cmake-args /tmp/stage_args.txt

    # Generate CMake args for specific projects
    python configure_stage.py --projects rocblas miopen --oneline
    # Output: -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_BLAS=ON -DTHEROCK_ENABLE_MIOPEN=ON

    # List available projects/subprojects
    python configure_stage.py --list-projects

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
import platform as platform_module
import sys
from pathlib import Path
from typing import List, Optional, Set

from _therock_utils.build_topology import BuildTopology
from github_actions.github_actions_api import gha_set_output
from github_actions.manylinux_config import (
    DIST_PYTHON_EXECUTABLES,
    SHARED_PYTHON_EXECUTABLES,
)


def log(msg: str):
    """Print message and flush."""
    print(msg, file=sys.stderr, flush=True)


def normalize_project_name(name: str) -> str:
    """Normalize a project name, handling paths like 'projects/hip' -> 'hip'.

    The changed_projects input from external repos may include paths like:
    - 'projects/hip' -> 'hip'
    - 'projects/rocblas' -> 'rocblas'
    - 'hip' -> 'hip' (already normalized)
    """
    # Strip 'projects/' prefix if present
    if name.startswith("projects/"):
        name = name[len("projects/") :]
    # Strip any trailing path components (e.g., 'hip/src/foo.cpp' -> 'hip')
    if "/" in name:
        name = name.split("/")[0]
    return name


def get_topology() -> BuildTopology:
    """Load the BUILD_TOPOLOGY.toml from the repository root."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    topology_path = repo_root / "BUILD_TOPOLOGY.toml"
    if not topology_path.exists():
        raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found at {topology_path}")
    return BuildTopology(str(topology_path))


def get_stage_features(
    topology: BuildTopology,
    stage_name: str,
    platform_name: str = "",
    enabled_flags: Optional[Set[str]] = None,
) -> Set[str]:
    """Get the set of feature names that should be enabled for a stage.

    This includes:
    1. Features for artifacts produced by this stage
    2. Features for artifacts that are inbound dependencies (needed but prebuilt)

    Artifacts disabled for platform_name are excluded.

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
            if topology.is_artifact_disabled_on_platform(
                artifact,
                platform_name,
                enabled_flags=enabled_flags,
            ):
                continue
            feature_name = topology.get_artifact_feature_name(artifact)
            features.add(feature_name)

    return features


def get_project_features(
    topology: BuildTopology,
    project_names: List[str],
    platform_name: str = "",
    build_dir: Path = None,
) -> Set[str]:
    """Resolve project names to CMake feature names."""
    return topology.resolve_projects_to_features(
        project_names, platform_name, build_dir
    )


def generate_cmake_args(
    stage_name: str,
    amdgpu_families: str,
    dist_amdgpu_families: str,
    topology: BuildTopology,
    include_comments: bool = False,
    platform_name: str = platform_module.system().lower(),
    manylinux: bool = False,
    project_names: List[str] = None,
    build_dir: Path = None,
) -> List[str]:
    """Generate CMake arguments for building a specific stage or projects."""
    args = []

    if stage_name and project_names:
        desc = f"stage {stage_name} + projects: {', '.join(project_names)}"
    elif stage_name:
        desc = stage_name
    else:
        desc = f"projects: {', '.join(project_names or [])}"
    if include_comments:
        args.append(f"# CMake arguments for {desc}")
        args.append("")

    # GPU families for shard-specific targets
    if amdgpu_families:
        args.append(f"-DTHEROCK_AMDGPU_FAMILIES={amdgpu_families}")

    # GPU families for dist targets (all architectures in the distribution)
    # Quote the value since it contains semicolons (CMake list separator)
    if dist_amdgpu_families:
        args.append(f'-DTHEROCK_DIST_AMDGPU_FAMILIES="{dist_amdgpu_families}"')

    # Manylinux Python executables for per-Python-version builds
    # Quote values since they contain semicolons (CMake list separator)
    if manylinux:
        args.append(f'-DTHEROCK_DIST_PYTHON_EXECUTABLES="{DIST_PYTHON_EXECUTABLES}"')
        args.append(
            f'-DTHEROCK_SHARED_PYTHON_EXECUTABLES="{SHARED_PYTHON_EXECUTABLES}"'
        )

    # Disable all features by default, then enable only what we need
    if include_comments:
        args.append("")
        args.append("# Disable all features by default")
    args.append("-DTHEROCK_ENABLE_ALL=OFF")

    # Get features to enable
    # --projects narrows down features; --stage alone enables all stage features
    if project_names:
        features = get_project_features(
            topology, project_names, platform_name=platform_name, build_dir=build_dir
        )
    elif stage_name:
        features = get_stage_features(topology, stage_name, platform_name=platform_name)
    else:
        features = set()

    if include_comments:
        args.append("")
        args.append(f"# Enable features for {desc}")

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
        default=None,
        help="Build stage name (e.g., compiler-runtime, math-libs)",
    )
    parser.add_argument(
        "--amdgpu-families",
        type=str,
        default="",
        help="Comma-separated GPU families for shard-specific targets (e.g., gfx94X-dcgpu)",
    )
    parser.add_argument(
        "--dist-amdgpu-families",
        type=str,
        default="",
        help="Semicolon-separated GPU families for dist targets (e.g., gfx94X-dcgpu;gfx110X-all)",
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
    parser.add_argument(
        "--platform",
        type=str,
        default=platform_module.system().lower(),
        help=f"Platform for platform-specific CMake args (default: {platform_module.system().lower()})",
    )
    parser.add_argument(
        "--manylinux",
        action="store_true",
        help="Add manylinux Python executable cmake args (for use inside "
        "the manylinux build container)",
    )
    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        metavar="PROJECT",
        help="Project/subproject names to enable (e.g., rocblas miopen hipfft). "
        "Enables building specific projects without requiring --stage.",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List available projects/subprojects and their artifacts",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="CMake build directory containing artifact_subprojects.json manifest. "
        "If provided, uses accurate CMake-generated mappings for project resolution.",
    )
    parser.add_argument(
        "--skip-stages",
        action="store_true",
        help="Output comma-separated list of stages to skip based on --projects. "
        "Stages not needed to build the specified projects will be listed.",
    )

    args = parser.parse_args(argv)

    if (
        not args.list_stages
        and not args.list_projects
        and not args.skip_stages
        and args.stage is None
        and args.projects is None
    ):
        parser.error(
            "--stage or --projects is required unless --list-stages, --list-projects, or --skip-stages is specified"
        )

    if args.skip_stages and not args.projects:
        parser.error("--skip-stages requires --projects")

    topology = get_topology()

    # List stages mode
    if args.list_stages:
        log("Available build stages:")
        for stage in topology.get_build_stages():
            log(f"  {stage.name} ({stage.type}): {stage.description}")
        return

    if args.list_projects:
        log("Available projects (artifact: subprojects -> cmake flag):")
        # Load manifest (from build_dir if provided, otherwise repo root)
        if args.build_dir:
            manifest = topology.load_subproject_manifest(
                args.build_dir / "artifact_subprojects.json"
            )
        else:
            manifest = topology.load_subproject_manifest()
        for artifact in sorted(topology.artifacts.values(), key=lambda a: a.name):
            feature = topology.get_artifact_feature_name(artifact)
            # Get subprojects from manifest or empty list
            subprojects = manifest.get(artifact.name, []) if manifest else []
            subs = sorted(set(subprojects + artifact.split_databases))
            subs_str = f" [{', '.join(subs)}]" if subs else ""
            log(f"  {artifact.name}{subs_str} -> THEROCK_ENABLE_{feature}")
        return

    # Validate stage if provided
    if args.stage and args.stage not in topology.build_stages:
        available = ", ".join(s.name for s in topology.get_build_stages())
        parser.error(f"Unknown stage '{args.stage}'. Available stages: {available}")

    # Normalize project names (handle paths like "projects/hip" -> "hip")
    if args.projects:
        args.projects = [normalize_project_name(p) for p in args.projects]

    # Validate projects if provided (fast-fail on unknown projects)
    if args.projects:
        alias_map = topology.get_alias_to_artifact_map(args.build_dir)
        unknown = [p for p in args.projects if p.lower() not in alias_map]
        if unknown:
            parser.error(f"Unknown project(s): {', '.join(unknown)}")

    # Output skip-stages if requested
    if args.skip_stages:
        required_stages = topology.get_stages_for_projects(
            args.projects, args.build_dir
        )
        all_stages = topology.get_all_stage_names()
        skip = sorted(all_stages - required_stages)
        print(",".join(skip))
        return

    # Generate arguments
    cmake_args = generate_cmake_args(
        stage_name=args.stage,
        amdgpu_families=args.amdgpu_families,
        dist_amdgpu_families=args.dist_amdgpu_families,
        topology=topology,
        include_comments=args.comments and not args.oneline,
        platform_name=args.platform,
        manylinux=args.manylinux,
        project_names=args.projects,
        build_dir=args.build_dir,
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
        # Get python requirements for this stage (only applicable for stage mode)
        if args.stage:
            python_requires = topology.get_python_requires_for_stage(args.stage)
            pip_install_cmd = " ".join(python_requires) if python_requires else ""
        else:
            pip_install_cmd = ""
        gha_set_output({"cmake_args": output, "pip_install_cmd": pip_install_cmd})
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
