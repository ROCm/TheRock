#!/usr/bin/env python3
"""
Generate CMake includes from BUILD_TOPOLOGY.toml.

This tool reads the build topology and generates CMake targets and variables
that can be included in the main build system.
"""

import argparse
import sys
from pathlib import Path
from typing import TextIO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from _therock_utils.build_topology import BuildTopology, Artifact


def write_cmake_header(f: TextIO):
    """Write CMake file header."""
    f.write("# Auto-generated from BUILD_TOPOLOGY.toml\n")
    f.write("# DO NOT EDIT MANUALLY\n\n")


def generate_artifact_targets(topology: BuildTopology, f: TextIO):
    """Generate CMake targets for individual artifacts."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Artifact targets\n")
    f.write(
        "# =============================================================================\n\n"
    )

    for artifact in topology.get_artifacts():
        f.write(f"# Artifact: {artifact.name}\n")
        f.write(f"add_custom_target(artifact-{artifact.name}\n")
        f.write(f'  COMMENT "Building artifact {artifact.name}"\n')
        f.write(")\n\n")


def generate_artifact_group_targets(topology: BuildTopology, f: TextIO):
    """Generate CMake targets for artifact groups."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Artifact group targets\n")
    f.write(
        "# =============================================================================\n\n"
    )

    for group in topology.get_artifact_groups():
        f.write(f"# Artifact group: {group.name}\n")
        f.write(f"add_custom_target(artifact-group-{group.name}\n")
        f.write(f'  COMMENT "Building artifact group {group.name}"\n')

        # Add dependencies on artifacts in this group
        artifacts = topology.get_artifacts_in_group(group.name)
        if artifacts:
            f.write("  DEPENDS\n")
            for artifact in artifacts:
                f.write(f"    artifact-{artifact.name}\n")

        f.write(")\n\n")


def generate_build_stage_targets(topology: BuildTopology, f: TextIO):
    """Generate CMake targets for build stages."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Build stage targets\n")
    f.write(
        "# =============================================================================\n\n"
    )

    for stage in topology.get_build_stages():
        f.write(f"# Build stage: {stage.name}\n")
        f.write(f"# Type: {stage.type}\n")
        f.write(f"# Description: {stage.description}\n")
        f.write(f"add_custom_target(stage-{stage.name}\n")
        f.write(f'  COMMENT "Building stage {stage.name}"\n')

        # Add dependencies on artifact groups in this stage
        if stage.artifact_groups:
            f.write("  DEPENDS\n")
            for group_name in stage.artifact_groups:
                f.write(f"    artifact-group-{group_name}\n")

        f.write(")\n\n")


def generate_dependency_variables(topology: BuildTopology, f: TextIO):
    """Generate CMake variables for dependency information."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Dependency information\n")
    f.write(
        "# =============================================================================\n\n"
    )

    # Generate lists of artifacts per stage
    for stage in topology.get_build_stages():
        produced = topology.get_produced_artifacts(stage.name)
        inbound = topology.get_inbound_artifacts(stage.name)

        # Produced artifacts
        f.write(f"# Stage {stage.name} - produced artifacts\n")
        f.write(f"set(THEROCK_STAGE_{stage.name.upper().replace('-', '_')}_ARTIFACTS\n")
        for artifact_name in sorted(produced):
            f.write(f"  {artifact_name}\n")
        f.write(")\n\n")

        # Inbound artifacts
        f.write(f"# Stage {stage.name} - inbound artifacts\n")
        f.write(f"set(THEROCK_STAGE_{stage.name.upper().replace('-', '_')}_DEPS\n")
        for artifact_name in sorted(inbound):
            f.write(f"  {artifact_name}\n")
        f.write(")\n\n")


def generate_build_order(topology: BuildTopology, f: TextIO):
    """Generate the build order based on dependencies."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Build order\n")
    f.write(
        "# =============================================================================\n\n"
    )

    build_order = topology.get_build_order()
    f.write("# Stages in dependency order:\n")
    for i, stage_name in enumerate(build_order, 1):
        f.write(f"#   {i}. {stage_name}\n")
    f.write("\n")

    f.write("set(THEROCK_BUILD_ORDER\n")
    for stage_name in build_order:
        f.write(f"  {stage_name}\n")
    f.write(")\n\n")


def generate_feature_declarations(topology: BuildTopology, f: TextIO):
    """Generate therock_add_feature() calls from artifacts."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Feature declarations from artifacts\n")
    f.write(
        "# =============================================================================\n\n"
    )

    f.write("# Note: therock_features is already included in main CMakeLists.txt\n\n")

    # We need to generate features in dependency order to satisfy CMake's requirement
    # that dependencies are defined before they're referenced
    # Use the build order which is already topologically sorted
    artifacts_in_order = []
    for stage_name in topology.get_build_order():
        stage = topology.build_stages.get(stage_name)
        if stage:
            for group_name in stage.artifact_groups:
                for artifact in topology.get_artifacts_in_group(group_name):
                    if artifact not in artifacts_in_order:
                        artifacts_in_order.append(artifact)

    for artifact in artifacts_in_order:
        feature_name = topology.get_artifact_feature_name(artifact)
        feature_group = topology.get_artifact_feature_group(artifact)

        # Map artifact dependencies to feature names
        requires = []
        for dep_name in artifact.artifact_deps:
            dep_artifact = topology.artifacts.get(dep_name)
            if dep_artifact:
                dep_feature = topology.get_artifact_feature_name(dep_artifact)
                requires.append(dep_feature)

        # Generate the feature declaration
        f.write(f"therock_add_feature({feature_name}\n")
        f.write(f"  GROUP {feature_group}\n")
        f.write(f'  DESCRIPTION "Enables {artifact.name}"\n')

        if requires:
            f.write(f"  REQUIRES {' '.join(requires)}\n")

        if artifact.disable_platforms:
            f.write(f"  DISABLE_PLATFORMS {' '.join(artifact.disable_platforms)}\n")

        f.write(")\n\n")


def generate_validation_metadata(topology: BuildTopology, f: TextIO):
    """Generate validation metadata for fail-fast checking."""
    f.write(
        "# =============================================================================\n"
    )
    f.write("# Validation metadata\n")
    f.write(
        "# =============================================================================\n\n"
    )

    # Generate list of all valid artifacts
    f.write("# List of all valid artifacts defined in topology\n")
    f.write("set(THEROCK_TOPOLOGY_ARTIFACTS\n")
    for artifact in topology.get_artifacts():
        f.write(f"  {artifact.name}\n")
    f.write(")\n\n")

    # Generate artifact to group mapping
    f.write("# Mapping of artifacts to their groups\n")
    for artifact in topology.get_artifacts():
        if artifact.artifact_group:
            f.write(
                f"set(THEROCK_ARTIFACT_GROUP_{artifact.name.replace('-', '_')} "
                f'"{artifact.artifact_group}")\n'
            )
    f.write("\n")

    # Generate list of artifacts per group
    f.write("# List of artifacts in each group\n")
    for group in topology.get_artifact_groups():
        f.write(f"set(THEROCK_GROUP_ARTIFACTS_{group.name.replace('-', '_')}\n")
        artifacts_in_group = topology.get_artifacts_in_group(group.name)
        for artifact in artifacts_in_group:
            f.write(f"  {artifact.name}\n")
        f.write(")\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CMake includes from BUILD_TOPOLOGY.toml"
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="BUILD_TOPOLOGY.toml",
        help="Path to BUILD_TOPOLOGY.toml file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cmake/therock_topology_generated.cmake",
        help="Output CMake file path",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the topology without generating output",
    )
    parser.add_argument(
        "--print-graph", action="store_true", help="Print the dependency graph as JSON"
    )

    args = parser.parse_args()

    # Find the topology file
    if not Path(args.topology).is_absolute():
        # Look for it relative to the script location
        script_dir = Path(__file__).parent.parent
        topology_path = script_dir / args.topology
    else:
        topology_path = Path(args.topology)

    if not topology_path.exists():
        print(f"Error: Topology file not found: {topology_path}", file=sys.stderr)
        sys.exit(1)

    # Load and validate the topology
    try:
        topology = BuildTopology(str(topology_path))
    except Exception as e:
        print(f"Error loading topology: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    errors = topology.validate_topology()
    if errors:
        print("Topology validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print("Topology validation successful!")
        return

    if args.print_graph:
        import json

        graph = topology.get_dependency_graph()
        print(json.dumps(graph, indent=2))
        return

    # Generate CMake output
    output_path = Path(args.output)
    if not output_path.is_absolute():
        script_dir = Path(__file__).parent.parent
        output_path = script_dir / output_path

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        write_cmake_header(f)
        generate_validation_metadata(topology, f)
        generate_feature_declarations(topology, f)
        generate_artifact_targets(topology, f)
        generate_artifact_group_targets(topology, f)
        generate_build_stage_targets(topology, f)
        generate_dependency_variables(topology, f)
        generate_build_order(topology, f)

    print(f"Generated CMake includes at: {output_path}")

    # Print summary
    stages = topology.get_build_stages()
    groups = topology.get_artifact_groups()
    artifacts = topology.get_artifacts()

    print(f"\nTopology summary:")
    print(f"  Build stages: {len(stages)}")
    print(f"  Artifact groups: {len(groups)}")
    print(f"  Artifacts: {len(artifacts)}")


if __name__ == "__main__":
    main()
