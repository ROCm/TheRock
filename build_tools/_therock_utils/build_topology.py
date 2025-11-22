"""
Build topology parsing and manipulation for TheRock CI/CD pipeline.

This module provides classes and utilities for parsing BUILD_TOPOLOGY.toml
and computing artifact dependencies for sharded build pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class BuildStage:
    """Represents a build stage (CI/CD pipeline job)."""

    name: str
    description: str
    artifact_groups: List[str]
    type: str = "generic"  # "generic" or "per-arch"


@dataclass
class ArtifactGroup:
    """Represents a logical grouping of related artifacts."""

    name: str
    description: str
    type: str  # "generic" or "per-arch"
    artifact_group_deps: List[str] = field(default_factory=list)


@dataclass
class Artifact:
    """Represents an individual build output."""

    name: str
    artifact_group: str
    type: str  # "target-neutral" or "target-specific"
    artifact_deps: List[str] = field(default_factory=list)
    platform: Optional[str] = None  # e.g., "windows"
    feature_name: Optional[str] = None  # Override default feature name
    feature_group: Optional[str] = None  # Override default feature group
    disable_platforms: List[str] = field(
        default_factory=list
    )  # Platforms where disabled


class BuildTopology:
    """
    Parses and provides operations on BUILD_TOPOLOGY.toml.

    This is the main interface for CI/CD pipelines to understand
    build dependencies and artifact relationships.
    """

    def __init__(self, toml_path: str):
        """
        Load and parse BUILD_TOPOLOGY.toml.

        Args:
            toml_path: Path to BUILD_TOPOLOGY.toml file
        """
        self.toml_path = Path(toml_path)
        self.build_stages: Dict[str, BuildStage] = {}
        self.artifact_groups: Dict[str, ArtifactGroup] = {}
        self.artifacts: Dict[str, Artifact] = {}

        self._load_topology()

    def _load_topology(self):
        """Load and parse the TOML file."""
        # Python version compatibility for TOML parsing
        try:
            import tomllib
        except ModuleNotFoundError:
            # Python <= 3.10 compatibility (requires install of 'tomli' package)
            import tomli as tomllib

        with open(self.toml_path, "rb") as f:
            data = tomllib.load(f)

        # Parse build stages
        for stage_name, stage_data in data.get("build_stages", {}).items():
            self.build_stages[stage_name] = BuildStage(
                name=stage_name,
                description=stage_data.get("description", ""),
                artifact_groups=stage_data.get("artifact_groups", []),
                type=stage_data.get("type", "generic"),
            )

        # Parse artifact groups
        for group_name, group_data in data.get("artifact_groups", {}).items():
            self.artifact_groups[group_name] = ArtifactGroup(
                name=group_name,
                description=group_data.get("description", ""),
                type=group_data.get("type", "generic"),
                artifact_group_deps=group_data.get("artifact_group_deps", []),
            )

        # Parse artifacts
        for artifact_name, artifact_data in data.get("artifacts", {}).items():
            self.artifacts[artifact_name] = Artifact(
                name=artifact_name,
                artifact_group=artifact_data.get("artifact_group", ""),
                type=artifact_data.get("type", "target-neutral"),
                artifact_deps=artifact_data.get("artifact_deps", []),
                platform=artifact_data.get("platform"),
                feature_name=artifact_data.get("feature_name"),
                feature_group=artifact_data.get("feature_group"),
                disable_platforms=artifact_data.get("disable_platforms", []),
            )

    def get_build_stages(self) -> List[BuildStage]:
        """Get all build stages."""
        return list(self.build_stages.values())

    def get_artifact_groups(self) -> List[ArtifactGroup]:
        """Get all artifact groups."""
        return list(self.artifact_groups.values())

    def get_artifacts(self) -> List[Artifact]:
        """Get all artifacts."""
        return list(self.artifacts.values())

    def get_artifact_feature_name(self, artifact: Artifact) -> str:
        """Get the effective feature name for an artifact."""
        if artifact.feature_name:
            return artifact.feature_name
        # Default rule: uppercase and replace - with _
        return artifact.name.upper().replace("-", "_")

    def get_artifact_feature_group(self, artifact: Artifact) -> str:
        """Get the effective feature group for an artifact."""
        if artifact.feature_group:
            return artifact.feature_group
        # Default rule: uppercase artifact_group and replace - with _
        return artifact.artifact_group.upper().replace("-", "_")

    def get_artifacts_in_group(self, group_name: str) -> List[Artifact]:
        """Get all artifacts belonging to a specific artifact group."""
        return [a for a in self.artifacts.values() if a.artifact_group == group_name]

    def get_inbound_artifacts(self, build_stage: str) -> Set[str]:
        """
        Get all artifacts needed by a build stage from previous stages.

        This is the key method for CI/CD pipelines to determine what
        artifacts need to be fetched from S3 before building.

        Args:
            build_stage: Name of the build stage

        Returns:
            Set of artifact names that this stage depends on
        """
        if build_stage not in self.build_stages:
            raise ValueError(f"Build stage '{build_stage}' not found")

        stage = self.build_stages[build_stage]
        inbound_artifacts = set()

        # Get all artifact groups this stage contains
        stage_groups = set(stage.artifact_groups)

        # For each artifact group in this stage, collect its dependencies
        for group_name in stage_groups:
            if group_name not in self.artifact_groups:
                continue

            group = self.artifact_groups[group_name]

            # Get all artifacts from dependent groups (transitively)
            for dep_group_name in group.artifact_group_deps:
                dep_artifacts = self.get_artifacts_in_group(dep_group_name)
                inbound_artifacts.update(a.name for a in dep_artifacts)

        # Also collect direct artifact dependencies from artifacts in this stage
        # This includes transitive artifact dependencies
        for artifact in self.artifacts.values():
            if artifact.artifact_group in stage_groups:
                # Add direct dependencies
                for dep_name in artifact.artifact_deps:
                    inbound_artifacts.add(dep_name)
                    # Also add transitive dependencies
                    self._collect_transitive_artifact_deps(dep_name, inbound_artifacts)

        # Remove artifacts that are produced by this stage itself
        produced = self.get_produced_artifacts(build_stage)
        inbound_artifacts -= produced

        return inbound_artifacts

    def _collect_transitive_artifact_deps(
        self, artifact_name: str, collected: Set[str]
    ):
        """
        Recursively collect all transitive dependencies of an artifact.

        Args:
            artifact_name: Name of the artifact to get dependencies for
            collected: Set to add dependencies to (modified in place)
        """
        if artifact_name not in self.artifacts:
            return

        artifact = self.artifacts[artifact_name]
        for dep_name in artifact.artifact_deps:
            if dep_name not in collected:
                # Add to collected set BEFORE recursing to prevent revisiting
                # the same node in diamond dependency patterns
                collected.add(dep_name)
                self._collect_transitive_artifact_deps(dep_name, collected)

    def get_produced_artifacts(self, build_stage: str) -> Set[str]:
        """
        Get all artifacts produced by a build stage.

        Args:
            build_stage: Name of the build stage

        Returns:
            Set of artifact names produced by this stage
        """
        if build_stage not in self.build_stages:
            raise ValueError(f"Build stage '{build_stage}' not found")

        stage = self.build_stages[build_stage]
        produced_artifacts = set()

        # Collect all artifacts from the groups in this stage
        for group_name in stage.artifact_groups:
            artifacts_in_group = self.get_artifacts_in_group(group_name)
            produced_artifacts.update(a.name for a in artifacts_in_group)

        return produced_artifacts

    def validate_topology(self) -> List[str]:
        """
        Validate topology for cycles, missing references, etc.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for missing artifact group references in stages
        for stage in self.build_stages.values():
            for group_name in stage.artifact_groups:
                if group_name not in self.artifact_groups:
                    errors.append(
                        f"Stage '{stage.name}' references unknown artifact group '{group_name}'"
                    )

        # Check for missing artifact group references in dependencies
        for group in self.artifact_groups.values():
            for dep_name in group.artifact_group_deps:
                if dep_name not in self.artifact_groups:
                    errors.append(
                        f"Artifact group '{group.name}' depends on unknown group '{dep_name}'"
                    )

        # Check for missing artifact references
        for artifact in self.artifacts.values():
            # Check artifact group reference
            if (
                artifact.artifact_group
                and artifact.artifact_group not in self.artifact_groups
            ):
                errors.append(
                    f"Artifact '{artifact.name}' references unknown group '{artifact.artifact_group}'"
                )

            # Check artifact dependencies
            for dep_name in artifact.artifact_deps:
                if dep_name not in self.artifacts:
                    errors.append(
                        f"Artifact '{artifact.name}' depends on unknown artifact '{dep_name}'"
                    )

        # Check for circular dependencies in artifact groups
        visited = set()
        rec_stack = set()

        def has_cycle(group_name: str) -> bool:
            visited.add(group_name)
            rec_stack.add(group_name)

            if group_name in self.artifact_groups:
                for dep_name in self.artifact_groups[group_name].artifact_group_deps:
                    if dep_name not in visited:
                        if has_cycle(dep_name):
                            return True
                    elif dep_name in rec_stack:
                        errors.append(
                            f"Circular dependency detected involving artifact group '{dep_name}'"
                        )
                        return True

            rec_stack.remove(group_name)
            return False

        for group_name in self.artifact_groups:
            if group_name not in visited:
                has_cycle(group_name)

        # Check for circular dependencies in artifacts
        visited_artifacts = set()
        rec_stack_artifacts = set()

        def has_artifact_cycle(artifact_name: str) -> bool:
            visited_artifacts.add(artifact_name)
            rec_stack_artifacts.add(artifact_name)

            if artifact_name in self.artifacts:
                for dep_name in self.artifacts[artifact_name].artifact_deps:
                    if dep_name not in visited_artifacts:
                        if has_artifact_cycle(dep_name):
                            return True
                    elif dep_name in rec_stack_artifacts:
                        errors.append(
                            f"Circular dependency detected involving artifact '{dep_name}'"
                        )
                        return True

            rec_stack_artifacts.remove(artifact_name)
            return False

        for artifact_name in self.artifacts:
            if artifact_name not in visited_artifacts:
                has_artifact_cycle(artifact_name)

        return errors

    def get_dependency_graph(self) -> Dict:
        """
        Generate full dependency graph for visualization.

        Returns:
            Dictionary representation of the dependency graph
        """
        graph = {"build_stages": {}, "artifact_groups": {}, "artifacts": {}}

        # Build stages graph
        for stage in self.build_stages.values():
            graph["build_stages"][stage.name] = {
                "type": stage.type,
                "artifact_groups": stage.artifact_groups,
                "inbound_artifacts": list(self.get_inbound_artifacts(stage.name)),
                "produced_artifacts": list(self.get_produced_artifacts(stage.name)),
            }

        # Artifact groups graph
        for group in self.artifact_groups.values():
            graph["artifact_groups"][group.name] = {
                "type": group.type,
                "depends_on": group.artifact_group_deps,
                "artifacts": [a.name for a in self.get_artifacts_in_group(group.name)],
            }

        # Artifacts graph
        for artifact in self.artifacts.values():
            graph["artifacts"][artifact.name] = {
                "type": artifact.type,
                "artifact_group": artifact.artifact_group,
                "depends_on": artifact.artifact_deps,
                "platform": artifact.platform,
            }

        return graph

    def get_build_order(self) -> List[str]:
        """
        Get the build order for stages based on dependencies.

        Returns:
            List of build stage names in order they should be built
        """
        # Build a dependency graph for stages based on artifact groups
        stage_deps = {}
        for stage_name, stage in self.build_stages.items():
            deps = set()
            for group_name in stage.artifact_groups:
                if group_name in self.artifact_groups:
                    group = self.artifact_groups[group_name]
                    # Find which stages produce the dependent groups
                    for dep_group in group.artifact_group_deps:
                        for other_stage_name, other_stage in self.build_stages.items():
                            if dep_group in other_stage.artifact_groups:
                                deps.add(other_stage_name)
            stage_deps[stage_name] = deps

        # Topological sort
        visited = set()
        order = []

        def visit(stage_name: str):
            if stage_name in visited:
                return
            visited.add(stage_name)
            for dep in stage_deps.get(stage_name, set()):
                visit(dep)
            order.append(stage_name)

        for stage_name in self.build_stages:
            visit(stage_name)

        return order
