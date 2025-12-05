"""
Build topology parsing and manipulation for TheRock CI/CD pipeline.

This module provides classes and utilities for parsing BUILD_TOPOLOGY.toml
and computing artifact dependencies for sharded build pipelines.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class Submodule:
    """Represents a git submodule with checkout configuration.

    This class is designed to be extended with additional fields:
    - sparse_checkout: List[str] - paths to include in sparse checkout
    - recursive: bool - whether to recursively init submodules
    - depth: int - shallow clone depth
    """

    name: str
    # Future fields for sparse checkout, recursive settings, etc.

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Submodule):
            return self.name == other.name
        return False


@dataclass
class SourceSet:
    """Represents a grouping of submodules for partial checkouts."""

    name: str
    description: str
    submodules: List[Submodule] = field(default_factory=list)


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
    source_sets: List[str] = field(default_factory=list)


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
    python_requires: List[str] = field(
        default_factory=list
    )  # pip install args (e.g., ["-r path/to/req.txt"])


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
        self.source_sets: Dict[str, SourceSet] = {}
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

        # Parse source sets
        for set_name, set_data in data.get("source_sets", {}).items():
            # Convert submodule names to Submodule objects
            submodule_names = set_data.get("submodules", [])
            submodules = [Submodule(name=name) for name in submodule_names]
            self.source_sets[set_name] = SourceSet(
                name=set_name,
                description=set_data.get("description", ""),
                submodules=submodules,
            )

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
                source_sets=group_data.get("source_sets", []),
            )

        # Parse artifacts
        for artifact_name, artifact_data in data.get("artifacts", {}).items():
            python_requires = artifact_data.get("python_requires", [])
            if python_requires and not isinstance(python_requires, list):
                raise ValueError(
                    f"Artifact '{artifact_name}' python_requires must be a list, "
                    f"got {type(python_requires).__name__}"
                )
            self.artifacts[artifact_name] = Artifact(
                name=artifact_name,
                artifact_group=artifact_data.get("artifact_group", ""),
                type=artifact_data.get("type", "target-neutral"),
                artifact_deps=artifact_data.get("artifact_deps", []),
                platform=artifact_data.get("platform"),
                feature_name=artifact_data.get("feature_name"),
                feature_group=artifact_data.get("feature_group"),
                disable_platforms=artifact_data.get("disable_platforms", []),
                python_requires=python_requires,
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

    def _validate_naming_conventions(self) -> List[str]:
        """
        Validate naming conventions for all topology entities.

        Conventions:
        - Entity names (stages, groups, artifacts): lowercase with hyphens
        - feature_name: UPPERCASE with underscores
        - feature_group: UPPERCASE with underscores
        - type values: lowercase
        - platform values: lowercase

        Returns:
            List of validation error messages
        """
        import re

        errors = []

        # Pattern for entity names: lowercase letters, numbers, and hyphens
        entity_pattern = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
        # Pattern for feature names/groups: uppercase letters, numbers, and underscores
        feature_pattern = re.compile(r"^[A-Z0-9]+(_[A-Z0-9]+)*$")

        # Valid type values
        valid_stage_types = {"generic", "per-arch"}
        valid_artifact_types = {"target-neutral", "target-specific"}
        valid_platforms = {"windows", "linux"}

        # Validate build stage names and types
        for stage_name, stage in self.build_stages.items():
            if not entity_pattern.match(stage_name):
                errors.append(
                    f"Build stage '{stage_name}' should be lowercase-with-hyphens"
                )
            if stage.type not in valid_stage_types:
                errors.append(
                    f"Build stage '{stage_name}' has invalid type '{stage.type}' "
                    f"(expected: {valid_stage_types})"
                )

        # Validate artifact group names and types
        for group_name, group in self.artifact_groups.items():
            if not entity_pattern.match(group_name):
                errors.append(
                    f"Artifact group '{group_name}' should be lowercase-with-hyphens"
                )
            if group.type not in valid_stage_types:
                errors.append(
                    f"Artifact group '{group_name}' has invalid type '{group.type}' "
                    f"(expected: {valid_stage_types})"
                )

        # Validate artifact names, types, and feature overrides
        for artifact_name, artifact in self.artifacts.items():
            if not entity_pattern.match(artifact_name):
                errors.append(
                    f"Artifact '{artifact_name}' should be lowercase-with-hyphens"
                )
            if artifact.type not in valid_artifact_types:
                errors.append(
                    f"Artifact '{artifact_name}' has invalid type '{artifact.type}' "
                    f"(expected: {valid_artifact_types})"
                )
            if artifact.feature_name and not feature_pattern.match(
                artifact.feature_name
            ):
                errors.append(
                    f"Artifact '{artifact_name}' feature_name '{artifact.feature_name}' "
                    f"should be UPPERCASE_WITH_UNDERSCORES"
                )
            if artifact.feature_group and not feature_pattern.match(
                artifact.feature_group
            ):
                errors.append(
                    f"Artifact '{artifact_name}' feature_group '{artifact.feature_group}' "
                    f"should be UPPERCASE_WITH_UNDERSCORES"
                )
            if artifact.platform and artifact.platform not in valid_platforms:
                errors.append(
                    f"Artifact '{artifact_name}' has invalid platform '{artifact.platform}' "
                    f"(expected: {valid_platforms})"
                )
            for platform in artifact.disable_platforms:
                if platform not in valid_platforms:
                    errors.append(
                        f"Artifact '{artifact_name}' has invalid disable_platform '{platform}' "
                        f"(expected: {valid_platforms})"
                    )

        return errors

    def validate_topology(self) -> List[str]:
        """
        Validate topology for cycles, missing references, naming conventions, etc.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate naming conventions
        errors.extend(self._validate_naming_conventions())

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

    def get_source_sets(self) -> List[SourceSet]:
        """Get all source sets."""
        return list(self.source_sets.values())

    def get_submodules_for_source_set(self, source_set_name: str) -> List[Submodule]:
        """
        Get the submodules for a specific source set.

        Args:
            source_set_name: Name of the source set

        Returns:
            List of Submodule objects
        """
        if source_set_name not in self.source_sets:
            raise ValueError(f"Source set '{source_set_name}' not found")
        return self.source_sets[source_set_name].submodules

    def get_submodules_for_stage(self, build_stage: str) -> List[Submodule]:
        """
        Get all submodules needed to build a specific stage.

        This collects source_sets from all artifact_groups in the stage,
        deduplicating by submodule name. When sparse checkout is added,
        this will need to merge specs for the same submodule.

        Args:
            build_stage: Name of the build stage

        Returns:
            List of Submodule objects needed for this stage
        """
        if build_stage not in self.build_stages:
            raise ValueError(f"Build stage '{build_stage}' not found")

        stage = self.build_stages[build_stage]
        # Use dict to dedupe by name while preserving order
        submodules_by_name: Dict[str, Submodule] = {}

        for group_name in stage.artifact_groups:
            if group_name not in self.artifact_groups:
                continue
            group = self.artifact_groups[group_name]
            for source_set_name in group.source_sets:
                if source_set_name in self.source_sets:
                    for submodule in self.source_sets[source_set_name].submodules:
                        # TODO: When adding sparse_checkout, merge specs here
                        if submodule.name not in submodules_by_name:
                            submodules_by_name[submodule.name] = submodule

        return list(submodules_by_name.values())

    def get_all_submodules(self) -> List[Submodule]:
        """
        Get all submodules defined across all source sets.

        Returns:
            List of all Submodule objects (deduplicated by name)
        """
        submodules_by_name: Dict[str, Submodule] = {}
        for source_set in self.source_sets.values():
            for submodule in source_set.submodules:
                if submodule.name not in submodules_by_name:
                    submodules_by_name[submodule.name] = submodule
        return list(submodules_by_name.values())

    def get_python_requires_for_stage(self, build_stage: str) -> List[str]:
        """
        Get all python_requires for artifacts produced by a build stage.

        Collects python_requires from all artifacts in the stage's artifact groups,
        returning them as a deduplicated list suitable for passing to pip install.

        Args:
            build_stage: Name of the build stage

        Returns:
            List of pip install arguments (e.g., ["-r path/to/req.txt", "package"])
        """
        if build_stage not in self.build_stages:
            raise ValueError(f"Build stage '{build_stage}' not found")

        stage = self.build_stages[build_stage]
        seen: Set[str] = set()
        requires: List[str] = []

        # Collect python_requires from artifacts in this stage's groups
        for group_name in stage.artifact_groups:
            for artifact in self.get_artifacts_in_group(group_name):
                for req in artifact.python_requires:
                    if req not in seen:
                        seen.add(req)
                        requires.append(req)

        return requires
