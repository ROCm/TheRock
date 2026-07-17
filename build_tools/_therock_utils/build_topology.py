# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Build topology parsing and manipulation for TheRock CI/CD pipeline.

This module provides classes and utilities for parsing BUILD_TOPOLOGY.toml
and computing artifact dependencies for sharded build pipelines.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


def get_topology(topology_path: Optional[Path] = None) -> "BuildTopology":
    """Load the BUILD_TOPOLOGY.toml from the default or specified location."""
    if topology_path is None:
        # Default: BUILD_TOPOLOGY.toml in repository root
        # This module is at build_tools/_therock_utils/build_topology.py
        script_dir = Path(__file__).parent
        repo_root = script_dir.parent.parent
        topology_path = repo_root / "BUILD_TOPOLOGY.toml"

    if not topology_path.exists():
        raise FileNotFoundError(f"BUILD_TOPOLOGY.toml not found at {topology_path}")

    return BuildTopology(str(topology_path))


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
class ExternalGitSource:
    """Represents an externally managed git checkout.

    External git sources are not git submodules. They are fetched into ignored
    locations under optional-sources/ and pinned by commit hash.
    """

    name: str
    origin: str
    commit: str
    path: str

    def __hash__(self):
        return hash((self.name, self.path))


@dataclass
class SourceSet:
    """Represents a grouping of submodules for partial checkouts."""

    name: str
    description: str
    submodules: List[Submodule] = field(default_factory=list)
    external_git_sources: List[ExternalGitSource] = field(default_factory=list)
    disable_platforms: List[str] = field(default_factory=list)
    path_prefixes: List[str] = field(default_factory=list)


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
    disable_platforms_if_flags_not_set: Dict[str, str] = field(
        default_factory=dict
    )  # Platforms disabled unless the named build flag is set
    python_requires: List[str] = field(
        default_factory=list
    )  # pip install args (e.g., ["-r path/to/req.txt"])
    split_databases: List[str] = field(
        default_factory=list
    )  # Database handlers to use when splitting artifacts (e.g., ["rocblas", "hipblaslt"])
    test_artifacts: List[str] = field(
        default_factory=list
    )  # Artifacts needed for testing this artifact (e.g., ["core-hiptests"])


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
            external_git_sources = [
                ExternalGitSource(
                    name=source_data.get("name", ""),
                    origin=source_data.get("origin", ""),
                    commit=source_data.get("commit", ""),
                    path=source_data.get("path", ""),
                )
                for source_data in set_data.get("external_git_sources", [])
            ]
            self.source_sets[set_name] = SourceSet(
                name=set_name,
                description=set_data.get("description", ""),
                submodules=submodules,
                external_git_sources=external_git_sources,
                disable_platforms=set_data.get("disable_platforms", []),
                path_prefixes=set_data.get("path_prefixes", []),
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
            disable_platforms_if_flags_not_set = artifact_data.get(
                "disable_platforms_if_flags_not_set", {}
            )
            if disable_platforms_if_flags_not_set and not isinstance(
                disable_platforms_if_flags_not_set, dict
            ):
                raise ValueError(
                    f"Artifact '{artifact_name}' disable_platforms_if_flags_not_set "
                    f"must be a table, got {type(disable_platforms_if_flags_not_set).__name__}"
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
                disable_platforms_if_flags_not_set=disable_platforms_if_flags_not_set,
                python_requires=python_requires,
                split_databases=artifact_data.get("split_databases", []),
                test_artifacts=artifact_data.get("test_artifacts", []),
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

    def is_artifact_disabled_on_platform(
        self,
        artifact: Artifact,
        platform_name: str,
        enabled_flags: Optional[Set[str]] = None,
    ) -> bool:
        """Return whether an artifact is disabled for a platform and flag set."""
        if not platform_name:
            return False
        if platform_name in artifact.disable_platforms:
            return True
        required_flag = artifact.disable_platforms_if_flags_not_set.get(platform_name)
        if not required_flag:
            return False
        enabled_flags = enabled_flags or set()
        return required_flag not in enabled_flags

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
                for artifact in dep_artifacts:
                    inbound_artifacts.add(artifact.name)
                    self._collect_transitive_artifact_deps(
                        artifact.name, inbound_artifacts
                    )

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
            for platform, flag in artifact.disable_platforms_if_flags_not_set.items():
                if platform not in valid_platforms:
                    errors.append(
                        f"Artifact '{artifact_name}' has invalid conditional disable_platform '{platform}' "
                        f"(expected: {valid_platforms})"
                    )
                if not isinstance(flag, str) or not feature_pattern.match(flag):
                    errors.append(
                        f"Artifact '{artifact_name}' disable_platforms_if_flags_not_set "
                        f"entry for '{platform}' should be UPPERCASE_WITH_UNDERSCORES"
                    )

        # Validate source set disable_platforms
        for source_set_name, source_set in self.source_sets.items():
            if not entity_pattern.match(source_set_name):
                errors.append(
                    f"Source set '{source_set_name}' should be lowercase-with-hyphens"
                )
            for platform in source_set.disable_platforms:
                if platform not in valid_platforms:
                    errors.append(
                        f"Source set '{source_set_name}' has invalid disable_platform '{platform}' "
                        f"(expected: {valid_platforms})"
                    )
            for source in source_set.external_git_sources:
                if not source.name:
                    errors.append(
                        f"Source set '{source_set_name}' has external git source with missing name"
                    )
                elif not entity_pattern.match(source.name):
                    errors.append(
                        f"External git source '{source.name}' should be lowercase-with-hyphens"
                    )
                if not source.origin:
                    errors.append(
                        f"External git source '{source.name}' in source set "
                        f"'{source_set_name}' has missing origin"
                    )
                if not source.commit:
                    errors.append(
                        f"External git source '{source.name}' in source set "
                        f"'{source_set_name}' has missing commit"
                    )
                if not source.path:
                    errors.append(
                        f"External git source '{source.name}' in source set "
                        f"'{source_set_name}' has missing path"
                    )
                else:
                    path = Path(source.path)
                    if path.is_absolute():
                        errors.append(
                            f"External git source '{source.name}' path '{source.path}' "
                            "must be relative"
                        )
                    path_parts = path.parts
                    if (
                        not path_parts
                        or path_parts[0] != "optional-sources"
                        or ".." in path_parts
                        or len(path_parts) < 2
                    ):
                        errors.append(
                            f"External git source '{source.name}' path '{source.path}' "
                            "must be under optional-sources/"
                        )
            for prefix in source_set.path_prefixes:
                if not prefix:
                    errors.append(
                        f"Source set '{source_set_name}' has an empty path_prefix"
                    )
                    continue

                prefix_path = Path(prefix)
                if prefix_path.is_absolute():
                    errors.append(
                        f"Source set '{source_set_name}' path_prefix '{prefix}' "
                        "must be relative"
                    )
                if ".." in prefix_path.parts:
                    errors.append(
                        f"Source set '{source_set_name}' path_prefix '{prefix}' "
                        "must not contain '..'"
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
            for source_set_name in group.source_sets:
                if source_set_name not in self.source_sets:
                    errors.append(
                        f"Artifact group '{group.name}' references unknown source set "
                        f"'{source_set_name}'"
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

        # Check for conflicting external source paths.
        external_sources_by_path: Dict[str, str] = {}
        for source_set in self.source_sets.values():
            for source in source_set.external_git_sources:
                previous_source_set = external_sources_by_path.get(source.path)
                if previous_source_set and previous_source_set != source_set.name:
                    errors.append(
                        f"External git source path '{source.path}' is used by both "
                        f"source sets '{previous_source_set}' and '{source_set.name}'"
                    )
                else:
                    external_sources_by_path[source.path] = source_set.name

        # Check for conflicting submodule ownership.
        submodule_owner: Dict[str, str] = {}
        for source_set in self.source_sets.values():
            for submodule in source_set.submodules:
                previous_source_set = submodule_owner.get(submodule.name)
                if previous_source_set and previous_source_set != source_set.name:
                    errors.append(
                        f"Submodule '{submodule.name}' is used by both "
                        f"source sets '{previous_source_set}' and '{source_set.name}'"
                    )
                else:
                    submodule_owner[submodule.name] = source_set.name

        # Check for conflicting path prefix ownership.
        path_prefix_owner: Dict[str, str] = {}
        for source_set in self.source_sets.values():
            for prefix in source_set.path_prefixes:
                normalized_prefix = prefix.rstrip("/")
                previous_source_set = path_prefix_owner.get(normalized_prefix)

                if previous_source_set and previous_source_set != source_set.name:
                    errors.append(
                        f"Path prefix '{normalized_prefix}' is used by both "
                        f"source sets '{previous_source_set}' and '{source_set.name}'"
                    )
                else:
                    path_prefix_owner[normalized_prefix] = source_set.name

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

    def get_source_set_to_artifact_groups(self) -> Dict[str, List[str]]:
        """
        Get a reverse index from source set names to artifact group names.

        Returns:
            Dictionary mapping source set names to artifact group names that
            reference them. Known source sets with no artifact group references
            are included with an empty list.
        """
        artifact_groups_by_source_set = {
            source_set_name: [] for source_set_name in self.source_sets
        }
        for group in self.artifact_groups.values():
            for source_set_name in group.source_sets:
                artifact_groups_by_source_set.setdefault(source_set_name, []).append(
                    group.name
                )
        return artifact_groups_by_source_set

    def get_artifact_group_to_artifacts(self) -> Dict[str, List[str]]:
        """
        Get an index from artifact group names to artifact names.

        Returns:
            Dictionary mapping artifact group names to artifact names in that
            group. Known artifact groups with no artifacts are included with an
            empty list.
        """
        artifacts_by_group = {group_name: [] for group_name in self.artifact_groups}
        for artifact in self.artifacts.values():
            artifacts_by_group.setdefault(artifact.artifact_group, []).append(
                artifact.name
            )
        return artifacts_by_group

    def get_artifact_group_to_build_stages(self) -> Dict[str, List[str]]:
        """
        Get a reverse index from artifact group names to producer build stages.

        Returns:
            Dictionary mapping artifact group names to build stage names that
            list the group. Known artifact groups with no producing stage are
            included with an empty list.
        """
        stages_by_group = {group_name: [] for group_name in self.artifact_groups}
        for stage in self.build_stages.values():
            for group_name in stage.artifact_groups:
                stages_by_group.setdefault(group_name, []).append(stage.name)
        return stages_by_group

    def get_artifact_to_producer_stages(self) -> Dict[str, List[str]]:
        """
        Get an index from artifact names to producer build stages.

        Returns:
            Dictionary mapping artifact names to build stage names that produce
            their artifact group. Artifacts in unmapped groups are included with
            an empty list.
        """
        stages_by_group = self.get_artifact_group_to_build_stages()
        return {
            artifact.name: list(stages_by_group.get(artifact.artifact_group, []))
            for artifact in self.artifacts.values()
        }

    def get_stage_to_source_sets(
        self, platform: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get an index from build stages to source set names.

        Args:
            platform: Current platform (e.g., "linux", "windows"). If provided,
                source_sets with this platform in disable_platforms are skipped.

        Returns:
            Dictionary mapping build stage names to source set names used by the
            stage's artifact groups.
        """
        return {
            stage_name: [
                source_set.name
                for source_set in self.get_source_sets_for_stage(
                    stage_name, platform=platform
                )
            ]
            for stage_name in self.build_stages
        }

    def get_source_set_to_stages(
        self, platform: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get a reverse index from source set names to build stages.

        Args:
            platform: Current platform (e.g., "linux", "windows"). If provided,
                source_sets with this platform in disable_platforms are skipped.

        Returns:
            Dictionary mapping source set names to build stage names that use
            them through artifact groups. Known source sets with no build stage
            usage are included with an empty list.
        """
        stages_by_source_set = {
            source_set_name: [] for source_set_name in self.source_sets
        }
        for stage_name, source_set_names in self.get_stage_to_source_sets(
            platform=platform
        ).items():
            for source_set_name in source_set_names:
                stages_by_source_set.setdefault(source_set_name, []).append(stage_name)
        return stages_by_source_set

    def get_submodule_to_source_set(self) -> Dict[str, str]:
        """
        Get a reverse index from submodule names to source set names.
        """
        mapping: Dict[str, str] = {}
        for source_set in self.source_sets.values():
            for submodule in source_set.submodules:
                mapping[submodule.name] = source_set.name
        return mapping

    def get_source_sets(self) -> List[SourceSet]:
        """Get all source sets."""
        return list(self.source_sets.values())

    def get_source_set_for_submodule(
        self, submodule_name: str, platform: Optional[str] = None
    ) -> Optional[SourceSet]:
        """
        Return the first source set whose submodules include `submodule_name`.

        Args:
            submodule_name: Name of the git submodule.
            platform: Optional platform filter.

        Returns:
            Matching SourceSet, or None if no match is found.
        """
        for source_set in self.source_sets.values():
            if platform and platform in source_set.disable_platforms:
                continue

            for submodule in source_set.submodules:
                if submodule.name == submodule_name:
                    return source_set

        return None

    def get_source_set_for_path(
        self, path: str, platform: Optional[str] = None
    ) -> Optional[SourceSet]:
        """
        Return the first source set whose path_prefixes match the given path.
        """
        normalized_path = path.strip().lstrip("./")

        for source_set in self.source_sets.values():
            if platform and platform in source_set.disable_platforms:
                continue

            for prefix in source_set.path_prefixes:
                prefix = prefix.rstrip("/")
                if normalized_path == prefix or normalized_path.startswith(
                    prefix + "/"
                ):
                    return source_set

        return None

    def get_source_sets_for_submodules(
        self, submodule_names: List[str], platform: Optional[str] = None
    ) -> List[SourceSet]:
        """
        Return deduplicated source sets for a list of submodule names.
        """
        source_sets_by_name: Dict[str, SourceSet] = {}

        for submodule_name in submodule_names:
            source_set = self.get_source_set_for_submodule(
                submodule_name, platform=platform
            )
            if source_set and source_set.name not in source_sets_by_name:
                source_sets_by_name[source_set.name] = source_set

        return list(source_sets_by_name.values())

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

    def get_external_git_sources_for_source_set(
        self, source_set_name: str
    ) -> List[ExternalGitSource]:
        """
        Get the external git sources for a specific source set.

        Args:
            source_set_name: Name of the source set

        Returns:
            List of ExternalGitSource objects
        """
        if source_set_name not in self.source_sets:
            raise ValueError(f"Source set '{source_set_name}' not found")
        return self.source_sets[source_set_name].external_git_sources

    def get_source_sets_for_stage(
        self, build_stage: str, platform: Optional[str] = None
    ) -> List[SourceSet]:
        """
        Get all source sets needed to build a specific stage.

        Args:
            build_stage: Name of the build stage
            platform: Current platform (e.g., "linux", "windows"). If provided,
                source_sets with this platform in disable_platforms are skipped.

        Returns:
            List of SourceSet objects needed for this stage
        """
        if build_stage not in self.build_stages:
            raise ValueError(f"Build stage '{build_stage}' not found")

        stage = self.build_stages[build_stage]
        source_sets_by_name: Dict[str, SourceSet] = {}

        for group_name in stage.artifact_groups:
            if group_name not in self.artifact_groups:
                continue
            group = self.artifact_groups[group_name]
            for source_set_name in group.source_sets:
                if source_set_name in self.source_sets:
                    source_set = self.source_sets[source_set_name]
                    if platform and platform in source_set.disable_platforms:
                        continue
                    if source_set.name not in source_sets_by_name:
                        source_sets_by_name[source_set.name] = source_set

        return list(source_sets_by_name.values())

    def get_submodules_for_stage(
        self, build_stage: str, platform: Optional[str] = None
    ) -> List[Submodule]:
        """
        Get all submodules needed to build a specific stage.

        This collects source_sets from all artifact_groups in the stage,
        deduplicating by submodule name. When sparse checkout is added,
        this will need to merge specs for the same submodule.

        Args:
            build_stage: Name of the build stage
            platform: Current platform (e.g., "linux", "windows"). If provided,
                source_sets with this platform in disable_platforms are skipped.

        Returns:
            List of Submodule objects needed for this stage
        """
        # Use dict to dedupe by name while preserving order
        submodules_by_name: Dict[str, Submodule] = {}

        for source_set in self.get_source_sets_for_stage(
            build_stage, platform=platform
        ):
            for submodule in source_set.submodules:
                # TODO: When adding sparse_checkout, merge specs here
                if submodule.name not in submodules_by_name:
                    submodules_by_name[submodule.name] = submodule

        return list(submodules_by_name.values())

    def get_external_git_sources_for_stage(
        self, build_stage: str, platform: Optional[str] = None
    ) -> List[ExternalGitSource]:
        """
        Get all external git sources needed to build a specific stage.

        Args:
            build_stage: Name of the build stage
            platform: Current platform (e.g., "linux", "windows"). If provided,
                source_sets with this platform in disable_platforms are skipped.

        Returns:
            List of ExternalGitSource objects needed for this stage
        """
        sources_by_path: Dict[str, ExternalGitSource] = {}
        for source_set in self.get_source_sets_for_stage(
            build_stage, platform=platform
        ):
            for source in source_set.external_git_sources:
                if source.path not in sources_by_path:
                    sources_by_path[source.path] = source
        return list(sources_by_path.values())

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

    def get_all_external_git_sources(self) -> List[ExternalGitSource]:
        """
        Get all external git sources defined across all source sets.

        Returns:
            List of all ExternalGitSource objects (deduplicated by path)
        """
        sources_by_path: Dict[str, ExternalGitSource] = {}
        for source_set in self.source_sets.values():
            for source in source_set.external_git_sources:
                if source.path not in sources_by_path:
                    sources_by_path[source.path] = source
        return list(sources_by_path.values())

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

    def load_subproject_manifest(
        self, manifest_path: Optional[Path] = None
    ) -> Optional[Dict[str, List[str]]]:
        """Load artifact_subprojects.json from manifest_path or build_tools/."""
        if manifest_path is None:
            manifest_path = (
                self.toml_path.parent / "build_tools" / "artifact_subprojects.json"
            )
        if not manifest_path.exists():
            return None
        with manifest_path.open() as f:
            return json.load(f)

    def _load_json_manifest(self, filename: str) -> Optional[Dict]:
        """Load a JSON manifest file from build_tools/."""
        manifest_path = self.toml_path.parent / "build_tools" / filename
        if not manifest_path.exists():
            return None
        with manifest_path.open() as f:
            return json.load(f)

    def _load_project_mappings(self) -> Optional[Dict]:
        """Load project_mappings.json from build_tools/."""
        return self._load_json_manifest("project_mappings.json")

    def get_subproject_to_feature_map(
        self, build_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Map subproject names directly to feature names."""
        feature_map: Dict[str, str] = {}
        mappings = self._load_project_mappings()
        if mappings and "subproject_features" in mappings:
            for subproject, feature in mappings["subproject_features"].items():
                feature_map[subproject.lower()] = feature
        return feature_map

    def get_alias_to_artifact_map(
        self, build_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Map subproject/artifact names to artifact names."""
        alias_map: Dict[str, str] = {}

        manifest = None
        if build_dir:
            build_manifest = build_dir / "artifact_subprojects.json"
            if build_manifest.exists():
                manifest = self.load_subproject_manifest(build_manifest)
        if manifest is None:
            manifest = self.load_subproject_manifest()

        for artifact in self.artifacts.values():
            alias_map[artifact.name.lower()] = artifact.name

            if manifest and artifact.name in manifest:
                for alias in manifest[artifact.name]:
                    alias_lower = alias.lower()
                    if alias_lower not in alias_map or alias_lower == artifact.name:
                        alias_map[alias_lower] = artifact.name

            for db_name in artifact.split_databases:
                alias_map[db_name.lower()] = artifact.name

        # Include rocm-systems project mappings (e.g., "hip" -> "core-hip")
        mappings = self._load_project_mappings()
        if mappings and "rocm_systems_projects" in mappings:
            for project_name, artifact_name in mappings[
                "rocm_systems_projects"
            ].items():
                project_lower = project_name.lower()
                if project_lower not in alias_map:
                    alias_map[project_lower] = artifact_name

        return alias_map

    def resolve_project_to_artifact(
        self, project_name: str, build_dir: Optional[Path] = None
    ) -> Optional[str]:
        """Resolve a project name to its artifact name."""
        return self.get_alias_to_artifact_map(build_dir).get(project_name.lower())

    def resolve_projects_to_features(
        self,
        project_names: List[str],
        platform_name: str = "",
        build_dir: Optional[Path] = None,
    ) -> Set[str]:
        """Resolve project names to CMake feature names."""
        features: Set[str] = set()
        feature_map = self.get_subproject_to_feature_map(build_dir)
        alias_map = self.get_alias_to_artifact_map(build_dir)

        for project in project_names:
            project_lower = project.lower()

            # First check direct subproject -> feature mapping
            if project_lower in feature_map:
                features.add(feature_map[project_lower])
                continue

            # Fall back to artifact mapping
            artifact_name = alias_map.get(project_lower)
            if artifact_name and artifact_name in self.artifacts:
                artifact = self.artifacts[artifact_name]
                if platform_name and platform_name in artifact.disable_platforms:
                    continue
                features.add(self.get_artifact_feature_name(artifact))

        return features

    def get_stage_for_artifact(self, artifact_name: str) -> Optional[str]:
        """Get the build stage that produces a given artifact."""
        if artifact_name not in self.artifacts:
            return None
        artifact = self.artifacts[artifact_name]
        artifact_group = artifact.artifact_group

        for stage in self.build_stages.values():
            if artifact_group in stage.artifact_groups:
                return stage.name
        return None

    def get_stages_for_projects(
        self,
        project_names: List[str],
        build_dir: Optional[Path] = None,
    ) -> Set[str]:
        """Get build stages required to build the given projects."""
        # Resolve projects to artifacts
        alias_map = self.get_alias_to_artifact_map(build_dir)
        required_artifacts: Set[str] = set()
        for project in project_names:
            artifact_name = alias_map.get(project.lower())
            if artifact_name:
                required_artifacts.add(artifact_name)

        # Also include test_artifacts for each required artifact
        artifacts_to_process = list(required_artifacts)
        while artifacts_to_process:
            artifact_name = artifacts_to_process.pop()
            if artifact_name not in self.artifacts:
                continue
            artifact = self.artifacts[artifact_name]
            for test_artifact in artifact.test_artifacts:
                if test_artifact not in required_artifacts:
                    required_artifacts.add(test_artifact)
                    artifacts_to_process.append(test_artifact)

        # Get stages that produce these artifacts
        required_stages: Set[str] = set()
        for artifact_name in required_artifacts:
            stage_name = self.get_stage_for_artifact(artifact_name)
            if stage_name:
                required_stages.add(stage_name)

        # Include dependent stages (stages that produce artifacts we depend on)
        # Walk the dependency chain
        stages_to_check = list(required_stages)
        while stages_to_check:
            stage_name = stages_to_check.pop()
            if stage_name not in self.build_stages:
                continue
            stage = self.build_stages[stage_name]

            # Get all artifacts in this stage's groups
            for group_name in stage.artifact_groups:
                for artifact in self.get_artifacts_in_group(group_name):
                    # Check artifact dependencies
                    for dep_artifact_name in artifact.artifact_deps:
                        dep_stage = self.get_stage_for_artifact(dep_artifact_name)
                        if dep_stage and dep_stage not in required_stages:
                            required_stages.add(dep_stage)
                            stages_to_check.append(dep_stage)

        return required_stages

    def get_all_stage_names(self) -> Set[str]:
        """Get all build stage names."""
        return set(self.build_stages.keys())

    def filter_projects_for_stage(
        self,
        project_names: List[str],
        stage_name: str,
        build_dir: Optional[Path] = None,
    ) -> List[str]:
        """Return only projects produced by the given stage."""
        if stage_name not in self.build_stages:
            return []

        # Get artifacts produced by this stage (not inbound deps)
        produced_artifacts = self.get_produced_artifacts(stage_name)

        # Resolve each project and check if its artifact is produced by this stage
        alias_map = self.get_alias_to_artifact_map(build_dir)
        filtered: List[str] = []
        for project in project_names:
            artifact_name = alias_map.get(project.lower())
            if artifact_name and artifact_name in produced_artifacts:
                filtered.append(project)

        return filtered
