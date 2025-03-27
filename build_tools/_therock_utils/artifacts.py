"""Manipulates artifact directories.

Artifacts are the primary way that build outputs are broken down in the
project. See cmake/therock_artifacts.cmake.

In brief, the artifacts/ dir consists of directories with names like:
    {name}_{component}[_{target_family}]

Components are variable but are typically:
    * dev: Development files
    * doc: Documentation files
    * dbg: Debug files
    * lib: Files needed to use the artifact as a library
    * run: Files needed to use the artifact as a tool

Each valid artifact directory contains an `artifact_manifest.txt` file, which
contains one relative path per line. That path represents a path into a TheRock
build directory that its contents are subset from.
"""

from typing import Callable, Sequence

import re
from pathlib import Path

from .pattern_match import PatternMatcher, MatchPredicate


class ArtifactName:
    def __init__(self, name: str, component: str, target_family: str):
        self.name = name
        self.component = component
        self.target_family = target_family

    def __repr__(self):
        return f"Artifact({self.name}[{self.component}:{self.target_family}])"


class ArtifactCatalog:
    def __init__(
        self,
        artifact_dir: Path,
        filter: Callable[[ArtifactName], bool] = lambda _: True,
        includes: Sequence[str] = (),
        excludes: Sequence[str] = (),
    ):
        self.artifact_dir = artifact_dir
        self.artifact_basedirs: list[tuple[ArtifactName, Path]] = []
        self.pm = PatternMatcher(includes=includes, excludes=excludes)

        for subdir in self.artifact_dir.iterdir():
            if not subdir.is_dir():
                continue
            # Matches {name}_{component}_{target_family} with an optional
            # extra suffix that we ignore.
            m = re.match(r"^([^_]+)_([^_]+)_([^_]+)(_.+)?$", subdir.name)
            if not m:
                continue
            name = ArtifactName(m.group(1), m.group(2), m.group(3))
            if not filter(name):
                continue
            manifest = subdir / "artifact_manifest.txt"
            if not manifest.exists():
                continue
            manifest_lines = manifest.read_text().splitlines()
            for manifest_line in manifest_lines:
                if not manifest_line:
                    continue
                full_path = subdir / manifest_line
                if full_path.exists():
                    self.artifact_basedirs.append((name, full_path))
                    self.pm.add_basedir(full_path)

    @property
    def artifact_names(self) -> list[ArtifactName]:
        return [an for an, _ in self.artifact_basedirs]

    @property
    def all_target_families(self) -> set[str]:
        return set(
            an.target_family
            for an in self.artifact_names
            if an.target_family != "generic"
        )
