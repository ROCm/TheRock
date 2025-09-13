"""Manipulates artifact directories.

See `artifact_builder` for tools to construct artifacts from descriptors.

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

from typing import Callable, Optional, Sequence

import os
import re
from pathlib import Path, PurePosixPath
import tarfile

from .pattern_match import PatternMatcher, MatchPredicate


class ArtifactName:
    def __init__(self, name: str, component: str, target_family: str):
        self.name = name
        self.component = component
        self.target_family = target_family

    @staticmethod
    def from_path(path: Path) -> Optional["ArtifactName"]:
        filename = path.name
        if path.is_dir():
            # Matches {name}_{component}_{target_family}.
            m = re.match(r"^([^_]+)_([^_]+)_([^_]+)$", filename)
            if not m:
                return None
            return ArtifactName(m.group(1), m.group(2), m.group(3))
        else:
            return ArtifactName.from_filename(filename)

    @staticmethod
    def from_filename(filename: str) -> Optional["ArtifactName"]:
        # Matches {name}_{component}_{target_family} and an archive extension.
        m = re.match(r"^([^_]+)_([^_]+)_([^_]+)\.tar.xz$", filename)
        if not m:
            return None
        return ArtifactName(m.group(1), m.group(2), m.group(3))

    def __repr__(self):
        return f"Artifact({self.name}[{self.component}:{self.target_family}])"

    def __eq__(self, other):
        if not isinstance(other, ArtifactName):
            return NotImplemented
        return (
            self.name == other.name
            and self.component == other.component
            and self.target_family == other.target_family
        )

    def __hash__(self):
        return hash((self.name, self.component, self.target_family))


class ArtifactCatalog:
    """Scans a directory containing exploded artifact sub-directories.

    This is used for various packaging activities that need to operate on
    actual files in the file system (vs as part of compressed/remote archives).
    """

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
            name = ArtifactName.from_path(subdir)
            if not name:
                continue
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


class ArtifactPopulator:
    """Populates a list of artifacts into one output directory, optionally flattening.

    Returns:
    The set of all relative root paths from all encountered artifact manifests.
    These paths can be interpreted relative to the output_path to get a full
    populated path on the filesystem.
    """

    def __init__(
        self, *, output_path: Path, verbose: bool = False, flatten: bool = False
    ):
        self.output_path = output_path
        self.verbose = verbose
        self.flatten = flatten
        self.relpaths: set[str] = set()

    def on_relpath(self, relpath: str):
        """Callback that is invoked for every top-level relpath encountered."""
        if relpath not in self.relpaths:
            self.on_first_relpath(relpath)
        self.relpaths.add(relpath)

    def on_first_relpath(self, relpath: str):
        """Called on the first time that a relpath is encountered by this flattener."""
        pass

    def on_artifact_dir(self, artifact_dir: Path):
        """Callback that is invoked for every exploded artifact directory encountered."""
        pass

    def on_artifact_archive(self, artifact_archive: Path):
        """Callback that is invoked for every artifact archive encountered."""
        pass

    def __call__(self, *artifact_paths: Sequence[Path]):
        all_root_relpaths: set[str] = set()
        for artifact_path in artifact_paths:
            if artifact_path.is_dir():
                # Process an exploded artifact dir.
                self.on_artifact_dir(artifact_path)
                manifest_path: Path = artifact_path / "artifact_manifest.txt"
                relpaths = manifest_path.read_text().splitlines()
                for relpath in relpaths:
                    if not relpath:
                        continue
                    pm = PatternMatcher()
                    self.on_relpath(relpath)
                    source_dir = artifact_path / relpath
                    if not source_dir.exists():
                        continue
                    pm.add_basedir(source_dir)
                    destdir = (
                        self.output_path if self.flatten else self.output_path / relpath
                    )
                    pm.copy_to(destdir=destdir, verbose=self.verbose, remove_dest=False)
            else:
                # Process as an archive file.
                with tarfile.TarFile.open(artifact_path, mode="r:xz") as tf:
                    self.on_artifact_archive(artifact_path)
                    # Read manifest first.
                    manifest_member = tf.next()
                    if (
                        manifest_member is None
                        or manifest_member.name != "artifact_manifest.txt"
                    ):
                        raise IOError(
                            f"Artifact archive {artifact_path} must have artifact_manifest.txt as its first member"
                        )
                    with tf.extractfile(manifest_member) as mf_file:
                        relpaths = mf_file.read().decode().splitlines()
                        for relpath in relpaths:
                            self.on_relpath(relpath)
                    # Iterate over all remaining members.
                    while member := tf.next():
                        member_name = member.name
                        # Figure out which relpath prefix it is a part of.
                        for prefix_relpath in relpaths:
                            output_path = self.output_path
                            if not self.flatten:
                                output_path = output_path / prefix_relpath
                            prefix_relpath += "/"
                            if member_name.startswith(prefix_relpath):
                                scoped_path = member_name[len(prefix_relpath) :]
                                dest_path = output_path / PurePosixPath(scoped_path)
                                if dest_path.is_symlink() or (
                                    dest_path.exists() and not dest_path.is_dir()
                                ):
                                    os.unlink(dest_path)
                                dest_path.parent.mkdir(parents=True, exist_ok=True)
                                if member.isfile():
                                    exec_mask = member.mode & 0o111
                                    with tf.extractfile(member) as member_file:
                                        with open(
                                            dest_path,
                                            "wb",
                                        ) as out_file:
                                            out_file.write(member_file.read())
                                            st = os.fstat(out_file.fileno())
                                            if hasattr(os, "fchmod"):
                                                # Windows has no fchmod.
                                                new_mode = st.st_mode | exec_mask
                                                os.fchmod(out_file.fileno(), new_mode)
                                elif member.isdir():
                                    dest_path.mkdir(parents=True, exist_ok=True)
                                elif member.issym():
                                    dest_path.symlink_to(member.linkname)
                                else:
                                    raise IOError(f"Unhandled tar member: {member}")
                                break
                        else:
                            raise IOError(
                                f"Extracting tar artifact archive, encountered file not in manifest: {member}"
                            )
        return all_root_relpaths
