#!/usr/bin/env python
"""Exports a canonical source archive from a pristine git worktree.

Many downstream packagers operate off of source archives, and this utility exports
canonical source archives of different types for such consumption.

There are two types of source archives:

* Pure sources: These archives contain all source material needed to build the project.
  They contain a recursive export of all contained submodules that have been
  initialized. In the future, as well, they will optionally contain any files that
  may normally be fetched lazily as part of the build (i.e. third party dependency
  archives, pre-generated bundles, etc) so that the resulting package is self
  contained and completely sufficient to build.

* Prebuilt archives: These archives contain only an export of the super-project git
  repository but no source project submodules or build dependency bundles. These
  archives are used for cross generating various downstream packages without executing
  their own build step. These archives have a number of uses:

    * Generating DEB or RPM packages without a build step, using build artifacts from
      some other source (typically built on a neutral, old glibc version that is
      widely portable with many operating systems).
    * Generating binary installers (i.e. NSIS, etc) without actually performing a
      built.

There are several special files and locations in export archives:

* `GIT_REVISION`: Text file containing the HEAD commit hash at the time of export.
* `GIT_ORIGIN`: Text file containing the origin URL at the time of export (or empty).
* `PREBUILT`: Marker file, indicating that if present, prebuilt artifacts are present
  in the `build/artifacts` directory as the output of some prior, offline build
  process.

These files are standard locations and the build system may be hard-wired to incorporate
them in preference to other mechanisms of obtaining this information.

While not yet implemented, in the future, the cache of downloaded build deps will be
stored in the `.build_dep_cache/` with file names keyed by a `BUILD_DEP_ID` in the build
system. The build system will automatically use these if available.
"""

from abc import ABC, abstractmethod
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from _therock_utils.pattern_match import PatternMatcher

REPO_DIR = Path(__file__).resolve().parent.parent


class ArchiveWriter(ABC):
    @staticmethod
    def create(p: Path, compresslevel: int):
        name = p.name
        if name.endswith(".tar"):
            return TarArchiveWriter(
                tarfile.open(p, mode="w", compresslevel=compresslevel)
            )
        elif name.endswith(".tar.gz") or name.endswith(".tgz"):
            return TarArchiveWriter(
                tarfile.open(p, mode="w:gz", compresslevel=compresslevel)
            )
        elif name.endswith(".tar.bz2"):
            return TarArchiveWriter(
                tarfile.open(p, mode="w:bz2", compresslevel=compresslevel)
            )
        elif name.endswith(".tar.xz"):
            return TarArchiveWriter(
                tarfile.open(p, mode="w:xz", compresslevel=compresslevel)
            )
        else:
            raise ValueError(f"Unsupported archive file extension for: {name}")

    @abstractmethod
    def close(self): ...

    @abstractmethod
    def add_file(self, path: Path, arcname: str): ...

    @abstractmethod
    def add_directory(self, path: Path, arcname: str): ...

    @abstractmethod
    def add_text(self, text: str, arcname: str): ...


class TarArchiveWriter(ArchiveWriter):
    def __init__(self, archive: tarfile.TarFile):
        self.archive = archive

    def close(self):
        self.archive.close()

    def add_file(self, path: Path, arcname: str):
        self.archive.add(str(path), arcname, recursive=False)

    def add_directory(self, path: Path, arcname: str):
        self.archive.add(str(path), arcname, recursive=True)

    def add_text(self, text: str, arcname: str):
        with tempfile.NamedTemporaryFile(mode="wt") as tf:
            tf.write(text)
            self.archive.add(Path(tf.name), arcname)


def git_ls_files(repo_dir: Path, recurse_submodules: bool = False):
    cl = ["git", "ls-files"]
    if recurse_submodules:
        cl += ["--recurse-submodules"]
    cl += [str(repo_dir)]
    lines = subprocess.check_output(cl).decode()
    return lines.splitlines()


def git_head_revision(repo_dir: Path) -> str:
    cl = ["git", "rev-parse", "HEAD"]
    return subprocess.check_output(cl).decode().strip()


def git_origin(repo_dir: Path) -> str:
    cl = ["git", "remote", "get-url", "origin"]
    try:
        return subprocess.check_output(cl).decode().strip()
    except subprocess.CalledProcessError:
        # It is legal for there to be no origin.
        return ""


def progress_iter(iterable, desc: str | None):
    try:
        import tqdm
    except ImportError:
        return iterable
    return tqdm.tqdm(
        iterable,
        desc=desc,
        unit="file",
        bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt}{postfix}",
    )


def create_archive(
    writer: ArchiveWriter, source_dir: Path, prebuilt_artifacts_dir: Path | None
):
    # Write some metadata.
    head_revision = git_head_revision(source_dir)
    origin = git_origin(source_dir)
    writer.add_text(head_revision, "GIT_REVISION")
    writer.add_text(origin, "GIT_ORIGIN")

    # Write a marker file to indicate that the archive is a PREBUILT archive.
    # Uses the relative path to the artifacts dir as the content, but this
    # is also assumed fixed (just written here for posterity).
    if prebuilt_artifacts_dir is not None:
        writer.add_text("build/artifacts", "PREBUILT")

    # Get the source files. If generating a pre-built archive, we do not
    # recurse submodules. Otherwise, we do.
    source_files = git_ls_files(
        source_dir, recurse_submodules=prebuilt_artifacts_dir is None
    )
    for source_file in progress_iter(source_files, desc="Adding source files"):
        source_abs_path = source_dir / source_file
        writer.add_file(source_abs_path, source_file)

    if prebuilt_artifacts_dir:
        pm = PatternMatcher()
        pm.add_basedir(prebuilt_artifacts_dir)
        artifact_files = list(pm.matches())
        for relpath, direntry in progress_iter(
            artifact_files, desc="Adding prebuilt files"
        ):
            # We canonically organize source tarballs to have artifacts in
            # build/artifacts, no matter where they came from.
            writer.add_file(Path(direntry.path), f"build/artifacts/{relpath}")


def main(argv: list[str]):
    p = argparse.ArgumentParser("create_source_tarball")
    p.add_argument(
        "-s",
        "--source-dir",
        type=Path,
        default=REPO_DIR,
        help="Path to TheRock source directory",
    )
    p.add_argument(
        "--prebuilt-artifacts",
        type=Path,
        help="If specified, creates a tarball which contains pre-built artifacts vs sources (prebuilt)",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="Output tarball")
    p.add_argument(
        "--compress-level", type=int, default=4, help="Tar compression level"
    )
    args = p.parse_args(argv)
    writer = ArchiveWriter.create(args.output, compresslevel=args.compress_level)
    try:
        create_archive(writer, args.source_dir, args.prebuilt_artifacts)
    finally:
        writer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
