from typing import Generator, Sequence

import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import time

_IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------------
# File copy strategies for copy_to.
#
# Three mutually exclusive strategies for placing a regular file at destpath:
#
#   1. _hardlink_or_copy_from_source: hardlink dest to src (shares inode with
#      source), falling back to copy on failure (e.g. cross-device).
#      Used by default to avoid redundant copies within a build tree.
#
#   2. _copy_preserving_hardlink_groups: copy file content (new inode), but
#      re-hardlink files that shared an inode in the source so they still
#      share an inode in the destination. Used for artifact populate where
#      we must break sharing with the source tree but preserve internal
#      hardlink structure (e.g. libfoo.so.1 <-> libfoo.so.1.0.0).
#      Not available on Windows (st_dev/st_ino unreliable).
#
#   3. _plain_copy: shutil.copy2, no inode tracking. Used on Windows with
#      always_copy, where we can't reliably detect hardlink groups.
# ---------------------------------------------------------------------------


def _hardlink_or_copy_from_source(src: str, destpath: Path, verbose: bool) -> None:
    """Hardlink destpath to src, falling back to copy on failure."""
    try:
        if verbose:
            print(f"hardlink {src} -> {destpath}", file=sys.stderr, end="")
        os.link(src, destpath, follow_symlinks=False)
    except OSError:
        if verbose:
            print(" (falling back to copy) ", file=sys.stderr, end="")
        _plain_copy(src, destpath, verbose)


def _copy_preserving_hardlink_groups(
    src: str,
    destpath: Path,
    verbose: bool,
    copied_inodes: dict[tuple[int, int], Path],
) -> None:
    """Copy file, but hardlink to a previous copy if the source inode matches.

    This gives tar-like behavior: files hardlinked together in the source
    remain hardlinked together in the destination, but no destination file
    shares an inode with the source.
    """
    src_stat = os.stat(src)
    inode_key = (src_stat.st_dev, src_stat.st_ino)
    prev_dest = copied_inodes.get(inode_key)
    if prev_dest is not None:
        if verbose:
            print(
                f"hardlink (internal) {prev_dest} -> {destpath}",
                file=sys.stderr,
                end="",
            )
        os.link(prev_dest, destpath)
        return
    # First time seeing this inode: copy and record.
    if verbose:
        print(f"copy {src} -> {destpath}", file=sys.stderr, end="")
    shutil.copy2(src, destpath, follow_symlinks=False)
    copied_inodes[inode_key] = destpath


def _plain_copy(src: str, destpath: Path, verbose: bool) -> None:
    if verbose:
        print(f"copy {src} -> {destpath}", file=sys.stderr, end="")
    shutil.copy2(src, destpath, follow_symlinks=False)


class RecursiveGlobPattern:
    def __init__(self, glob: str):
        self.glob = glob
        pattern = f"^{re.escape(glob)}$"
        # Intermediate recursive directory match.
        pattern = pattern.replace("/\\*\\*/", "/(.*/)?")
        # First segment recursive directory match.
        pattern = pattern.replace("^\\*\\*/", "^(.*/)?")
        # Last segment recursive directory match.
        pattern = pattern.replace("/\\*\\*$", "(/.*)?$")
        # Intra-segment * match.
        pattern = pattern.replace("\\*", "[^/]*")
        # Intra-segment ? match.
        pattern = pattern.replace("\\?", "[^/]*")
        self.pattern = re.compile(pattern)

    def matches(self, relpath: str, direntry: os.DirEntry[str]) -> bool:
        m = self.pattern.match(relpath)
        return True if m else False


class MatchPredicate:
    def __init__(
        self,
        includes: Sequence[str] = (),
        excludes: Sequence[str] = (),
        force_includes: Sequence[str] = (),
    ):
        self.includes = [RecursiveGlobPattern(p) for p in includes]
        self.excludes = [RecursiveGlobPattern(p) for p in excludes]
        self.force_includes = [RecursiveGlobPattern(p) for p in force_includes]

    def matches(self, match_path: str, direntry: os.DirEntry[str]):
        includes = self.includes
        excludes = self.excludes
        force_includes = self.force_includes
        if force_includes:
            for force_include in force_includes:
                if force_include.matches(match_path, direntry):
                    return True
        if includes:
            for include in includes:
                if include.matches(match_path, direntry):
                    break
            else:
                return False
        for exclude in excludes:
            if exclude.matches(match_path, direntry):
                return False
        return True


class PatternMatcher:
    # Maximum number of attempts to retry removing the destination directory
    max_attempts: int = 5
    # Delay between retry attempts in seconds
    retry_delay_seconds: float = 0.2

    def __init__(
        self,
        includes: Sequence[str] = (),
        excludes: Sequence[str] = (),
        force_includes: Sequence[str] = (),
    ):
        self.predicate = MatchPredicate(includes, excludes, force_includes)
        # Dictionary of relative posix-style path to DirEntry.
        # Last relative path to entry.
        self.all: dict[str, os.DirEntry[str]] = {}

    def add_basedir(self, basedir: Path):
        all = self.all
        basedir = basedir.absolute()

        # Using scandir and being judicious about path concatenation/conversion
        # (versus using walk) is on the order of 10-50x faster. This is still
        # about 10x slower than an `ls -R` but gets us down to tens of
        # milliseconds for an LLVM install sized tree, which is acceptable.
        def scan_children(rootpath: str, prefix: str):
            with os.scandir(rootpath) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        relpath = f"{prefix}{entry.name}"
                        new_rootpath = os.path.join(rootpath, entry.name)
                        all[relpath] = entry
                        scan_children(new_rootpath, f"{relpath}/")
                    else:
                        relpath = f"{prefix}{entry.name}"
                        all[relpath] = entry

        scan_children(basedir, "")

    def add_entry(self, relpath: str, direntry: os.DirEntry):
        self.all[relpath] = direntry

    def matches(self) -> Generator[tuple[str, os.DirEntry[str]], None, None]:
        for match_path, direntry in self.all.items():
            if self.predicate.matches(match_path, direntry):
                yield match_path, direntry

    def copy_to(
        self,
        *,
        destdir: Path,
        destprefix: str = "",
        verbose: bool = False,
        always_copy: bool = False,
        remove_dest: bool = True,
    ):
        if remove_dest and destdir.exists():
            self._rmtree_with_retry(destdir, verbose)
        destdir.mkdir(parents=True, exist_ok=True)

        # Inode tracking for _copy_preserving_hardlink_groups.
        copied_inodes: dict[tuple[int, int], Path] = {}

        for relpath, direntry in self.matches():
            try:
                destpath = destdir / PurePosixPath(destprefix + relpath)
                if direntry.is_dir() and not direntry.is_symlink():
                    if verbose:
                        print(f"mkdir {destpath}", file=sys.stderr, end="")
                    destpath.mkdir(parents=True, exist_ok=True)
                elif direntry.is_symlink():
                    self._copy_symlink(direntry, destpath, remove_dest, verbose)
                else:
                    self._copy_regular_file(
                        direntry,
                        destpath,
                        always_copy,
                        remove_dest,
                        verbose,
                        copied_inodes,
                    )
            finally:
                if verbose:
                    print("", file=sys.stderr)

    def _rmtree_with_retry(self, path: Path, verbose: bool) -> None:
        for attempt in range(self.max_attempts):
            try:
                shutil.rmtree(path)
                if verbose:
                    print(f"rmtree {path}", file=sys.stderr)
                return
            except PermissionError:
                wait_time = self.retry_delay_seconds * (attempt + 2)
                if verbose:
                    print(
                        f"PermissionError calling shutil.rmtree('{path}') "
                        f"retrying after {wait_time}s",
                        file=sys.stderr,
                    )
                time.sleep(wait_time)
                if attempt == self.max_attempts - 1:
                    if verbose:
                        print(
                            f"rmtree failed after {self.max_attempts} "
                            f"attempts, failing",
                            file=sys.stderr,
                        )
                    raise

    @staticmethod
    def _copy_symlink(
        direntry: os.DirEntry[str],
        destpath: Path,
        remove_dest: bool,
        verbose: bool,
    ) -> None:
        if not remove_dest and (destpath.exists() or destpath.is_symlink()):
            os.unlink(destpath)
        targetpath = os.readlink(direntry.path)
        if verbose:
            print(f"symlink {targetpath} -> {destpath}", file=sys.stderr, end="")
        destpath.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(targetpath, destpath)

    @staticmethod
    def _copy_regular_file(
        direntry: os.DirEntry[str],
        destpath: Path,
        always_copy: bool,
        remove_dest: bool,
        verbose: bool,
        copied_inodes: dict[tuple[int, int], Path],
    ) -> None:
        # When hardlinking to source, another process may have already
        # created the link. On Windows files in use can't be removed, so
        # detect this and skip.
        if (
            not always_copy
            and destpath.exists()
            and os.stat(destpath).st_ino == os.stat(direntry.path).st_ino
        ):
            if verbose:
                print(
                    f"skipping (already hardlinked) {direntry.path}",
                    file=sys.stderr,
                    end="",
                )
            return

        if not remove_dest and (destpath.exists() or destpath.is_symlink()):
            os.unlink(destpath)
        destpath.parent.mkdir(parents=True, exist_ok=True)

        # Dispatch to the appropriate strategy.
        if not always_copy:
            _hardlink_or_copy_from_source(direntry.path, destpath, verbose)
        elif _IS_WINDOWS:
            _plain_copy(direntry.path, destpath, verbose)
        else:
            _copy_preserving_hardlink_groups(
                direntry.path, destpath, verbose, copied_inodes
            )
