# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from typing import Callable, Generator, Protocol, Sequence

import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import sys
import time

from .os_util import rmtree_with_retry

_IS_WINDOWS = platform.system() == "Windows"
_DIAGNOSTIC_PROGRESS_INTERVAL = 1000


class DiagnosticReporter(Protocol):
    def log(self, message: str) -> None: ...

    def set_current_operation(self, message: str) -> None: ...


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


def _hardlink_or_copy_from_source(
    src: str,
    destpath: Path,
    verbose: bool,
    on_hardlink_fallback: Callable[[str, Path, OSError], None],
) -> None:
    """Hardlink destpath to src, falling back to copy on failure."""
    try:
        if verbose:
            print(f"hardlink {src} -> {destpath}", file=sys.stderr, end="")
        os.link(src, destpath, follow_symlinks=False)
    except OSError as e:
        on_hardlink_fallback(src, destpath, e)
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

    def __init__(
        self,
        includes: Sequence[str] = (),
        excludes: Sequence[str] = (),
        force_includes: Sequence[str] = (),
        diagnostics: DiagnosticReporter | None = None,
    ):
        self.predicate = MatchPredicate(includes, excludes, force_includes)
        self.diagnostics = diagnostics
        # Dictionary of relative posix-style path to DirEntry.
        # Last relative path to entry.
        self.all: dict[str, os.DirEntry[str]] = {}

    def add_basedir(self, basedir: Path):
        all = self.all
        basedir = basedir.absolute()
        start_time = time.monotonic()
        entries_seen = 0
        replaced_entries = 0
        if self.diagnostics is not None:
            self.diagnostics.log(f"scan_begin basedir={basedir}")

        # Using scandir and being judicious about path concatenation/conversion
        # (versus using walk) is on the order of 10-50x faster. This is still
        # about 10x slower than an `ls -R` but gets us down to tens of
        # milliseconds for an LLVM install sized tree, which is acceptable.
        def record_entry(relpath: str, entry: os.DirEntry[str]) -> None:
            nonlocal entries_seen, replaced_entries
            entries_seen += 1
            if relpath in all:
                replaced_entries += 1
            if self.diagnostics is not None:
                self.diagnostics.set_current_operation(
                    f"scan basedir={basedir} relpath={relpath}"
                )
                if entries_seen % _DIAGNOSTIC_PROGRESS_INTERVAL == 0:
                    self.diagnostics.log(
                        f"scan_progress basedir={basedir} "
                        f"entries={entries_seen} relpath={relpath}"
                    )
            all[relpath] = entry

        def scan_children(rootpath: str, prefix: str):
            with os.scandir(rootpath) as it:
                for entry in it:
                    relpath = f"{prefix}{entry.name}"
                    record_entry(relpath, entry)
                    if entry.is_dir(follow_symlinks=False):
                        new_rootpath = os.path.join(rootpath, entry.name)
                        scan_children(new_rootpath, f"{relpath}/")

        try:
            scan_children(basedir, "")
        except Exception as e:
            if self.diagnostics is not None:
                self.diagnostics.log(
                    f"scan_error basedir={basedir} entries={entries_seen} error={e!r}"
                )
            raise
        if self.diagnostics is not None:
            self.diagnostics.log(
                f"scan_end basedir={basedir} entries={entries_seen} "
                f"replaced={replaced_entries} unique_total={len(all)} "
                f"duration={time.monotonic() - start_time:.3f}s"
            )

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
        start_time = time.monotonic()
        matched_entries = 0
        directory_count = 0
        symlink_count = 0
        file_count = 0
        hardlink_fallback_count = 0
        if self.diagnostics is not None:
            self.diagnostics.log(
                f"copy_begin dest={destdir} candidates={len(self.all)} "
                f"remove_dest={remove_dest} always_copy={always_copy}"
            )
        if remove_dest and destdir.exists():
            if self.diagnostics is not None:
                self.diagnostics.set_current_operation(f"rmtree dest={destdir}")
            rmtree_with_retry(destdir, verbose=verbose or self.diagnostics is not None)
        if self.diagnostics is not None:
            self.diagnostics.set_current_operation(f"mkdir dest={destdir}")
        destdir.mkdir(parents=True, exist_ok=True)

        # Inode tracking for _copy_preserving_hardlink_groups.
        copied_inodes: dict[tuple[int, int], Path] = {}

        def on_hardlink_fallback(
            source: str, destination: Path, error: OSError
        ) -> None:
            nonlocal hardlink_fallback_count
            hardlink_fallback_count += 1
            if self.diagnostics is not None and (
                hardlink_fallback_count <= 10
                or hardlink_fallback_count % _DIAGNOSTIC_PROGRESS_INTERVAL == 0
            ):
                self.diagnostics.log(
                    f"hardlink_fallback count={hardlink_fallback_count} "
                    f"source={source} dest={destination} error={error!r}"
                )

        for relpath, direntry in self.matches():
            matched_entries += 1
            try:
                destpath = destdir / PurePosixPath(destprefix + relpath)
                if self.diagnostics is not None:
                    self.diagnostics.set_current_operation(
                        f"copy source={direntry.path} dest={destpath}"
                    )
                    if matched_entries % _DIAGNOSTIC_PROGRESS_INTERVAL == 0:
                        self.diagnostics.log(
                            f"copy_progress dest={destdir} entries={matched_entries} "
                            f"relpath={relpath}"
                        )
                if direntry.is_dir() and not direntry.is_symlink():
                    directory_count += 1
                    if verbose:
                        print(f"mkdir {destpath}", file=sys.stderr, end="")
                    destpath.mkdir(parents=True, exist_ok=True)
                elif direntry.is_symlink():
                    symlink_count += 1
                    self._copy_symlink(direntry, destpath, remove_dest, verbose)
                else:
                    file_count += 1
                    self._copy_regular_file(
                        direntry,
                        destpath,
                        always_copy,
                        remove_dest,
                        verbose,
                        copied_inodes,
                        on_hardlink_fallback,
                    )
            except Exception as e:
                if self.diagnostics is not None:
                    self.diagnostics.log(
                        f"copy_error source={direntry.path} dest={destpath} error={e!r}"
                    )
                raise
            finally:
                if verbose:
                    print("", file=sys.stderr)
        if self.diagnostics is not None:
            self.diagnostics.set_current_operation(f"copy complete dest={destdir}")
            self.diagnostics.log(
                f"copy_end dest={destdir} entries={matched_entries} "
                f"directories={directory_count} symlinks={symlink_count} "
                f"files={file_count} hardlink_fallbacks={hardlink_fallback_count} "
                f"duration={time.monotonic() - start_time:.3f}s"
            )

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
        on_hardlink_fallback: Callable[[str, Path, OSError], None],
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
            _hardlink_or_copy_from_source(
                direntry.path, destpath, verbose, on_hardlink_fallback
            )
        elif _IS_WINDOWS:
            _plain_copy(direntry.path, destpath, verbose)
        else:
            _copy_preserving_hardlink_groups(
                direntry.path, destpath, verbose, copied_inodes
            )
