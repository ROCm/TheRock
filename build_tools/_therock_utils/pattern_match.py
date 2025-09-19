from typing import Generator, Sequence

import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys


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
            if verbose:
                print(f"rmtree {destdir}", file=sys.stderr)
            shutil.rmtree(destdir)
        destdir.mkdir(parents=True, exist_ok=True)

        for relpath, direntry in self.matches():
            try:
                destpath = destdir / PurePosixPath(destprefix + relpath)
                if direntry.is_dir() and not direntry.is_symlink():
                    # Directory.
                    if verbose:
                        print(f"mkdir {destpath}", file=sys.stderr, end="")
                    destpath.mkdir(parents=True, exist_ok=True)
                elif direntry.is_symlink():
                    # Symlink.
                    if not remove_dest and (destpath.exists() or destpath.is_symlink()):
                        os.unlink(destpath)
                    targetpath = os.readlink(direntry.path)
                    if verbose:
                        print(
                            f"symlink {targetpath} -> {destpath}",
                            file=sys.stderr,
                            end="",
                        )
                    destpath.parent.mkdir(parents=True, exist_ok=True)
                    os.symlink(targetpath, destpath)
                else:
                    # Regular file.

                    # Sometimes multiple processes try to link the same file.
                    # On Unix, we can safely unlink/remove a file and overwrite
                    # it. However, on Windows a file that is in use cannot be
                    # removed: https://docs.python.org/3/library/os.html#os.remove.
                    # Here we check if the inode matches (true for hardlinks)
                    # and avoid the unlink/link step in that case.
                    if (
                        destpath.exists()
                        and not always_copy
                        and os.stat(destpath).st_ino == os.stat(direntry.path).st_ino
                    ):
                        if verbose:
                            print(
                                f"skipping unlink and link for existing hardlink {direntry.path} -> {destpath}",
                                file=sys.stderr,
                                end="",
                            )
                        continue

                    if not remove_dest and (destpath.exists() or destpath.is_symlink()):
                        # Hopefully safe even on Windows given the check above.
                        os.unlink(destpath)

                    destpath.parent.mkdir(parents=True, exist_ok=True)
                    linked_file = False
                    if not always_copy:
                        # Attempt to link
                        try:
                            if verbose:
                                print(
                                    f"hardlink {direntry.path} -> {destpath}",
                                    file=sys.stderr,
                                    end="",
                                )
                            os.link(direntry.path, destpath, follow_symlinks=False)
                            linked_file = True
                        except OSError:
                            if verbose:
                                print(
                                    " (falling back to copy) ", file=sys.stderr, end=""
                                )
                    if not linked_file:
                        # Make a copy instead.
                        if verbose:
                            print(
                                f"copy {direntry.path} -> {destpath}",
                                file=sys.stderr,
                                end="",
                            )
                        shutil.copy2(direntry.path, destpath, follow_symlinks=False)
            finally:
                if verbose:
                    print("", file=sys.stderr)
