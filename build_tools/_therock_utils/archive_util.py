# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Utilities for reading and writing zstd/xz compressed tar archives."""

from typing import Callable
import functools
import os
from pathlib import Path
import subprocess
import tarfile
import time

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent


def _git_commit_timestamp() -> int | None:
    """Commit time of TheRock's HEAD, or None if not a usable git checkout.

    Deliberately the superproject rather than the submodule an artifact's
    content came from. TheRock pins every submodule, so its HEAD identifies the
    whole source state, whereas an artifact has no single source submodule to
    ask: a merged dist tree spans a subproject and its runtime deps, and several
    subprojects (zlib, zstd, bzip2, elfutils, ...) are tarballs downloaded from
    S3 with no git history at all.

    The tradeoff is over-invalidation: a docs-only commit here changes every
    artifact's hash even when the content is byte-identical. That is fine for
    detecting conflicting uploads within a run, which is what this is for, but
    it does mean archive hashes cannot be used to dedupe across commits. Set
    `SOURCE_DATE_EPOCH` to decide that policy somewhere better informed.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=THEROCK_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        # git is not installed.
        return None
    if result.returncode != 0:
        # Not a git checkout (e.g. building from an exported source archive) or
        # no commits yet.
        return None
    output = result.stdout.strip()
    return int(output) if output else None


@functools.cache
def get_archive_timestamp() -> int:
    """Resolves the mtime to stamp into reproducible archives.

    Deliberately not 0: extraction restores mtime (`tarfile.extract()` and
    `tar -x` both do, for root and non-root alike), so an epoch mtime makes
    freshly installed SDK headers and libraries look older than the objects a
    downstream project already built against the previous version. Timestamp
    driven build systems then skip work that needed doing.

    Resolution order:

    1. `SOURCE_DATE_EPOCH`, the reproducible-builds convention. Set this to pin
       the timestamp explicitly; it is the supported override.
    2. The commit time of HEAD. Deterministic for a given revision, so parallel
       CI jobs building the same commit still produce identical archives, and
       it advances as the source advances.
    3. The current time, if neither is available. Not reproducible, but a
       working build beats a deterministically broken one -- set
       `SOURCE_DATE_EPOCH` where reproducibility is required.

    Note that a dirty working tree still reports its last commit time. That is
    fine for CI, which is always clean; set `SOURCE_DATE_EPOCH` locally if a
    dirty tree's archives need to look newer than something.
    """
    env_value = os.environ.get("SOURCE_DATE_EPOCH")
    if env_value:
        try:
            timestamp = int(env_value)
        except ValueError:
            raise ValueError(
                "SOURCE_DATE_EPOCH must be an integer number of seconds since "
                f"the Unix epoch, got {env_value!r}"
            )
        if timestamp < 0:
            raise ValueError(f"SOURCE_DATE_EPOCH must not be negative, got {timestamp}")
        return timestamp

    commit_timestamp = _git_commit_timestamp()
    if commit_timestamp is not None:
        return commit_timestamp

    return int(time.time())


def normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalizes TarInfo metadata so identical content yields identical archives.

    Pass as the `filter` argument to `TarFile.add()`. The wall-clock build time
    and the uid/gid of the building user otherwise vary from build to build,
    giving the same content a different archive hash every time.

    Timestamps are replaced with `get_archive_timestamp()` rather than dropped,
    for the reasons given there. Permissions, sizes, symlink targets and hardlink
    identity are preserved. Extracting as a non-root user gives files owned by
    the extracting user; extracting as root with `-p` gives root:root.
    """
    tarinfo.mtime = get_archive_timestamp()
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = "root"
    tarinfo.gname = "root"
    return tarinfo


def add_tree(
    tf: tarfile.TarFile,
    source_dir: Path,
    *,
    relative_to: Path,
    on_add: Callable[[str], None] | None = None,
) -> None:
    """Adds every file and directory under `source_dir` to `tf` reproducibly.

    Member names are computed relative to `relative_to`. Entries are walked and
    emitted in sorted order and their metadata is normalized, so the same tree
    always produces the same archive bytes. `on_add`, if given, is called with
    each member name as it is added.
    """
    for root, dirnames, filenames in sorted(
        os.walk(source_dir), key=lambda entry: entry[0]
    ):
        for name in sorted(list(filenames) + list(dirnames)):
            file_path = os.path.join(root, name)
            arcname = os.path.relpath(file_path, relative_to)
            if on_add is not None:
                on_add(arcname)
            tf.add(
                file_path,
                arcname=arcname,
                recursive=False,
                filter=normalize_tarinfo,
            )


def _get_pyzstd():
    """Lazy import pyzstd with helpful error message."""
    try:
        import pyzstd

        return pyzstd
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pyzstd is required for zstd artifact compression. "
            "Install it with: pip install pyzstd"
        )


class ZstdTarFile(tarfile.TarFile):
    """TarFile wrapper that manages the underlying ZstdFile lifetime.

    When TarFile receives a fileobj it did not open, it does not close it.
    This leaves the OS file handle open, which on Windows prevents subsequent
    os.unlink() calls from succeeding.
    """

    def __init__(self, path: Path, mode: str = "rb", **zstd_kwargs) -> None:
        pyzstd = _get_pyzstd()
        self._zstd_file = pyzstd.ZstdFile(path, mode=mode, **zstd_kwargs)

        # Trim mode from ZstdFile format to TarFile format
        #   * https://pyzstd.readthedocs.io/en/stable/pyzstd.html#open
        #   * https://docs.python.org/3/library/tarfile.html#tarfile.open
        # "rb" -> "r", "wb" -> "w"
        mode_tarfile = mode[0]

        super().__init__(fileobj=self._zstd_file, mode=mode_tarfile)

    def close(self) -> None:
        super().close()
        self._zstd_file.close()


def open_archive_for_read(path: Path) -> tarfile.TarFile:
    """Open a tar archive for reading, auto-detecting compression from extension."""
    if path.name.endswith(".tar.zst"):
        return ZstdTarFile(path, mode="rb")
    elif path.name.endswith(".tar.xz"):
        return tarfile.TarFile.open(path, mode="r:xz")
    else:
        raise ValueError(f"Unknown archive format: {path}")


def open_archive_for_write(
    path: Path, compression_type: str, compression_level: int | None = None
) -> tarfile.TarFile:
    """Open a tar archive for writing with the specified compression."""
    if compression_type == "zstd":
        level = compression_level if compression_level is not None else 3
        return ZstdTarFile(path, "wb", level_or_option=level)
    elif compression_type == "xz":
        level = compression_level if compression_level is not None else 6
        return tarfile.TarFile.open(path, mode="x:xz", preset=level)
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")
