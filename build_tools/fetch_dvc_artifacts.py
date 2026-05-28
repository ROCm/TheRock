#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fetch large binary artifacts referenced by .dvc pointer files from an S3 remote.

Minimal in-tree replacement for `dvc pull` covering TheRock's only use of dvc:
pulling MD5-content-addressed binary artifacts (MIOpen kernel databases, PAL
prebuilt libs) from anonymous-public S3 buckets. Does not implement add/push/repro.

Pointer file schema (the only one we handle):
    outs:
    - md5: <32-char-hex>      # may end with `.dir` for directory hashes
      size: <bytes>
      hash: md5
      path: <relpath-from-pointer-file-dir>

Remote config (.dvc/config, INI):
    [core]
        remote = storage
    ['remote "storage"']
        url = s3://<bucket>/<prefix>
        allow_anonymous_login = true

S3 key scheme (dvc 3.x "new" layout, used by therock-dvc):
    `<prefix>/files/md5/<md5[:2]>/<md5[2:]>` for both files and `.dir` manifests.

Library:
    from fetch_dvc_artifacts import pull
    result = pull(Path("/path/to/project"))

CLI:
    python fetch_dvc_artifacts.py pull [DIR ...] [--jobs N] [--cache-dir PATH]
                                       [--no-cache] [-v]
"""

import argparse
import concurrent.futures
import configparser
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.client import Config


DEFAULT_JOBS = 8
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "therock-dvc"


class FetchError(Exception):
    """Raised on missing config, download failure, or MD5 mismatch."""


@dataclass
class PullResult:
    fetched: int = 0  # downloaded from remote
    cached: int = 0  # served from local content-addressed cache
    skipped: int = 0  # already at destination with matching md5

    def __iadd__(self, other: "PullResult") -> "PullResult":
        self.fetched += other.fetched
        self.cached += other.cached
        self.skipped += other.skipped
        return self


@dataclass(frozen=True)
class _Out:
    """One entry from a .dvc pointer file's `outs:` list."""

    md5: str
    size: int
    path: str

    @property
    def is_dir(self) -> bool:
        return self.md5.endswith(".dir")

    @property
    def bare_md5(self) -> str:
        return self.md5[:-4] if self.is_dir else self.md5


@dataclass(frozen=True)
class _Remote:
    bucket: str
    prefix: str
    anonymous: bool


# --------------------------------------------------------------------------
# .dvc YAML subset parser
# --------------------------------------------------------------------------


def _parse_dvc_pointer(path: Path) -> list[_Out]:
    """Parse a .dvc pointer file. Schema is fixed; full YAML is overkill."""
    text = path.read_text()
    outs: list[_Out] = []
    current: dict[str, str] | None = None
    in_outs = False
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not in_outs:
            if line == "outs:":
                in_outs = True
            continue
        if line.startswith("- "):
            if current is not None:
                outs.append(_to_out(current, path))
            current = {}
            line = line[2:]
        elif line.startswith("  "):
            line = line[2:]
        else:
            # A non-indented line ends the outs: section (e.g., another top-level key).
            in_outs = False
            continue
        if current is None:
            raise FetchError(f"{path}: 'outs:' content before list item: {raw_line!r}")
        if ":" not in line:
            raise FetchError(f"{path}: expected 'key: value', got {raw_line!r}")
        key, _, value = line.partition(":")
        current[key.strip()] = value.strip()
    if current is not None:
        outs.append(_to_out(current, path))
    if not outs:
        raise FetchError(f"{path}: no 'outs:' entries")
    return outs


def _to_out(d: dict[str, str], path: Path) -> _Out:
    try:
        md5 = d["md5"]
        size = int(d["size"])
        out_path = d["path"]
    except KeyError as e:
        raise FetchError(f"{path}: missing required field {e}") from None
    except ValueError as e:
        raise FetchError(f"{path}: bad size: {e}") from None
    hash_alg = d.get("hash", "md5")
    if hash_alg != "md5":
        raise FetchError(f"{path}: unsupported hash algorithm {hash_alg!r}")
    bare = md5[:-4] if md5.endswith(".dir") else md5
    if len(bare) != 32 or not all(c in "0123456789abcdef" for c in bare):
        raise FetchError(f"{path}: invalid md5 {md5!r}")
    return _Out(md5=md5, size=size, path=out_path)


# --------------------------------------------------------------------------
# .dvc/config parser
# --------------------------------------------------------------------------


def _parse_dvc_config(path: Path) -> _Remote:
    """Extract the configured remote's URL and anonymous flag from .dvc/config.

    DVC's .dvc/config uses INI section headers like ['remote "storage"'] -
    the single quotes are literally part of the section name in the file, and
    Python's configparser preserves them. We canonicalize by stripping the
    surrounding single quotes before matching `remote "<name>"`.
    """
    cp = configparser.ConfigParser()
    cp.read(path)
    if "core" not in cp:
        raise FetchError(f"{path}: missing [core] section")
    remote_name = cp["core"].get("remote")
    if not remote_name:
        raise FetchError(f"{path}: [core] missing 'remote' key")
    target = f'remote "{remote_name}"'
    section_name: str | None = None
    for s in cp.sections():
        if s == target or s.strip("'") == target:
            section_name = s
            break
    if section_name is None:
        raise FetchError(f"{path}: missing [{target}] section")
    url = cp[section_name].get("url")
    if not url:
        raise FetchError(f"{path}: [{target}] missing 'url'")
    parsed = urlparse(url)
    if parsed.scheme != "s3":
        raise FetchError(f"{path}: only s3:// remotes are supported, got {url!r}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    anonymous = cp[section_name].getboolean("allow_anonymous_login", fallback=False)
    return _Remote(bucket=bucket, prefix=prefix, anonymous=anonymous)


# --------------------------------------------------------------------------
# S3 key + cache helpers
# --------------------------------------------------------------------------


def _s3_key(remote: _Remote, md5: str) -> str:
    """Build the S3 key for a content-addressed blob.

    Uses the dvc 3.x "new" layout: <prefix>/files/md5/<first2>/<rest30>.
    The therock-dvc bucket was migrated to this layout; the older
    <prefix>/<first2>/<rest30> layout is not in use here.
    """
    bare = md5[:-4] if md5.endswith(".dir") else md5
    if len(bare) != 32:
        raise FetchError(f"invalid md5 length: {md5!r}")
    suffix = ".dir" if md5.endswith(".dir") else ""
    parts = [remote.prefix, "files", "md5", bare[:2], bare[2:] + suffix]
    return "/".join(p for p in parts if p)


def _cache_path(cache_dir: Path, md5: str) -> Path:
    bare = md5[:-4] if md5.endswith(".dir") else md5
    suffix = ".dir" if md5.endswith(".dir") else ""
    return cache_dir / bare[:2] / (bare[2:] + suffix)


def _md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _materialize_from_cache(cache_file: Path, dest: Path) -> None:
    """Hardlink cache_file to dest; fall back to copy across filesystems."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(cache_file, dest)
    except OSError:
        shutil.copy2(cache_file, dest)


def _store_in_cache(src: Path, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        return
    # Unique temp name: distinct pointer files referencing the same content hash
    # are materialized by separate workers, which would otherwise race on a
    # shared `<cache_file>.tmp` path. os.replace stays atomic and idempotent.
    tmp = cache_file.with_suffix(f"{cache_file.suffix}.{uuid.uuid4().hex}.tmp")
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, cache_file)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------
# Atomic-write download with MD5 verification
# --------------------------------------------------------------------------


def _download_blob(
    s3, remote: _Remote, md5: str, dest: Path, expected_size: int | None
) -> None:
    """Download s3://<bucket>/<key> to dest atomically, verifying md5 (and size).

    MD5 is computed by re-reading the file after download, not streamed during
    write. boto3's multipart download writes chunks out of order via seek+write,
    which breaks any sequential-hash wrapper. Re-reading costs one extra pass
    (~150 MB/s on SSD); negligible vs network time for our workload.
    """
    key = _s3_key(remote, md5)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with tmp.open("wb") as f:
            s3.download_fileobj(remote.bucket, key, f)
        actual_size = tmp.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise FetchError(
                f"{dest}: size mismatch (expected {expected_size}, got {actual_size})"
            )
        actual_md5 = _md5_of(tmp)
        bare_expected = md5[:-4] if md5.endswith(".dir") else md5
        if actual_md5 != bare_expected:
            raise FetchError(
                f"{dest}: md5 mismatch (expected {bare_expected}, got {actual_md5})"
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------
# Per-pointer-entry materialization
# --------------------------------------------------------------------------


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def _materialize_file(
    s3,
    remote: _Remote,
    md5: str,
    size: int | None,
    dest: Path,
    cache_dir: Path | None,
    log: Callable[[str], None],
) -> PullResult:
    """Materialize one file: destination check -> cache check -> download.

    `size` may be None for files referenced from a .dir manifest (which carries
    md5 only). In that case the size check is skipped; md5 verification still
    happens.
    """
    label = f"{dest.name}" if size is None else f"{dest.name} ({_human(size)})"

    # Fast path 1: destination already correct.
    if dest.exists():
        if size is None or dest.stat().st_size == size:
            if _md5_of(dest) == md5:
                log(f"  ok        {label} [present]")
                return PullResult(skipped=1)

    # Fast path 2: content-addressed cache hit.
    if cache_dir is not None:
        cache_file = _cache_path(cache_dir, md5)
        if cache_file.exists():
            if size is None or cache_file.stat().st_size == size:
                if _md5_of(cache_file) == md5:
                    _materialize_from_cache(cache_file, dest)
                    log(f"  cached    {label}")
                    return PullResult(cached=1)
            log(f"  warn      cache entry corrupt for {md5[:8]}..., discarding")
            cache_file.unlink()

    log(f"  fetching  {label}")
    _download_blob(s3, remote, md5, dest, expected_size=size)
    if cache_dir is not None:
        _store_in_cache(dest, _cache_path(cache_dir, md5))
    return PullResult(fetched=1)


def _materialize_dir(
    s3,
    remote: _Remote,
    out: _Out,
    dest: Path,
    cache_dir: Path | None,
    log: Callable[[str], None],
) -> PullResult:
    """Handle a `.dir` directory hash: fetch JSON manifest, materialize each entry."""
    log(f"  manifest  {dest}/ ({out.md5[:8]}...)")

    # Try cache for the manifest first.
    manifest_bytes: bytes | None = None
    cache_file = _cache_path(cache_dir, out.md5) if cache_dir is not None else None
    if cache_file is not None and cache_file.exists():
        if _md5_of(cache_file) == out.bare_md5:
            manifest_bytes = cache_file.read_bytes()

    if manifest_bytes is None:
        # Download manifest. If we have a cache, write it there directly (it's
        # an immutable content-addressed blob). Otherwise, use a tempfile.
        if cache_file is not None:
            _download_blob(s3, remote, out.md5, cache_file, expected_size=None)
            manifest_bytes = cache_file.read_bytes()
        else:
            with tempfile.TemporaryDirectory(prefix="therock-dvc-") as tmp_dir:
                manifest_path = Path(tmp_dir) / "manifest"
                _download_blob(s3, remote, out.md5, manifest_path, expected_size=None)
                manifest_bytes = manifest_path.read_bytes()

    try:
        entries = json.loads(manifest_bytes)
    except json.JSONDecodeError as e:
        raise FetchError(f"{dest}: manifest is not valid JSON: {e}") from None
    if not isinstance(entries, list):
        raise FetchError(f"{dest}: manifest is not a JSON array")

    result = PullResult()
    for entry in entries:
        if not isinstance(entry, dict) or "md5" not in entry or "relpath" not in entry:
            raise FetchError(f"{dest}: malformed manifest entry {entry!r}")
        # Manifest entries don't carry size; trust md5 verification alone.
        result += _materialize_file(
            s3,
            remote,
            md5=entry["md5"],
            size=None,
            dest=dest / entry["relpath"],
            cache_dir=cache_dir,
            log=log,
        )
    return result


# --------------------------------------------------------------------------
# Discovery + parallel orchestration
# --------------------------------------------------------------------------


def _walk_dvc_pointers(project_dir: Path) -> list[Path]:
    # rglob("*.dvc") matches the .dvc/ config directory too; filter to files.
    return sorted(p for p in project_dir.rglob("*.dvc") if p.is_file())


# Inner s3transfer max_concurrency is 10 by default; we bump the boto3
# connection pool to `jobs * INNER_S3TRANSFER_CONCURRENCY` so that multipart
# ranged GETs don't starve when many outer workers hit S3 simultaneously.
_INNER_S3TRANSFER_CONCURRENCY = 10
_MIN_POOL_CONNECTIONS = 50
_BOTO3_RETRIES = {"max_attempts": 5, "mode": "adaptive"}


def _make_s3_client(remote: _Remote, *, max_pool_connections: int):
    """Build an S3 client. Anonymous (UNSIGNED) when the remote allows it."""
    config_kwargs: dict[str, object] = {
        "max_pool_connections": max_pool_connections,
        "retries": _BOTO3_RETRIES,
    }
    if remote.anonymous:
        config_kwargs["signature_version"] = UNSIGNED
    return boto3.client("s3", config=Config(**config_kwargs))


def _process_pointer(
    s3,
    remote: _Remote,
    pointer: Path,
    cache_dir: Path | None,
    log: Callable[[str], None],
) -> PullResult:
    """Materialize every entry in one .dvc pointer file."""
    outs = _parse_dvc_pointer(pointer)
    result = PullResult()
    for out in outs:
        dest = pointer.parent / out.path
        if out.is_dir:
            result += _materialize_dir(s3, remote, out, dest, cache_dir, log)
        else:
            result += _materialize_file(
                s3,
                remote,
                md5=out.md5,
                size=out.size,
                dest=dest,
                cache_dir=cache_dir,
                log=log,
            )
    return result


def pull(
    project_dir: Path,
    *,
    jobs: int = DEFAULT_JOBS,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
    log: Callable[[str], None] = print,
) -> PullResult:
    """Materialize every .dvc-tracked file under project_dir from its S3 remote.

    Raises FetchError if .dvc/config is missing or any download fails.
    """
    project_dir = Path(project_dir).resolve()
    config_file = project_dir / ".dvc" / "config"
    if not config_file.exists():
        raise FetchError(f"no .dvc/config in {project_dir}")
    remote = _parse_dvc_config(config_file)

    pointers = _walk_dvc_pointers(project_dir)
    if not pointers:
        log(f"no .dvc pointer files found under {project_dir}")
        return PullResult()

    pool_size = max(jobs * _INNER_S3TRANSFER_CONCURRENCY, _MIN_POOL_CONNECTIONS)
    s3 = _make_s3_client(remote, max_pool_connections=pool_size)

    log(
        f"pull: {len(pointers)} pointer file(s) from "
        f"s3://{remote.bucket}/{remote.prefix}"
    )

    result = PullResult()
    errors: list[tuple[Path, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {
            ex.submit(_process_pointer, s3, remote, p, cache_dir, log): p
            for p in pointers
        }
        for fut in concurrent.futures.as_completed(futures):
            pointer = futures[fut]
            try:
                result += fut.result()
            except Exception as e:
                errors.append((pointer, e))

    if errors:
        for pointer, err in errors:
            log(f"  ERROR     {pointer}: {err}")
        raise FetchError(f"{len(errors)} file(s) failed to fetch (see log above)")

    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_dvc_artifacts",
        description="Fetch large binary artifacts referenced by .dvc files from S3.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_pull = sub.add_parser("pull", help="Pull all .dvc-referenced artifacts.")
    p_pull.add_argument(
        "directories",
        nargs="*",
        default=["."],
        help="Project directories to pull (each must contain .dvc/config). "
        "Default: cwd.",
    )
    p_pull.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"Parallel pointer-file workers (default {DEFAULT_JOBS}).",
    )
    p_pull.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Content-addressed cache directory (default {DEFAULT_CACHE_DIR}).",
    )
    p_pull.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the local content-addressed cache.",
    )
    p_pull.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable boto3/botocore debug logging.",
    )
    args = parser.parse_args(argv)

    if args.verbose:
        boto3.set_stream_logger("botocore", logging.INFO)

    cache_dir = None if args.no_cache else args.cache_dir
    total = PullResult()
    try:
        for d in args.directories:
            total += pull(Path(d), jobs=args.jobs, cache_dir=cache_dir)
    except FetchError as e:
        print(f"fetch_dvc_artifacts: {e}", file=sys.stderr)
        return 1

    print(
        f"summary: fetched={total.fetched} "
        f"cached={total.cached} skipped={total.skipped}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
