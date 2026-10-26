# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A local, directory-backed object store for local builds.

Phase 0 of the roadmap runs entirely on one machine with no CI and no network:
a local build populates a cache, a later local build reuses from it. This module
is the local fallback for S3 "immutable artifact storage" — it copies artifact
files into a content directory, records real SHA-256 digests, and satisfies the
``ObjectStore`` protocol used by ``restore`` (``exists`` / ``fetch_digest``).

Everything here is deliberately simple and provisional: the goal is to flush
design issues cheaply before any contract is frozen.
"""


from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile

from .manifest import ArtifactFile, OutputObject
from .diagnostics import trace, traced, span, ByteProgress

_URI_SCHEME = "local://"


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class LocalObjectStore:
    """Compatibility object-store adapter with immutable per-write generations.

    Managed component publication uses store_component below. This adapter
    does not publish database entries or claim a producing build succeeded.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path helpers ------------------------------------------------------
    def uri_for(self, key: str) -> str:
        return f"{_URI_SCHEME}{key}"

    def path_for(self, uri: str) -> Path:
        key = uri[len(_URI_SCHEME) :] if uri.startswith(_URI_SCHEME) else uri
        safe_relative(key)
        path = self.root / key
        if any(p.is_symlink() for p in (path, *path.parents) if p != self.root):
            raise ValueError("symlink in object-store path")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("object path escapes storage")
        return path

    # -- writes ------------------------------------------------------------
    def upload_file(self, src: Path, key: str | None = None) -> OutputObject:
        """Copy a built artifact file into the store; return its OutputObject.

        The returned object carries the real digest and size, so a later
        ``restore`` can verify integrity exactly as it will against S3.
        """
        src = Path(src)
        key = key or src.name
        safe_relative(key)
        if "/" in key:
            raise ValueError("object key must be a filename")
        generation = Path(tempfile.mkdtemp(prefix="object-", dir=self.root))
        try:
            staged = generation / ".pending"
            with regular_reader(src) as source, staged.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            record = _object_record(staged)
            dest = generation / key
            staged.rename(dest)
            return OutputObject(key=key, uri=self.uri_for(dest.relative_to(self.root).as_posix()),
                                digest=record.content_sha256, size_bytes=record.size_bytes)
        except BaseException:
            shutil.rmtree(generation)
            raise

    # -- ObjectStore protocol (reads) -------------------------------------
    def exists(self, uri: str) -> bool:
        return self.path_for(uri).exists()

    def fetch_digest(self, uri: str) -> str | None:
        path = self.path_for(uri)
        if not path.exists():
            return None
        return sha256_file(path)

    def download_file(self, uri: str, dest: Path) -> Path:
        """Copy an object out of the store (the 'download' in flowchart 2)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.path_for(uri), dest)
        return dest


# The relational local cache stores complete component exports as two objects.
# inventory.json also describes every unpacked file, directory and symlink.
# Archives preserve executable bits and links without changing the DB schema.
LOCAL_OBJECTS = ("inventory.json", "payload.tar")


def _file_state(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def regular_reader(path: Path):
    """Read a stable regular file; reject symlinks, pipes and replacements.

    Ancestor directories must be controlled by cooperating local processes.
    This is not a sandbox against another process with the same user rights.
    """
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"expected a regular file: {path}")
    flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _file_state(before) != _file_state(opened):
            raise ValueError(f"file changed before reading: {path}")
        yield stream
        if (_file_state(opened) != _file_state(os.fstat(stream.fileno()))
                or _file_state(opened) != _file_state(path.lstat())):
            raise ValueError(f"file changed while reading: {path}")


def safe_relative(path: str) -> str:
    # Reuse the database path contract, including Windows drive/path rejection.
    ArtifactFile(path, "0" * 64, 0)
    return path


def _check_link(relative: str, target: str) -> None:
    if (not target or target.startswith("/") or "\\" in target or ":" in target
            or any(ord(c) < 32 or ord(c) == 127 for c in target)):
        raise ValueError(f"unsupported symlink target: {relative} -> {target}")
    parts = relative.split("/")[:-1]
    for part in target.split("/"):
        if part == "..":
            if not parts:
                raise ValueError(f"symlink escapes artifact: {relative} -> {target}")
            parts.pop()
        elif part not in ("", "."):
            parts.append(part)


def _tree_state(directory: Path) -> dict[str, tuple]:
    result = {}
    def visit(parent):
        for path in sorted(parent.iterdir()):
            relative = safe_relative(path.relative_to(directory).as_posix())
            info = path.lstat()
            result[relative] = _file_state(info)
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            elif not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                raise ValueError(f"special file in artifact export: {path}")
    visit(directory)
    return result


def _object_record(path: Path) -> ArtifactFile:
    digest = hashlib.sha256()
    size = 0
    progress = ByteProgress("HASH_PROGRESS", path=path)
    with regular_reader(path) as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
            progress.advance(len(block))
    return ArtifactFile(path.name, digest.hexdigest(), size)


@dataclass(frozen=True)
class StoredComponent:
    directory: Path
    files: tuple[ArtifactFile, ...]


def store_component(source: Path, cache_root: Path) -> StoredComponent:
    """Store one complete export using the existing schema-1 inventory."""
    return _store_exports({"": Path(source)}, cache_root, artifact=Path(source).name)


def store_bundle(artifacts: Path, key, cache_root: Path) -> StoredComponent:
    """Store all requested components in one archive and immutable generation."""
    from .bundles import validate_bundle
    validate_bundle(key.artifact_identity, key.components)
    return _store_exports({name: Path(artifacts) / name for name in key.components},
                          cache_root, artifact=key.artifact_identity)


@traced("PACKAGE")
def _store_exports(sources, cache_root, *, artifact):
    cache_root = Path(cache_root).resolve()
    components = tuple(sorted(name for name in sources if name))
    def tree_state():
        state, locations = {}, {}
        for prefix, source in sources.items():
            if not stat.S_ISDIR(source.lstat().st_mode):
                raise ValueError(f"artifact export must be a real directory: {source}")
            root = source.resolve(strict=True)
            if root == cache_root or root in cache_root.parents or cache_root in root.parents:
                raise ValueError("artifact source and cache storage must not overlap")
            with regular_reader(source / "artifact_manifest.txt"):
                pass
            if prefix:
                state[prefix] = _file_state(source.lstat())
                locations[prefix] = (source, root, "")
            for relative, value in _tree_state(source).items():
                name = prefix + "/" + relative if prefix else relative
                state[name] = value
                locations[name] = (source / relative, root, relative)
        return state, locations
    before, locations = tree_state()
    trace("EXPORT_SCAN", artifact=artifact, components=list(components))
    total_bytes = sum(state[3] for state in before.values() if stat.S_ISREG(state[2]))
    trace("EXPORT_INVENTORY", artifact=artifact, members=len(before), file_bytes=total_bytes)
    progress = ByteProgress("PACK_PROGRESS", total_bytes=total_bytes, artifact=artifact)
    objects = cache_root / "objects"
    objects.mkdir(parents=True, exist_ok=True)
    generation = Path(tempfile.mkdtemp(prefix="artifact-", dir=objects))
    staging = generation / "staging"
    staging.mkdir()
    trace("GENERATION_CREATED", artifact=artifact, directory=generation, published=False)
    try:
        inventory = []
        trace("ARCHIVE_WRITE", artifact=artifact, path=staging / "payload.tar", compression="none")
        with tarfile.open(staging / "payload.tar", "w", format=tarfile.PAX_FORMAT) as archive:
            for relative in sorted(before):
                path, root, component_relative = locations[relative]
                info = path.lstat()
                if _file_state(info) != before[relative]:
                    raise ValueError(f"artifact changed during packaging: {path}")
                member = tarfile.TarInfo(relative)
                member.mode = stat.S_IMODE(info.st_mode) & 0o777
                member.mtime = int(info.st_mtime)
                item = {"path": relative, "mode": member.mode}
                if stat.S_ISDIR(info.st_mode):
                    member.type = tarfile.DIRTYPE
                    item["type"] = "directory"
                    archive.addfile(member)
                elif stat.S_ISLNK(info.st_mode):
                    target = os.readlink(path)
                    _check_link(component_relative, target)
                    if not path.resolve().is_relative_to(root):
                        raise ValueError(f"symlink resolves outside artifact: {relative}")
                    member.type, member.linkname = tarfile.SYMTYPE, target
                    item.update(type="symlink", target=target)
                    archive.addfile(member)
                else:
                    member.size = info.st_size
                    item.update(type="file", size=member.size)
                    with regular_reader(path) as reader:
                        digest = hashlib.sha256()
                        class HashReader:
                            def read(self, size=-1):
                                block = reader.read(size)
                                digest.update(block)
                                progress.advance(len(block))
                                return block
                        archive.addfile(member, HashReader())
                    item["sha256"] = digest.hexdigest()
                inventory.append(item)
        (staging / "inventory.json").write_text(
            json.dumps({"schema_version": 2 if components else 1, "members": inventory,
                        **({"components": list(components)} if components else {})}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        trace("INVENTORY_WRITTEN", artifact=artifact, path=staging / "inventory.json",
              members=len(inventory), files=sum(i["type"] == "file" for i in inventory),
              directories=sum(i["type"] == "directory" for i in inventory),
              symlinks=sum(i["type"] == "symlink" for i in inventory))
        if before != tree_state()[0]:
            raise ValueError("artifact export changed during packaging")
        # Read back the exact stored archive and check every member's bytes.
        unpack_component(staging, None, components=components)
        with span("OBJECT_DIGESTS", artifact=artifact):
            files = tuple(_object_record(staging / name) for name in LOCAL_OBJECTS)
        for name in LOCAL_OBJECTS:
            with (staging / name).open("rb") as stream:
                os.fsync(stream.fileno())
        final = generation / "files"
        staging.rename(final)
        for item in files:
            trace("OBJECT_SAVED", artifact=artifact, path=final / item.relative_path,
                  size_bytes=item.size_bytes, sha256=item.content_sha256,
                  ready_entry_committed=False)
        return StoredComponent(final, files)
    except BaseException:
        shutil.rmtree(generation)
        trace("GENERATION_REMOVED", directory=generation, reason="packaging failed")
        raise


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate inventory JSON field: {key}")
        result[key] = value
    return result


@traced("INVENTORY_VERIFY")
def unpack_component(objects: Path, destination: Path | None, *, components=()) -> None:
    """Verify the exact member inventory, optionally extracting into a private tree.

    No tarfile.extract/extractall is used. Links are created last, only inside
    the artifact tree. Hardlinks and special files are rejected; packaging
    serializes source hardlinks as regular files.
    """
    with regular_reader(objects / "inventory.json") as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("artifact member inventory is too large")
    value = json.loads(raw, object_pairs_hook=_unique_pairs)
    expected_fields = {"schema_version", "members"} | ({"components"} if components else set())
    if (not isinstance(value, dict) or set(value) != expected_fields
            or type(value["schema_version"]) is not int or value["schema_version"] != (2 if components else 1)
            or (components and value.get("components") != list(components))
            or not isinstance(value["members"], list)):
        raise ValueError("unsupported artifact member inventory")
    entries = {}
    for item in value["members"]:
        if not isinstance(item, dict):
            raise ValueError("invalid artifact inventory member")
        path = safe_relative(item.get("path"))
        kind = item.get("type")
        fields = {"path", "type", "mode"}
        if kind == "file":
            fields |= {"sha256", "size"}
            ArtifactFile(path, item.get("sha256"), item.get("size"))
        elif kind == "symlink":
            fields.add("target")
            if not isinstance(item.get("target"), str):
                raise ValueError("invalid symlink target")
            if components and "/" not in path:
                raise ValueError("bundle component root must be a directory")
            _check_link(path.split("/", 1)[1] if components else path, item["target"])
        elif kind != "directory":
            raise ValueError("unsupported artifact member type")
        if (set(item) != fields or type(item["mode"]) is not int
                or not 0 <= item["mode"] <= 0o777 or path in entries):
            raise ValueError("invalid or duplicate artifact inventory member")
        entries[path] = item
    if components:
        if {name.split("/", 1)[0] for name in entries} != set(components):
            raise ValueError("bundle inventory does not contain exactly the requested components")
        for component in components:
            if (entries.get(component, {}).get("type") != "directory"
                    or entries.get(component + "/artifact_manifest.txt", {}).get("type") != "file"):
                raise ValueError(f"bundle component export is incomplete: {component}")
    elif entries.get("artifact_manifest.txt", {}).get("type") != "file":
        raise ValueError("artifact_manifest.txt is missing from the export")
    for path in entries:
        for parent in Path(path).parents:
            if parent == Path("."):
                continue
            if entries.get(parent.as_posix(), {}).get("type") != "directory":
                raise ValueError(f"member parent is not an inventoried directory: {path}")

    total_bytes = sum(item.get("size", 0) for item in entries.values())
    trace("INVENTORY_CHECK", objects=objects, members=len(entries), file_bytes=total_bytes,
          action="extract_to_private_directory" if destination is not None else "read_back_only",
          destination=destination)
    progress = ByteProgress("INVENTORY_PROGRESS", total_bytes=total_bytes, objects=objects)
    seen = set()
    with regular_reader(objects / "payload.tar") as reader:
        with tarfile.open(fileobj=reader, mode="r:") as archive:
            for member in archive:
                path = safe_relative(member.name)
                if path in seen or path not in entries:
                    raise ValueError(f"unexpected or duplicate archive member: {path}")
                seen.add(path)
                item = entries[path]
                if member.mode != item["mode"]:
                    raise ValueError(f"archive mode mismatch: {path}")
                dest = destination / path if destination is not None else None
                if item["type"] == "file":
                    if not member.isfile() or member.size != item["size"]:
                        raise ValueError(f"archive file type/size mismatch: {path}")
                    if dest is not None:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as contents, (
                        dest.open("xb") if dest is not None else nullcontext()
                    ) as writer:
                        for block in iter(lambda: contents.read(1 << 20), b""):
                            digest.update(block)
                            progress.advance(len(block))
                            if writer is not None:
                                writer.write(block)
                    if digest.hexdigest() != item["sha256"]:
                        raise ValueError(f"archive member digest mismatch: {path}")
                    if dest is not None:
                        dest.chmod(item["mode"])
                elif item["type"] == "directory":
                    if not member.isdir():
                        raise ValueError(f"archive directory type mismatch: {path}")
                    if dest is not None:
                        dest.mkdir(parents=True, exist_ok=True)
                elif not member.issym() or member.linkname != item["target"]:
                    raise ValueError(f"archive symlink mismatch: {path}")
    if seen != set(entries):
        raise ValueError("archive is missing required inventory members")
    if destination is not None:
        for path, item in entries.items():
            if item["type"] == "symlink":
                dest = destination / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(item["target"])
        # Lexical checks alone do not catch '..' traversing another symlink.
        for path, item in entries.items():
            if item["type"] == "symlink":
                try:
                    resolved = (destination / path).resolve()
                except (RuntimeError, OSError) as exc:
                    raise ValueError(f"unresolvable artifact symlink: {path}") from exc
                component_root = destination / path.split("/", 1)[0] if components else destination
                if not resolved.is_relative_to(component_root.resolve()):
                    raise ValueError(f"symlink resolves outside artifact: {path}")
        for path in sorted(entries, key=lambda s: s.count("/"), reverse=True):
            if entries[path]["type"] == "directory":
                (destination / path).chmod(entries[path]["mode"])
    trace("INVENTORY_VERIFIED", objects=objects, members=len(seen), file_bytes=total_bytes)
