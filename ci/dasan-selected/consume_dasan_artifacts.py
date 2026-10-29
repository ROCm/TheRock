#!/usr/bin/env python3
"""Consume only pinned GitHub DASan producers into a fresh ROCm prefix.

The output directory IS the SDK root. Component payloads are selected from
artifacts/<group>_<component>_<family>/<manifest-root>, then flattened exactly
like TheRock ArtifactPopulator(flatten=True). Primary groups use their pinned
producer; remaining dependency-closure groups use that stage archive. Empty
bootstrap placeholders are recorded. Compiler files are authoritative. Requires Python
3.9+ and GH_TOKEN (or GITHUB_TOKEN) for GitHub artifact downloads. No packages.
"""
import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PINNED_RUNS = {
    "compiler-runtime": 40443090683, "math-libs": 40545400942,
    "runtime-tests": 40586626409, "comm-libs": 40586632932,
    "storage-libs": 40586640605, "profiler-apps": 40586651063,
    "media-libs": 40584081938, "dctools-core": 40584069915,
    "cv-libs": 40584055579, "emulation": 40583338049,
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Unsafe archive path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe archive path: " + value)
    return p.as_posix()


def digest_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("Missing pinned sha256 digest")
    return value[7:]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def api_request(url, token):
    # Authorization is sent only to the fixed GitHub API, never a storage URL.
    if urllib.parse.urlsplit(url).netloc != "api.github.com":
        raise ValueError("Unexpected API host")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "pinned-dasan-consumer",
               "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.build_opener(NoRedirect).open(req, timeout=120)


def artifact_zip(producer, cache, evidence, token, logs=False):
    aid = int(producer["logs_artifact_id" if logs else "artifact_id"])
    expected = digest_value(producer["logs_artifact_digest" if logs else "artifact_digest"])
    base = "https://api.github.com/repos/ROCm/TheRock/actions/artifacts/" + str(aid)
    # Verify identity even on cache hits. Never substitute by latest/name/run.
    with api_request(base, token) as response:
        metadata = json.load(response)
    expected_name = producer.get("logs_artifact_name", "dasan-build-logs-and-tests") if logs else producer["artifact_name"]
    workflow = metadata.get("workflow_run") or {}
    checks = (metadata.get("id") == aid, metadata.get("name") == expected_name,
              workflow.get("id") == producer["run_id"],
              workflow.get("head_sha") == producer["head_sha"],
              metadata.get("digest") == "sha256:" + expected,
              not metadata.get("expired", True))
    if not all(checks):
        raise ValueError("Artifact metadata differs from pin for ID " + str(aid))
    # Metadata has no access token or signed storage URL.
    write_json(evidence / (str(aid) + "-metadata.json"), metadata)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (str(aid) + "-" + expected + ".zip")
    if archive.exists() and sha256(archive) != expected:
        raise ValueError("Cached archive digest mismatch: " + archive.name)
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        if partial.exists():
            raise ValueError("Incomplete/concurrent download exists: " + partial.name)
        try:
            try:
                response = api_request(base + "/zip", token)
            except urllib.error.HTTPError as exc:
                if exc.code not in (301, 302, 303, 307, 308):
                    raise
                location = exc.headers.get("Location", "")
                if urllib.parse.urlsplit(location).scheme != "https":
                    raise ValueError("Unsafe artifact redirect") from None
                # Fresh request: explicitly no Authorization header.
                response = urllib.request.urlopen(location, timeout=120)
            with response, partial.open("xb") as output:
                shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            if sha256(partial) != expected:
                raise ValueError("Downloaded archive digest mismatch for ID " + str(aid))
            partial.replace(archive)
        finally:
            if partial.exists():
                partial.unlink()
    return archive


def unpack_inner(archive, basename, destination):
    safe_name(basename)
    with zipfile.ZipFile(archive) as z:
        matches = []
        seen = set()
        for info in z.infolist():
            name = safe_name(info.filename)
            if name in seen:
                raise ValueError("Duplicate ZIP member: " + name)
            seen.add(name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP symlink unsupported: " + name)
            if name == basename:
                matches.append(info)
        if len(matches) != 1 or matches[0].is_dir():
            raise ValueError("Expected exactly one ZIP member: " + basename)
        with z.open(matches[0]) as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    return sha256(destination)


class Overlay:
    def __init__(self, prefix, evidence):
        self.prefix = prefix
        self.evidence = evidence
        self.files = {}
        self.counts = {}
        self.overlaps = []

    def destination(self, rel):
        rel = safe_name(rel)
        dest = self.prefix / rel
        # Never write through a symlink, even when it currently points in-prefix.
        current = self.prefix
        for part in PurePosixPath(rel).parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError("Archive write through symlink: " + rel)
            if current.exists() and not current.is_dir():
                raise ValueError("Archive parent is not a directory: " + rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def install(self, tf, member, rel, origin, hard_target=None):
        dest = self.destination(rel)
        if member.isdir():
            if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
                raise ValueError("Directory collision: " + rel)
            dest.mkdir(exist_ok=True)
            return
        item = dict(origin, path=rel, mode=oct(member.mode & 0o777))
        temporary = None
        if member.isfile():
            item["type"] = "file"
            h = hashlib.sha256()
            with tf.extractfile(member) as source, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as output:
                temporary = Path(output.name)
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    h.update(block)
            item.update(sha256=h.hexdigest(), size=member.size)
        elif member.issym():
            target = member.linkname
            if not target or "\\" in target or ":" in target or "\x00" in target or target.startswith("/"):
                raise ValueError("Unsafe symlink: " + rel)
            normalized = posixpath.normpath(posixpath.join(posixpath.dirname(rel), target))
            if normalized == ".." or normalized.startswith("../"):
                raise ValueError("Escaping symlink: " + rel)
            item.update(type="symlink", target=target)
        elif member.islnk():
            if hard_target is None or hard_target not in self.files:
                raise ValueError("Unresolved hardlink: " + rel)
            original = self.files[hard_target]
            if original["type"] != "file":
                raise ValueError("Hardlink to non-file: " + rel)
            item.update(type="file", sha256=original["sha256"], size=original["size"], hardlink=hard_target)
        else:
            raise ValueError("Unsupported archive entry: " + rel)
        try:
            previous = self.files.get(rel)
            if previous is not None:
                equal = all(previous.get(k) == item.get(k) for k in ("type", "sha256", "target", "mode"))
                protected = previous["stage"] == "compiler-runtime"
                self.overlaps.append({"path": rel, "existing": previous, "incoming": item,
                                      "action": "identical" if equal else "protected-compiler" if protected else "conflict"})
                if not equal and not protected:
                    raise ValueError("Different non-compiler payloads overlap: " + rel)
                return
            if dest.exists() or dest.is_symlink():
                raise ValueError("Untracked destination collision: " + rel)
            if member.isfile():
                temporary.chmod(member.mode & 0o777)
                temporary.replace(dest)
                temporary = None
            elif member.issym():
                dest.symlink_to(member.linkname)
            else:
                os.link(self.prefix / hard_target, dest)
            self.files[rel] = item
            self.counts[origin["stage"]] = self.counts.get(origin["stage"], 0) + 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def save(self):
        self.evidence.mkdir(parents=True, exist_ok=True)
        with (self.evidence / "installed-files.jsonl").open("w") as stream:
            for key in sorted(self.files):
                stream.write(json.dumps(self.files[key], sort_keys=True) + "\n")
        write_json(self.evidence / "overlaps.json", self.overlaps)
        write_json(self.evidence / "availability.json", {
            "counts": self.counts,
            "paths": sorted(self.files),
            "test_candidates": sorted(p for p in self.files if any(x in p.lower() for x in ("test", "ctest", "lit.py"))),
        })


def component_roots(tarpath, groups, required=None):
    required = groups if required is None else required
    roots = {}
    found = set()
    omitted = set()
    manifests = {}
    with tarfile.open(tarpath, "r|*") as tf:
        for member in tf:
            name = safe_name(member.name)
            parts = PurePosixPath(name).parts
            if len(parts) < 2 or parts[0] != "artifacts":
                continue
            match = re.fullmatch(r"([^_]+)_([^_]+)_([^_]+)", parts[1])
            if not match:
                continue
            group, component, family = match.groups()
            if group not in groups or family not in ("generic", "gfx950"):
                omitted.add(parts[1])
                continue
            if len(parts) == 3 and parts[2] == "artifact_manifest.txt":
                if not member.isfile() or member.size > 1024 * 1024:
                    raise ValueError("Invalid component manifest: " + name)
                if parts[1] in manifests:
                    raise ValueError("Duplicate component manifest: " + name)
                text = tf.extractfile(member).read().decode("utf-8")
                entries = [safe_name(line.strip()) for line in text.splitlines() if line.strip()]
                manifests[parts[1]] = entries
                found.add(group)
                for entry in entries:
                    key = "artifacts/" + parts[1] + "/" + entry
                    roots[key] = {"group": group, "component": component, "family": family,
                                  "manifest_root": entry, "artifact_directory": parts[1]}
    if required - found:
        raise ValueError("Selected groups have no manifests: " + ", ".join(sorted(required - found)))
    return roots, {"manifests": manifests, "omitted_artifacts": sorted(omitted),
                   "absent_dependency_groups": sorted(groups - found)}


def consume_tar(tarpath, overlay, producer, roots, required=None):
    origin = {k: producer[k] for k in ("stage", "repository", "run_id", "artifact_id", "artifact_name", "artifact_digest")}
    origin["inner_archive"] = tarpath.name
    origin["inner_sha256"] = sha256(tarpath)
    ordered = sorted(roots, key=len, reverse=True)
    def mapped(name):
        name = safe_name(name)
        for root in ordered:
            if name.startswith(root + "/"):
                return name[len(root) + 1:], roots[root]
        return None
    pending = []
    seen = set()
    payload_counts = {}
    with tarfile.open(tarpath, "r|*") as tf:
        for member in tf:
            name = safe_name(member.name)
            mapping = mapped(name)
            if not mapping:
                continue
            if name in seen:
                raise ValueError("Duplicate TAR member: " + name)
            seen.add(name)
            rel, details = mapping
            full_origin = dict(origin, **details)
            if not member.isdir():
                key = details.get("group", "compiler-runtime")
                payload_counts[key] = payload_counts.get(key, 0) + 1
            if member.islnk():
                target = mapped(member.linkname)
                safe_name(member.linkname)
                pending.append((member, rel, full_origin, target[0] if target else None))
            else:
                overlay.install(tf, member, rel, full_origin)
    # GNU tar may hardlink files across artifact directories. Resolve bytes from
    # this same pinned archive without installing the unselected target group.
    if pending:
        with tarfile.open(tarpath, "r:*") as tf:
            for member, rel, full_origin, target in pending:
                existing = overlay.files.get(target) if target else None
                if existing and existing.get("inner_sha256") == origin["inner_sha256"]:
                    overlay.install(None, member, rel, full_origin, target)
                    continue
                source = tf.getmember(safe_name(member.linkname))
                visited = {member.name}
                while source.islnk():
                    if source.name in visited:
                        raise ValueError("Archive hardlink cycle")
                    visited.add(source.name)
                    source = tf.getmember(safe_name(source.linkname))
                if not source.isfile():
                    raise ValueError("Archive hardlink target is not a file")
                full_origin = dict(full_origin, archive_hardlink_target=member.linkname)
                overlay.install(tf, source, rel, full_origin)
    expected = {d.get("group", "compiler-runtime") for d in roots.values()}
    missing = (expected if required is None else required) - payload_counts.keys()
    if missing:
        raise ValueError("Selected artifact groups contain no payload: " + ", ".join(sorted(missing)))
    return dict(origin, payload_counts=payload_counts, empty_dependency_groups=sorted(expected - payload_counts.keys()))


def validation_bundle(archive, destination):
    if destination.exists():
        raise ValueError("Validation output already exists")
    destination.mkdir(parents=True)
    inventory = []
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for info in z.infolist():
            name = safe_name(info.filename)
            if name in seen:
                raise ValueError("Duplicate validation ZIP entry")
            seen.add(name)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Validation ZIP symlink unsupported")
            # Logs remain evidence; no file from this ZIP enters the SDK prefix.
            path = destination / name
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
                path.chmod((info.external_attr >> 16) & 0o777 or 0o644)
                inventory.append({"path": name, "sha256": sha256(path), "size": info.file_size})
    if not (destination / "dasan-lit-bundle/llvm-lit.py").is_file():
        raise ValueError("Compiler validation bundle is absent")
    write_json(destination.parent / "compiler-validation-inventory.json", inventory)


def run(args):
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    status = {"stage": args.stage, "state": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": sha256(args.manifest), "prefix": str(args.output_dir.resolve()), "consumed": []}
    overlay = None
    try:
        manifest = json.loads(args.manifest.read_text())
        producers = {}
        for p in manifest["producers"]:
            stage = p["stage"]
            if stage in producers or PINNED_RUNS.get(stage) != p["run_id"] or p["repository"] != "ROCm/TheRock":
                raise ValueError("Unselected or duplicate producer in manifest")
            digest_value(p["artifact_digest"])
            producers[stage] = p
        selected = producers[args.stage]
        compiler = producers["compiler-runtime"]
        prefix = args.output_dir.resolve()
        if args.output_dir.is_symlink() or (prefix.exists() and any(prefix.iterdir())):
            raise ValueError("Output must be a fresh empty directory")
        prefix.mkdir(parents=True, exist_ok=True)
        overlay = Overlay(prefix, evidence)
        write_json(evidence / "consume-status.json", status)
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        # Resolve every explicitly selected primary group to its pinned producer.
        # Non-primary closure groups may come only from a selected producer's
        # own artifact archive. Empty bootstrap placeholders are recorded.
        primary_owners = {}
        for producer in producers.values():
            for group in producer.get("primary_artifacts") or []:
                if group in primary_owners:
                    raise ValueError("Ambiguous authoritative primary group: " + group)
                primary_owners[group] = producer["stage"]
        desired = set(selected.get("primary_artifacts") or []) | set(args.dependency_group)
        if set(args.dependency_group) - primary_owners.keys():
            raise ValueError("Explicit dependency group has no selected primary producer")
        assignments = {}
        required_groups = set(desired)
        expanded = set()
        queue = [(g, args.stage) for g in sorted(desired)]
        while queue:
            group, requester = queue.pop(0)
            owner = primary_owners.get(group, requester)
            if group in assignments:
                continue
            assignments[group] = owner
            if owner in expanded:
                continue
            expanded.add(owner)
            producer = producers[owner]
            closure = set(producer["sources"].get("dependency_closure", []))
            # A producer's other independent primary groups are not implicit deps.
            closure -= set(producer.get("primary_artifacts") or [])
            for dependency in sorted(closure):
                queue.append((dependency, owner))
                if dependency in primary_owners:
                    required_groups.add(dependency)
        selections = [(compiler, None, None)]
        stages = sorted(set(assignments.values()) - {"compiler-runtime", args.stage})
        if args.stage != "compiler-runtime":
            stages.append(args.stage)
        for stage in stages:
            groups = {g for g, owner in assignments.items() if owner == stage}
            if groups:
                selections.append((producers[stage], groups, groups & required_groups))
        write_json(evidence / "dependency-selection.json", {
            "assignments": assignments, "required_groups": sorted(required_groups),
            "explicit_extra_groups": args.dependency_group,
            "policy": "authoritative primary producers; own closure fallback; compiler files protected",
        })
        for producer, groups, required in selections:
            print("Consuming pinned " + producer["stage"] + " run " + str(producer["run_id"]), flush=True)
            archive = artifact_zip(producer, args.cache_dir, evidence, token)
            filename = producer["installation_archive"] if groups is None else producer["component_artifacts_archive"]
            with tempfile.TemporaryDirectory(prefix="dasan-payload-", dir=args.cache_dir) as temporary:
                tarpath = Path(temporary) / filename
                unpack_inner(archive, filename, tarpath)
                if groups is None:
                    roots = {safe_name(producer["installation_root"]): {}}
                else:
                    roots, details = component_roots(tarpath, groups, required)
                    write_json(evidence / (producer["stage"] + "-selection.json"), dict(details, selected_groups=sorted(groups)))
                status["consumed"].append(consume_tar(tarpath, overlay, producer, roots, required))
        if args.include_validation_bundle:
            archive = artifact_zip(compiler, args.cache_dir, evidence, token, logs=True)
            validation_bundle(archive, evidence / "compiler-validation")
        missing = [pattern for pattern in args.require_path if not any(prefix.glob(pattern))]
        status["missing_required_paths"] = missing
        if missing:
            raise ValueError("Required test/install paths absent: " + ", ".join(missing))
        status["state"] = "ready"
        return 0
    except Exception as exc:
        status["state"] = "failed"
        # Never include signed URLs or auth headers in diagnostics.
        status["error"] = ("HTTP status " + str(exc.code)) if isinstance(exc, urllib.error.HTTPError) else str(exc).split("https://")[0]
        print("Artifact consumption failed: " + status["error"], file=sys.stderr)
        return 1
    finally:
        if overlay:
            overlay.save()
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(evidence / "consume-status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(PINNED_RUNS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--include-validation-bundle", action="store_true")
    parser.add_argument("--require-path", action="append", default=[])
    parser.add_argument("--dependency-group", action="append", default=[],
                        help="Additional runtime dependency primary group from a pinned producer (e.g. rccl)")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
