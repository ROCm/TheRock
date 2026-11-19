#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Plan reuse of prebuilt artifacts using a static, prototype cache index.

This is a prototype of the lookup step of the fingerprint-based artifact cache
(https://github.com/ROCm/TheRock/issues/8651). It does not download or restore
artifact archives.

The record subcommand reads a CI run's artifact fingerprints (.fprint) and
archive checksums (.sha256sum) and saves them, with the archive locations, in a
static JSON index. The lookup subcommand compares configure-time fingerprints
with that index and writes a build or populate plan.

Artifacts are selected with include/exclude regular expressions in the same way
as fetch_artifacts.py. Patterns are searched in "<name>_<component>_<target>"
(for example "amd-llvm_lib_generic"). With no include pattern only "amd-llvm"
artifacts are used; pass "." to select every artifact.

Examples (run from the repository root):

    Add a CI run's amd-llvm artifacts to the index:
        python build_tools/artifact_cache/artifact_cache.py record \
            --run-id 43936062480 \
            --platform linux

    Add everything except debug components:
        python build_tools/artifact_cache/artifact_cache.py record \
            --run-id 43936062480 \
            --platform linux \
            . --exclude _dbg_

    Look for artifacts matching the current build's fingerprints:
        python build_tools/artifact_cache/artifact_cache.py lookup \
            --fingerprints-dir build/artifacts \
            --platform linux \
            --output build/artifact-cache-plan.json

Both commands use build_tools/artifact_cache/prototype_artifact_cache_index.json
by default. Use --index to select another file.

Artifacts are read through the artifact backend in _therock_utils, so set
THEROCK_LOCAL_STAGING_DIR to record from a local directory instead of S3.

A lookup hit is a candidate for population; that step must verify availability
and checksums before reuse.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import os
import platform as platform_module
import re
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

# Add build_tools to the path for sibling package imports.
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent.parent))

from _therock_utils.artifact_backend import (
    ARTIFACT_EXTENSIONS,
    ArtifactBackend,
    LocalDirectoryBackend,
    S3Backend,
)
from _therock_utils.artifacts import ArtifactName
from _therock_utils.workflow_outputs import WorkflowOutputRoot
from fetch_artifacts import filter_artifacts

SCHEMA_VERSION = 1
PLAN_VERSION = 1
INDEX_STATUS = "prototype"
DEFAULT_INCLUDES = ("amd-llvm",)
DEFAULT_REPOSITORY = "ROCm/TheRock"
DEFAULT_INDEX = Path(__file__).with_name("prototype_artifact_cache_index.json")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
NAME_PART = re.compile(r"[^_\s/]+\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
RUN_ID = re.compile(r"[0-9]+\Z")


class CacheError(ValueError):
    """An expected failure: invalid input, index, or CI run metadata."""


class ArtifactId(NamedTuple):
    """Identity of one artifact: matches ArtifactName in _therock_utils."""

    name: str
    component: str
    target: str

    @property
    def stem(self) -> str:
        return f"{self.name}_{self.component}_{self.target}"


class GroupKey(NamedTuple):
    """The components of one artifact on one platform share one fingerprint."""

    platform: str
    name: str
    target: str


class CacheKey(NamedTuple):
    platform: str
    artifact: ArtifactId
    fingerprint: str


@dataclass
class Provenance:
    repository: str
    run_id: str


@dataclass
class Output:
    key: str  # Archive filename, e.g. amd-llvm_lib_generic.tar.zst
    uri: str  # Where to get the archive.
    digest: str  # SHA-256 of the archive, as lowercase hex.


@dataclass
class CacheEntry:
    platform: str
    name: str
    component: str
    target: str
    fingerprint: str
    provenance: Provenance
    outputs: list[Output]
    # THEROCK_FPRINT_EPOCH the producer used. The epoch is already part of the
    # fingerprint; recording it makes a bump easy to see and to check.
    epoch: str | None = None

    @property
    def artifact(self) -> ArtifactId:
        return ArtifactId(name=self.name, component=self.component, target=self.target)

    @property
    def group(self) -> GroupKey:
        return GroupKey(platform=self.platform, name=self.name, target=self.target)

    @property
    def key(self) -> CacheKey:
        return CacheKey(
            platform=self.platform, artifact=self.artifact, fingerprint=self.fingerprint
        )


@dataclass
class CacheIndex:
    schema_version: int
    status: str
    entries: list[CacheEntry]


@dataclass
class ArtifactMatch:
    name: str
    component: str
    target: str
    status: str
    reason: str
    fingerprint: str | None = None
    provenance: Provenance | None = None
    outputs: list[Output] | None = None


@dataclass
class LookupPlan:
    plan_version: int
    platform: str
    status: str
    action: str
    reason: str
    includes: list[str]
    excludes: list[str]
    artifacts: list[ArtifactMatch]


@dataclass
class SkippedArtifact:
    artifact: str
    reason: str


@dataclass
class CollectedRun:
    entries: list[CacheEntry]
    skipped: list[SkippedArtifact]


@dataclass
class RecordSummary:
    added: int = 0
    replaced: int = 0
    unchanged: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.replaced)


def require(condition: bool, message: str) -> None:
    """Raise a clear error when a required condition is not met."""
    if not condition:
        raise CacheError(message)


def text_field(value: object, name: str) -> str:
    """Check that a field is nonempty text without surrounding whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise CacheError(f"{name} must be a nonempty string")
    require(value == value.strip(), f"{name} must not have surrounding whitespace")
    return value


def name_part(value: object, name: str) -> str:
    """Check a name, component, or target: it must not contain underscores."""
    text = text_field(value, name)
    require(NAME_PART.match(text) is not None, f"{name} is not a valid artifact part")
    return text


def check_run_id(run_id: str) -> None:
    """Check that a CI run ID is numeric."""
    require(RUN_ID.match(run_id) is not None, "run-id must be numeric")


def digest(value: object, name: str) -> str:
    """Check that a value has the format of a lowercase SHA-256 hash."""
    if not isinstance(value, str) or SHA256.match(value) is None:
        raise CacheError(f"{name} must be a lowercase SHA-256 digest")
    return value


def artifact_id_from_name(parsed: ArtifactName) -> ArtifactId:
    """Convert the repo's ArtifactName, whose target is called target_family."""
    return ArtifactId(
        name=parsed.name, component=parsed.component, target=parsed.target_family
    )


def archive_names(artifact: ArtifactId) -> tuple[str, ...]:
    return tuple(artifact.stem + suffix for suffix in ARTIFACT_EXTENSIONS)


def check_output(output: Output, artifact: ArtifactId) -> None:
    """Check that an output is a supported archive with an absolute location."""
    require(output.key in archive_names(artifact), "Output key must match its artifact")
    uri = urlsplit(output.uri)
    require(
        uri.scheme in ("s3", "https", "file"),
        "output uri must be an s3://, https://, or file:// location",
    )
    require(
        not uri.query and not uri.fragment and not uri.username and not uri.password,
        "output uri must not contain credentials, a query, or a fragment",
    )
    require(
        (
            bool(uri.netloc)
            if uri.scheme != "file"
            else uri.netloc in ("", "localhost") and uri.path.startswith("/")
        ),
        "output uri must identify an absolute location",
    )
    require(
        uri.path.rsplit("/", 1)[-1] == output.key,
        "output uri must end with the output key",
    )


def to_json(value: CacheIndex | LookupPlan) -> dict[str, object]:
    """Convert an index or plan to JSON data, omitting unset optional fields."""

    def drop_none(item: object) -> object:
        if isinstance(item, dict):
            return {k: drop_none(v) for k, v in item.items() if v is not None}
        if isinstance(item, list):
            return [drop_none(v) for v in item]
        return item

    return drop_none(asdict(value))


def load_index(path: Path) -> CacheIndex:
    """Read the index and check its fields while constructing typed entries."""
    index_data = json.loads(path.read_text(encoding="utf-8"))
    try:
        require(
            type(index_data["schema_version"]) is int
            and index_data["schema_version"] == SCHEMA_VERSION,
            "Unsupported index schema_version",
        )
        require(isinstance(index_data["entries"], list), "Index entries must be a list")
        entries: list[CacheEntry] = []
        for entry_data in index_data["entries"]:
            provenance_data = entry_data["provenance"]
            epoch = entry_data.get("epoch")
            entry = CacheEntry(
                platform=text_field(entry_data["platform"], "platform"),
                name=name_part(entry_data["name"], "name"),
                component=name_part(entry_data["component"], "component"),
                target=name_part(entry_data["target"], "target"),
                fingerprint=digest(entry_data["fingerprint"], "fingerprint"),
                epoch=None if epoch is None else text_field(epoch, "epoch"),
                provenance=Provenance(
                    repository=text_field(
                        provenance_data["repository"], "provenance.repository"
                    ),
                    run_id=text_field(provenance_data["run_id"], "provenance.run_id"),
                ),
                outputs=[
                    Output(
                        key=text_field(output["key"], "output.key"),
                        uri=text_field(output["uri"], "output.uri"),
                        digest=digest(output["digest"], "output.digest"),
                    )
                    for output in entry_data["outputs"]
                ],
            )
            require(entry.outputs, "An entry needs at least one output")
            for output in entry.outputs:
                check_output(output, entry.artifact)
            entries.append(entry)
        require(
            len({entry.key for entry in entries}) == len(entries),
            "Index contains duplicate cache keys",
        )
        return CacheIndex(
            schema_version=SCHEMA_VERSION,
            status=text_field(index_data.get("status", INDEX_STATUS), "status"),
            entries=entries,
        )
    except (KeyError, TypeError) as error:
        raise CacheError(f"Invalid index structure in {path}: {error}") from error


def write_json(path: Path, value: CacheIndex | LookupPlan) -> None:
    """Replace a report/index only after its complete content has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(to_json(value), output_file, indent=2)
            output_file.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def select_artifacts(
    artifacts: set[ArtifactId], includes: list[str], excludes: list[str]
) -> list[ArtifactId]:
    """Apply include/exclude patterns, as fetch_artifacts.py does, to artifacts."""
    by_stem = {artifact.stem: artifact for artifact in artifacts}
    selected = filter_artifacts(set(by_stem), includes, excludes)
    return sorted(by_stem[stem] for stem in selected)


def read_local_fingerprints(fingerprints_dir: Path) -> dict[ArtifactId, str | None]:
    """Read *.fprint files. A value of None marks an unreadable fingerprint."""
    fingerprints: dict[ArtifactId, str | None] = {}
    for fingerprint_path in sorted(fingerprints_dir.glob("*.fprint")):
        # ArtifactName understands archive names, so give it one.
        parsed = ArtifactName.from_filename(
            fingerprint_path.stem + ARTIFACT_EXTENSIONS[0]
        )
        if parsed is None:
            continue
        artifact = artifact_id_from_name(parsed)
        try:
            fingerprints[artifact] = digest(
                fingerprint_path.read_text(encoding="utf-8").strip(),
                fingerprint_path.name,
            )
        except (UnicodeError, CacheError):
            fingerprints[artifact] = None
    return fingerprints


def lookup_artifacts_by_fingerprint(
    index: CacheIndex,
    fingerprints_dir: Path,
    platform: str,
    includes: list[str],
    excludes: list[str],
    epoch: str | None = None,
) -> LookupPlan:
    """Compare the current build's fingerprints with the static cache index.

    Only artifacts the build produces are checked. Those are the selected
    .fprint files in fingerprints_dir and, for each (name, target) group with
    such a file, the selected index entries for the same group on the platform.
    An indexed component with no local fingerprint is a miss, but a group with
    no local fingerprint files at all is not part of the plan. Match each
    artifact by name, component, target, fingerprint, and platform.

    All components of one group must match together: the group shares one
    fingerprint, and a partly cached group is never reused. So every artifact in
    a group with a miss, or with differing fingerprints, is a miss. A miss for
    one group does not change the result of another. Artifacts that have no
    .fprint file, such as split per-target artifacts, never appear in the plan.

    Missing or malformed fingerprints count as cache misses. Other file-read
    errors are raised to the caller. This function returns a plan without
    downloading or building artifacts.
    """
    text_field(platform, "platform")
    local = read_local_fingerprints(fingerprints_dir)
    entries_by_key = {entry.key: entry for entry in index.entries}
    selected_local = select_artifacts(set(local), includes, excludes)
    local_groups = {(artifact.name, artifact.target) for artifact in selected_local}
    indexed = {
        entry.artifact
        for entry in index.entries
        if entry.platform == platform and (entry.name, entry.target) in local_groups
    }
    matches: list[ArtifactMatch] = []
    for artifact in select_artifacts({*selected_local, *indexed}, includes, excludes):
        match = ArtifactMatch(
            name=artifact.name,
            component=artifact.component,
            target=artifact.target,
            status="miss",
            reason="missing_expected_fingerprint",
        )
        matches.append(match)
        if artifact not in local:
            continue
        match.fingerprint = local[artifact]
        if match.fingerprint is None:
            match.reason = "invalid_expected_fingerprint"
            continue
        entry = entries_by_key.get(
            CacheKey(
                platform=platform, artifact=artifact, fingerprint=match.fingerprint
            )
        )
        if entry is None:
            match.reason = "no_matching_index_entry"
        elif epoch is not None and entry.epoch not in (None, epoch):
            match.reason = "epoch_mismatch"
        else:
            match.status, match.reason = "hit", "matching_index_entry"
            match.provenance, match.outputs = entry.provenance, entry.outputs

    # Every component of an artifact shares one aggregate fingerprint. Differing
    # fingerprints mean the files are left over from different configurations.
    groups: dict[tuple[str, str], list[ArtifactMatch]] = {}
    for match in matches:
        groups.setdefault((match.name, match.target), []).append(match)
    for group in groups.values():
        fingerprints = {m.fingerprint for m in group if m.fingerprint is not None}
        reason = None
        if len(fingerprints) > 1:
            reason = "inconsistent_group_fingerprints"
        elif any(m.status == "miss" for m in group):
            reason = "incomplete_group"
        if reason is not None:
            for match in group:
                if match.status == "hit":
                    match.status, match.reason = "miss", reason
                    match.provenance = match.outputs = None

    complete_hit = bool(matches) and all(m.status == "hit" for m in matches)
    if not local:
        reason = "no_local_fingerprints"
    elif not matches:
        reason = "no_artifacts_selected"
    else:
        reason = "complete_index_match" if complete_hit else "incomplete_index_match"
    return LookupPlan(
        plan_version=PLAN_VERSION,
        platform=platform,
        status="hit" if complete_hit else "miss",
        action="populate" if complete_hit else "build",
        reason=reason,
        includes=includes,
        excludes=excludes,
        artifacts=matches,
    )


def create_source_backend(
    run_id: str, platform: str, repository: str
) -> ArtifactBackend:
    """Create a backend for reading another CI run's artifacts.

    Like artifact_manager.py, this looks up the workflow run for S3 so that the
    bucket is right for forks and for runs from before a bucket change. With
    THEROCK_LOCAL_STAGING_DIR set it reads a local directory instead.
    """
    local_staging_dir = os.getenv("THEROCK_LOCAL_STAGING_DIR")
    if local_staging_dir:
        return LocalDirectoryBackend(
            staging_dir=Path(local_staging_dir),
            output_root=WorkflowOutputRoot.for_local(run_id=run_id, platform=platform),
        )
    return S3Backend(
        output_root=WorkflowOutputRoot.from_workflow_run(
            run_id=run_id,
            platform=platform,
            github_repository=repository,
            lookup_workflow_run=True,
        )
    )


def archive_uri(backend: ArtifactBackend, archive: str) -> str:
    """Return the location to record for an archive in a backend."""
    if isinstance(backend, S3Backend):
        return backend.output_root.artifact(archive).https_url
    if isinstance(backend, LocalDirectoryBackend):
        return (backend.base_path / archive).resolve().as_uri()
    raise CacheError(f"Cannot record locations for {type(backend).__name__}")


def read_checksum(path: Path, archive: str) -> str:
    """Read the SHA-256 from a .sha256sum file: "<hash>" or "<hash> [*]<name>"."""
    fields = path.read_text(encoding="utf-8").strip().split()
    require(bool(fields), f"Empty checksum file: {path.name}")
    checksum, *filenames = fields
    require(len(filenames) <= 1, f"Invalid checksum file: {path.name}")
    for filename in filenames:
        require(
            filename.removeprefix("*") == archive,
            f"Checksum filename mismatch: {path.name}",
        )
    return digest(checksum, path.name)


def collect_ci_run_entries(
    backend: ArtifactBackend,
    platform: str,
    repository: str,
    run_id: str,
    includes: list[str],
    excludes: list[str],
    epoch: str | None = None,
) -> CollectedRun:
    """Read the selected artifacts' fingerprints and checksums from a CI run.

    Only the small .fprint and .sha256sum files are downloaded, never the
    archives. An artifact without a fingerprint is skipped and reported, because
    some artifacts have none. A fingerprint without a checksum, or an invalid
    file, raises an error.
    """
    archives: dict[ArtifactId, str] = {}
    # Visit preferred archive suffixes first so they win if a run has several.
    for filename in sorted(
        backend.list_artifacts(),
        key=lambda f: (not f.endswith(ARTIFACT_EXTENSIONS[0]), f),
    ):
        parsed = ArtifactName.from_filename(filename)
        if parsed is not None:
            archives.setdefault(artifact_id_from_name(parsed), filename)
    entries: list[CacheEntry] = []
    skipped: list[SkippedArtifact] = []
    with tempfile.TemporaryDirectory(prefix="therock-cache-metadata-") as temp_dir:
        for artifact in select_artifacts(set(archives), includes, excludes):
            archive = archives[artifact]
            fingerprint_key = artifact.stem + ".fprint"
            checksum_key = archive + ".sha256sum"
            if not backend.artifact_exists(fingerprint_key):
                skipped.append(
                    SkippedArtifact(artifact=artifact.stem, reason="no_fingerprint")
                )
                continue
            require(
                backend.artifact_exists(checksum_key),
                f"{artifact.stem} has a fingerprint but no checksum file {checksum_key}",
            )
            paths = {}
            for key in (fingerprint_key, checksum_key):
                paths[key] = Path(temp_dir) / key
                backend.download_artifact(key, paths[key])
            entry = CacheEntry(
                platform=platform,
                name=artifact.name,
                component=artifact.component,
                target=artifact.target,
                fingerprint=digest(
                    paths[fingerprint_key].read_text(encoding="utf-8").strip(),
                    fingerprint_key,
                ),
                epoch=epoch,
                provenance=Provenance(repository=repository, run_id=run_id),
                outputs=[
                    Output(
                        key=archive,
                        uri=archive_uri(backend, archive),
                        digest=read_checksum(paths[checksum_key], archive),
                    )
                ],
            )
            check_output(entry.outputs[0], artifact)
            entries.append(entry)
    return CollectedRun(entries=entries, skipped=skipped)


def record_artifacts_from_ci_run(
    index: CacheIndex, new_entries: list[CacheEntry]
) -> tuple[CacheIndex, RecordSummary]:
    """Merge a CI run's entries into the index.

    An entry replaces the entry with the same platform, name, component, and
    target when its fingerprint differs. A matching fingerprint keeps the
    existing entry and its original provenance. Other entries are untouched.

    Components of one (platform, name, target) must share a fingerprint, in the
    new entries and in the merged result, so record all components of an
    artifact together. Return an updated index for the caller to save, leaving
    the supplied index unchanged.
    """
    for entry in new_entries:
        require(
            REPOSITORY.match(entry.provenance.repository) is not None,
            "repository must be owner/repository",
        )
        check_run_id(entry.provenance.run_id)
    summary = RecordSummary()
    kept = list(index.entries)
    to_add: list[CacheEntry] = []
    for entry in new_entries:
        same_artifact = [
            e
            for e in kept
            if (e.platform, e.artifact) == (entry.platform, entry.artifact)
        ]
        if any(e.fingerprint == entry.fingerprint for e in same_artifact):
            summary.unchanged += 1
            kept = [e for e in kept if e not in same_artifact or e.key == entry.key]
            continue
        kept = [e for e in kept if e not in same_artifact]
        to_add.append(entry)
        if same_artifact:
            summary.replaced += 1
        else:
            summary.added += 1
    merged = [*kept, *to_add]
    for group in {entry.group for entry in new_entries}:
        fingerprints = {e.fingerprint for e in merged if e.group == group}
        require(
            len(fingerprints) <= 1,
            f"Components of {group.name} ({group.target}) on {group.platform} "
            "would have different fingerprints; record all components together",
        )
    return (
        CacheIndex(schema_version=SCHEMA_VERSION, status=index.status, entries=merged),
        summary,
    )


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by record and lookup."""
    parser.add_argument(
        "include",
        nargs="*",
        help="Regular expressions of artifacts to include; one must match "
        f"(default: {' '.join(DEFAULT_INCLUDES)}; use '.' for all artifacts)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Regular expressions of artifacts to exclude",
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--platform",
        choices=["linux", "windows"],
        default=os.getenv("THEROCK_PLATFORM", platform_module.system().lower()),
        help="Platform of the artifacts (default: THEROCK_PLATFORM or this platform)",
    )
    parser.add_argument(
        "--epoch",
        help="THEROCK_FPRINT_EPOCH to record, or to require when it is recorded",
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and read command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    lookup_parser = subparsers.add_parser(
        "lookup", help="Write a hit/miss plan for the selected artifacts"
    )
    lookup_parser.add_argument("--fingerprints-dir", type=Path, required=True)
    lookup_parser.add_argument("--output", type=Path, required=True)
    add_selection_arguments(lookup_parser)
    record_parser = subparsers.add_parser(
        "record", help="Import a CI run's fingerprints and checksums"
    )
    record_parser.add_argument("--run-id", required=True)
    record_parser.add_argument(
        "--run-github-repo",
        type=str,
        default=None,
        help="GitHub repository for --run-id in 'owner/repo' format (e.g. "
        f"'{DEFAULT_REPOSITORY}'). Defaults to GITHUB_REPOSITORY env var or "
        f"'{DEFAULT_REPOSITORY}'",
    )
    add_selection_arguments(record_parser)
    return parser.parse_args(argv)


def run_record(args: argparse.Namespace) -> None:
    """Record the selected artifacts and report what changed."""
    check_run_id(args.run_id)
    index = (
        load_index(args.index)
        if args.index.exists()
        else CacheIndex(schema_version=SCHEMA_VERSION, status=INDEX_STATUS, entries=[])
    )
    repository = (
        args.run_github_repo or os.getenv("GITHUB_REPOSITORY") or DEFAULT_REPOSITORY
    )
    backend = create_source_backend(
        run_id=args.run_id, platform=args.platform, repository=repository
    )
    collected = collect_ci_run_entries(
        backend=backend,
        platform=args.platform,
        repository=repository,
        run_id=args.run_id,
        includes=args.include or list(DEFAULT_INCLUDES),
        excludes=args.exclude or [],
        epoch=args.epoch,
    )
    require(
        collected.entries, f"No artifacts with fingerprints found in run {args.run_id}"
    )
    updated_index, summary = record_artifacts_from_ci_run(
        index=index, new_entries=collected.entries
    )
    for skipped in collected.skipped:
        print(f"Skipped {skipped.artifact}: {skipped.reason}")
    if summary.changed:
        write_json(args.index, updated_index)
    print(
        f"Run {args.run_id} ({args.platform}): {summary.added} added, "
        f"{summary.replaced} replaced, {summary.unchanged} already indexed"
        + ("" if summary.changed else "; index unchanged")
    )


def run_lookup(args: argparse.Namespace) -> None:
    """Replace the reuse plan and report cache hits and misses."""
    require(
        args.output.resolve() != args.index.resolve(),
        "Plan output must not overwrite the index",
    )
    # Remove an old plan first: an unsuccessful lookup must not leave an
    # earlier hit in place for another step to consume.
    args.output.unlink(missing_ok=True)
    index = load_index(args.index)
    plan = lookup_artifacts_by_fingerprint(
        index=index,
        fingerprints_dir=args.fingerprints_dir,
        platform=args.platform,
        includes=args.include or list(DEFAULT_INCLUDES),
        excludes=args.exclude or [],
        epoch=args.epoch,
    )
    write_json(args.output, plan)
    hit_count = sum(match.status == "hit" for match in plan.artifacts)
    print(f"{plan.status.upper()} ({hit_count}/{len(plan.artifacts)} artifacts)")
    if not plan.artifacts:
        print(f"  No selected fingerprints in {args.fingerprints_dir}: {plan.reason}")
    for match in plan.artifacts:
        artifact = ArtifactId(
            name=match.name, component=match.component, target=match.target
        )
        print(f"  {artifact.stem}: {match.status.upper()} ({match.reason})")
    print(f"Action: {plan.action}. Plan: {args.output}")


def main(argv: list[str] | None = None) -> int:
    """Run the selected command and return its exit status, reporting errors."""
    args = parse_arguments(argv)
    try:
        if args.command == "record":
            run_record(args)
        else:
            run_lookup(args)
        return 0  # Both cache hits and ordinary misses are successful queries.
    except (CacheError, json.JSONDecodeError, OSError, UnicodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
