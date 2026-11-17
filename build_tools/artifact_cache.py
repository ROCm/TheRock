#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Plan reuse of the complete amd-llvm artifact using a static cache index.

The record command downloads a CI run's fingerprints and archive checksums,
or reads local metadata. The lookup command compares existing configure-time
fingerprints with the index and writes a build or populate plan.

Examples (run from the repository root):

    Add a CI run's artifact metadata to the index:
        python build_tools/artifact_cache.py record \
            --run-id 43738549005 \
            --platform linux --build-variant release

    Look for artifacts matching the current build's fingerprints:
        python build_tools/artifact_cache.py lookup \
            --fingerprints-dir build/artifacts \
            --platform linux --build-variant release \
            --output build/llvm-cache-plan.json

Both commands use build_tools/artifact_cache_index.json by default.
Use --index to select another index file.

A lookup hit is a candidate for population; that step must verify availability
and checksums before reuse. These commands do not download artifact archives.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


SCHEMA_VERSION = 1
# Keep in sync with therock_provide_artifact(amd-llvm) in compiler/CMakeLists.txt.
# This artifact includes the amd-llvm, amd-comgr, and hipcc subproject stages.
LLVM_COMPONENTS = ("dbg", "dev", "doc", "lib", "run")
LLVM_ARTIFACTS = tuple(f"amd-llvm_{component}_generic" for component in LLVM_COMPONENTS)
ARCHIVE_SUFFIXES = (".tar.zst", ".tar.xz")
DEFAULT_INDEX = Path(__file__).with_name("artifact_cache_index.json")
METADATA_DOWNLOAD_TIMEOUT_SECONDS = 30
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CacheKey(NamedTuple):
    platform: str
    build_variant: str
    artifact: str
    fingerprint: str


@dataclass
class Producer:
    repository: str
    run_id: str


@dataclass
class ArtifactSource:
    root: str
    archive: str
    sha256: str


@dataclass
class CacheEntry:
    platform: str
    build_variant: str
    artifact: str
    fingerprint: str
    producer: Producer
    source: ArtifactSource


@dataclass
class CacheIndex:
    schema_version: int
    entries: list[CacheEntry]


@dataclass
class ArtifactMatch:
    artifact: str
    expected_fingerprint: str | None
    status: str
    reason: str
    producer: Producer | None = None
    source: ArtifactSource | None = None


@dataclass
class CopyRequest:
    artifact: str
    fingerprint: str
    producer: Producer
    source: ArtifactSource
    fingerprint_file: str
    checksum_file: str


@dataclass
class LookupPlan:
    schema_version: int
    artifact: str
    platform: str
    build_variant: str
    status: str
    action: str
    reason: str
    artifacts: list[ArtifactMatch]
    copy_requests: list[CopyRequest]


def require(condition: bool, message: str) -> None:
    """Raise a clear error when a required condition is not met."""
    if not condition:
        raise ValueError(message)


def text_field(value: object, name: str) -> str:
    """Check that a field is nonempty text without surrounding whitespace."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    require(value == value.strip(), f"{name} must not have surrounding whitespace")
    return value


def digest(value: object, name: str) -> str:
    """Check that a value has the format of a lowercase SHA-256 hash."""
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def source_root(location: object) -> str:
    """Check the artifact source location and remove its trailing slash."""
    location = text_field(location, "source.root")
    parsed_location = urlsplit(location)
    require(
        parsed_location.scheme in ("s3", "https", "file"),
        "source.root must be an s3://, https://, or file:// location",
    )
    require(
        not parsed_location.query
        and not parsed_location.fragment
        and not parsed_location.username
        and not parsed_location.password,
        "source.root must not contain credentials, a query, or a fragment",
    )
    require(
        (
            bool(parsed_location.netloc)
            if parsed_location.scheme != "file"
            else parsed_location.path.startswith("/")
        ),
        "source.root must identify an absolute location",
    )
    return location.rstrip("/")


def entry_key(entry: CacheEntry) -> CacheKey:
    """Return the platform, variant, artifact, and fingerprint used for matching."""
    return CacheKey(
        platform=entry.platform,
        build_variant=entry.build_variant,
        artifact=entry.artifact,
        fingerprint=entry.fingerprint,
    )


def to_json(value: CacheIndex | LookupPlan) -> dict[str, object]:
    """Convert an index or plan to JSON data, omitting unavailable match details."""
    json_data = asdict(value)
    if isinstance(value, LookupPlan):
        for match in json_data["artifacts"]:
            for optional_field in ("producer", "source"):
                if match[optional_field] is None:
                    del match[optional_field]
    return json_data


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
            producer_data, source_data = entry_data["producer"], entry_data["source"]
            entry = CacheEntry(
                artifact=text_field(entry_data["artifact"], "artifact"),
                fingerprint=digest(entry_data["fingerprint"], "fingerprint"),
                platform=text_field(entry_data["platform"], "platform"),
                build_variant=text_field(entry_data["build_variant"], "build_variant"),
                producer=Producer(
                    repository=text_field(
                        producer_data["repository"], "producer.repository"
                    ),
                    run_id=text_field(producer_data["run_id"], "producer.run_id"),
                ),
                source=ArtifactSource(
                    root=text_field(source_data["root"], "source.root"),
                    archive=text_field(source_data["archive"], "source.archive"),
                    sha256=digest(source_data["sha256"], "source.sha256"),
                ),
            )
            require(entry.artifact in LLVM_ARTIFACTS, "Unsupported LLVM artifact")
            require(
                entry.source.archive
                in tuple(entry.artifact + suffix for suffix in ARCHIVE_SUFFIXES),
                "Archive name must match its artifact and a supported extension",
            )
            source_root(entry.source.root)
            entries.append(entry)
        require(
            len({entry_key(entry) for entry in entries}) == len(entries),
            "Index contains duplicate cache keys",
        )
        return CacheIndex(schema_version=SCHEMA_VERSION, entries=entries)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid index structure in {path}: {error}") from error


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


def lookup_artifacts_by_fingerprint(
    index: CacheIndex, fingerprints_dir: Path, platform: str, build_variant: str
) -> LookupPlan:
    """Compare the current build's fingerprints with the static cache index.

    Read each required LLVM component's .fprint file from fingerprints_dir.
    Match entries by artifact name, fingerprint, platform, and build variant.
    Return a populate plan with copy requests only when all components match
    and their fingerprints describe the same bundle. Otherwise, return a
    build plan with no copy requests.

    Missing or malformed fingerprints count as cache misses. Other file-read
    errors are raised to the caller. This function returns a plan without
    downloading or building artifacts.
    """
    text_field(platform, "platform")
    text_field(build_variant, "build_variant")
    entries_by_key = {entry_key(entry): entry for entry in index.entries}
    matches: list[ArtifactMatch] = []
    copy_requests: list[CopyRequest] = []
    for artifact in LLVM_ARTIFACTS:
        fingerprint_path = fingerprints_dir / f"{artifact}.fprint"
        match = ArtifactMatch(
            artifact=artifact,
            expected_fingerprint=None,
            status="miss",
            reason="missing_expected_fingerprint",
        )
        try:
            expected_fingerprint = digest(
                fingerprint_path.read_text(encoding="utf-8").strip(),
                fingerprint_path.name,
            )
        except FileNotFoundError:
            pass  # The default result is already a missing-fingerprint miss.
        except (UnicodeError, ValueError):
            match.reason = "invalid_expected_fingerprint"
        else:
            match.expected_fingerprint = expected_fingerprint
            entry = entries_by_key.get(
                CacheKey(
                    platform=platform,
                    build_variant=build_variant,
                    artifact=artifact,
                    fingerprint=expected_fingerprint,
                )
            )
            if entry is None:
                match.reason = "no_matching_index_entry"
            else:
                match.status, match.reason = "hit", "matching_index_entry"
                match.producer, match.source = entry.producer, entry.source
                copy_requests.append(
                    CopyRequest(
                        artifact=artifact,
                        fingerprint=expected_fingerprint,
                        producer=entry.producer,
                        source=entry.source,
                        fingerprint_file=artifact + ".fprint",
                        checksum_file=entry.source.archive + ".sha256sum",
                    )
                )
        matches.append(match)
    # CMake currently gives every component of one artifact the same aggregate
    # fingerprint. Inconsistent files may be left over from different configs.
    bundle_fingerprints = {
        match.expected_fingerprint
        for match in matches
        if match.expected_fingerprint is not None
    }
    if len(bundle_fingerprints) > 1:
        for match in matches:
            match.status = "miss"
            match.reason = "inconsistent_expected_bundle_fingerprints"
            match.producer = match.source = None
    complete_hit = all(match.status == "hit" for match in matches)
    return LookupPlan(
        schema_version=SCHEMA_VERSION,
        artifact="amd-llvm",
        platform=platform,
        build_variant=build_variant,
        status="hit" if complete_hit else "miss",
        action="populate" if complete_hit else "build",
        reason="complete_index_match" if complete_hit else "incomplete_index_match",
        artifacts=matches,
        # Partial hits are diagnostic only, never a partial stage replacement.
        copy_requests=copy_requests if complete_hit else [],
    )


def download_metadata(root: str, metadata_dir: Path, archive_suffix: str) -> None:
    """Download a producer's fingerprint and checksum files for recording.

    Fetch the .fprint and archive checksum file for every required LLVM
    component from root, using HTTPS or a public S3 location. Save the files
    in the existing metadata_dir. The archive_suffix selects the checksum
    filenames for .tar.zst or .tar.xz archives.

    Artifact archives are not downloaded. A failed download or local file
    write raises an error to the caller.
    """
    root = source_root(root)
    require(archive_suffix in ARCHIVE_SUFFIXES, "Unsupported archive suffix")
    parsed_location = urlsplit(root)
    if parsed_location.scheme == "s3":
        root = (
            f"https://{parsed_location.netloc}.s3.amazonaws.com{parsed_location.path}"
        )
    require(
        root.startswith("https://"),
        "Downloads require an HTTPS or public S3 source root",
    )
    print(f"Downloading metadata from {root}")
    for artifact in LLVM_ARTIFACTS:
        for suffix in (".fprint", archive_suffix + ".sha256sum"):
            metadata_filename = artifact + suffix
            download_url = f"{root}/{metadata_filename}"
            try:
                with urlopen(
                    download_url, timeout=METADATA_DOWNLOAD_TIMEOUT_SECONDS
                ) as response:
                    (metadata_dir / metadata_filename).write_bytes(response.read())
            except URLError as error:
                raise ValueError(f"Cannot download {download_url}: {error}") from error


def record_artifacts_from_ci_run(
    index: CacheIndex,
    metadata_dir: Path,
    platform: str,
    build_variant: str,
    repository: str,
    run_id: str,
    root: str,
    archive_suffix: str,
) -> CacheIndex:
    """Add LLVM artifact metadata from a previous CI run to the static cache index.

    Read fingerprint and checksum files from metadata_dir for every required
    component. Check that the files are valid and the component fingerprints
    describe the same LLVM bundle. Add missing entries with their platform,
    build variant, producer repository/run ID, archive location, and checksum.
    Keep existing entries when the artifact, fingerprint, platform, and build
    variant already match.

    Return an updated index for the caller to save, leaving the supplied index
    unchanged. Missing or invalid metadata raises an error.
    """
    text_field(platform, "platform")
    text_field(build_variant, "build_variant")
    require(
        re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text_field(repository, "repository")
        )
        is not None,
        "repository must be owner/repository",
    )
    require(
        re.fullmatch(r"[0-9]+", text_field(run_id, "run_id")) is not None,
        "run_id must be numeric",
    )
    root = source_root(root)
    require(archive_suffix in ARCHIVE_SUFFIXES, "Unsupported archive suffix")
    producer_entries: list[CacheEntry] = []
    for artifact in LLVM_ARTIFACTS:
        fingerprint_path = metadata_dir / f"{artifact}.fprint"
        fingerprint = digest(
            fingerprint_path.read_text(encoding="utf-8").strip(), fingerprint_path.name
        )
        archive = artifact + archive_suffix
        checksum_path = metadata_dir / f"{archive}.sha256sum"
        checksum_fields = checksum_path.read_text(encoding="utf-8").strip().split()
        require(
            len(checksum_fields) in (1, 2), f"Invalid checksum file: {checksum_path}"
        )
        archive_checksum = digest(checksum_fields[0], checksum_path.name)
        if len(checksum_fields) == 2:
            require(
                checksum_fields[1].removeprefix("*") == archive,
                f"Checksum filename mismatch: {checksum_path}",
            )
        producer_entries.append(
            CacheEntry(
                artifact=artifact,
                fingerprint=fingerprint,
                platform=platform,
                build_variant=build_variant,
                producer=Producer(repository=repository, run_id=run_id),
                source=ArtifactSource(
                    root=root, archive=archive, sha256=archive_checksum
                ),
            )
        )
    require(
        len({entry.fingerprint for entry in producer_entries}) == 1,
        "Producer component fingerprints must describe one LLVM bundle",
    )
    existing_keys = {entry_key(entry) for entry in index.entries}
    new_entries = [
        entry for entry in producer_entries if entry_key(entry) not in existing_keys
    ]
    return CacheIndex(
        schema_version=SCHEMA_VERSION,
        entries=[*index.entries, *new_entries],
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and read command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    lookup_parser = subparsers.add_parser(
        "lookup", help="Write a complete LLVM hit/miss plan"
    )
    lookup_parser.add_argument("--fingerprints-dir", type=Path, required=True)
    lookup_parser.add_argument("--output", type=Path, required=True)
    record_parser = subparsers.add_parser(
        "record", help="Import producer fingerprints and checksums"
    )
    record_parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Read existing local sidecars instead of downloading them",
    )
    record_parser.add_argument("--repository", default="ROCm/TheRock")
    record_parser.add_argument("--run-id", required=True)
    record_parser.add_argument(
        "--source-root",
        help="Producer artifact root (default: standard TheRock CI location)",
    )
    record_parser.add_argument(
        "--archive-suffix", choices=ARCHIVE_SUFFIXES, default=".tar.zst"
    )
    for command_parser in (lookup_parser, record_parser):
        command_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
        command_parser.add_argument("--platform", required=True)
        command_parser.add_argument(
            "--build-variant",
            required=True,
            help="Explicit CI compatibility label, shared by producer and consumer",
        )
    return parser.parse_args(argv)


def run_record(args: argparse.Namespace) -> None:
    """Import producer metadata and save newly indexed components."""
    index = load_index(args.index)
    require(
        re.fullmatch(r"[0-9]+", args.run_id) is not None,
        "run-id must be numeric",
    )
    source_location = args.source_root
    if source_location is None:
        require(
            args.repository == "ROCm/TheRock" and args.platform in ("linux", "windows"),
            "Specify --source-root for other repositories or platforms",
        )
        source_location = f"https://therock-ci-artifacts.s3.amazonaws.com/{args.run_id}-{args.platform}"
    with tempfile.TemporaryDirectory(
        prefix="therock-cache-metadata-"
    ) as temporary_metadata_dir:
        metadata_dir = (
            args.metadata_dir
            if args.metadata_dir is not None
            else Path(temporary_metadata_dir)
        )
        if args.metadata_dir is None:
            download_metadata(
                root=source_location,
                metadata_dir=metadata_dir,
                archive_suffix=args.archive_suffix,
            )
        updated_index = record_artifacts_from_ci_run(
            index=index,
            metadata_dir=metadata_dir,
            platform=args.platform,
            build_variant=args.build_variant,
            repository=args.repository,
            run_id=args.run_id,
            root=source_location,
            archive_suffix=args.archive_suffix,
        )
    added_count = len(updated_index.entries) - len(index.entries)
    already_indexed_count = len(LLVM_ARTIFACTS) - added_count
    if added_count:
        write_json(args.index, updated_index)
        print(f"Recorded {added_count} LLVM components from run {args.run_id}")
        if already_indexed_count:
            print(
                f"Already indexed: {already_indexed_count} components; "
                "keeping existing entries."
            )
    else:
        print(
            f"Already indexed: all {already_indexed_count} LLVM components "
            "have matching fingerprints. Index unchanged."
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
        build_variant=args.build_variant,
    )
    write_json(args.output, plan)
    hit_count = sum(match.status == "hit" for match in plan.artifacts)
    print(
        f"amd-llvm: {plan.status.upper()} ({hit_count}/{len(LLVM_ARTIFACTS)} components)"
    )
    for match in plan.artifacts:
        print(f"  {match.artifact}: {match.status.upper()} ({match.reason})")
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
    except (OSError, UnicodeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
