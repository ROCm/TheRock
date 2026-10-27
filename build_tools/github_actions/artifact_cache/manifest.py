# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Python representations of artifact metadata; no database initialization.

LocalManifest, ArtifactKey and ArtifactFile represent the relational schema.
Manifest and its supporting types describe object references and producer
metadata for model checks. Their presence does not establish reusable bytes.

Per the design feedback (gist item 6), the cache value should be a manifest,
not just an S3 URL:

    versioned input key -> output manifest -> immutable artifact objects

The manifest describes output identities, digests, sizes, dependency
identities, producer provenance, and publication/validation state. That gives
us integrity checks, explainable hit/miss decisions, and a way to assemble a
build from objects produced by different runs without losing their origin.

Trust and validation state are first-class so the cache can distinguish
"built on a trusted push to main" from "built on an untrusted PR", and
"build passed" from "tests passed" -- the read policy (see ``cache_db``) then
decides which entries a given consumer is allowed to reuse.
"""


from dataclasses import dataclass, field
from .cache_key import current_cache_epoch, validate_cache_epoch
from enum import Enum
import json
from pathlib import PurePosixPath
import re
from typing import Any

from .cache_key import ArtifactIdentity


class TrustLevel(str, Enum):
    """Where an entry came from, for read-policy purposes.

    History classifies only push/schedule runs on main as trusted, subject to
    the repository check in history_ingest. Other branches/events are untrusted.
    Local publication defaults to untrusted; an explicit local trust label is
    metadata, not authentication. Read policy decides which labels to accept.
    """

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"

    def meets(self, minimum: "TrustLevel") -> bool:
        order = {TrustLevel.UNTRUSTED: 0, TrustLevel.TRUSTED: 1}
        return order[self] >= order[minimum]


class ValidationState(str, Enum):
    """How far the producing run validated the outputs.

    A cache hit should not by itself suppress testing (gist item 5); this state
    lets the read policy require, e.g., build-passed for reuse while test
    selection is decided independently.
    """

    UNVALIDATED = "unvalidated"  # objects exist, producing job state unknown
    BUILD_PASSED = "build_passed"  # producing build job succeeded
    TESTS_PASSED = "tests_passed"  # relevant tests also passed

    def meets(self, minimum: "ValidationState") -> bool:
        order = {
            ValidationState.UNVALIDATED: 0,
            ValidationState.BUILD_PASSED: 1,
            ValidationState.TESTS_PASSED: 2,
        }
        return order[self] >= order[minimum]


class PublicationState(str, Enum):
    """Publication state recorded by the object-reference metadata model."""

    PENDING = "pending"  # objects may still be uploading; do NOT serve
    PUBLISHED = "published"  # metadata marked published; bytes still need verification
    REVOKED = "revoked"  # withdrawn (bad build discovered, retention, ...)


@dataclass(frozen=True)
class OutputObject:
    """One immutable object that makes up an artifact unit."""

    key: str  # object/archive filename, e.g. "blas_lib_gfx94X-dcgpu.tar.zst"
    uri: str  # pointer into existing S3 storage (not duplicated)
    digest: str | None = None  # sha256 of the archive, when known
    size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "uri": self.uri,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OutputObject":
        return OutputObject(
            key=d["key"],
            uri=d["uri"],
            digest=d.get("digest"),
            size_bytes=d.get("size_bytes"),
        )


@dataclass(frozen=True)
class Provenance:
    """Where an entry was produced -- for explainable hits and debugging."""

    repository: str
    run_id: str
    branch: str
    commit: str
    workflow: str
    created_at: str  # ISO 8601
    html_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "run_id": self.run_id,
            "branch": self.branch,
            "commit": self.commit,
            "workflow": self.workflow,
            "created_at": self.created_at,
            "html_url": self.html_url,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Provenance":
        return Provenance(
            repository=d["repository"],
            run_id=d["run_id"],
            branch=d["branch"],
            commit=d["commit"],
            workflow=d["workflow"],
            created_at=d["created_at"],
            html_url=d.get("html_url", ""),
        )


@dataclass(frozen=True)
class Manifest:
    """Object-reference metadata for model checks; not a verified local entry."""

    cache_key: str  # CacheKey.value
    artifact: ArtifactIdentity
    outputs: tuple[OutputObject, ...]
    provenance: Provenance
    build_variant: str = "release"
    dep_identities: tuple[str, ...] = ()
    trust: TrustLevel = TrustLevel.UNTRUSTED
    validation: ValidationState = ValidationState.UNVALIDATED
    publication: PublicationState = PublicationState.PENDING

    def is_complete(self) -> bool:
        """At least one output reference, with a URI for every recorded output."""
        return bool(self.outputs) and all(o.uri for o in self.outputs)

    def is_servable(self) -> bool:
        """Published, complete metadata; physical contents remain unverified."""
        return self.publication is PublicationState.PUBLISHED and self.is_complete()

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "artifact": list(self.artifact.as_tuple()),
            "outputs": [o.to_dict() for o in self.outputs],
            "provenance": self.provenance.to_dict(),
            "build_variant": self.build_variant,
            "dep_identities": list(self.dep_identities),
            "trust": self.trust.value,
            "validation": self.validation.value,
            "publication": self.publication.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Manifest":
        name, component, target_family = d["artifact"]
        return Manifest(
            cache_key=d["cache_key"],
            artifact=ArtifactIdentity(name, component, target_family),
            outputs=tuple(OutputObject.from_dict(o) for o in d["outputs"]),
            provenance=Provenance.from_dict(d["provenance"]),
            build_variant=d.get("build_variant", "release"),
            dep_identities=tuple(d.get("dep_identities", ())),
            trust=TrustLevel(d.get("trust", "untrusted")),
            validation=ValidationState(d.get("validation", "unvalidated")),
            publication=PublicationState(d.get("publication", "pending")),
        )

    @staticmethod
    def from_json(text: str) -> "Manifest":
        return Manifest.from_dict(json.loads(text))


# Full output identities are the relational storage contract. Policy metadata is
# recorded explicitly in producer parameters, never inferred from an identity.
@dataclass(frozen=True)
class ArtifactKey:
    """The exact lookup identity used by artifacts + artifact_keys."""

    artifact_identity: str
    fingerprint: str
    epoch: int = field(default_factory=current_cache_epoch, kw_only=True)

    # Empty for component keys. Sorted complete output identities for a bundle.
    components: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        validate_cache_epoch(self.epoch)
        if not isinstance(self.components, tuple):
            raise ValueError("components must be a tuple")
        for name in ("artifact_identity",):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be nonempty text without edge whitespace")
            if any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ValueError(f"{name} must not contain control characters")
            value.encode("utf-8")
        if self.components:
            from .bundles import validate_bundle
            validate_bundle(self.artifact_identity, self.components)
        elif "/bundle/" in self.artifact_identity:
            raise ValueError("bundle keys require an explicit component list")
        if not isinstance(self.fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.fingerprint
        ):
            raise ValueError("fingerprint must be a lowercase SHA-256 hexadecimal value")


@dataclass(frozen=True)
class ArtifactFile:
    """One expected local output; its digest identifies bytes, not build inputs."""

    relative_path: str
    content_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        path = self.relative_path
        if (
            not isinstance(path, str)
            or PurePosixPath(path).is_absolute()
            or any(part in ("", ".", "..") for part in path.split("/"))
            or "\\" in path
            or ":" in path
            or any(ord(c) < 32 or ord(c) == 127 for c in path)
        ):
            raise ValueError("artifact file path must be a safe relative POSIX path")
        if not isinstance(self.content_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.content_sha256
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 value")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")


@dataclass(frozen=True)
class LocalManifest:
    """A ready database candidate reconstructed from relational rows.

    This describes metadata only. A consumer must check physical file
    containment, completeness and bytes before importing any output.
    CI trust, remote URIs and workflow provenance are not invented here.
    """

    cache_entry_id: int
    artifact_key_id: int
    key: ArtifactKey
    producer_run_id: int
    producer_commit: str
    producer_label: str
    local_directory: str
    expected_files: tuple[str, ...]
    files: tuple[ArtifactFile, ...]
    validation_state: str
    validated_at: int
    created_at: int
    expires_at: int | None
    producer_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.producer_metadata, dict):
            raise ValueError("producer metadata must be an object")
        expected = self.expected_files
        actual = tuple(f.relative_path for f in self.files)
        if (
            not expected
            or len(set(expected)) != len(expected)
            or len(set(actual)) != len(actual)
            or set(expected) != set(actual)
        ):
            raise ValueError("manifest must contain every expected file exactly once")
        if self.validation_state not in ("build_passed", "tests_passed"):
            raise ValueError("manifest must refer to build-passed output metadata")
        if self.validated_at is None:
            raise ValueError("manifest must have a validation timestamp")

    @property
    def payloads_verified(self) -> bool:
        """Loading database rows does not verify stored bytes."""
        return False

    @property
    def trust(self) -> TrustLevel:
        metadata = self.producer_metadata.get("cache_provenance", {})
        if isinstance(metadata, dict) and metadata.get("trust") == "trusted":
            return TrustLevel.TRUSTED
        return TrustLevel.UNTRUSTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_entry_id": self.cache_entry_id,
            "artifact_key_id": self.artifact_key_id,
            "epoch": self.key.epoch,
            "artifact_identity": self.key.artifact_identity,
            "fingerprint": self.key.fingerprint,
            "components": list(self.key.components),
            "producer_run_id": self.producer_run_id,
            "producer_commit": self.producer_commit,
            "producer_label": self.producer_label,
            "local_directory": self.local_directory,
            "expected_files": list(self.expected_files),
            "files": [
                {"relative_path": f.relative_path,
                 "content_sha256": f.content_sha256, "size_bytes": f.size_bytes}
                for f in self.files
            ],
            "validation_state": self.validation_state,
            "validated_at": self.validated_at,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "trust": self.trust.value,
            "producer_metadata": self.producer_metadata,
            "lookup_scope": "database_metadata",
            "payloads_verified": False,
        }
