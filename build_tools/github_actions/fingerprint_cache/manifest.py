# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The cache *value*: an output manifest.

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
from enum import Enum
import json
from typing import Any

from .cache_key import ArtifactIdentity


class TrustLevel(str, Enum):
    """Where an entry came from, for read-policy purposes.

    Writes are expected to be limited to trusted events (push/schedule on a
    default/release branch); PR builds produce untrusted entries that a
    conservative consumer may decline to reuse.
    """

    TRUSTED = "trusted"  # push/schedule on a protected branch
    UNTRUSTED = "untrusted"  # pull_request and other unprivileged events

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
    """Publish ordering (flowchart 3): objects first, entry last."""

    PENDING = "pending"  # objects may still be uploading; do NOT serve
    PUBLISHED = "published"  # manifest + all objects complete; safe to serve
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
    """A complete cache entry: key -> outputs + metadata."""

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
        """All outputs present with a URI, and at least one output."""
        return bool(self.outputs) and all(o.uri for o in self.outputs)

    def is_servable(self) -> bool:
        """Safe to hand to a consumer as a cache hit."""
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
