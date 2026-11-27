# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish freshly built outputs to the cache (flowchart 3).

Write ordering is the whole point of this module:

    built outputs + fingerprint + provenance
        -> outputs complete?                      (no  -> do NOT publish)
        -> producer & validation policy satisfied? (no  -> do NOT publish)
        -> upload immutable objects FIRST
        -> publish manifest + fingerprint entry LAST

Publishing the fingerprint entry last guarantees a reader never resolves a key
to a manifest whose objects are not yet fully uploaded. ``CacheDB.publish``
enforces the "complete + PUBLISHED" invariant; this function enforces the
policy gates in front of it and composes the cache key via the one seam in
``cache_key``.
"""


from typing import Callable, Sequence

from .cache_key import ArtifactIdentity, compute_cache_key
from .cache_db import CacheDB
from .manifest import (
    Manifest,
    OutputObject,
    Provenance,
    PublicationState,
    TrustLevel,
    ValidationState,
)

# A producer policy decides whether an entry is allowed to be published at all,
# given its trust and validation state. Returns True to publish.
ProducerPolicy = Callable[[TrustLevel, ValidationState], bool]


def default_producer_policy(trust: TrustLevel, validation: ValidationState) -> bool:
    """Publish only build-passed outputs; record trust for read-time filtering.

    We intentionally allow *untrusted* (PR) entries to be published -- they are
    tagged ``UNTRUSTED`` and the read policy decides who may reuse them -- but we
    never publish outputs whose build did not pass.
    """
    return validation.meets(ValidationState.BUILD_PASSED)


class ObjectUploader:
    """Uploads immutable objects before the manifest is published.

    This uploader is a no-op recorder: in production, artifacts already live
    in the ``ci-artifacts`` buckets, so "upload" is often just confirming the
    pointer. Subclass / replace ``upload`` to copy into immutable storage.
    """

    def upload(self, obj: OutputObject) -> OutputObject:
        return obj


def publish_build_outputs(
    *,
    db: CacheDB,
    artifact: ArtifactIdentity,
    outputs: Sequence[OutputObject],
    provenance: Provenance,
    source_fingerprint: str,
    build_variant: str = "release",
    trust: TrustLevel = TrustLevel.UNTRUSTED,
    validation: ValidationState = ValidationState.BUILD_PASSED,
    dep_identities: Sequence[str] = (),
    cmake_files_hash: str | None = None,
    producer_policy: ProducerPolicy = default_producer_policy,
    uploader: ObjectUploader | None = None,
) -> Manifest | None:
    """Publish one unit's outputs, or return ``None`` if policy forbids it.

    Returns the published :class:`Manifest` on success, or ``None`` when the
    outputs are incomplete or the producer policy is not satisfied (the two
    "do not publish cache entry" branches of flowchart 3).
    """
    # Gate 1: outputs complete?
    if not outputs or not all(o.uri for o in outputs):
        return None

    # Gate 2: producer & validation policy satisfied?
    if not producer_policy(trust, validation):
        return None

    # Upload immutable objects FIRST.
    uploader = uploader or ObjectUploader()
    uploaded = tuple(uploader.upload(o) for o in outputs)

    key = compute_cache_key(
        source_fingerprint=source_fingerprint,
        build_variant=build_variant,
        cmake_files_hash=cmake_files_hash,
    )

    manifest = Manifest(
        cache_key=key.value,
        artifact=artifact,
        outputs=uploaded,
        provenance=provenance,
        build_variant=build_variant,
        dep_identities=tuple(dep_identities),
        trust=trust,
        validation=validation,
        publication=PublicationState.PUBLISHED,  # objects are up; entry goes last
    )

    # Publish manifest + fingerprint entry LAST.
    db.publish(manifest)
    return manifest
