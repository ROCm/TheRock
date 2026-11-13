# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fingerprint-based CI artifact cache.

This package implements the system described in
https://github.com/ROCm/TheRock/issues/8651: replacing the "baseline run"
stage-reuse mechanism (GitHub-API run-history traversal + commit ancestry +
S3 archive-filename matching) with a direct, O(1) **fingerprint -> manifest**
cache lookup.

The code is organized around the three flowcharts from the design:

* ``planner``  -- per-CI-run planning: changed paths -> required artifact units
  and stages -> compute cache keys -> classify hits/misses -> decide which
  stages can be skipped entirely (flowchart 1).
* ``restore``  -- resolve a single required unit against the cache: lookup ->
  compatibility/completeness/read-policy -> download + verify digests ->
  reusable vs. needs-building (flowchart 2).
* ``publish``  -- write a freshly built unit back to the cache: outputs
  complete? producer/validation policy satisfied? -> upload immutable objects
  first -> publish manifest + fingerprint entry last (flowchart 3).

Supporting modules:

* ``cache_key``       -- the cache-key contract and the single seam where a
  real artifact ``.fprint`` will replace the fallback derived key.
* ``manifest``        -- the cache *value*: an output manifest with digests,
  provenance, trust and publication/validation state.
* ``cache_db``        -- SQLite-backed store with batched lookup, ordered
  publish, and eviction (abstracted so it can be repointed at ClickHouse).
* ``history_ingest``  -- populate the cache from real workflow-run history.
* ``replay``          -- point-in-time replay harness to quantify reuse.

NOTE: The cache key is currently *derived* from
signals available in historical runs (see ``cache_key.derive_fallback_fingerprint``)
because real artifact fingerprints are not yet uploaded with artifacts. The
``cache_key.compute_cache_key`` seam is the single place that changes when real
fingerprints become available.
"""

from .cache_key import (
    CACHE_EPOCH,
    ArtifactIdentity,
    CacheKey,
    compute_cache_key,
    derive_fallback_fingerprint,
)
from .manifest import (
    Manifest,
    OutputObject,
    Provenance,
    PublicationState,
    TrustLevel,
    ValidationState,
)
from .cache_db import CacheDB, ReadPolicy
from .planner import CachePlan, PlanEntry, RequiredUnit, StagePlan, plan_cache
from .restore import RestoreResult, InMemoryObjectStore, restore_units
from .publish import publish_build_outputs
from .local_store import LocalObjectStore

__all__ = [
    "CACHE_EPOCH",
    "ArtifactIdentity",
    "CacheKey",
    "compute_cache_key",
    "derive_fallback_fingerprint",
    "Manifest",
    "OutputObject",
    "Provenance",
    "PublicationState",
    "TrustLevel",
    "ValidationState",
    "CacheDB",
    "ReadPolicy",
    "CachePlan",
    "PlanEntry",
    "RequiredUnit",
    "StagePlan",
    "plan_cache",
    "RestoreResult",
    "InMemoryObjectStore",
    "restore_units",
    "publish_build_outputs",
    "LocalObjectStore",
]
