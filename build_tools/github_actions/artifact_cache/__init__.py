# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Artifact-cache APIs, loaded lazily so collection stays independent of network helpers."""

from importlib import import_module


_EXPORTS = {
    "build_and_publish": ("publish", "build_and_publish"),
    "restore_local_outputs": ("restore", "restore_local_outputs"),
    "cache_build": ("local_cache", "cache_build"),
    "TopologyAdapter": ("inventory", "TopologyAdapter"),
    "ingest_runs": ("history_ingest", "ingest_runs"),
    "replay": ("replay", "replay"),
    "CACHE_EPOCH": ("cache_key", "CACHE_EPOCH"),
    "ArtifactIdentity": ("cache_key", "ArtifactIdentity"),
    "CacheKey": ("cache_key", "CacheKey"),
    "compute_cache_key": ("cache_key", "compute_cache_key"),
    "derive_fallback_fingerprint": ("cache_key", "derive_fallback_fingerprint"),
    "Manifest": ("manifest", "Manifest"),
    "OutputObject": ("manifest", "OutputObject"),
    "Provenance": ("manifest", "Provenance"),
    "PublicationState": ("manifest", "PublicationState"),
    "TrustLevel": ("manifest", "TrustLevel"),
    "ValidationState": ("manifest", "ValidationState"),
    "CacheDB": ("cache_db", "CacheDB"),
    "ReadPolicy": ("cache_db", "ReadPolicy"),
    "CachePlan": ("planner", "CachePlan"),
    "PlanEntry": ("planner", "PlanEntry"),
    "RequiredUnit": ("planner", "RequiredUnit"),
    "StagePlan": ("planner", "StagePlan"),
    "plan_cache": ("planner", "plan_cache"),
    "RestoreResult": ("restore", "RestoreResult"),
    "InMemoryObjectStore": ("restore", "InMemoryObjectStore"),
    "restore_units": ("restore", "restore_units"),
    "publish_build_outputs": ("publish", "publish_build_outputs"),
    "LocalObjectStore": ("local_store", "LocalObjectStore"),
    "ArtifactKey": ("manifest", "ArtifactKey"),
    "ArtifactFile": ("manifest", "ArtifactFile"),
    "LocalManifest": ("manifest", "LocalManifest"),
    "SchemaError": ("cache_db", "SchemaError"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
