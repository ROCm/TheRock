# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the adapted module tests.

Use the real SQLite schema, filesystem objects and archive verifier. Producer
records created here are explicitly synthetic metadata, not evidence of an
actual compiler build. The managed-workflow tests cover that separate boundary.
"""

import json

from build_tools.github_actions.artifact_cache.cache_db import CacheDB
from build_tools.github_actions.artifact_cache.local_store import store_component
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey, TrustLevel
from build_tools.github_actions.tests import artifact_cache_collector_test as collector_tests

FPRINT = collector_tests.FPRINT
IDENTITY = "blas_lib_gfx94X-dcgpu"


class RelationalFixture(collector_tests.CacheFixture):
    def key(self, identity=IDENTITY, fingerprint=FPRINT):
        return ArtifactKey(identity, fingerprint)

    def seed(self, identity=IDENTITY, *, trust=TrustLevel.UNTRUSTED, **kwargs):
        """Seed lookup-only metadata; its dummy archive cannot be restored."""
        entry = self.seed_candidate(identity, **kwargs)
        with self.db:
            self.db.execute(
                "UPDATE prototype_runs SET parameters_json=? WHERE prototype_run_id="
                "(SELECT producer_run_id FROM cache_entries WHERE cache_entry_id=?)",
                (json.dumps({"cache_provenance": {"trust": trust.value}}), entry),
            )
        return entry

    def producer(self, identity=IDENTITY, *, trust=TrustLevel.UNTRUSTED,
                 successful_build=True):
        self.outputs((identity, FPRINT))
        parameters = {"cache_provenance": {"trust": trust.value}}
        if successful_build:
            parameters["managed_build"] = {
                "command": ["synthetic-test-producer"],
                "returncode": 0,
                "inputs_stable": True,
            }
        return self.collect(parameters=parameters)["run_id"]

    def stored(self, identity=IDENTITY, payload=b"artifact version one\n"):
        export = self.artifacts / identity
        (export / "stage").mkdir(parents=True, exist_ok=True)
        (export / "artifact_manifest.txt").write_text("stage\n", encoding="utf-8")
        (export / "stage/payload.txt").write_bytes(payload)
        return store_component(export, self.root / "objects")

    def publish_fixture(self, identity=IDENTITY, *, trust=TrustLevel.UNTRUSTED,
                        payload=b"artifact version one\n"):
        run = self.producer(identity, trust=trust)
        stored = self.stored(identity, payload)
        with CacheDB(self.database) as db:
            return db.register_local_outputs(
                producer_run_id=run,
                outputs=[(self.key(identity), stored.directory, stored.files)],
            )[0]

    def publish_copy(self, cache, payload):
        """Publish a synthetic producer's real archive into a chosen store."""
        run = self.producer()
        component = self.artifacts / IDENTITY
        (component / "stage").mkdir(parents=True, exist_ok=True)
        (component / "artifact_manifest.txt").write_text(
            "stage\n", encoding="utf-8"
        )
        (component / "stage/file").write_bytes(payload)
        stored = store_component(component, cache)
        with CacheDB(self.database) as db:
            return db.register_local_outputs(
                producer_run_id=run,
                outputs=[(self.key(), stored.directory, stored.files)],
            )[0]

def history_run(run_id, commit="c1", *, branch="main", event="push",
                start="2026-09-01T00:00:00Z", end="2026-09-01T00:10:00Z",
                conclusion="success", fingerprints=None):
    run = {
        "id": str(run_id), "head_sha": commit, "head_branch": branch,
        "head_repository": {"full_name": "ROCm/TheRock"}, "event": event,
        "status": "completed", "conclusion": conclusion,
        "created_at": start, "completed_at": end,
        "name": "multi_arch_ci.yml",
    }
    if fingerprints is not None:
        run["source_fingerprints"] = fingerprints
    return run
