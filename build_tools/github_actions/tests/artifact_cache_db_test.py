# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Relational DB tests: schema, lookup, policy, publication, history and eviction.

Combines the original module contracts and the integrated schema regressions.
Fixtures use isolated databases; metadata lookup never verifies payload bytes.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import (
    CacheDB, ReadPolicy, SchemaError,
)
from build_tools.github_actions.artifact_cache.manifest import (
    ArtifactFile, ArtifactKey, LocalManifest, TrustLevel, ValidationState,
)
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures
from build_tools.github_actions.tests import _artifact_cache_fixtures as relational


class CacheDBTest(fixtures.CacheFixture):
    def key(self, identity="compiler_lib_generic"):
        return ArtifactKey(identity, fixtures.FPRINT)

    def test_open_preserves_schema_and_enforces_foreign_keys(self):
        before = self.db.execute("SELECT type, name, sql FROM sqlite_master").fetchall()
        with CacheDB(self.database) as db:
            self.assertEqual(db.stats()["cache_entries"], 0)
            self.assertEqual(db._conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        after = self.db.execute("SELECT type, name, sql FROM sqlite_master").fetchall()
        self.assertEqual(before, after)
        self.assertNotIn("manifests", {row[1] for row in after})

    def test_legacy_schema_rejected_without_creating_relational_tables(self):
        path = self.root / "legacy.db"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE manifests (cache_key TEXT PRIMARY KEY)")
        with self.assertRaisesRegex(SchemaError, "missing table artifacts"):
            CacheDB(path)
        with sqlite3.connect(path) as conn:
            self.assertEqual(
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall(),
                [("manifests",)],
            )

    def test_same_hash_preserves_output_identity_and_newer_production_wins(self):
        self.seed_candidate("amd-llvm_lib_generic")
        dev = self.seed_candidate("amd-llvm_dev_generic")
        newer = self.seed_candidate("amd-llvm_lib_generic", created_at=20)
        keys = [self.key("amd-llvm_lib_generic"), self.key("amd-llvm_dev_generic"), self.key("absent")]
        with CacheDB(self.database) as db:
            result = db.lookup_batch(keys)
        self.assertEqual([result[key].cache_entry_id if result[key] else None for key in keys],
                         [newer, dev, None])
        self.assertEqual(self.count("artifact_keys"), 2)
        self.assertFalse(result[keys[0]].payloads_verified)

    def test_manifest_reconstructs_files_and_actual_provenance(self):
        entry = self.seed_candidate("compiler_lib_generic")
        with CacheDB(self.database) as db:
            manifest = db.get_manifest(entry)
        self.assertIsInstance(manifest, LocalManifest)
        self.assertEqual(manifest.key, self.key())
        self.assertEqual(manifest.expected_files, ("artifact.tar",))
        payload = (Path(manifest.local_directory) / "artifact.tar").read_bytes()
        self.assertEqual(manifest.files, (
            ArtifactFile("artifact.tar", hashlib.sha256(payload).hexdigest(), len(payload)),
        ))
        self.assertEqual(manifest.producer_commit, "synthetic-old-commit")
        self.assertEqual(manifest.producer_label, "synthetic producer")
        self.assertEqual(manifest.validation_state, "build_passed")
        self.assertFalse(manifest.payloads_verified)
        encoded = json.loads(json.dumps(manifest.to_dict()))
        self.assertEqual(encoded["lookup_scope"], "database_metadata")
        self.assertEqual(encoded["trust"], "untrusted")
        self.assertEqual(encoded["producer_metadata"], {})
        self.assertNotIn("branch", encoded)
        self.assertNotIn("workflow", encoded)

    def test_lookup_metadata_does_not_check_or_claim_file_integrity(self):
        entry = self.seed_candidate("compiler_lib_generic")
        with CacheDB(self.database) as db:
            manifest = db.get_manifest(entry)
            (Path(manifest.local_directory) / "artifact.tar").unlink()
            result = db.lookup(self.key())
        self.assertIsNotNone(result)
        self.assertFalse(result.payloads_verified)

    def test_duplicate_keys_and_empty_batch_are_supported(self):
        self.seed_candidate("compiler_lib_generic")
        with CacheDB(self.database) as db:
            self.assertEqual(db.lookup_batch([]), {})
            result = db.lookup_batch([self.key(), self.key()])
            self.assertEqual(list(result), [self.key()])
            self.assertIsNotNone(result[self.key()])

    def test_large_batch_does_not_use_per_output_queries(self):
        keys = [self.key(f"output-{n}") for n in range(1200)]
        with CacheDB(self.database) as db:
            statements = []
            db._conn.set_trace_callback(statements.append)
            result = db.lookup_batch(keys)
        self.assertEqual(len(result), 1200)
        self.assertTrue(all(value is None for value in result.values()))
        self.assertLess(len(statements), 10)

    def test_read_lookup_does_not_create_keys_runs_or_observations(self):
        with CacheDB(self.database) as db:
            before = db.stats()
            self.assertIsNone(db.lookup(self.key()))
            self.assertEqual(db.stats(), before)

    def test_newer_pending_entry_does_not_hide_ready_entry(self):
        first = self.seed_candidate("compiler_lib_generic", created_at=10)
        pending = self.seed_candidate("compiler_lib_generic", state="pending", created_at=20)
        with CacheDB(self.database) as db:
            self.assertEqual(db.lookup(self.key()).cache_entry_id, first)
            self.assertIsNone(db.get_manifest(pending))

    def test_missing_file_metadata_rejects_only_its_candidate(self):
        bad = self.seed_candidate("one")
        good = self.seed_candidate("two")
        with self.db:
            self.db.execute("DELETE FROM artifact_files WHERE cache_entry_id = ?", (bad,))
        with CacheDB(self.database) as db:
            result = db.lookup_batch([self.key("one"), self.key("two")])
        self.assertIsNone(result[self.key("one")])
        self.assertEqual(result[self.key("two")].cache_entry_id, good)

    def test_expired_invalid_and_corrupt_candidates_are_not_returned(self):
        entries = [
            self.seed_candidate("expired", expires_at=1),
            self.seed_candidate("invalid", state="invalid"),
            self.seed_candidate("corrupt", availability="corrupt"),
        ]
        with CacheDB(self.database) as db:
            for entry in entries:
                self.assertIsNone(db.get_manifest(entry))
            self.assertIsNone(db.get_manifest(999999))

    def test_unparsed_output_identity_is_retained(self):
        identity = "opaque/output:component_with_underscores"
        self.seed_candidate(identity)
        with CacheDB(self.database) as db:
            self.assertEqual(db.lookup(self.key(identity)).key.artifact_identity, identity)

    def test_legacy_string_keys_rejected_and_policy_supported(self):
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(TypeError, "ArtifactKey"):
                db.lookup("e1:release:old-key")
            self.assertIsNone(db.lookup(self.key(), ReadPolicy()))
            with self.assertRaisesRegex(NotImplementedError, "local publication"):
                db.publish(None)
            self.assertEqual(db.stats()["cache_entries"], 0)

    def test_write_transaction_rolls_back_on_exception(self):
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with db.transaction():
                    db._conn.execute("INSERT INTO artifacts (artifact_identity) VALUES ('temporary')")
                    raise RuntimeError("injected")
            self.assertEqual(db.stats()["artifacts"], 0)

    def test_nested_transaction_rejected_without_committing_outer_work(self):
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with db.transaction():
                    db._conn.execute("INSERT INTO artifacts (artifact_identity) VALUES ('temporary')")
                    with db.transaction():
                        pass
            self.assertEqual(db.stats()["artifacts"], 0)

    def test_record_observations_requires_transaction(self):
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(RuntimeError, "active transaction"):
                db.record_observations(
                    [], label="test", commit="test", configure_state="succeeded", parameters={}, source_revisions={},
                    overlay_identity=None, collect_ms=0, lookup_ms=0, plan_ms=0,
                )

    def test_invalid_raw_key_is_rejected(self):
        for value in ("", "not-a-sha256", "A" * 64, "srcfp1:" + "a" * 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ArtifactKey("compiler", value)

    def test_manifest_file_paths_reject_traversal_and_absolute_paths(self):
        for name in ("", "../outside", "/outside", "a/../outside", "C:/outside", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ArtifactFile(name, fixtures.FPRINT, 0)

    def test_direct_and_module_collector_entry_points(self):
        self.outputs(("compiler_lib_generic", fixtures.FPRINT))
        root = fixtures.TOOLS_DIR.parent
        args = ["collect", "--db", str(self.database), "--expected", str(self.manifest),
                "--artifacts-dir", str(self.artifacts), "--label", "CLI fixture", "--commit", "fixture", "--configure-state", "succeeded"]
        for command in (
            [sys.executable, "build_tools/github_actions/artifact_cache/collector.py"],
            [sys.executable, "-m", "build_tools.github_actions.artifact_cache.collector"],
        ):
            completed = subprocess.run(command + args, cwd=root, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["summary"]["miss"], 1)

    def test_collector_import_does_not_load_planner_or_network_modules(self):
        command = [sys.executable, "-c", (
            "import sys; from build_tools.github_actions.artifact_cache import collector; "
            "assert not any(name.endswith(('.planner', '.history_ingest', '.topology_view')) "
            "for name in sys.modules)"
        )]
        completed = subprocess.run(command, cwd=fixtures.TOOLS_DIR.parent, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


class PublishGuardTest(relational.RelationalFixture):
    def test_pending_entry_is_never_returned(self):
        self.seed(state="pending")
        with CacheDB(self.database) as db:
            self.assertIsNone(db.lookup(self.key()))
            self.assertIsNone(db.lookup(self.key(), ReadPolicy(
                min_trust=TrustLevel.UNTRUSTED, min_validation=ValidationState.UNVALIDATED)))


    def test_republication_reuses_key_and_preserves_both_generations(self):
        # The old overwrite/idempotence assertion is intentionally replaced:
        # one input key may have multiple immutable producing generations.
        first = self.publish_fixture()
        second = self.publish_fixture(payload=b"second generation\n")
        self.assertEqual(first.artifact_key_id, second.artifact_key_id)
        self.assertNotEqual(first.cache_entry_id, second.cache_entry_id)
        self.assertNotEqual(first.local_directory, second.local_directory)
        with CacheDB(self.database) as db:
            self.assertEqual(db.stats()["artifact_keys"], 1)
            self.assertEqual(db.count(), 2)
            self.assertEqual(db.lookup(self.key()).cache_entry_id, second.cache_entry_id)


class ReadPolicyTest(relational.RelationalFixture):
    def test_trust_filter(self):
        self.seed(trust=TrustLevel.UNTRUSTED)
        with CacheDB(self.database) as db:
            self.assertIsNone(db.lookup(self.key(), ReadPolicy(min_trust=TrustLevel.TRUSTED)))
            self.assertIsNotNone(db.lookup(self.key(), ReadPolicy(min_trust=TrustLevel.UNTRUSTED)))

    def test_validation_filter(self):
        self.seed(trust=TrustLevel.TRUSTED)
        with CacheDB(self.database) as db:
            self.assertIsNone(db.lookup(self.key(), ReadPolicy(min_validation=ValidationState.TESTS_PASSED)))
            self.assertIsNotNone(db.lookup(self.key(), ReadPolicy(min_validation=ValidationState.BUILD_PASSED)))

    def test_point_in_time_filter(self):
        self.seed(trust=TrustLevel.TRUSTED, created_at=20)
        with CacheDB(self.database) as db:
            before = db.lookup(self.key(), ReadPolicy(created_before="1970-01-01T00:00:20Z"))
            after = db.lookup(self.key(), ReadPolicy(created_before="1970-01-01T00:00:21Z"))
        self.assertIsNone(before)
        self.assertIsNotNone(after)


class EvictionTest(relational.RelationalFixture):
    def test_evict_before_preserves_history_rows(self):
        old = self.seed(created_at=10)
        new = self.seed("base_run_generic", created_at=30)
        with CacheDB(self.database) as db:
            self.assertEqual(db.evict_before("1970-01-01T00:00:20Z"), 1)
            self.assertIsNone(db.get_manifest(old))
            self.assertIsNotNone(db.get_manifest(new))
            self.assertEqual(db.stats()["cache_entries"], 2)
            self.assertEqual(db.count(), 1)

    def test_evict_where_missing(self):
        present = self.seed()
        gone = self.seed("base_run_generic")
        with CacheDB(self.database) as db:
            manifest = db.get_manifest(gone)
            (Path(manifest.local_directory) / manifest.files[0].relative_path).unlink()
            self.assertEqual(db.evict_where_missing(lambda path: Path(path).is_file()), 1)
            self.assertIsNotNone(db.get_manifest(present))
            self.assertIsNone(db.get_manifest(gone))


class StatsAndDiagnosticsTest(relational.RelationalFixture):
    def test_stats_and_producing_run_are_retained(self):
        manifest = self.publish_fixture()
        with CacheDB(self.database) as db:
            read = db.get_manifest(manifest.cache_entry_id)
            self.assertEqual(read.producer_run_id, manifest.producer_run_id)
            self.assertEqual(read.producer_commit, "synthetic-commit-a")
            self.assertEqual(db.stats()["cache_entries"], 1)
            self.assertEqual(db.stats()["artifact_files"], 2)

if __name__ == "__main__":
    unittest.main()
