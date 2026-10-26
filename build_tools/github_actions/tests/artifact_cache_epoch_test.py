#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Epoch identity, migration, publication and restore regression tests.

These use temporary SQLite databases and actual local archives. Producer
evidence in the metadata-only tests is synthetic, not an attested real build.
"""

import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache import cache_key, migrate_schema
from build_tools.github_actions.artifact_cache.cache_db import CacheDB, SchemaError
from build_tools.github_actions.artifact_cache.inventory import scan_build
from build_tools.github_actions.artifact_cache.local_store import store_component
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey
from build_tools.github_actions.artifact_cache.planner import plan_cache
from build_tools.github_actions.artifact_cache.restore import restore_local_key
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures
from build_tools.github_actions.tests import artifact_cache_local_test as local


IDENTITY = "core-amdsmi_lib_generic"


class EpochKeyTest(fixtures.CacheFixture):
    def key(self, epoch):
        return ArtifactKey(IDENTITY, fixtures.FPRINT, epoch=epoch)

    def test_default_epoch_and_strict_validation(self):
        with patch.object(cache_key, "CACHE_EPOCH", 2):
            self.assertEqual(ArtifactKey(IDENTITY, fixtures.FPRINT).epoch, 2)
        self.assertNotEqual(self.key(1), self.key(2))
        self.assertEqual(len({self.key(1), self.key(2)}), 2)
        for invalid in (None, True, False, 0, -1, 1.0, "1", 2**63):
            with self.subTest(epoch=invalid), self.assertRaises(ValueError):
                self.key(invalid)

    def test_same_fingerprint_epochs_coexist_and_lookup_matches_exactly(self):
        first = self.seed_candidate(IDENTITY, epoch=1, created_at=10)
        second = self.seed_candidate(IDENTITY, epoch=2, created_at=20)
        with CacheDB(self.database) as db:
            matches = db.lookup_batch([self.key(1), self.key(2), self.key(3)])
            self.assertEqual(matches[self.key(1)].cache_entry_id, first)
            self.assertEqual(matches[self.key(2)].cache_entry_id, second)
            self.assertIsNone(matches[self.key(3)])
        self.assertEqual(self.count("artifact_keys"), 2)

    def test_loading_old_manifest_preserves_stored_epoch_after_bump(self):
        entry = self.seed_candidate(IDENTITY, epoch=1)
        with patch.object(cache_key, "CACHE_EPOCH", 2), CacheDB(self.database) as db:
            manifest = db.get_manifest(entry)
            self.assertEqual(manifest.key.epoch, 1)
            self.assertEqual(manifest.to_dict()["epoch"], 1)
            self.assertIsNone(db.lookup(ArtifactKey(IDENTITY, fixtures.FPRINT)))

    def test_collector_bump_creates_new_key_and_repeat_recognizes_it(self):
        self.outputs((IDENTITY, fixtures.FPRINT))
        self.seed_candidate(IDENTITY, epoch=1)
        before = self.collect()
        with patch.object(cache_key, "CACHE_EPOCH", 2):
            after = self.collect()
            repeated = self.collect()
        self.assertEqual(before["summary"]["hit"], 1)
        self.assertEqual(after["epoch"], 2)
        self.assertEqual(after["summary"]["fingerprints_seen_before"], 0)
        self.assertEqual(after["summary"]["miss"], 1)
        self.assertEqual(repeated["summary"]["fingerprints_seen_before"], 1)
        self.assertEqual(repeated["summary"]["miss"], 1)
        rows = self.db.execute(
            "SELECT o.prototype_run_id, k.epoch FROM artifact_observations o "
            "JOIN artifact_keys k USING (artifact_key_id) ORDER BY o.prototype_run_id"
        ).fetchall()
        self.assertEqual([r["epoch"] for r in rows], [1, 2, 2])
        self.assertEqual(self.count("cache_entries"), 1)

    def test_scan_and_planner_use_current_epoch(self):
        self.outputs((IDENTITY, fixtures.FPRINT))
        self.seed_candidate(IDENTITY, epoch=1)
        with patch.object(cache_key, "CACHE_EPOCH", 2):
            records = scan_build(self.artifacts.parent, self.manifest)
            with CacheDB(self.database) as db:
                plan = plan_cache(db=db, records=records)
        self.assertEqual(records[0][1].epoch, 2)
        self.assertEqual(plan.entries[0].status, "miss")
        self.assertEqual(plan.to_dict()["stages"][0]["entries"][0]["epoch"], 2)

    def test_sql_rejects_omitted_and_invalid_epochs_and_duplicate_keys(self):
        self.seed_candidate(IDENTITY, epoch=1)
        artifact_id = self.db.execute("SELECT artifact_id FROM artifacts").fetchone()[0]
        for epoch in (None, 0, -1, 1.5):
            with self.subTest(epoch=epoch), self.assertRaises(sqlite3.IntegrityError):
                with self.db:
                    self.db.execute(
                        "INSERT INTO artifact_keys (artifact_id,fingerprint,epoch) "
                        "VALUES (?,?,?)", (artifact_id, fixtures.FPRINT, epoch)
                    )
        with self.assertRaises(sqlite3.IntegrityError), self.db:
            self.db.execute(
                "INSERT INTO artifact_keys (artifact_id,fingerprint) VALUES (?,?)",
                (artifact_id, fixtures.FPRINT)
            )
        with self.assertRaises(sqlite3.IntegrityError), self.db:
            self.db.execute(
                "INSERT INTO artifact_keys (artifact_id,fingerprint,epoch) VALUES (?,?,1)",
                (artifact_id, fixtures.FPRINT)
            )

    def test_publication_requires_observation_in_same_epoch_and_restore_isolates(self):
        self.outputs((IDENTITY, fixtures.FPRINT))
        # Synthetic successful-build evidence exercises only the DB boundary.
        run = self.collect(parameters={"managed_build": {
            "returncode": 0, "command": ["synthetic-test"], "inputs_stable": True}})
        component = self.root / "component"
        component.mkdir()
        (component / "artifact_manifest.txt").write_text("stage\n")
        (component / "stage").mkdir()
        (component / "stage/payload").write_bytes(b"real archive bytes; synthetic build")
        cache = self.root / "objects"
        stored = store_component(component, cache)
        with CacheDB(self.database) as db:
            with self.assertRaisesRegex(ValueError, "exact artifact key"):
                db.register_local_outputs(producer_run_id=run["run_id"], outputs=[
                    (self.key(2), stored.directory, stored.files)])
            self.assertEqual(db.stats()["cache_entries"], 0)
            published = db.register_local_outputs(producer_run_id=run["run_id"], outputs=[
                (self.key(1), stored.directory, stored.files)])[0]
            missed = restore_local_key(db, self.key(2), cache, self.root / "missed")
            self.assertEqual(missed["outcome"], "needs_build")
            self.assertFalse((self.root / "missed").exists())
            restored = restore_local_key(db, self.key(1), cache, self.root / "restored")
        self.assertEqual(published.key.epoch, 1)
        self.assertEqual(restored["outcome"], "restored")
        self.assertEqual(restored["epoch"], 1)
        self.assertEqual((self.root / "restored/stage/payload").read_bytes(),
                         b"real archive bytes; synthetic build")


# Literal revision-3 key table, independent of the migration implementation.
V3_KEYS = """CREATE TABLE artifact_keys (
    artifact_key_id INTEGER PRIMARY KEY,
    artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id),
    key_namespace TEXT NOT NULL CHECK (length(key_namespace) > 0),
    fingerprint TEXT NOT NULL CHECK (
        length(fingerprint) = 64 AND fingerprint NOT GLOB '*[^0-9a-f]*'
    ),
    UNIQUE (key_namespace, artifact_id, fingerprint),
    UNIQUE (artifact_key_id, artifact_id)
);"""


class EpochMigrationTest(fixtures.CacheFixture):
    def setUp(self):
        super().setUp()
        self.legacy = self.root / "legacy.db"
        self.backup = self.root / "before-epoch.db"
        current = fixtures.SCHEMA.read_text()
        start = current.index("CREATE TABLE artifact_keys (")
        end = current.index(";", start) + 1
        legacy = current[:start] + V3_KEYS + current[end:]
        with sqlite3.connect(self.legacy) as db:
            db.executescript(legacy)
            db.execute("PRAGMA user_version = 3")
            db.execute("INSERT INTO artifacts VALUES (7, ?)", (IDENTITY,))
            db.execute("INSERT INTO artifact_keys VALUES (17, 7, ?, ?)",
                       ("legacy-test", fixtures.FPRINT))
            db.execute("INSERT INTO prototype_runs (prototype_run_id,label,repository_commit,"
                       "configure_state,source_state) VALUES (23,'old producer','old commit','succeeded','prepared')")
            db.execute("INSERT INTO artifact_observations (prototype_run_id,artifact_id,artifact_key_id,"
                       "fingerprint_state,lookup_result) VALUES (23,7,17,'valid','miss')")
            db.execute("INSERT INTO cache_entries (cache_entry_id,artifact_key_id,producer_run_id,"
                       "local_directory,expected_files_json) VALUES (42,17,23,?,'[\"payload.tar\"]')",
                       (str(self.root / "stored"),))
            db.execute("INSERT INTO artifact_files VALUES (42,'payload.tar',?,7,'available')",
                       (hashlib.sha256(b"payload").hexdigest(),))
            db.execute("UPDATE cache_entries SET state='ready',validation_state='build_passed',"
                       "validated_at=1 WHERE cache_entry_id=42")
            db.execute("UPDATE artifact_observations SET lookup_result='hit', selected_entry_id=42")
            db.execute("CREATE INDEX extra_fingerprint_index ON artifact_keys(fingerprint)")
            db.execute("CREATE VIEW extra_key_view AS SELECT artifact_key_id FROM artifact_keys")
        self.before = self.snapshot(self.legacy)

    def snapshot(self, path):
        with sqlite3.connect(path) as db:
            tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            return {name: db.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
                    for name in tables}

    def test_migration_preserves_ids_productions_observations_and_backup(self):
        with patch.object(cache_key, "CACHE_EPOCH", 2):
            result = migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(result["keys_after"], 1)
        self.assertEqual(result["legacy_epoch"], 1)
        self.assertEqual(self.snapshot(self.backup), self.before)
        after = self.snapshot(self.legacy)
        for table in after:
            if table != "artifact_keys":
                self.assertEqual(after[table], self.before[table], table)
        with CacheDB(self.legacy) as db:
            key = ArtifactKey(IDENTITY, fixtures.FPRINT, epoch=1)
            self.assertEqual(db.lookup(key).cache_entry_id, 42)
            self.assertIsNone(db.lookup(ArtifactKey(IDENTITY, fixtures.FPRINT, epoch=2)))
            self.assertEqual(db._conn.execute("SELECT artifact_key_id FROM extra_key_view").fetchone()[0], 17)
            self.assertEqual(db._conn.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(db._conn.execute("PRAGMA user_version").fetchone()[0], 6)
            with self.assertRaises(sqlite3.IntegrityError), db.transaction():
                db._conn.execute("UPDATE artifact_keys SET epoch=2 WHERE artifact_key_id=17")
            with db.transaction():
                db._conn.execute("INSERT INTO artifact_keys (artifact_id,fingerprint,epoch) "
                                 "VALUES (7,?,2)", (fixtures.FPRINT,))
        second_backup = self.root / "unused.db"
        self.assertEqual(migrate_schema.migrate(self.legacy, second_backup)["status"], "already_current")
        self.assertFalse(second_backup.exists())

    def test_open_old_schema_requires_explicit_migration_without_writing(self):
        with self.assertRaisesRegex(SchemaError, "namespace removal migration"):
            CacheDB(self.legacy)
        self.assertEqual(self.snapshot(self.legacy), self.before)

    def test_adding_column_without_replacing_uniqueness_is_rejected(self):
        with sqlite3.connect(self.legacy) as db:
            db.execute("ALTER TABLE artifact_keys ADD COLUMN epoch INTEGER NOT NULL DEFAULT 1")
        with self.assertRaisesRegex(SchemaError, "namespace removal migration"):
            CacheDB(self.legacy)

    def test_migration_failure_rolls_back_and_retains_backup(self):
        original = migrate_schema._validate_schema
        def fail_final(db, *, legacy=False, namespaced=False, component_only=False):
            if not legacy and not namespaced:
                raise SchemaError("injected post-migration validation failure")
            original(db, legacy=True)
        with patch.object(migrate_schema, "_validate_schema", side_effect=fail_final):
            with self.assertRaisesRegex(SchemaError, "injected"):
                migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.snapshot(self.legacy), self.before)
        self.assertEqual(self.snapshot(self.backup), self.before)
        with sqlite3.connect(self.legacy) as db:
            self.assertNotIn("epoch", [r[1] for r in db.execute("PRAGMA table_info(artifact_keys)")])
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 3)

    def test_existing_backup_and_missing_database_are_not_overwritten(self):
        self.backup.write_bytes(b"keep me")
        with self.assertRaises(FileExistsError):
            migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.backup.read_bytes(), b"keep me")
        self.assertEqual(self.snapshot(self.legacy), self.before)
        with self.assertRaises(FileNotFoundError):
            migrate_schema.migrate(self.root / "typo.db", self.root / "other.db")
        self.assertFalse((self.root / "typo.db").exists())

    def test_unrecognized_unique_index_refused_before_migration(self):
        with sqlite3.connect(self.legacy) as db:
            db.execute("CREATE UNIQUE INDEX custom_unique ON artifact_keys(fingerprint)")
        with self.assertRaisesRegex(SchemaError, "unrecognized unique"):
            migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.snapshot(self.legacy), self.before)
        self.assertFalse(self.backup.exists())


@unittest.skipUnless(all(local.shutil.which(x) for x in ("cmake", "ninja", "git")),
                     "requires CMake, Ninja and Git")
class EpochManagedBuildTest(unittest.TestCase):
    setUp = local.LocalCacheBuildTest.setUp
    run_command = local.LocalCacheBuildTest.run_command
    git = local.LocalCacheBuildTest.git
    commit = local.LocalCacheBuildTest.commit
    configure = local.LocalCacheBuildTest.configure

    def test_epoch_bump_rebuilds_same_fingerprint_then_warm_reuses(self):
        from build_tools.github_actions.artifact_cache.local_cache import cache_build
        def run():
            return cache_build(database=self.database, build=self.build, cache=self.cache,
                               expected=self.expected)
        first = run()
        fingerprint = (self.build / "artifacts" / (local.IDENTITY + ".fprint")).read_text()
        with patch.object(cache_key, "CACHE_EPOCH", 2):
            second = run()
            count = (self.build / "build-count").read_text()
            warm = run()
        self.assertEqual(first["epoch"], 1)
        self.assertEqual(second["epoch"], 2)
        self.assertTrue(second["build_executed"])
        self.assertEqual(second["summary"]["stored"], 1)
        self.assertFalse(warm["build_executed"])
        self.assertEqual(warm["summary"]["verified_cached"], 1)
        self.assertEqual((self.build / "build-count").read_text(), count)
        self.assertEqual((self.build / "artifacts" / (local.IDENTITY + ".fprint")).read_text(), fingerprint)
        with CacheDB(self.database) as db:
            self.assertEqual({m.key.epoch for m in db.all_manifests()}, {1, 2})


if __name__ == "__main__":
    unittest.main()
