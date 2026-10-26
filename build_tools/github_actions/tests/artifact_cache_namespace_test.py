#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Remove namespaces without losing productions, observations or file records."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache import collector as artifact_cache
from build_tools.github_actions.artifact_cache import cli, local_cache, migrate_schema
from build_tools.github_actions.artifact_cache.cache_db import CacheDB, SchemaError
from build_tools.github_actions.artifact_cache.local_store import store_component
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey, TrustLevel, ValidationState
from build_tools.github_actions.artifact_cache.cache_db import ReadPolicy
from build_tools.github_actions.artifact_cache.restore import restore_local_key
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures

IDENTITY = "core-amdsmi_lib_generic"
# Literal old key contract, independent of the migration code.
V4_KEYS = """CREATE TABLE artifact_keys (
    artifact_key_id INTEGER PRIMARY KEY,
    artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id),
    key_namespace TEXT NOT NULL CHECK (length(key_namespace) > 0),
    epoch INTEGER NOT NULL CHECK (typeof(epoch) = 'integer' AND epoch > 0),
    fingerprint TEXT NOT NULL CHECK (length(fingerprint) = 64 AND fingerprint NOT GLOB '*[^0-9a-f]*'),
    UNIQUE (key_namespace, epoch, artifact_id, fingerprint),
    UNIQUE (artifact_key_id, artifact_id)
);"""


class NamespaceMigrationTest(fixtures.CacheFixture):
    def setUp(self):
        super().setUp()
        self.legacy = self.root / "v4.db"
        self.backup = self.root / "before-removal.db"
        schema = fixtures.SCHEMA.read_text()
        start = schema.index("CREATE TABLE artifact_keys (")
        end = schema.index(";", start) + 1
        schema = schema[:start] + V4_KEYS + schema[end:]
        with sqlite3.connect(self.legacy) as db:
            db.executescript(schema)
            db.execute("PRAGMA user_version=4")
            db.executemany("INSERT INTO artifacts VALUES (?,?)", [(7, IDENTITY), (8, "core-amdsmi_dev_generic")])
            for key, artifact, namespace, epoch, fp in (
                (10, 7, "experiment-a", 1, fixtures.FPRINT),
                (20, 7, "experiment-b", 1, fixtures.FPRINT),
                (30, 7, "experiment-b", 2, fixtures.FPRINT),
                (40, 8, "experiment-a", 1, fixtures.FPRINT),
                (50, 7, "experiment-a", 1, "b" * 64),
            ):
                db.execute("INSERT INTO artifact_keys VALUES (?,?,?,?,?)", (key, artifact, namespace, epoch, fp))
                db.execute("INSERT INTO prototype_runs(prototype_run_id,label,repository_commit,configure_state,source_state) "
                           "VALUES (?,?,?,'succeeded','prepared')", (key, namespace, "commit-" + str(key)))
                db.execute("INSERT INTO cache_entries(cache_entry_id,artifact_key_id,producer_run_id,local_directory,expected_files_json,created_at) "
                           "VALUES (?,?,?,?,?,?)", (key + 100, key, key, str(self.root / f'objects/{key}'), '["payload.tar"]', key))
                db.execute("INSERT INTO artifact_files VALUES (?, 'payload.tar', ?, 7, 'available')", (key + 100, fixtures.FPRINT))
                db.execute("UPDATE cache_entries SET state='ready', validation_state='build_passed', validated_at=1 WHERE cache_entry_id=?", (key + 100,))
                db.execute("INSERT INTO artifact_observations(prototype_run_id,artifact_id,artifact_key_id,fingerprint_state,lookup_result,selected_entry_id) "
                           "VALUES (?,?,?,'valid','hit',?)", (key, artifact, key, key + 100))
            # Both states must survive; a newer invalid duplicate cannot hide ready entry 110.
            db.execute("UPDATE cache_entries SET state='invalid' WHERE cache_entry_id=120")
            db.execute("CREATE INDEX custom_fp_index ON artifact_keys(fingerprint)")
            db.execute("CREATE VIEW custom_key_view AS SELECT artifact_key_id FROM artifact_keys")
        self.before = self.snapshot(self.legacy)

    def snapshot(self, path):
        with sqlite3.connect(path) as db:
            result = {}
            for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                db.row_factory = sqlite3.Row
                result[name] = [dict(r) for r in db.execute(f'SELECT * FROM "{name}" ORDER BY rowid')]
                db.row_factory = None
            result['schema'] = db.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()
            result['version'] = db.execute("PRAGMA user_version").fetchone()[0]
            return result

    def test_merges_only_equal_keys_relinks_both_foreign_keys_preserves_all_entries(self):
        result = migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual((result['keys_before'], result['keys_after'], result['merged_keys']), (5, 4, 1))
        self.assertEqual(self.snapshot(self.backup), self.before)
        after = self.snapshot(self.legacy)
        for table in ('artifacts', 'prototype_runs', 'artifact_files'):
            self.assertEqual(after[table], self.before[table])
        for table in ('cache_entries', 'artifact_observations'):
            expected = [dict(r, artifact_key_id=10 if r['artifact_key_id'] == 20 else r['artifact_key_id'])
                        for r in self.before[table]]
            self.assertEqual(after[table], expected)
        self.assertEqual([r['artifact_key_id'] for r in after['artifact_keys']], [10, 30, 40, 50])
        self.assertTrue(all('key_namespace' not in r for r in after['artifact_keys']))
        with CacheDB(self.legacy) as db:
            self.assertEqual(db.lookup(ArtifactKey(IDENTITY, fixtures.FPRINT)).cache_entry_id, 110)
            self.assertEqual(db.lookup(ArtifactKey(IDENTITY, fixtures.FPRINT, epoch=2)).cache_entry_id, 130)
            self.assertEqual(db._conn.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(db._conn.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            self.assertEqual(db._conn.execute('SELECT count(*) FROM custom_key_view').fetchone()[0], 4)
            self.assertEqual(db._conn.execute('PRAGMA user_version').fetchone()[0], 6)
            # Restored identity triggers still protect entries and keys.
            with self.assertRaises(sqlite3.IntegrityError), db.transaction():
                db._conn.execute('UPDATE cache_entries SET artifact_key_id=30 WHERE cache_entry_id=110')
            with self.assertRaises(sqlite3.IntegrityError), db.transaction():
                db._conn.execute('UPDATE artifact_keys SET epoch=2 WHERE artifact_key_id=10')
        unused = self.root / 'unused-backup.db'
        self.assertEqual(migrate_schema.migrate(self.legacy, unused)['status'], 'already_current')
        self.assertFalse(unused.exists())

    def test_old_schema_requires_explicit_migration_without_writes(self):
        with self.assertRaisesRegex(SchemaError, 'namespace removal migration'):
            CacheDB(self.legacy)
        self.assertEqual(self.snapshot(self.legacy), self.before)

    def test_failure_after_relinking_rolls_back_data_and_schema(self):
        original = migrate_schema._validate_schema
        def fail_final(db, **kwargs):
            if not kwargs:
                raise SchemaError('injected validation failure')
            return original(db, **kwargs)
        with patch.object(migrate_schema, '_validate_schema', side_effect=fail_final):
            with self.assertRaisesRegex(SchemaError, 'injected'):
                migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.snapshot(self.legacy), self.before)
        self.assertEqual(self.snapshot(self.backup), self.before)

    def test_namespace_dependent_custom_objects_refused_without_changes(self):
        with sqlite3.connect(self.legacy) as db:
            db.execute('CREATE VIEW extra_namespace_view AS SELECT key_namespace FROM artifact_keys')
        before = self.snapshot(self.legacy)
        with self.assertRaisesRegex(SchemaError, 'extra_namespace_view'):
            migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.snapshot(self.legacy), before)
        self.assertFalse(self.backup.exists())

    def test_unexpected_foreign_key_to_merged_key_rolls_back(self):
        with sqlite3.connect(self.legacy) as db:
            db.execute('CREATE TABLE extension(key_id REFERENCES artifact_keys(artifact_key_id))')
            db.execute('INSERT INTO extension VALUES (20)')
        before = self.snapshot(self.legacy)
        with self.assertRaisesRegex(SchemaError, 'foreign-key check failed'):
            migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.snapshot(self.legacy), before)
        self.assertEqual(self.snapshot(self.backup), before)

    def test_existing_backup_never_overwritten(self):
        self.backup.write_bytes(b'keep this file')
        with self.assertRaises(FileExistsError):
            migrate_schema.migrate(self.legacy, self.backup)
        self.assertEqual(self.backup.read_bytes(), b'keep this file')
        self.assertEqual(self.snapshot(self.legacy), self.before)

    def test_wal_backup_includes_committed_pages_and_remains_readable(self):
        with sqlite3.connect(self.legacy) as writer:
            writer.execute('PRAGMA journal_mode=WAL')
            writer.execute("UPDATE prototype_runs SET label='latest WAL value' WHERE prototype_run_id=10")
            writer.commit()
            before = self.snapshot(self.legacy)
            migrate_schema.migrate(self.legacy, self.backup)
            self.assertEqual(self.snapshot(self.backup), before)
        with CacheDB(self.legacy) as db:
            self.assertEqual(db.get_manifest(110).producer_label, 'latest WAL value')


class NamespaceFreeKeyTest(fixtures.CacheFixture):
    def test_key_and_json_no_longer_expose_namespace(self):
        key = ArtifactKey(IDENTITY, fixtures.FPRINT, epoch=1)
        with self.assertRaises(TypeError):
            ArtifactKey('old-namespace', IDENTITY, fixtures.FPRINT)
        with self.assertRaises(TypeError):
            ArtifactKey(IDENTITY, fixtures.FPRINT, key_namespace='old')
        entry = self.seed_candidate(IDENTITY)
        with CacheDB(self.database) as db:
            self.assertEqual(db.lookup(key).cache_entry_id, entry)
            self.assertNotIn('key_namespace', db.lookup(key).to_dict())
            self.assertNotIn('key_namespace', [r['name'] for r in db._conn.execute('PRAGMA table_info(artifact_keys)')])
        self.outputs((IDENTITY, fixtures.FPRINT))
        self.assertEqual(self.collect()['report_version'], 3)

    def test_cli_help_does_not_offer_namespace(self):
        for main, args in ((local_cache.main, ['cache']), (local_cache.main, ['keys']),
                           (cli.main, ['lookup']), (cli.main, ['publish']), (cli.main, ['restore']),
                           (cli.main, ['plan']), (artifact_cache.main, ['collect']),
                           (artifact_cache.main, ['build-publish']), (artifact_cache.main, ['restore'])):
            with self.subTest(command=args), redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as caught:
                    main(args + ['--help'])
                self.assertEqual(caught.exception.code, 0)
                self.assertNotIn('--namespace', out.getvalue())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main(['lookup', '--db', str(self.database), '--artifact', IDENTITY,
                      '--fingerprint', fixtures.FPRINT, '--namespace', 'old'])
        self.assertEqual(caught.exception.code, 2)

    def publish_copy(self, cache, payload):
        self.outputs((IDENTITY, fixtures.FPRINT))
        run = self.collect(parameters={'managed_build': {'returncode': 0, 'command': ['fixture'], 'inputs_stable': True}})
        component = self.root / 'component'
        (component / 'stage').mkdir(parents=True, exist_ok=True)
        (component / 'artifact_manifest.txt').write_text('stage\n')
        (component / 'stage/file').write_bytes(payload)
        stored = store_component(component, cache)
        with CacheDB(self.database) as db:
            return db.register_local_outputs(producer_run_id=run['run_id'], outputs=[
                (ArtifactKey(IDENTITY, fixtures.FPRINT), stored.directory, stored.files)])[0]

    def test_shared_key_selects_correct_store_before_ranking(self):
        cache = self.root / 'objects'
        first = self.publish_copy(cache, b'first store')
        second = self.publish_copy(self.root / 'objects-other', b'second store')
        key = ArtifactKey(IDENTITY, fixtures.FPRINT)
        with CacheDB(self.database) as db:
            self.assertEqual(db.stats()['artifact_keys'], 1)
            self.assertEqual(db.lookup(key).cache_entry_id, second.cache_entry_id)
            self.assertEqual(db.lookup_batch([key, key], cache_root=cache)[key].cache_entry_id, first.cache_entry_id)
            self.assertIsNone(db.lookup(key, cache_root=self.root / 'empty-store'))
            restored = restore_local_key(db, key, cache, self.root / 'restored')
            self.assertEqual(restored['cache_entry_id'], first.cache_entry_id)
            self.assertIsNone(db.lookup(key, ReadPolicy(min_trust=TrustLevel.TRUSTED), cache_root=cache))
            self.assertIsNone(db.lookup(key, ReadPolicy(min_validation=ValidationState.TESTS_PASSED), cache_root=cache))
            self.assertEqual(db.stats()['reusable_cache_entries'], 2)
        self.assertEqual((self.root / 'restored/stage/file').read_bytes(), b'first store')

    def test_corruption_retries_within_selected_store_without_retiring_other_store(self):
        cache = self.root / 'objects'
        first = self.publish_copy(cache, b'good')
        bad = self.publish_copy(cache, b'bad later copy')
        other = self.publish_copy(self.root / 'elsewhere', b'other store')
        (Path(bad.local_directory) / 'payload.tar').write_bytes(b'corrupt')
        with CacheDB(self.database) as db:
            restored = restore_local_key(db, ArtifactKey(IDENTITY, fixtures.FPRINT), cache, self.root / 'restored')
            self.assertEqual(restored['cache_entry_id'], first.cache_entry_id)
            self.assertEqual([e['cache_entry_id'] for e in restored['rejected_entries']], [bad.cache_entry_id])
            self.assertIsNotNone(db.get_manifest(other.cache_entry_id))
            self.assertIsNone(db.get_manifest(bad.cache_entry_id))


if __name__ == '__main__':
    unittest.main()
