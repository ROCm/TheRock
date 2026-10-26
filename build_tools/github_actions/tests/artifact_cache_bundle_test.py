#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Bundle membership, SQLite migration, real archive integrity and orchestration.

Most tests use real SQLite/filesystem storage with simulated build commands.
RealNinjaBundleTest additionally runs a miniature CMake/Ninja/Git project when
those tools are installed; no ROCm compiler or GPU is required.
"""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache import bundles, cli, inventory, local_cache, migrate_schema, publish
from build_tools.github_actions.artifact_cache.cache_db import CacheDB, SchemaError
from build_tools.github_actions.artifact_cache.local_store import store_bundle, unpack_component
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey
from build_tools.github_actions.artifact_cache.planner import plan_cache
from build_tools.github_actions.artifact_cache.reporting import log_report
from build_tools.github_actions.artifact_cache.restore import restore_local_key, restore_local_outputs
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures
from build_tools.github_actions.tests import artifact_cache_local_test as local

MEMBERS = ("rpp_dev_generic", "rpp_lib_generic")
BUNDLE = "rpp/bundle/generic"


class BundleFixture(fixtures.CacheFixture):
    def setUp(self):
        super().setUp()
        self.outputs(*[(name, fixtures.FPRINT) for name in MEMBERS])
        self.cache = self.root / "objects"
        for name in MEMBERS:
            export = self.artifacts / name
            (export / "stage").mkdir(parents=True)
            (export / "artifact_manifest.txt").write_text("stage\n")
            (export / "stage/payload").write_text(name)
        self.key = ArtifactKey(BUNDLE, fixtures.FPRINT, components=MEMBERS)

    def publish_fixture(self, key=None):
        key = key or self.key
        # Explicit synthetic build evidence: this helper tests the storage API,
        # not whether an arbitrary local directory was actually built.
        run = self.collect(parameters={"managed_build": {
            "command": ["synthetic-test"], "returncode": 0, "inputs_stable": True}})
        stored = store_bundle(self.artifacts, key, self.cache)
        with CacheDB(self.database) as db:
            return db.register_local_outputs(producer_run_id=run["run_id"],
                outputs=[(key, stored.directory, stored.files)])[0]


class BundleStorageTest(BundleFixture):
    def test_shared_raw_fingerprint_groups_components_and_topology(self):
        records = inventory.scan_build(self.artifacts.parent, self.manifest, granularity="bundle")
        self.assertEqual(len(records), 1)
        record, key = records[0]
        self.assertEqual(key, self.key)
        self.assertEqual(key.fingerprint, fixtures.FPRINT)
        self.assertEqual(record["components"], list(MEMBERS))
        class Topology:
            def get_build_order(self): return ["image-libs"]
            def get_produced_artifacts(self, stage): return {"rpp"}
            def get_artifacts_for_path(self, path): return {"rpp"}
        self.publish_fixture()
        with CacheDB(self.database) as db:
            topology = inventory.TopologyAdapter(Topology())
            plan = plan_cache(db=db, records=records, topology=topology)
            self.assertEqual(plan.entries[0].status, "candidate")
            self.assertFalse(plan.stage_plans[0].can_skip)
            self.assertEqual(plan.entries[0].unit.stage, "image-libs")
            forced = plan_cache(db=db, records=records, topology=topology, changed_paths=["rpp/change"])
            self.assertEqual(forced.entries[0].status, "miss")

    def test_different_artifacts_and_targets_never_merge_even_with_same_fingerprint(self):
        names = (*MEMBERS, "rpp_lib_gfx110X-all", "rpp_dev_gfx110X-all", "other_lib_generic")
        self.outputs(*[(name, fixtures.FPRINT) for name in names])
        records = inventory.scan_build(self.artifacts.parent, self.manifest, granularity="bundle")
        self.assertEqual({key.artifact_identity for _, key in records},
                         {BUNDLE, "rpp/bundle/gfx110X-all", "other/bundle/generic"})

    def test_missing_or_invalid_fingerprint_skips_the_entire_bundle(self):
        sidecar = self.artifacts / (MEMBERS[0] + ".fprint")
        for contents in (None, "invalid", "b" * 64):
            with self.subTest(contents=contents):
                if contents is None: sidecar.unlink()
                else: sidecar.write_text(contents)
                records = inventory.scan_build(self.artifacts.parent, self.manifest, granularity="bundle")
                self.assertIsNone(records[0][1])
                self.assertEqual(records[0][0]["components"], list(MEMBERS))

    def test_discovery_cannot_shrink_bundle_when_both_directory_and_sidecar_are_missing(self):
        (self.artifacts.parent / "CMakeCache.txt").write_text(
            "CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_MAKE_PROGRAM:FILEPATH=ninja\n")
        shutil.rmtree(self.artifacts / MEMBERS[0])
        (self.artifacts / (MEMBERS[0] + ".fprint")).unlink()
        output = "\n".join(f"artifacts/{name}/artifact_manifest.txt: CUSTOM_COMMAND" for name in MEMBERS)
        with patch.object(bundles.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
            record, key = inventory.scan_build(self.artifacts.parent, granularity="bundle")[0]
        self.assertIsNone(key)
        self.assertEqual(record["fingerprint_state"], "missing")
        self.assertEqual(record["components"], list(MEMBERS))

    def test_unknown_configured_membership_is_not_published_from_observed_subset(self):
        (self.artifacts.parent / "CMakeCache.txt").write_text("CMAKE_GENERATOR:INTERNAL=Other\n")
        record, key = inventory.scan_build(self.artifacts.parent, granularity="bundle")[0]
        self.assertIsNone(key)
        self.assertIn("complete component list is unknown", record["detail"])

    def test_ninja_target_paths_are_deduplicated_and_require_direct_component_manifests(self):
        (self.artifacts.parent / "CMakeCache.txt").write_text(
            "CMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_MAKE_PROGRAM:FILEPATH=ninja\n")
        lines = [f"artifacts/{MEMBERS[0]}/artifact_manifest.txt: CUSTOM_COMMAND",
                 f"{self.artifacts}/{MEMBERS[0]}/artifact_manifest.txt: CUSTOM_COMMAND",
                 f"artifacts/{MEMBERS[1]}/stage/artifact_manifest.txt: CUSTOM_COMMAND",
                 "other/artifact_manifest.txt: phony"]
        with patch.object(bundles.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "\n".join(lines), "")):
            self.assertEqual(bundles.declared_ninja_outputs(self.artifacts.parent), {MEMBERS[0]})

    def test_key_rejects_unsafe_mixed_duplicate_or_unsorted_members(self):
        for components in ((), MEMBERS[::-1], (MEMBERS[0], MEMBERS[0]),
                           ("../rpp_lib_generic",), ("other_lib_generic",),
                           ("rpp_lib_gfx110X-all",), [MEMBERS[0]]):
            with self.subTest(components=components), self.assertRaises(ValueError):
                ArtifactKey(BUNDLE, fixtures.FPRINT, components=components)

    def test_one_entry_two_objects_exact_members_and_separate_component_identity(self):
        manifest = self.publish_fixture()
        self.assertEqual(self.count("cache_entries"), 1)
        self.assertEqual(self.count("artifact_files"), 2)
        self.assertEqual(self.count("artifact_observations"), 3)
        self.assertEqual(manifest.key, self.key)
        with CacheDB(self.database) as db:
            partial = replace(self.key, components=(MEMBERS[1],))
            self.assertIsNone(db.lookup(partial))
            self.assertIsNone(db.lookup(replace(self.key, epoch=2)))
            self.assertIsNone(db.lookup(ArtifactKey(MEMBERS[1], fixtures.FPRINT)))
            self.assertEqual(db.lookup(self.key).key, self.key)
        objects = Path(manifest.local_directory)
        meta = json.loads((objects / "inventory.json").read_text())
        self.assertEqual(meta["schema_version"], 2)
        self.assertEqual(meta["components"], list(MEMBERS))
        target = self.root / "extracted"
        target.mkdir()
        unpack_component(objects, target, components=MEMBERS)
        self.assertEqual(sorted(p.name for p in target.iterdir()), list(MEMBERS))
        for name in MEMBERS:
            self.assertEqual((target / name / "stage/payload").read_text(), name)

    def test_partial_and_full_membership_coexist_as_distinct_keys(self):
        first = self.publish_fixture(replace(self.key, components=(MEMBERS[1],)))
        second = self.publish_fixture()
        with CacheDB(self.database) as db:
            found = db.lookup_batch([first.key, second.key])
        self.assertNotEqual(first.artifact_key_id, second.artifact_key_id)
        self.assertEqual(found[first.key].cache_entry_id, first.cache_entry_id)
        self.assertEqual(found[second.key].cache_entry_id, second.cache_entry_id)

    def test_missing_component_export_never_becomes_a_ready_bundle(self):
        shutil.rmtree(self.artifacts / MEMBERS[0])
        with self.assertRaises(FileNotFoundError): self.publish_fixture()
        self.assertEqual(self.count("cache_entries"), 0)

    def test_publication_requires_every_component_observed_with_the_same_fingerprint(self):
        (self.artifacts / (MEMBERS[0] + ".fprint")).write_text("b" * 64)
        with self.assertRaisesRegex(ValueError, "every bundle component"):
            self.publish_fixture()
        self.assertEqual(self.count("cache_entries"), 0)
        self.assertEqual(self.count("artifact_files"), 0)
        self.assertEqual(self.db.execute("SELECT count(*) FROM artifacts WHERE artifact_identity=?", (BUNDLE,)).fetchone()[0], 0)

    def test_corrupt_newest_bundle_retries_older_whole_bundle(self):
        older = self.publish_fixture()
        newer = self.publish_fixture()
        (Path(newer.local_directory) / "payload.tar").write_bytes(b"corrupt")
        dest = self.root / "restored"
        with CacheDB(self.database) as db:
            result = restore_local_key(db, self.key, self.cache, dest)
            self.assertEqual(result["cache_entry_id"], older.cache_entry_id)
            self.assertIsNone(db.get_manifest(newer.cache_entry_id))
        self.assertEqual(sorted(p.name for p in dest.iterdir()), list(MEMBERS))

    def test_incomplete_archive_or_membership_mismatch_is_rejected(self):
        stored = store_bundle(self.artifacts, self.key, self.cache)
        with self.assertRaises(ValueError): unpack_component(stored.directory, None, components=(MEMBERS[1],))
        with tarfile.open(stored.directory / "payload.tar", "w"):
            pass
        with self.assertRaisesRegex(ValueError, "missing required"):
            unpack_component(stored.directory, None, components=MEMBERS)

    @unittest.skipIf(os.name == "nt", "symlink privileges differ on Windows")
    def test_links_may_not_escape_one_component_into_another(self):
        (self.artifacts / MEMBERS[0] / "stage/link").symlink_to(f"../../{MEMBERS[1]}/stage/payload")
        with self.assertRaises(ValueError): store_bundle(self.artifacts, self.key, self.cache)

    def test_terminal_report_has_one_line_per_bundle_with_members_and_missing_reason(self):
        output = io.StringIO()
        with redirect_stderr(output):
            log_report({"operation": "cache", "granularity": "bundle", "summary": {
                "stored": 1, "verified_cached": 0, "uncacheable": 1}, "artifacts": [
                    {"artifact_identity": BUNDLE, "components": list(MEMBERS), "outcome": "stored"},
                    {"artifact_identity": "blas/bundle/gfx1100", "components": ["blas_lib_gfx1100"],
                     "outcome": "uncacheable", "fingerprint_state": "missing",
                     "reason": "blas_lib_gfx1100: expected fingerprint file is missing"}]})
        text = output.getvalue()
        self.assertEqual(text.count("MISS -> STORED rpp/bundle/generic"), 1)
        self.assertIn("components=dev,lib", text)
        self.assertIn("NO_FINGERPRINT blas/bundle/gfx1100", text)
        self.assertIn("units=bundles", text)


class BundleManagedFlowTest(BundleFixture):
    def setUp(self):
        super().setUp()
        self.build = self.artifacts.parent
        self.source = self.root / "source"
        self.source.mkdir()
        self.input = self.source / "input"
        self.input.write_text("version one")
        self.builds = 0
        self.fail = False
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={self.source}\nCMAKE_COMMAND:INTERNAL=fixture-cmake\n"
            "CMAKE_BUILD_TYPE:STRING=Release\n")
        def revisions(_): return {".": hashlib.sha256(self.input.read_bytes()).hexdigest()}
        def command(args, **kwargs):
            if "--build" in args:
                self.builds += 1
                if self.fail: raise subprocess.CalledProcessError(17, args)
                for name in MEMBERS:
                    (self.artifacts / name / "stage/payload").write_bytes(self.input.read_bytes())
            else:
                fp = hashlib.sha256(self.input.read_bytes()).hexdigest()
                for name in MEMBERS: (self.artifacts / (name + ".fprint")).write_text(fp)
        for mocked in (patch.object(publish, "_source_revisions", side_effect=revisions),
                       patch.object(publish, "_run", side_effect=command)):
            mocked.start()
            self.addCleanup(mocked.stop)

    def cache_command(self):
        return local_cache.cache_build(database=self.database, build=self.build, cache=self.cache,
                                       expected=self.manifest, granularity="bundle")

    def test_cold_warm_restore_and_source_change_use_whole_bundles(self):
        cold = self.cache_command()
        self.assertEqual(cold["summary"], {"stored": 1, "verified_cached": 0, "uncacheable": 0})
        warm = self.cache_command()
        self.assertFalse(warm["build_executed"])
        self.assertEqual(warm["summary"]["verified_cached"], 1)
        self.assertEqual(self.builds, 1)
        destination = self.root / "restored"
        report = restore_local_outputs(database=self.database, expected_manifest=self.manifest,
            build_dir=self.build, cache_root=self.cache, destination=destination, granularity="bundle")
        self.assertEqual(report["restored"], 1)
        self.assertEqual(sorted(p.name for p in destination.iterdir()), list(MEMBERS))
        for name in MEMBERS: self.assertEqual((destination / name / "stage/payload").read_text(), "version one")
        self.input.write_text("version two")
        changed = self.cache_command()
        self.assertEqual(changed["summary"]["stored"], 1)
        self.assertEqual(self.builds, 2)
        self.assertEqual(self.count("cache_entries"), 2)
        with CacheDB(self.database) as db:
            self.assertEqual({m.key.fingerprint for m in db.all_manifests()},
                             {hashlib.sha256(x.encode()).hexdigest() for x in ("version one", "version two")})

    def test_failed_build_after_configure_cannot_publish_stale_bundle(self):
        self.cache_command()
        self.input.write_text("new inputs")
        self.fail = True
        with self.assertRaises(subprocess.CalledProcessError): self.cache_command()
        self.assertEqual(self.count("cache_entries"), 1)

    def test_cli_defaults_to_bundle_and_keeps_json_separate(self):
        output, logs = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(logs):
            code = local_cache.main(["cache", "--db", str(self.database), "--cache", str(self.cache),
                "--build", str(self.build), "--expected", str(self.manifest)])
        self.assertEqual(code, 0, logs.getvalue())
        report = json.loads(output.getvalue())
        self.assertEqual(report["granularity"], "bundle")
        self.assertEqual(report["artifacts"][0]["artifact_identity"], BUNDLE)
        self.assertEqual(logs.getvalue().count("MISS -> STORED rpp/bundle/generic"), 1)


V5_KEYS = """CREATE TABLE artifact_keys (
    artifact_key_id INTEGER PRIMARY KEY,
    artifact_id INTEGER NOT NULL REFERENCES artifacts(artifact_id),
    epoch INTEGER NOT NULL CHECK(typeof(epoch)='integer' AND epoch>0),
    fingerprint TEXT NOT NULL CHECK(length(fingerprint)=64 AND fingerprint NOT GLOB '*[^0-9a-f]*'),
    UNIQUE(epoch,artifact_id,fingerprint), UNIQUE(artifact_key_id,artifact_id)
);"""


class BundleMigrationTest(fixtures.CacheFixture):
    def setUp(self):
        super().setUp()
        self.old = self.root / "v5.db"
        self.backup = self.root / "backup.db"
        text = fixtures.SCHEMA.read_text()
        start = text.index("CREATE TABLE artifact_keys (")
        end = text.index("\n);", start) + 3
        with sqlite3.connect(self.old) as db:
            db.executescript(text[:start] + V5_KEYS + text[end:])
            db.execute("PRAGMA user_version=5")
            db.execute("INSERT INTO artifacts VALUES(9,'rpp_lib_generic')")
            db.execute("INSERT INTO artifact_keys VALUES(17,9,2,?)", (fixtures.FPRINT,))
            db.execute("INSERT INTO prototype_runs(prototype_run_id,label,repository_commit,configure_state,source_state) "
                       "VALUES(19,'old producer','old-commit','succeeded','prepared')")
            db.execute("INSERT INTO cache_entries(cache_entry_id,artifact_key_id,producer_run_id,local_directory,expected_files_json) "
                       "VALUES(27,17,19,?,'[\"payload.tar\"]')", (str(self.root / "old-objects"),))
            db.execute("INSERT INTO artifact_files VALUES(27,'payload.tar',?,42,'available')", (fixtures.FPRINT,))
            db.execute("UPDATE cache_entries SET state='ready',validation_state='build_passed',validated_at=1 WHERE cache_entry_id=27")
            db.execute("INSERT INTO artifact_observations(prototype_run_id,artifact_id,artifact_key_id,fingerprint_state,lookup_result,selected_entry_id) "
                       "VALUES(19,9,17,'valid','hit',27)")
        with sqlite3.connect(self.old) as db: self.before = list(db.iterdump())

    def test_v5_migration_preserves_key_ids_and_assigns_empty_component_list(self):
        with self.assertRaisesRegex(SchemaError, "bundle membership migration"): CacheDB(self.old)
        result = migrate_schema.migrate(self.old, self.backup)
        self.assertEqual(result["schema_version"], 6)
        self.assertEqual(result["merged_keys"], 0)
        with sqlite3.connect(self.backup) as db: self.assertEqual(list(db.iterdump()), self.before)
        with CacheDB(self.old) as db:
            row = db._conn.execute("SELECT * FROM artifact_keys").fetchone()
            self.assertEqual(tuple(row), (17, 9, 2, fixtures.FPRINT, "[]"))
            self.assertEqual(db._conn.execute("PRAGMA foreign_key_check").fetchall(), [])
            manifest = db.lookup(ArtifactKey("rpp_lib_generic", fixtures.FPRINT, epoch=2))
            self.assertEqual(manifest.cache_entry_id, 27)
            self.assertEqual(manifest.key.components, ())
            self.assertEqual(manifest.local_directory, str(self.root / "old-objects"))
            for table in ("artifacts", "prototype_runs", "artifact_observations", "cache_entries", "artifact_files"):
                with sqlite3.connect(self.backup) as old:
                    self.assertEqual([tuple(r) for r in db._conn.execute(f"SELECT * FROM {table}")],
                                     old.execute(f"SELECT * FROM {table}").fetchall())

    def test_failure_rolls_back_v5_schema_and_preserves_backup(self):
        validate = migrate_schema._validate_schema
        def fail_final(db, **kwargs):
            if not kwargs: raise SchemaError("injected after rebuild")
            return validate(db, **kwargs)
        with patch.object(migrate_schema, "_validate_schema", side_effect=fail_final):
            with self.assertRaisesRegex(SchemaError, "injected"):
                migrate_schema.migrate(self.old, self.backup)
        for path in (self.old, self.backup):
            with sqlite3.connect(path) as db: self.assertEqual(list(db.iterdump()), self.before)


@unittest.skipUnless(all(shutil.which(x) for x in ("cmake", "ninja", "git")),
                     "requires CMake, Ninja and Git")
class RealNinjaBundleTest(unittest.TestCase):
    setUp = local.LocalCacheBuildTest.setUp
    run_command = local.LocalCacheBuildTest.run_command
    git = local.LocalCacheBuildTest.git
    commit = local.LocalCacheBuildTest.commit
    configure = local.LocalCacheBuildTest.configure

    def test_real_declared_two_component_bundle_cold_warm_and_restore(self):
        # Extend the existing miniature source project, not TheRock's CMake.
        cmake = self.source / "CMakeLists.txt"
        with cmake.open("a") as stream:
            stream.write('''
file(WRITE "${CMAKE_BINARY_DIR}/artifacts/sample_dev_generic.fprint" "${fingerprint}")
add_custom_command(TARGET exports POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${CMAKE_BINARY_DIR}/artifacts/sample_lib_generic"
    "${CMAKE_BINARY_DIR}/artifacts/sample_dev_generic"
  BYPRODUCTS
    "${CMAKE_BINARY_DIR}/artifacts/sample_lib_generic/artifact_manifest.txt"
    "${CMAKE_BINARY_DIR}/artifacts/sample_dev_generic/artifact_manifest.txt")
''')
        self.commit()
        self.configure()
        args = dict(database=self.database, build=self.build, cache=self.cache, granularity="bundle")
        cold = local_cache.cache_build(**args)
        self.assertEqual(cold["summary"]["stored"], 1)
        self.assertEqual(cold["artifacts"][0]["components"], ["sample_dev_generic", "sample_lib_generic"])
        warm = local_cache.cache_build(**args)
        self.assertEqual(warm["summary"]["verified_cached"], 1)
        self.assertFalse(warm["build_executed"])
        self.assertEqual((self.build / "build-count").read_text(), "1")
        self.expected.write_text(json.dumps({"schema_version": 1, "artifacts": [
            {"artifact_identity": name, "fingerprint_path": name + ".fprint"}
            for name in ("sample_dev_generic", "sample_lib_generic")]}))
        restored = restore_local_outputs(database=self.database, expected_manifest=self.expected,
            build_dir=self.build, cache_root=self.cache, destination=self.dest, granularity="bundle")
        self.assertEqual(restored["restored"], 1)
        for name in ("sample_dev_generic", "sample_lib_generic"):
            self.assertEqual((self.dest / name / "stage/bin/tool").read_text(), "version one\n")


if __name__ == "__main__":
    unittest.main()
