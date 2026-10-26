"""Regression tests for the local collector against the checked-in relational schema.

All source fingerprints and payloads here are synthetic fixtures. These tests
exercise SQLite and filesystem ingestion, not actual ROCm builds or restores.

From TheRock's root:
    THEROCK_CACHE_SCHEMA=~/therock_fingerprint_cache.sql python3 -m unittest \
        discover -s build_tools/github_actions/tests -p artifact_cache_collector_test.py -v

Alternatively place the existing schema at
build_tools/therock_fingerprint_cache.sql.
"""

import contextlib
import concurrent.futures
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache import collector as artifact_cache


TOOLS_DIR = Path(__file__).resolve().parents[2]
SCHEMA = Path(
    os.environ.get("THEROCK_CACHE_SCHEMA", str(TOOLS_DIR / "therock_fingerprint_cache.sql"))
).expanduser()
FPRINT = "a" * 64


class CacheFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "build" / "artifacts"
        self.artifacts.mkdir(parents=True)
        self.manifest = self.root / "expected.json"
        self.database = self.root / "cache.db"
        if not SCHEMA.is_file():
            self.fail(
                f"Schema missing: {SCHEMA}. Set THEROCK_CACHE_SCHEMA to the existing SQL file."
            )
        self.db = sqlite3.connect(self.database)
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA.read_text(encoding="utf-8"))

    def outputs(self, *items):
        """Supply explicit expectations, including entries with no file."""
        expected = []
        for identity, fingerprint in items:
            filename = f"{identity}.fprint"
            expected.append({"artifact_identity": identity, "fingerprint_path": filename})
            if fingerprint is not None:
                (self.artifacts / filename).write_text(fingerprint + "\n", encoding="utf-8")
        self.manifest.write_text(
            json.dumps({"schema_version": 1, "artifacts": expected}), encoding="utf-8"
        )

    def collect(self, **overrides):
        arguments = dict(
            database=self.database,
            expected_manifest=self.manifest,
            artifacts_dir=self.artifacts,
            label="test build",
            commit="synthetic-commit-a",
            configure_state="succeeded",
        )
        arguments.update(overrides)
        return artifact_cache.collect(**arguments)

    def count(self, table):
        # Table identifiers are test-owned constants, never user input.
        return self.db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def seed_candidate(
        self, identity, fingerprint=FPRINT, *, state="ready", availability="available", expires_at=None, created_at=10,
        epoch=1,
    ):
        """Publish synthetic fixture metadata using the real schema constraints."""
        payload = b"synthetic artifact archive; not a ROCm build"
        location = self.root / f"candidate-{self.count('cache_entries')}"
        location.mkdir()
        (location / "artifact.tar").write_bytes(payload)
        with self.db:
            producer = self.db.execute(
                """INSERT INTO prototype_runs
                (label, repository_commit, configure_state, source_state)
                VALUES ('synthetic producer', 'synthetic-old-commit', 'succeeded', 'prepared')"""
            ).lastrowid
            self.db.execute(
                "INSERT INTO artifacts (artifact_identity) VALUES (?) ON CONFLICT DO NOTHING",
                (identity,),
            )
            artifact_id = self.db.execute(
                "SELECT artifact_id FROM artifacts WHERE artifact_identity = ?", (identity,)
            ).fetchone()[0]
            self.db.execute(
                """INSERT INTO artifact_keys (artifact_id, fingerprint, epoch)
                VALUES (?, ?, ?) ON CONFLICT DO NOTHING""",
                (artifact_id, fingerprint, epoch),
            )
            key_id = self.db.execute(
                """SELECT artifact_key_id FROM artifact_keys
                WHERE artifact_id = ? AND fingerprint = ? AND epoch = ?""",
                (artifact_id, fingerprint, epoch),
            ).fetchone()[0]
            entry_id = self.db.execute(
                """INSERT INTO cache_entries (artifact_key_id, producer_run_id,
                local_directory, expected_files_json, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (key_id, producer, str(location), '["artifact.tar"]', expires_at, created_at),
            ).lastrowid
            self.db.execute(
                """INSERT INTO artifact_files
                (cache_entry_id, relative_path, content_sha256, size_bytes)
                VALUES (?, 'artifact.tar', ?, ?)""",
                (entry_id, hashlib.sha256(payload).hexdigest(), len(payload)),
            )
            if state == "ready":
                self.db.execute(
                    """UPDATE cache_entries SET state = 'ready',
                    validation_state = 'build_passed', validated_at = 1
                    WHERE cache_entry_id = ?""", (entry_id,)
                )
            elif state == "invalid":
                self.db.execute(
                    "UPDATE cache_entries SET state = 'invalid' WHERE cache_entry_id = ?",
                    (entry_id,),
                )
            if availability != "available":
                self.db.execute(
                    "UPDATE artifact_files SET availability = ? WHERE cache_entry_id = ?",
                    (availability, entry_id),
                )
        return entry_id


class ArtifactCacheTest(CacheFixture):
    def test_first_collection_records_real_file_and_miss(self):
        self.outputs(("compiler_lib_generic", FPRINT))
        report = self.collect()
        self.assertEqual(report["summary"], {
            "expected": 1, "fingerprints_seen_before": 0, "hit": 0, "miss": 1, "uncacheable": 0,
        })
        row = self.db.execute("SELECT * FROM artifact_observations").fetchone()
        self.assertEqual(row["fingerprint_state"], "valid")
        self.assertEqual(row["lookup_result"], "miss")
        self.assertEqual(self.count("cache_entries"), 0)
        self.assertEqual(self.count("artifact_files"), 0)

    def test_repeated_import_reuses_key_but_does_not_invent_payload_hit(self):
        self.outputs(("compiler_lib_generic", FPRINT))
        first = self.collect()
        second = self.collect()
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(self.count("artifacts"), 1)
        self.assertEqual(self.count("artifact_keys"), 1)
        self.assertEqual(self.count("artifact_observations"), 2)
        self.assertEqual(second["summary"]["fingerprints_seen_before"], 1)
        self.assertEqual(second["summary"]["miss"], 1)

    def test_partial_hits_preserve_other_misses_and_uncacheable_outputs(self):
        entry = self.seed_candidate("compiler_lib_generic")
        self.outputs(
            ("compiler_lib_generic", FPRINT),
            ("blas_lib_gfx-test", "b" * 64),
            ("missing_lib_generic", None),
        )
        report = self.collect()
        self.assertEqual(
            [r["lookup_result"] for r in report["artifacts"]],
            ["hit", "miss", "uncacheable"],
        )
        self.assertEqual(report["artifacts"][0]["selected_entry_id"], entry)
        self.assertEqual(self.count("artifact_observations"), 3)

    def test_same_hash_does_not_merge_packaging_or_target_identities(self):
        self.seed_candidate("blas_lib_gfx-a")
        self.outputs(
            ("blas_lib_gfx-a", FPRINT),
            ("blas_dev_gfx-a", FPRINT),
            ("blas_lib_gfx-b", FPRINT),
        )
        self.assertEqual(
            [r["lookup_result"] for r in self.collect()["artifacts"]],
            ["hit", "miss", "miss"],
        )
        self.assertEqual(self.count("artifact_keys"), 3)

    def test_run_label_does_not_split_artifact_keys(self):
        self.outputs(("compiler_lib_generic", FPRINT))
        self.collect(label="first-experiment")
        report = self.collect(label="second-experiment")
        self.assertEqual(self.count("artifact_keys"), 1)
        self.assertTrue(report["artifacts"][0]["fingerprint_seen_before"])
        self.assertNotIn("key_namespace", report)

    def test_commit_is_provenance_not_part_of_lookup_key(self):
        self.seed_candidate("compiler_lib_generic")
        self.outputs(("compiler_lib_generic", FPRINT))
        self.assertEqual(self.collect(commit="another-commit")["summary"]["hit"], 1)

    def test_changed_fingerprint_misses(self):
        self.seed_candidate("compiler_lib_generic")
        self.outputs(("compiler_lib_generic", "b" * 64))
        self.assertEqual(self.collect()["summary"]["miss"], 1)

    def test_uppercase_hex_normalizes_to_same_key(self):
        self.seed_candidate("compiler_lib_generic")
        self.outputs(("compiler_lib_generic", FPRINT.upper()))
        report = self.collect()
        self.assertEqual(report["artifacts"][0]["fingerprint"], FPRINT)
        self.assertEqual(report["summary"]["hit"], 1)

    def test_missing_and_invalid_fingerprints_never_create_keys(self):
        self.outputs(
            ("missing", None), ("invalid", "not a fingerprint"),
            ("empty", ""), ("long", "a" * 4097),
        )
        report = self.collect()
        self.assertEqual(report["summary"]["uncacheable"], 4)
        self.assertEqual(self.count("artifact_keys"), 0)
        self.assertEqual(self.count("artifact_observations"), 4)
        self.assertEqual(report["artifacts"][0]["fingerprint_state"], "missing")
        self.assertTrue(all(r["detail"] for r in report["artifacts"]))

    def test_failed_or_unrun_configure_ignores_stale_valid_files(self):
        self.outputs(("compiler_lib_generic", FPRINT))
        for state in ("failed", "not_run"):
            with self.subTest(state=state):
                report = self.collect(configure_state=state)
                self.assertEqual(report["summary"]["uncacheable"], 1)
                self.assertEqual(report["artifacts"][0]["next_step"], "fix_configuration")
                self.assertIsNone(report["artifacts"][0]["fingerprint"])
        self.assertEqual(self.count("artifact_keys"), 0)

    def test_only_expected_files_are_ingested(self):
        self.outputs(("requested", FPRINT))
        (self.artifacts / "stale_extra.fprint").write_text("b" * 64)
        self.assertEqual(self.collect()["summary"]["expected"], 1)
        self.assertEqual(self.count("artifacts"), 1)

    def test_newer_ineligible_candidate_does_not_hide_older_hit(self):
        eligible = self.seed_candidate("compiler_lib_generic", created_at=10)
        self.seed_candidate("compiler_lib_generic", state="pending", created_at=20)
        self.seed_candidate("compiler_lib_generic", availability="corrupt", created_at=30)
        self.outputs(("compiler_lib_generic", FPRINT))
        self.assertEqual(self.collect()["artifacts"][0]["selected_entry_id"], eligible)

    def test_newest_eligible_candidate_wins(self):
        self.seed_candidate("compiler_lib_generic", created_at=10)
        latest = self.seed_candidate("compiler_lib_generic", created_at=20)
        self.outputs(("compiler_lib_generic", FPRINT))
        self.assertEqual(self.collect()["artifacts"][0]["selected_entry_id"], latest)

    def test_expired_invalid_pending_and_unavailable_candidates_miss(self):
        cases = {
            "expired": {"expires_at": 1}, "invalid": {"state": "invalid"},
            "pending": {"state": "pending"}, "missing": {"availability": "missing"},
            "corrupt": {"availability": "corrupt"},
        }
        for identity, options in cases.items():
            self.seed_candidate(identity, **options)
        self.outputs(*[(identity, FPRINT) for identity in cases])
        report = self.collect()
        self.assertEqual(report["summary"]["miss"], len(cases))
        self.assertEqual(report["summary"]["fingerprints_seen_before"], len(cases))

    def test_missing_inventory_row_invalidates_only_its_candidate(self):
        bad = self.seed_candidate("one")
        good = self.seed_candidate("two")
        with self.db:
            self.db.execute("DELETE FROM artifact_files WHERE cache_entry_id = ?", (bad,))
        self.outputs(("one", FPRINT), ("two", FPRINT))
        report = self.collect()
        self.assertEqual([r["lookup_result"] for r in report["artifacts"]], ["miss", "hit"])
        self.assertEqual(report["artifacts"][1]["selected_entry_id"], good)

    def test_metadata_hit_does_not_claim_physical_verification(self):
        entry = self.seed_candidate("compiler_lib_generic")
        directory = self.db.execute(
            "SELECT local_directory FROM cache_entries WHERE cache_entry_id = ?", (entry,)
        ).fetchone()[0]
        (Path(directory) / "artifact.tar").unlink()
        self.outputs(("compiler_lib_generic", FPRINT))
        report = self.collect()
        self.assertEqual(report["summary"]["hit"], 1)
        self.assertFalse(report["payloads_verified"])
        self.assertEqual(report["artifacts"][0]["next_step"], "verify_before_restore")

    def test_duplicate_identity_rejected_without_database_changes(self):
        self.outputs(("duplicate", FPRINT), ("duplicate", FPRINT))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.collect()
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_unsafe_manifest_paths_rejected(self):
        paths = (
            "../outside.fprint", "/tmp/outside.fprint",
            "dir/../file.fprint", "C:\\x.fprint",
        )
        for path in paths:
            with self.subTest(path=path):
                self.manifest.write_text(json.dumps({"schema_version": 1, "artifacts": [
                    {"artifact_identity": "test", "fingerprint_path": path}
                ]}))
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    self.collect()
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_symlink_outside_artifacts_directory_is_uncacheable(self):
        outside = self.root / "outside.fprint"
        outside.write_text(FPRINT)
        self.outputs(("link", None))
        try:
            (self.artifacts / "link.fprint").symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable on this host")
        report = self.collect()
        self.assertEqual(report["summary"]["uncacheable"], 1)
        self.assertEqual(self.count("artifact_keys"), 0)

    def test_sql_like_identity_is_bound_as_data(self):
        self.outputs(("safe_path", FPRINT))
        manifest = json.loads(self.manifest.read_text())
        identity = "value'); DROP TABLE artifacts; --"
        manifest["artifacts"][0]["artifact_identity"] = identity
        self.manifest.write_text(json.dumps(manifest))
        self.collect()
        stored_identity = self.db.execute(
            "SELECT artifact_identity FROM artifacts"
        ).fetchone()[0]
        self.assertEqual(stored_identity, identity)

    def test_no_database_created_on_typo(self):
        self.outputs(("compiler", FPRINT))
        missing = self.root / "does-not-exist.db"
        with self.assertRaises(sqlite3.OperationalError):
            self.collect(database=missing)
        self.assertFalse(missing.exists())

    def test_failure_rolls_back_entire_import(self):
        self.outputs(("compiler", FPRINT))
        self.db.executescript("""
        CREATE TRIGGER test_reject_observation BEFORE INSERT ON artifact_observations
        BEGIN SELECT RAISE(ABORT, 'injected write failure'); END;
        """)
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected write failure"):
            self.collect()
        for table in ("prototype_runs", "artifacts", "artifact_keys", "artifact_observations"):
            self.assertEqual(self.count(table), 0)

    def test_more_than_999_requests_use_batch_lookup(self):
        self.outputs(*[(f"artifact_{index}", FPRINT) for index in range(1100)])
        report = self.collect()
        self.assertEqual(report["summary"]["miss"], 1100)
        self.assertEqual(self.count("artifact_observations"), 1100)

    def test_timings_do_not_claim_source_configure_or_end_to_end_preflight(self):
        self.outputs(("compiler", FPRINT))
        report = self.collect(
            parameters={"build_type": "Release"},
            source_revisions={"llvm-project": "fixture-revision"},
        )
        row = self.db.execute("SELECT * FROM prototype_runs").fetchone()
        for key in ("source_prepare_ms", "cmake_configure_ms", "preflight_ms", "restore_ms"):
            self.assertIsNone(row[key])
        for key in ("fingerprint_collect_ms", "lookup_ms", "plan_ms"):
            self.assertGreaterEqual(row[key], 0)
        self.assertEqual(row["source_state"], "prepared")
        self.assertEqual(json.loads(row["parameters_json"]), {"build_type": "Release"})
        self.assertIsNone(report["timings_ms"]["preflight"])

    def cli_arguments(self):
        return [
            "collect", "--db", str(self.database), "--expected", str(self.manifest),
            "--artifacts-dir", str(self.artifacts), "--label", "cli fixture", "--commit", "fixture-sha", "--configure-state", "succeeded",
        ]

    def test_cli_returns_json(self):
        self.outputs(("compiler", FPRINT))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = artifact_cache.main(self.cli_arguments())
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())["summary"]["miss"], 1)

    def test_cli_failed_configure_is_recorded_but_returns_failure(self):
        self.outputs(("compiler", FPRINT))
        arguments = self.cli_arguments()
        arguments[-1] = "failed"
        with contextlib.redirect_stdout(io.StringIO()):
            code = artifact_cache.main(arguments)
        self.assertEqual(code, 1)
        self.assertEqual(self.count("prototype_runs"), 1)
        self.assertEqual(self.count("artifact_keys"), 0)

    def test_cli_malformed_input_has_clear_error_and_no_partial_run(self):
        self.manifest.write_text("not JSON")
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = artifact_cache.main(self.cli_arguments())
        self.assertEqual(code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("artifact-cache:", errors.getvalue())
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_json_rejects_duplicate_fields_and_non_finite_numbers(self):
        cases = (
            '{"schema_version": 1, "schema_version": 1, "artifacts": []}',
            '{"schema_version": 1, "artifacts": NaN}',
            '{"schema_version": 1, "artifacts": Infinity}',
            '{"schema_version": 1, "artifacts": 1e999}',
        )
        for contents in cases:
            with self.subTest(contents=contents):
                self.manifest.write_text(contents)
                with self.assertRaises(artifact_cache.InputError):
                    self.collect()
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_manifest_rejects_unknown_fields_and_invalid_versions(self):
        self.outputs(("compiler", FPRINT))
        original = json.loads(self.manifest.read_text())
        cases = []
        for version in (True, 1.0, "1", 2):
            cases.append({**original, "schema_version": version})
        cases.append({**original, "unknown": True})
        cases.append({**original, "artifacts": [
            {**original["artifacts"][0], "fingerprint_paths": "typo.fprint"}
        ]})
        for contents in cases:
            with self.subTest(contents=contents):
                self.manifest.write_text(json.dumps(contents))
                with self.assertRaises(artifact_cache.InputError):
                    self.collect()
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_non_json_metadata_fails_before_database_writes(self):
        self.outputs(("compiler", FPRINT))
        for parameters in ({"number": float("nan")}, {"bytes": b"invalid"}):
            with self.subTest(parameters=parameters):
                with self.assertRaises(artifact_cache.InputError):
                    self.collect(parameters=parameters)
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_non_ascii_fingerprint_is_uncacheable_without_rejecting_valid_output(self):
        self.outputs(("binary", None), ("valid", FPRINT))
        (self.artifacts / "binary.fprint").write_bytes(b"\xff" * 64)
        report = self.collect()
        self.assertEqual(report["summary"]["uncacheable"], 1)
        self.assertEqual(report["summary"]["miss"], 1)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX named pipes")
    def test_non_regular_fingerprint_does_not_wait_for_a_pipe_writer(self):
        self.outputs(("pipe", None))
        os.mkfifo(self.artifacts / "pipe.fprint")
        report = self.collect()
        self.assertEqual(report["summary"]["uncacheable"], 1)
        self.assertIn("regular file", report["artifacts"][0]["detail"])

    def test_missing_required_trigger_rejects_schema_without_writing(self):
        self.outputs(("compiler", FPRINT))
        self.db.execute("DROP TRIGGER artifact_keys_no_update")
        self.db.commit()
        with self.assertRaisesRegex(artifact_cache.SchemaError, "missing triggers"):
            self.collect()
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_incompatible_columns_are_reported_before_writing(self):
        self.outputs(("compiler", FPRINT))
        self.db.execute("ALTER TABLE artifact_observations RENAME COLUMN detail TO obsolete")
        self.db.commit()
        with self.assertRaisesRegex(artifact_cache.SchemaError, "missing detail"):
            self.collect()
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_ignored_observation_rolls_back_every_row(self):
        self.outputs(("compiler", FPRINT))
        self.db.executescript("""
        CREATE TRIGGER ignore_observation BEFORE INSERT ON artifact_observations
        BEGIN SELECT RAISE(IGNORE); END;
        """)
        with self.assertRaisesRegex(artifact_cache.SchemaError, "incomplete"):
            self.collect()
        for table in ("prototype_runs", "artifacts", "artifact_keys", "artifact_observations"):
            self.assertEqual(self.count(table), 0)

    def test_database_lock_timeout_does_not_leave_partial_data(self):
        self.outputs(("compiler", FPRINT))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                self.collect(database_timeout=0)
        finally:
            self.db.rollback()
        self.assertEqual(self.count("prototype_runs"), 0)
        self.assertEqual(self.collect()["summary"]["miss"], 1)

    def test_timeout_validation_rejects_non_finite_or_unbounded_waits(self):
        self.outputs(("compiler", FPRINT))
        for timeout in (True, "5", -1, float("nan"), float("inf"), 61):
            with self.subTest(timeout=timeout):
                with self.assertRaises(artifact_cache.InputError):
                    self.collect(database_timeout=timeout)
        self.assertEqual(self.count("prototype_runs"), 0)

    def test_concurrent_collectors_keep_unique_keys_and_complete_runs(self):
        self.outputs(("compiler", FPRINT))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.collect(), range(2)))
        self.assertEqual(self.count("prototype_runs"), 2)
        self.assertEqual(self.count("artifact_keys"), 1)
        self.assertEqual(self.count("artifact_observations"), 2)
        self.assertEqual(
            sorted(result["summary"]["fingerprints_seen_before"] for result in results),
            [0, 1],
        )

    def test_report_output_failure_identifies_the_committed_run(self):
        self.outputs(("compiler", FPRINT))
        output = mock.Mock()
        output.write.side_effect = OSError("output storage unavailable")
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = artifact_cache.main(self.cli_arguments())
        self.assertEqual(status, 3)
        self.assertEqual(self.count("prototype_runs"), 1)
        self.assertEqual(self.count("artifact_observations"), 1)
        self.assertIn("run_id=1 committed", errors.getvalue())

    def test_report_uses_versioned_public_run_identifier(self):
        self.outputs(("compiler", FPRINT))
        report = self.collect()
        self.assertEqual(report["report_version"], 3)
        self.assertIn("run_id", report)
        self.assertNotIn("prototype_run_id", report)
        self.assertEqual(report["lookup_scope"], "database_metadata")

    def test_empty_configured_output_set_records_an_empty_run(self):
        self.outputs()
        report = self.collect()
        self.assertEqual(report["summary"], {
            "expected": 0, "fingerprints_seen_before": 0,
            "hit": 0, "miss": 0, "uncacheable": 0,
        })
        self.assertEqual(self.count("prototype_runs"), 1)
        self.assertEqual(self.count("artifact_observations"), 0)


if __name__ == "__main__":
    unittest.main()
