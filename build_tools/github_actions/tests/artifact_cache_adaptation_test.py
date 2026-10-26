#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Relational planner, history, policy, CLI and managed local workflow tests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB, ReadPolicy
from build_tools.github_actions.artifact_cache.cli import main
from build_tools.github_actions.artifact_cache.history_ingest import ingest_runs, trust_for_run
from build_tools.github_actions.artifact_cache.inventory import TopologyAdapter, scan_build
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey, OutputObject, TrustLevel, ValidationState
from build_tools.github_actions.artifact_cache.planner import plan_cache
from build_tools.github_actions.artifact_cache.replay import replay
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures
from build_tools.github_actions.tests import artifact_cache_local_test as local


class Topology:
    def get_build_order(self): return ["compiler", "base"]
    def get_produced_artifacts(self, stage): return [stage]
    def get_artifacts_for_path(self, path): return ["base"] if path == "base/file" else []


def history_run(number, commit="a", start="00:00:00", end="00:10:00", fingerprint=None, event="push"):
    result = {"id": number, "head_sha": commit, "head_branch": "main", "event": event,
              "status": "completed", "conclusion": "success",
              "created_at": "2026-01-01T" + start + "Z", "completed_at": "2026-01-01T" + end + "Z"}
    if fingerprint:
        result["source_fingerprints"] = {"compiler_lib_generic": fingerprint}
    return result


def lister(run):
    return [OutputObject("compiler_lib_generic.tar.zst", f"s3://history/{run['id']}/compiler_lib_generic.tar.zst")]


class AdaptationTest(fixtures.CacheFixture):
    def key(self, name="compiler_lib_generic"):
        return ArtifactKey(name, fixtures.FPRINT)

    def mark_trust(self, entry, trust):
        with self.db:
            self.db.execute("UPDATE prototype_runs SET parameters_json=? WHERE prototype_run_id="
                            "(SELECT producer_run_id FROM cache_entries WHERE cache_entry_id=?)",
                            (json.dumps({"cache_provenance": {"trust": trust}}), entry))

    def test_policy_filters_before_ranking_and_absent_trust_is_untrusted(self):
        older = self.seed_candidate("compiler_lib_generic", created_at=10)
        self.mark_trust(older, "trusted")
        newer = self.seed_candidate("compiler_lib_generic", created_at=20)
        with CacheDB(self.database) as db:
            self.assertEqual(db.lookup(self.key()).cache_entry_id, newer)
            self.assertEqual(db.lookup(self.key(), ReadPolicy()).cache_entry_id, older)
            self.assertEqual(db.get_manifest(newer).trust, TrustLevel.UNTRUSTED)

    def test_validation_and_time_policy_do_not_hide_older_candidate(self):
        old = self.seed_candidate("compiler_lib_generic", created_at=10)
        self.seed_candidate("compiler_lib_generic", created_at=20)
        with self.db:
            self.db.execute("UPDATE cache_entries SET validation_state='tests_passed' WHERE cache_entry_id=?", (old,))
        with CacheDB(self.database) as db:
            policy = ReadPolicy(min_trust=TrustLevel.UNTRUSTED, min_validation=ValidationState.TESTS_PASSED)
            self.assertEqual(db.lookup(self.key(), policy).cache_entry_id, old)
            policy = ReadPolicy(min_trust=TrustLevel.UNTRUSTED, created_before="1970-01-01T00:00:15Z")
            self.assertEqual(db.lookup(self.key(), policy).cache_entry_id, old)
            with self.assertRaisesRegex(ValueError, "timezone"):
                ReadPolicy(created_before="2026-01-01")

    def test_planner_components_are_candidates_and_never_unverified_stage_skips(self):
        self.seed_candidate("compiler_lib_generic")
        records = [({"artifact_identity": name}, self.key(name))
                   for name in ("compiler_lib_generic", "compiler_dev_generic")]
        with CacheDB(self.database) as db:
            plan = plan_cache(db=db, records=records, topology=TopologyAdapter(Topology()))
        self.assertEqual([e.status for e in plan.entries], ["candidate", "miss"])
        self.assertFalse(plan.stage_plans[0].can_skip)
        self.assertFalse(plan.entries[0].is_hit)
        self.assertFalse(plan.to_dict()["payloads_verified"])

    def test_planner_retains_unmapped_and_uncacheable_outputs(self):
        with CacheDB(self.database) as db:
            plan = plan_cache(db=db, records=[({"artifact_identity": "unknown_lib_generic"}, None)],
                              topology=TopologyAdapter(Topology()))
        self.assertEqual(plan.entries[0].status, "uncacheable")
        self.assertEqual(plan.entries[0].unit.stage, "unmapped")

    def test_planner_unmapped_change_is_conservative_and_duplicate_rejected(self):
        self.seed_candidate("compiler_lib_generic")
        record = ({"artifact_identity": "compiler_lib_generic"}, self.key())
        with CacheDB(self.database) as db:
            plan = plan_cache(db=db, records=[record], topology=TopologyAdapter(Topology()), changed_paths=["unknown"])
            self.assertEqual(plan.entries[0].status, "miss")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                plan_cache(db=db, records=[record, record])

    def test_history_import_idempotent_and_never_publishes_payloads(self):
        with CacheDB(self.database) as db:
            first = ingest_runs(db=db, runs=[history_run(1)], artifact_lister=lister)
            repeat = ingest_runs(db=db, runs=[history_run(1)], artifact_lister=lister)
            self.assertEqual(first.runs_indexed, 1)
            self.assertEqual(repeat.runs_duplicate, 1)
            self.assertEqual(db.stats()["artifact_keys"], 0)
            self.assertEqual(db.stats()["cache_entries"], 0)
            self.assertEqual(db.stats()["artifact_observations"], 0)
            self.assertTrue(db.history()[0]["units"][0]["fingerprint"].startswith("srcfp1:"))

    def test_history_requires_completion_and_scopes_repository_trust(self):
        unknown = history_run(1)
        del unknown["completed_at"]
        fork = history_run(2)
        fork["head_repository"] = {"full_name": "someone/fork"}
        with CacheDB(self.database) as db:
            result = ingest_runs(db=db, runs=[unknown, fork], artifact_lister=lister, require_trusted=True)
            self.assertEqual(result.runs_skipped, 2)
            self.assertEqual(db.history(), [])

    def test_history_trust_requires_main_push_or_schedule(self):
        for branch in ("main", "release/10.2", "feature/cache", "main-extra", "", None):
            for event in ("push", "schedule", "pull_request", "pull_request_target",
                          "workflow_dispatch", "workflow_run", "", None):
                with self.subTest(branch=branch, event=event):
                    run = history_run(1, event=event)
                    run["head_branch"] = branch
                    run["head_repository"] = {"full_name": "ROCm/TheRock"}
                    expected = (TrustLevel.TRUSTED if branch == "main" and
                                event in ("push", "schedule") else TrustLevel.UNTRUSTED)
                    self.assertEqual(trust_for_run(run, "ROCm/TheRock"), expected)
        self.assertEqual(trust_for_run({}, "ROCm/TheRock"), TrustLevel.UNTRUSTED)
        for event in ("push", "schedule"):
            with self.subTest(foreign_repository=True, event=event):
                run = history_run(1, event=event)
                run["head_repository"] = {"full_name": "someone/fork"}
                self.assertEqual(trust_for_run(run, "ROCm/TheRock"), TrustLevel.UNTRUSTED)

    def test_trusted_history_import_skips_other_branches_and_events_before_listing(self):
        specs = [("main", "push"), ("main", "schedule"), ("release/10.2", "push"),
                 ("release/10.2", "schedule"), ("feature/cache", "push"),
                 ("main", "pull_request"), ("main", "workflow_dispatch")]
        runs = []
        for number, (branch, event) in enumerate(specs, start=1):
            run = history_run(number, event=event)
            run["head_branch"] = branch
            runs.append(run)
        listed = []

        def record_listing(run):
            listed.append(run["id"])
            return lister(run)

        with CacheDB(self.database) as db:
            result = ingest_runs(db=db, runs=runs, artifact_lister=record_listing,
                                 require_trusted=True)
            self.assertEqual(result.runs_indexed, 2)
            self.assertEqual(result.runs_skipped, 5)
            self.assertEqual(listed, [1, 2])
            self.assertEqual({r["run_id"]: r["trust"] for r in db.history()},
                             {"1": "trusted", "2": "trusted"})
            self.assertEqual(db.stats()["cache_entries"], 0)

    def test_release_history_is_untrusted_evidence_when_filter_is_disabled(self):
        run = history_run(1)
        run["head_branch"] = "release/10.2"
        with CacheDB(self.database) as db:
            result = ingest_runs(db=db, runs=[run], artifact_lister=lister)
            self.assertEqual(result.runs_indexed, 1)
            self.assertEqual(db.history()[0]["trust"], "untrusted")
            self.assertEqual(db.stats()["cache_entries"], 0)

    def test_history_invalid_raw_fingerprint_has_no_partial_run(self):
        with CacheDB(self.database) as db:
            with self.assertRaises(ValueError):
                ingest_runs(db=db, runs=[history_run(1, fingerprint="invalid")], artifact_lister=lister)
            self.assertEqual(db.stats()["prototype_runs"], 0)

    def test_replay_cross_commit_reuse_requires_matching_real_fingerprints(self):
        runs = [history_run(1, fingerprint=fixtures.FPRINT),
                history_run(2, commit="b", start="00:20:00", end="00:30:00", fingerprint=fixtures.FPRINT)]
        with CacheDB(self.database) as db:
            ingest_runs(db=db, runs=runs, artifact_lister=lister)
            report = replay(db.history()).to_dict()
        self.assertEqual(report["potential_hits"], 1)
        self.assertEqual(report["same_commit_hits"], 0)
        self.assertFalse(report["payloads_verified"])

    def test_replay_no_lookahead_for_overlapping_runs_or_failed_producers(self):
        runs = [history_run(1, end="00:50:00"), history_run(2, start="00:20:00", end="00:30:00"),
                history_run(3, start="00:35:00", end="00:40:00")]
        runs[1]["conclusion"] = "failure"
        with CacheDB(self.database) as db:
            ingest_runs(db=db, runs=list(reversed(runs)), artifact_lister=lister)
            self.assertEqual(replay(db.history()).to_dict()["potential_hits"], 0)

    def test_replay_scopes_platform_and_counts_only_produced_components(self):
        first, second = history_run(1), history_run(2, start="00:20:00", end="00:30:00")
        with CacheDB(self.database) as db:
            ingest_runs(db=db, runs=[first], artifact_lister=lister, platform="linux")
            ingest_runs(db=db, runs=[second], artifact_lister=lister, platform="windows")
            self.assertEqual(replay(db.history()).to_dict()["potential_hits"], 0)
            ingest_runs(db=db, runs=[second], artifact_lister=lambda r: lister(r) + [
                OutputObject("compiler_dev_generic.tar.zst", "s3://history/dev")], platform="linux")
            report = replay(db.history()).to_dict()
            self.assertEqual(report["potential_hits"], 1)
            self.assertEqual(report["same_commit_hits"], 1)

    def test_cli_ingest_replay_lookup_and_both_module_entrypoints(self):
        runs = self.root / "runs.json"
        run = history_run(1)
        run["artifacts"] = [o.to_dict() for o in lister(run)]
        runs.write_text(json.dumps([run]))
        for args in (["ingest", "--runs-json", str(runs)], ["replay"], ["stats"]):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main([*args, "--db", str(self.database)]), 0)
            json.loads(stdout.getvalue())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["lookup", "--db", str(self.database), "--artifact", "absent_lib_generic", "--fingerprint", fixtures.FPRINT]), 1)
        root = Path(__file__).resolve().parents[3]
        for module, cwd in (("build_tools.github_actions.artifact_cache", root),
                            ("artifact_cache", root / "build_tools/github_actions")):
            result = subprocess.run([sys.executable, "-m", module, "stats", "--db", str(self.database)],
                                    cwd=cwd, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("history_runs", json.loads(result.stdout))

    def test_eviction_preserves_observations_and_payloads(self):
        entry = self.seed_candidate("compiler_lib_generic")
        self.outputs(("compiler_lib_generic", fixtures.FPRINT))
        self.collect()
        with CacheDB(self.database) as db:
            directory = Path(db.get_manifest(entry).local_directory)
            self.assertEqual(db.evict_before("1970-01-01T00:00:20Z"), 1)
            self.assertIsNone(db.lookup(self.key()))
            self.assertEqual(db.stats()["artifact_observations"], 1)
        self.assertTrue((directory / "artifact.tar").exists())

    def test_scan_reports_missing_and_invalid_separately_without_split_assumption(self):
        build = self.root / "build"
        artifacts = build / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "compiler_lib_generic.fprint").write_text(fixtures.FPRINT)
        (artifacts / "dirty_lib_generic.fprint").write_text("")
        (artifacts / "missing_lib_generic").mkdir()
        records = {r["artifact_identity"]: (r, key) for r, key in scan_build(build)}
        self.assertIsNotNone(records["compiler_lib_generic"][1])
        self.assertIsNone(records["dirty_lib_generic"][1])
        self.assertIsNone(records["missing_lib_generic"][1])

    def test_local_scan_rejects_opaque_identity_as_a_restore_path(self):
        expected = self.root / "unsafe-expected.json"
        expected.write_text(json.dumps({"schema_version": 1, "artifacts": [
            {"artifact_identity": "nested/component", "fingerprint_path": "safe.fprint"}]}))
        with self.assertRaisesRegex(ValueError, "emitted component"):
            scan_build(self.root / "build", expected_manifest=expected)

    def test_legacy_store_writes_do_not_overwrite_older_objects(self):
        from build_tools.github_actions.artifact_cache.local_store import LocalObjectStore
        store = LocalObjectStore(self.root / "objects")
        payload = self.root / "payload"
        payload.write_bytes(b"first")
        first = store.upload_file(payload, "artifact.tar")
        payload.write_bytes(b"second")
        second = store.upload_file(payload, "artifact.tar")
        self.assertNotEqual(first.uri, second.uri)
        self.assertEqual(store.path_for(first.uri).read_bytes(), b"first")
        with self.assertRaises(ValueError):
            store.path_for("local://../outside")

    def test_restore_simulation_rejects_unknown_digests(self):
        from build_tools.github_actions.artifact_cache.restore import InMemoryObjectStore, restore_manifest
        from build_tools.github_actions.artifact_cache.manifest import Manifest, Provenance, PublicationState
        from build_tools.github_actions.artifact_cache.cache_key import ArtifactIdentity
        manifest = Manifest("model", ArtifactIdentity("a", "lib", "generic"),
                            (OutputObject("a", "s3://a", "a" * 64),),
                            Provenance("repo", "1", "main", "sha", "wf", "date"),
                            publication=PublicationState.PUBLISHED)
        self.assertFalse(restore_manifest(manifest, InMemoryObjectStore()).reusable)



@unittest.skipUnless(local.shutil.which("cmake") and local.shutil.which("ninja") and local.shutil.which("git"),
                     "requires CMake, Ninja and Git")
class AdaptedLocalWorkflowTest(unittest.TestCase):
    # Reuse setup only, rather than rerunning all the base class's tests.
    setUp = local.LocalCacheBuildTest.setUp
    run_command = local.LocalCacheBuildTest.run_command
    git = local.LocalCacheBuildTest.git
    commit = local.LocalCacheBuildTest.commit
    configure = local.LocalCacheBuildTest.configure

    def cache_command(self):
        from build_tools.github_actions.artifact_cache.local_cache import cache_build
        return cache_build(database=self.database, build=self.build, cache=self.cache,
                           expected=self.expected)

    def test_cold_warm_and_changed_keys_require_real_build_before_publication(self):
        cold = self.cache_command()
        self.assertEqual(cold["summary"]["stored"], 1)
        count = (self.build / "build-count").read_text()
        warm = self.cache_command()
        self.assertEqual(warm["summary"]["verified_cached"], 1)
        self.assertFalse(warm["build_executed"])
        self.assertEqual((self.build / "build-count").read_text(), count)
        (self.source / "input.txt").write_text("new bytes\n")
        subprocess.run(["git", "-C", str(self.source), "commit", "-am", "Change input"], check=True, capture_output=True)
        changed = self.cache_command()
        self.assertTrue(changed["build_executed"])
        self.assertEqual(changed["summary"]["stored"], 1)
        with CacheDB(self.database) as db:
            self.assertEqual(db.stats()["cache_entries"], 2)

    def test_corrupt_warm_object_rebuilds_and_replaces_only_that_production(self):
        self.cache_command()
        with CacheDB(self.database) as db:
            manifest = db.all_manifests()[0]
        (Path(manifest.local_directory) / "payload.tar").write_bytes(b"corrupt")
        result = self.cache_command()
        self.assertTrue(result["build_executed"])
        with CacheDB(self.database) as db:
            self.assertIsNone(db.get_manifest(manifest.cache_entry_id))
            self.assertEqual(db.stats()["reusable_cache_entries"], 1)

    def test_failed_build_after_key_change_does_not_publish_stale_export(self):
        self.cache_command()
        (self.source / "input.txt").write_text("changed\n")
        (self.source / "fail").touch()
        subprocess.run(["git", "-C", str(self.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.source), "commit", "-m", "Fail build"], check=True, capture_output=True)
        with self.assertRaises(subprocess.CalledProcessError):
            self.cache_command()
        with CacheDB(self.database) as db:
            self.assertEqual(db.stats()["cache_entries"], 1)

    def test_restore_policy_refuses_untrusted_then_accepts_explicit_provenance(self):
        from build_tools.github_actions.artifact_cache.restore import restore_local_key
        self.cache_command()
        with CacheDB(self.database) as db:
            manifest = db.all_manifests()[0]
            result = restore_local_key(db, manifest.key, self.cache, self.dest, policy=ReadPolicy())
            self.assertEqual(result["outcome"], "needs_build")
            self.assertFalse(self.dest.exists())
            with db.transaction():
                db._conn.execute("UPDATE prototype_runs SET parameters_json = json_set(parameters_json, "
                                 "'$.cache_provenance.trust', 'trusted') WHERE prototype_run_id=?",
                                 (manifest.producer_run_id,))
            result = restore_local_key(db, manifest.key, self.cache, self.dest, policy=ReadPolicy())
            self.assertEqual(result["outcome"], "restored")

    def test_package_plan_and_restore_share_the_relational_keys(self):
        self.cache_command()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["plan", "--granularity", "component", "--db", str(self.database), "--build", str(self.build),
                         "--expected", str(self.expected), ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["summary"]["candidate"], 1)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["restore", "--granularity", "component", "--db", str(self.database), "--build", str(self.build),
                         "--expected", str(self.expected), "--cache", str(self.cache), "--destination", str(self.dest)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["restored"], 1)
        self.assertEqual((self.dest / local.IDENTITY / "stage/bin/tool").read_text(), "version one\n")



@unittest.skipUnless(local.shutil.which("cmake") and local.shutil.which("ninja") and local.shutil.which("git"),
                     "requires CMake, Ninja and Git")
class SynthWorkflowTest(unittest.TestCase):
    def test_fixture_cold_warm_and_single_component_change(self):
        import tempfile
        import sqlite3
        from build_tools.github_actions.artifact_cache.local_cache import synth, cache_build
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            database, build, cache = root / "cache.db", root / "build", root / "objects"
            with sqlite3.connect(database) as db:
                db.executescript(local.SCHEMA.read_text())
            synth(build)
            first = cache_build(database=database, build=build, cache=cache, variant="release")
            self.assertEqual(first["summary"], {"stored": 5, "verified_cached": 0, "uncacheable": 1})
            warm = cache_build(database=database, build=build, cache=cache, variant="release")
            self.assertEqual(warm["summary"]["verified_cached"], 5)
            synth(build, "blas")
            changed = cache_build(database=database, build=build, cache=cache, variant="release")
            self.assertEqual(changed["summary"], {"stored": 1, "verified_cached": 4, "uncacheable": 1})
            with self.assertRaisesRegex(ValueError, "match"):
                cache_build(database=database, build=build, cache=cache, variant="debug")
            ordinary = root / "ordinary"
            ordinary.mkdir()
            with self.assertRaisesRegex(ValueError, "new directory"):
                synth(ordinary)


if __name__ == "__main__":
    unittest.main()
