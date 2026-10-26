#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Detailed logs must preserve cache results, failure semantics and JSON stdout.

Pipeline checks use real SQLite, archive storage, extraction and subprocess I/O.
Only Git provenance and the CMake command are replaced by tiny fixture outputs.
The existing artifact_cache_local_test.py covers real CMake when available.
"""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache import collector as artifact_cache
from build_tools.github_actions.artifact_cache import (
    cli, diagnostics, local_cache, publish,
)
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures


IDENTITY = "core-amdsmi_lib_generic"
SPLIT = "blas_lib_gfx1100"


class DiagnosticSessionTest(unittest.TestCase):
    def test_disabled_logging_is_silent_and_nested_sessions_restore_context(self):
        output = io.StringIO()
        with redirect_stderr(output):
            diagnostics.trace("DISABLED")
            with diagnostics.logging_session(verbose=True):
                diagnostics.trace("OUTER", path="name\nwith\x1bcontrol")
                with diagnostics.logging_session(verbose=True):
                    diagnostics.trace("INNER")
                diagnostics.trace("OUTER_AGAIN")
            diagnostics.trace("DISABLED_AGAIN")
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("#1 OUTER ", lines[0])
        self.assertIn("#1 INNER", lines[1])
        self.assertIn("#2 OUTER_AGAIN", lines[2])
        self.assertNotIn("\x1b", output.getvalue())
        self.assertIn("\\n", lines[0])
        self.assertRegex(lines[0], r"\d{4}-\d\d-\d\dT.*Z \+\d+\.\d{3}s")

    def test_byte_progress_is_rate_limited_without_sleep(self):
        output = io.StringIO()
        with redirect_stderr(output), diagnostics.logging_session(verbose=True):
            with patch.object(diagnostics.time, "monotonic", return_value=0):
                progress = diagnostics.ByteProgress("BYTES", total_bytes=100)
            with patch.object(diagnostics.time, "monotonic", return_value=1):
                progress.advance(10)
                progress.advance(20)
            self.assertEqual(output.getvalue(), "")
            with patch.object(diagnostics.time, "monotonic", return_value=5):
                progress.advance(30)
            self.assertIn("processed_bytes=60", output.getvalue())
            with patch.object(diagnostics.time, "monotonic", return_value=6):
                progress.advance(10)
            self.assertEqual(output.getvalue().count(" BYTES "), 1)

    def test_write_failure_does_not_fail_cache_work(self):
        class BrokenLog:
            def write(self, value):
                raise OSError("disk full")

        output = io.StringIO()
        with redirect_stderr(output), diagnostics.logging_session(verbose=True):
            session = diagnostics._active.get()
            session.stream = BrokenLog()
            diagnostics.trace("FIRST")
            diagnostics.trace("SECOND")
        self.assertEqual(output.getvalue().count("Diagnostic file write failed"), 1)
        self.assertIn("SECOND", output.getvalue())

    def test_span_reports_failure_and_reraises_original_exception(self):
        error = ValueError("bad archive")
        output = io.StringIO()
        with redirect_stderr(output), diagnostics.logging_session(verbose=True):
            with self.assertRaises(ValueError) as caught:
                with diagnostics.span("VERIFY"):
                    raise error
        self.assertIs(caught.exception, error)
        self.assertIn("VERIFY_FAILED", output.getvalue())
        self.assertNotIn("VERIFY_DONE", output.getvalue())


class DetailedPipelineTest(fixtures.CacheFixture):
    def setUp(self):
        super().setUp()
        self.outputs((IDENTITY, fixtures.FPRINT), (SPLIT, None))
        (self.artifacts / SPLIT).mkdir()
        self.build = self.artifacts.parent
        self.source = self.root / "source"
        self.source.mkdir()
        (self.build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={self.source}\nCMAKE_COMMAND:INTERNAL=cmake\n"
        )
        self.cache = self.root / "objects"
        self.expected_one = self.root / "one.json"
        self.expected_one.write_text(json.dumps({"schema_version": 1, "artifacts": [
            {"artifact_identity": IDENTITY, "fingerprint_path": IDENTITY + ".fprint"}
        ]}))
        self.commands = []
        self.fail_build = False
        real_run = publish._run

        def fixture_run(command, **kwargs):
            self.commands.append(command)
            if "--build" in command:
                script = "raise SystemExit(17)" if self.fail_build else (
                    "from pathlib import Path\n"
                    f"export = Path({str(self.artifacts / IDENTITY)!r})\n"
                    "(export / 'stage').mkdir(parents=True, exist_ok=True)\n"
                    "(export / 'artifact_manifest.txt').write_text('stage\\n')\n"
                    "(export / 'stage/payload').write_text('built bytes\\n')\n"
                    "print('[1/1] building fixture')\n"
                )
            else:
                script = "print('-- Configuring fixture')"
            real_run([sys.executable, "-c", script], **kwargs)

        for mock in (
            patch.object(publish, "_source_revisions", return_value={".": "a" * 40}),
            patch.object(publish, "_run", side_effect=fixture_run),
        ):
            mock.start()
            self.addCleanup(mock.stop)

    def invoke(self, main, args):
        stdout = io.StringIO()
        # subprocess stderr needs a real file descriptor; all controller logs
        # are captured in the same stream as build output.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(args)
            stderr.seek(0)
            output = stderr.read()
        return code, stdout.getvalue(), output

    def cache_args(self, *extra):
        return ["cache", "--granularity", "component", "--db", str(self.database), "--build", str(self.build),
                "--cache", str(self.cache), *extra]

    def publication(self):
        return publish.build_and_publish(database=self.database,
            expected_manifest=self.expected_one, build_dir=self.build,
            cache_root=self.cache, label="fixture")

    def test_cold_warm_and_restore_detail_preserves_json_and_final_report(self):
        cold_log = self.root / "cold.log"
        code, stdout, stderr = self.invoke(local_cache.main,
            self.cache_args("--verbose", "--log-file", str(cold_log)))
        self.assertEqual(code, 0, stderr)
        report = json.loads(stdout)
        self.assertEqual(report["summary"], {"stored": 1, "verified_cached": 0, "uncacheable": 1})
        for event in ("DISCOVERY", "FINGERPRINT", "KEY", "DB_LOOKUP", "CACHE_DECISION",
                      "BUILD_DECISION", "ARCHIVE_WRITE", "INVENTORY_WRITTEN", "INVENTORY_VERIFIED",
                      "OBJECT_SAVED", "PRODUCER_RECORD_COMMITTED", "ENTRY_COMMITTED", "FINAL_KEYS_VERIFIED"):
            self.assertIn(event, stderr)
            self.assertIn(event, cold_log.read_text())
        self.assertIn('mode="observed_outputs"', stderr)
        self.assertIn(fixtures.FPRINT, stderr)
        self.assertIn(str(self.cache / "objects"), stderr)
        self.assertIn("size_bytes=", stderr)
        self.assertIn("sha256=", stderr)
        self.assertLess(stderr.index("OBJECT_SAVED"), stderr.index("ENTRY_COMMITTED"))
        self.assertLess(stderr.index("ENTRY_COMMITTED"), stderr.index("MISS -> STORED"))
        self.assertLess(stderr.index("FINAL_KEYS_VERIFIED"), stderr.index("MISS -> STORED"))
        self.assertEqual(stderr.count("[artifact-cache] SUMMARY"), 1)
        self.assertEqual(stderr.count("[artifact-cache] MISS -> STORED"), 1)
        self.assertEqual(stderr.count("[artifact-cache] NO_FINGERPRINT"), 1)

        self.commands.clear()
        warm_log = self.root / "warm.log"
        code, stdout, stderr = self.invoke(local_cache.main,
            self.cache_args("--log-file", str(warm_log)))
        self.assertEqual(code, 0, stderr)
        warm = json.loads(stdout)
        self.assertFalse(warm["build_executed"])
        self.assertFalse(any("--build" in command for command in self.commands))
        self.assertEqual(warm["summary"]["verified_cached"], 1)
        self.assertIn("[artifact-cache] REUSED", stderr)
        self.assertNotIn("CANDIDATE_CHECK", stderr)
        saved_log = warm_log.read_text()
        for event in ("CANDIDATE_CHECK", "OBJECT_VERIFIED", "CANDIDATE_VERIFIED", "CACHE_VERIFIED"):
            self.assertIn(event, saved_log)
        self.assertIn("temporary verification; not installed into build", saved_log)
        self.assertIn("SUMMARY", saved_log)
        self.assertIn('exit_code=0', saved_log)

        destination = self.root / "restored"
        code, stdout, stderr = self.invoke(cli.main, [
            "restore", "--granularity", "component", "--db", str(self.database), "--build", str(self.build),
            "--cache", str(self.cache), "--expected", str(self.expected_one), "--destination", str(destination), "--verbose",
        ])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["restored"], 1)
        self.assertIn("RESTORE_DESTINATION_READY", stderr)
        self.assertEqual((destination / IDENTITY / "stage/payload").read_text(), "built bytes\n")

    def test_build_failure_is_logged_and_publishes_nothing(self):
        self.fail_build = True
        log = self.root / "failure.log"
        code, stdout, stderr = self.invoke(local_cache.main,
            self.cache_args("--verbose", "--log-file", str(log)))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("COMMAND_FAILED", stderr)
        self.assertIn("BUILD_PUBLISH_FAILED", stderr)
        self.assertIn("exit_code=2", log.read_text())
        for marker in ("ENTRY_COMMITTED", "MISS -> STORED", "[artifact-cache] SUMMARY"):
            self.assertNotIn(marker, stderr)
        self.assertEqual(self.db.execute("SELECT count(*) FROM cache_entries").fetchone()[0], 0)

    def test_corrupt_newest_candidate_logs_retry_and_uses_older_generation(self):
        older = self.publication()["published"][0]
        newer = self.publication()["published"][0]
        (Path(newer["local_directory"]) / "payload.tar").write_bytes(b"broken")
        code, stdout, stderr = self.invoke(local_cache.main, self.cache_args("--verbose"))
        self.assertEqual(code, 0, stderr)
        report = json.loads(stdout)
        artifact = next(a for a in report["artifacts"] if a["artifact_identity"] == IDENTITY)
        self.assertEqual(artifact["cache_entry_id"], older["cache_entry_id"])
        self.assertFalse(report["build_executed"])
        self.assertIn("CANDIDATE_REJECTED", stderr)
        self.assertIn('action="invalidate_then_retry"', stderr)
        self.assertIn("ENTRY_INVALIDATED", stderr)
        self.assertIn("digest mismatch", stderr)
        self.assertLess(stderr.index("CANDIDATE_REJECTED"), stderr.index("CANDIDATE_VERIFIED"))

    def test_existing_log_file_is_preserved_and_work_does_not_start(self):
        previous = self.database.read_bytes()
        code, stdout, stderr = self.invoke(local_cache.main,
            self.cache_args("--log-file", str(self.database)))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("File exists", stderr)
        self.assertEqual(self.commands, [])
        self.assertEqual(self.database.read_bytes(), previous)

    def test_rollback_log_does_not_claim_entries_were_committed(self):
        # Abort publication after objects and producer observations were saved.
        # The log must distinguish those durable steps from the rolled-back entries.
        self.db.execute("CREATE TRIGGER fail_publish BEFORE INSERT ON cache_entries "
                        "BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END")
        self.db.commit()
        code, stdout, stderr = self.invoke(local_cache.main, self.cache_args("--verbose"))
        self.assertEqual(code, 2, stderr)
        self.assertEqual(stdout, "")
        self.assertIn("OBJECT_SAVED", stderr)
        self.assertIn("PRODUCER_RECORD_COMMITTED", stderr)
        self.assertIn("DB_TRANSACTION_ROLLED_BACK", stderr)
        self.assertNotIn("ENTRY_COMMITTED", stderr)
        self.assertNotIn("[artifact-cache] SUMMARY", stderr)

    def test_collector_entry_point_supports_file_logging_and_explicit_inventory(self):
        log = self.root / "collector.log"
        code, stdout, stderr = self.invoke(artifact_cache.main, [
            "collect", "--db", str(self.database), "--expected", str(self.expected_one),
            "--artifacts-dir", str(self.artifacts), "--label", "trace", "--commit", "a" * 40, "--configure-state", "succeeded",
            "--log-file", str(log),
        ])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["summary"]["expected"], 1)
        self.assertEqual(stderr, "")
        self.assertIn("FINGERPRINT", log.read_text())
        self.assertIn("DB_TRANSACTION_COMMITTED", log.read_text())

    def test_final_check_failure_keeps_committed_event_but_no_success_report(self):
        real_scan = local_cache.prepared_scan
        scans = 0

        @contextmanager
        def scan(*args, **kwargs):
            nonlocal scans
            scans += 1
            with real_scan(*args, **kwargs) as records:
                yield records
            if scans == 2:
                raise ValueError("injected final check failure")

        with patch.object(local_cache, "prepared_scan", side_effect=scan):
            code, stdout, stderr = self.invoke(local_cache.main, self.cache_args("--verbose"))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ENTRY_COMMITTED", stderr)
        self.assertIn("injected final check failure", stderr)
        self.assertIn("exit_code=2", stderr)
        self.assertNotIn("[artifact-cache] SUMMARY", stderr)
        self.assertNotIn("[artifact-cache] MISS -> STORED", stderr)
        self.assertEqual(self.db.execute("SELECT count(*) FROM cache_entries").fetchone()[0], 1)

    def test_explicit_scan_and_plan_logs_do_not_claim_reuse_or_stage_skips(self):
        code, stdout, stderr = self.invoke(cli.main, [
            "plan", "--db", str(self.database), "--build", str(self.build),
            "--expected", str(self.expected_one), "--verbose",
        ])
        self.assertEqual(code, 0, stderr)
        report = json.loads(stdout)
        self.assertEqual(report["inventory_scope"], "explicit_outputs")
        self.assertFalse(report["payloads_verified"])
        self.assertEqual(report["skippable_stages"], [])
        self.assertIn('mode="explicit_outputs"', stderr)
        self.assertIn("PLAN_ARTIFACT", stderr)
        self.assertIn("can_skip_stage=false", stderr)
        self.assertNotIn("REUSED", stderr)

    def test_direct_publish_entry_point_accepts_verbose_and_file_flags(self):
        log = self.root / "publish.log"
        code, stdout, stderr = self.invoke(artifact_cache.main, [
            "build-publish", "--db", str(self.database), "--expected", str(self.expected_one),
            "--build-dir", str(self.build), "--cache-root", str(self.cache),
            "--label", "trace", "--verbose", "--log-file", str(log),
        ])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(len(json.loads(stdout)["published"]), 1)
        self.assertIn("ENTRY_COMMITTED", stderr)
        self.assertIn("[artifact-cache] STORED", stderr)
        self.assertIn("RUN_END", log.read_text())


if __name__ == "__main__":
    unittest.main()
