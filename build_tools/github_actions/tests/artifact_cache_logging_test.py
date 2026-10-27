#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI status logs must preserve JSON reports and failure exit statuses."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache import collector as artifact_cache
from build_tools.github_actions.artifact_cache import cli, local_cache
from build_tools.github_actions.tests import artifact_cache_collector_test as fixtures


IDENTITY = "core-amdsmi_lib_generic"
COMMON = ["--db", "cache.db", "--build", "build", "--cache", "objects",
          "--expected", "expected.json"]


class CacheLoggingTest(unittest.TestCase):
    def invoke(self, main, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cold_warm_and_uncacheable_keep_stdout_json(self):
        for outcome, reason in (("stored", None), ("verified_cached", None),
                                ("uncacheable", "expected fingerprint file is missing"),
                                ("uncacheable", "fingerprint must contain 64 hexadecimal characters")):
            with self.subTest(outcome=outcome, reason=reason):
                artifact = {"artifact_identity": IDENTITY, "outcome": outcome}
                if outcome != "uncacheable":
                    artifact["cache_entry_id"] = 2
                if outcome == "verified_cached":
                    artifact["payloads_verified"] = True
                if outcome == "uncacheable":
                    artifact["reason"] = reason
                report = {"operation": "cache", "epoch": 1,
                          "build_executed": outcome == "stored",
                          "artifacts": [artifact], "summary": {
                              state: int(state == outcome) for state in
                              ("stored", "verified_cached", "uncacheable")}}
                def cache_result(**kwargs):
                    self.assertTrue(kwargs["show_configure"])
                    self.assertTrue(callable(kwargs["progress"]))
                    # The CLI callback must not print intermediate artifact
                    # outcomes, even though Python callers can receive them.
                    for status in ("MISS", "STORING", "STORED", "VERIFIED_CACHED", "UNCACHEABLE"):
                        kwargs["progress"](status, IDENTITY)
                    return report
                with patch.object(local_cache, "cache_build", side_effect=cache_result):
                    code, stdout, stderr = self.invoke(local_cache.main, ["cache", "--granularity", "component", *COMMON])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(stdout), report)
                display = {"stored": "MISS -> STORED", "verified_cached": "REUSED",
                           "uncacheable": ("NO_FINGERPRINT" if reason ==
                                           "expected fingerprint file is missing" else "CACHE_SKIPPED")}[outcome]
                self.assertIn(f"[artifact-cache] {display} {IDENTITY}", stderr)
                self.assertEqual(stderr.count(IDENTITY), 1)
                self.assertIn("[artifact-cache] SUMMARY", stderr)
                self.assertNotIn("skip rebuild", stderr)
                self.assertNotIn("STORING", stderr)
                self.assertNotIn("UNCACHEABLE", stderr)
                if outcome == "uncacheable":
                    self.assertIn(".fprint file missing" if display == "NO_FINGERPRINT" else reason, stderr)
                    self.assertIn("cache restore and storage skipped", stderr)
                    self.assertIn("cache_skipped=1", stderr)
                    self.assertNotIn("STORED", stderr)
                    if display == "CACHE_SKIPPED":
                        self.assertNotIn("NO_FINGERPRINT", stderr)
                if outcome == "verified_cached":
                    self.assertIn("matching cached files verified", stderr)
                    self.assertIn("reused=1", stderr)
                if outcome == "stored":
                    self.assertIn("no usable cache entry; saved after successful build", stderr)

    def test_restore_status_and_miss_exit_code(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                outcome = "needs_build" if missing else "restored"
                report = {"operation": "restore", "destination": "restored",
                          "restored": int(not missing), "needs_build": int(missing),
                          "artifacts": [{"artifact_identity": IDENTITY, "epoch": 1,
                                         "outcome": outcome, "payloads_verified": not missing}]}
                with patch("build_tools.github_actions.artifact_cache.restore.restore_local_outputs",
                           return_value=report):
                    code, stdout, stderr = self.invoke(
                        cli.main, ["restore", "--granularity", "component", *COMMON, "--destination", "restored"])
                self.assertEqual(code, int(missing))
                self.assertEqual(json.loads(stdout), report)
                self.assertIn(f"[artifact-cache] {outcome.upper()} {IDENTITY}", stderr)
                if missing:
                    self.assertNotIn("payloads_verified=true", stderr)

    def test_both_publication_entry_points_log_stored(self):
        report = {"operation": "build-publish", "run_id": 8, "build_executed": True,
                  "published": [{"artifact_identity": IDENTITY, "epoch": 1,
                                 "cache_entry_id": 2, "payloads_verified": False}]}
        collector_args = ["build-publish", "--db", "cache.db", "--build-dir", "build",
                  "--cache-root", "objects", "--expected", "expected.json",
                  "--label", "test"]
        for main, args in ((cli.main, ["publish", "--granularity", "component", *COMMON]), (artifact_cache.main, collector_args)):
            with self.subTest(entry_point=main.__module__):
                def publish_result(**kwargs):
                    self.assertIsNone(kwargs.get("progress"))
                    return report
                with patch("build_tools.github_actions.artifact_cache.publish.build_and_publish",
                           side_effect=publish_result):
                    code, stdout, stderr = self.invoke(main, args)
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(stdout), report)
                self.assertIn(f"STORED {IDENTITY} (epoch=1, entry=2)", stderr)
                self.assertIn("run_id=8", stderr)
                self.assertNotIn("payloads_verified=true", stderr)
                self.assertNotIn("MISS", stderr)
                self.assertNotIn("STORING", stderr)
                self.assertEqual(stderr.count(IDENTITY), 1)

    def test_failed_build_does_not_log_success(self):
        with patch.object(local_cache, "cache_build",
                          side_effect=subprocess.CalledProcessError(1, ["cmake", "--build", "build"])):
            code, stdout, stderr = self.invoke(local_cache.main, ["cache", "--granularity", "component", *COMMON])
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("ERROR:", stderr)
        self.assertNotIn("STORED", stderr)
        self.assertNotIn("VERIFIED_CACHED", stderr)
        self.assertNotIn("REUSED", stderr)
        self.assertNotIn("SUMMARY", stderr)

    def test_metadata_only_lookup_does_not_claim_verified_reuse(self):
        report = {"artifact_identity": IDENTITY, "payloads_verified": False}
        with patch.object(cli, "execute", return_value=(report, 0)):
            code, stdout, stderr = self.invoke(
                cli.main, ["lookup", "--db", "cache.db", "--artifact", IDENTITY,
                           "--fingerprint", "a" * 64])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout), report)
        self.assertEqual(stderr, "")


class CacheReportTimingTest(fixtures.CacheFixture):
    def test_cache_cli_streams_first_configure_and_build_for_cold_and_warm_runs(self):
        from build_tools.github_actions.artifact_cache import publish
        self.outputs((IDENTITY, fixtures.FPRINT), ("split_lib_gfx1100", None))
        (self.artifacts / "split_lib_gfx1100").mkdir()
        build = self.artifacts.parent
        source = self.root / "source"
        source.mkdir()
        (build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\nCMAKE_COMMAND:INTERNAL=cmake\n")
        real_run = publish._run
        commands = []

        def run_fixture(command, **kwargs):
            # Real subprocess output with the real DB/store/publisher pipeline.
            # Only CMake and Git provenance are replaced by this tiny fixture.
            commands.append(command)
            if "--build" in command:
                script = (
                    "from pathlib import Path\n"
                    f"export = Path({str(self.artifacts / IDENTITY)!r})\n"
                    "(export / 'stage').mkdir(parents=True, exist_ok=True)\n"
                    "(export / 'artifact_manifest.txt').write_text('stage\\n')\n"
                    "(export / 'stage/payload').write_text('built output\\n')\n"
                    "print('[1/1] fixture build output')\n"
                )
            else:
                script = f"print('-- Build files have been written to: {build.as_posix()}')"
            real_run([sys.executable, "-c", script], **kwargs)

        args = ["cache", "--granularity", "component", "--db", str(self.database), "--build", str(build),
                "--cache", str(self.root / "objects"), ]
        with patch.object(publish, "_source_revisions", return_value={".": "a" * 40}), \
                patch.object(publish, "_run", side_effect=run_fixture):
            for phase in ("cold", "warm"):
                commands.clear()
                stdout = io.StringIO()
                with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        code = local_cache.main(args)
                    stderr.seek(0)
                    output = stderr.read()
                self.assertEqual(code, 0, output)
                report = json.loads(stdout.getvalue())
                self.assertEqual(report["build_executed"], phase == "cold")
                self.assertEqual(output.count("-- Build files have been written to:"), 1)
                self.assertEqual(output.count("[artifact-cache] SUMMARY"), 1)
                self.assertEqual(output.count("NO_FINGERPRINT split_lib_gfx1100"), 1)
                self.assertNotIn("STORING", output)
                if phase == "cold":
                    self.assertEqual(sum("-S" in cmd for cmd in commands), 4)
                    self.assertEqual(sum("--build" in cmd for cmd in commands), 1)
                    self.assertEqual(output.count("MISS -> STORED " + IDENTITY), 1)
                    self.assertLess(output.index("BUILD: running"), output.index("fixture build output"))
                    self.assertLess(output.index("fixture build output"), output.index("BUILD PASSED:"))
                    self.assertLess(output.index("validating final artifact keys"),
                                    output.index("MISS -> STORED"))
                else:
                    self.assertEqual(sum("-S" in cmd for cmd in commands), 1)
                    self.assertFalse(any("--build" in cmd for cmd in commands))
                    self.assertIn("BUILD NOT RUN:", output)
                    self.assertEqual(output.count("REUSED " + IDENTITY), 1)
                    self.assertNotIn("BUILD PASSED:", output)
                    self.assertNotIn("fixture build output", output)
        logs = list((build / "logs/artifact-cache").glob("configure-*.log"))
        self.assertEqual(len(logs), 3)
        self.assertTrue(all("Build files have been written to:" in p.read_text() for p in logs))

    def test_all_discovered_statuses_are_reported_once_after_final_checks(self):
        self._cache_report(fail_recheck=False)

    def test_failed_final_check_does_not_print_a_success_report(self):
        self._cache_report(fail_recheck=True)

    def _cache_report(self, *, fail_recheck):
        from build_tools.github_actions.artifact_cache.inventory import scan_build
        from build_tools.github_actions.artifact_cache import publish, restore
        hit, miss, invalid = "hit_lib_generic", "miss_lib_generic", "split_lib_gfx1100"
        self.outputs((hit, fixtures.FPRINT), (miss, fixtures.FPRINT), (invalid, None))
        (self.artifacts / invalid).mkdir()
        entry = self.seed_candidate(hit)

        scans = 0

        def assert_no_artifact_report():
            for marker in ("MISS -> STORED", "REUSED ", "NO_FINGERPRINT ", "SUMMARY"):
                self.assertNotIn(marker, stderr.getvalue())

        @contextmanager
        def scan(*args, show_configure=False, granularity="component"):
            nonlocal scans
            scans += 1
            self.assertEqual(show_configure, scans == 1)
            yield scan_build(self.artifacts.parent)
            assert_no_artifact_report()
            if fail_recheck and scans == 2:
                raise ValueError("inputs changed during cache inspection; retry")

        stdout, stderr = io.StringIO(), io.StringIO()

        def build(**kwargs):
            # Artifact outcomes must wait for the final input checks.
            assert_no_artifact_report()
            kwargs["progress"]("BUILD_START")
            kwargs["progress"]("BUILD_PASSED")
            kwargs["progress"]("STORING", miss)
            return {"run_id": 10, "published": [{"artifact_identity": miss, "cache_entry_id": 99}]}

        def verified(*args, **kwargs):
            assert_no_artifact_report()
            return {"artifact_identity": hit, "epoch": 1, "outcome": "restored",
                    "cache_entry_id": entry, "payloads_verified": True}

        with patch.object(local_cache, "prepared_scan", side_effect=scan), \
                patch.object(restore, "restore_local_key", side_effect=verified), \
                patch.object(publish, "build_and_publish", side_effect=build), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = local_cache.main(["cache", "--granularity", "component", "--db", str(self.database),
                                    "--build", str(self.artifacts.parent),
                                    "--cache", str(self.root),
                                    ])
        if fail_recheck:
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("inputs changed", stderr.getvalue())
            assert_no_artifact_report()
            return
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["summary"],
                         {"stored": 1, "verified_cached": 1, "uncacheable": 1})
        lines = [line for line in stderr.getvalue().splitlines()
                 if any(marker in line for marker in
                        ("REUSED ", "MISS -> STORED", "NO_FINGERPRINT ", "SUMMARY"))]
        self.assertEqual(len(lines), 4)
        self.assertIn("CONFIGURE:", stderr.getvalue())
        self.assertIn("CACHE CHECK:", stderr.getvalue())
        self.assertIn("BUILD:", stderr.getvalue())
        self.assertIn("BUILD PASSED:", stderr.getvalue())
        self.assertNotIn("STORING", stderr.getvalue())
        self.assertIn("REUSED " + hit, lines[0])
        self.assertIn("MISS -> STORED " + miss, lines[1])
        self.assertIn("NO_FINGERPRINT " + invalid, lines[2])
        self.assertIn(".fprint file missing; cache restore and storage skipped", lines[2])
        self.assertEqual(lines[3], "[artifact-cache] SUMMARY stored=1 reused=1 "
                         "cache_skipped=1")

    def test_publisher_reports_stored_only_after_database_commit(self):
        self._publisher_progress(fail=False)

    def test_publisher_build_failure_never_reports_stored(self):
        self._publisher_progress(fail=True)

    def _publisher_progress(self, *, fail):
        from build_tools.github_actions.artifact_cache import publish
        from build_tools.github_actions.artifact_cache.cache_db import CacheDB
        from build_tools.github_actions.artifact_cache.reporting import log_progress
        self.outputs((IDENTITY, fixtures.FPRINT))
        build = self.artifacts.parent
        source = self.root / "source"
        source.mkdir()
        (build / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\nCMAKE_COMMAND:INTERNAL=cmake\n")
        stderr = io.StringIO()
        events = []

        def run(command, *, configure_log_dir=None):
            if "--build" in command:
                self.assertIsNone(configure_log_dir)
                self.assertIn("BUILD_START", stderr.getvalue())
                self.assertNotIn("STORING", stderr.getvalue())
                if fail:
                    raise subprocess.CalledProcessError(1, command)
                exported = self.artifacts / IDENTITY
                (exported / "stage").mkdir(parents=True)
                (exported / "artifact_manifest.txt").write_text("stage\n")
                (exported / "stage/payload").write_text("built output\n")
            else:
                self.assertEqual(configure_log_dir, build / "logs" / "artifact-cache")

        def progress(status, identity=None, **details):
            events.append(status)
            if status == "STORED":
                # A different connection must see the committed ready entry.
                with CacheDB(self.database) as db:
                    self.assertIsNotNone(db.get_manifest(details["entry"]))
            log_progress(status, identity, **details)

        with patch.object(publish, "_source_revisions", return_value={".": "a" * 40}), \
                patch.object(publish, "_run", side_effect=run), redirect_stderr(stderr):
            def invoke():
                return publish.build_and_publish(
                    database=self.database, expected_manifest=self.manifest,
                    build_dir=build, cache_root=self.root.parent / (self.root.name + "-objects"),
                    label="logging-test", progress=progress)
            # Use a sibling directory: publication forbids storage inside build.
            import shutil
            self.addCleanup(shutil.rmtree, self.root.parent / (self.root.name + "-objects"), True)
            if fail:
                with self.assertRaises(subprocess.CalledProcessError):
                    invoke()
                self.assertIn("BUILD_FAILED", events)
                self.assertNotIn("STORED", events)
                self.assertEqual(self.count("cache_entries"), 0)
            else:
                report = invoke()
                self.assertEqual(events, ["CONFIGURING", "BUILD_START", "BUILD_PASSED",
                                          "SAVING_OUTPUTS", "STORING", "STORED"])
                self.assertEqual(len(report["published"]), 1)


class ConfigureOutputTest(unittest.TestCase):
    def test_successful_configures_are_quiet_and_keep_separate_logs(self):
        from build_tools.github_actions.artifact_cache.publish import _run
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp) / "logs"
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                for phase in ("before", "after"):
                    _run([sys.executable, "-c",
                          f"import sys; print('{phase} configure'); "
                          "print('configure warning', file=sys.stderr)"],
                         configure_log_dir=logs)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")
            contents = [p.read_text() for p in logs.glob("configure-*.log")]
            self.assertEqual(len(contents), 2)
            self.assertTrue(any("before configure" in text for text in contents))
            self.assertTrue(any("after configure" in text for text in contents))
            self.assertTrue(all("configure warning" in text for text in contents))

    def test_failed_configure_replays_diagnostics_and_preserves_failure(self):
        from build_tools.github_actions.artifact_cache.publish import _run
        with tempfile.TemporaryDirectory() as temp:
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    _run([sys.executable, "-c",
                          "import sys; print('configure context'); "
                          "print('CMake Error: missing dependency', file=sys.stderr); sys.exit(7)"],
                         configure_log_dir=Path(temp))
            self.assertEqual(caught.exception.returncode, 7)
            self.assertEqual(stdout.getvalue(), "")
            log, = Path(temp).glob("configure-*.log")
            self.assertIn(str(log), stderr.getvalue())
            self.assertIn("configure context", stderr.getvalue())
            self.assertIn("CMake Error: missing dependency", stderr.getvalue())
            self.assertIn(log.read_text(), stderr.getvalue())

    def test_missing_configure_executable_is_not_silenced(self):
        from build_tools.github_actions.artifact_cache.publish import _run
        with tempfile.TemporaryDirectory() as temp:
            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(OSError):
                _run([str(Path(temp) / "missing-cmake")], configure_log_dir=Path(temp))
            self.assertIn("CMake configuration failed. Full log:", stderr.getvalue())

    def test_build_output_still_streams_to_stderr(self):
        from build_tools.github_actions.artifact_cache.publish import _run
        # A real file provides the descriptor needed by the child process.
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
            stdout = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                _run([sys.executable, "-c",
                      "import sys; print('build progress'); print('build warning', file=sys.stderr)"])
            stderr.seek(0)
            output = stderr.read()
            self.assertIn("build progress", output)
            self.assertIn("build warning", output)
            self.assertEqual(stdout.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
