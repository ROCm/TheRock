#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Local publication/restore tests with real CMake builds, SQLite and file I/O.

The miniature project needs Git, CMake and Ninja, but no compiler, ROCm GPU,
submodules or network. Its payload is deliberately tiny; this is not a full
TheRock build or a test of CMake prebuilt-stage scheduling.
"""

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
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.github_actions.artifact_cache.cache_db import CacheDB
from build_tools.github_actions.artifact_cache.local_store import (
    _object_record, store_component, unpack_component,
)
from build_tools.github_actions.artifact_cache.manifest import ArtifactKey


REPO = Path(__file__).resolve().parents[3]
SCHEMA = Path(os.environ.get("THEROCK_CACHE_SCHEMA", REPO / "build_tools/therock_fingerprint_cache.sql"))
IDENTITY = "sample_lib_generic"


class LocalObjectStorageTest(unittest.TestCase):
    """Byte-storage tests also run where CMake is not installed."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "export"
        self.source.mkdir()
        (self.source / "artifact_manifest.txt").write_text("stage\n")
        (self.source / "stage").mkdir()
        (self.source / "stage/payload").write_bytes(b"payload\x00bytes\n")

    def test_store_round_trip_and_immutable_generations(self):
        first = store_component(self.source, self.root / "cache")
        (self.source / "stage/payload").write_bytes(b"new payload")
        second = store_component(self.source, self.root / "cache")
        self.assertNotEqual(first.directory, second.directory)
        restored = self.root / "restored"
        restored.mkdir()
        unpack_component(first.directory, restored)
        self.assertEqual((restored / "stage/payload").read_bytes(), b"payload\x00bytes\n")
        for file in first.files:
            self.assertEqual(_object_record(first.directory / file.relative_path), file)

    def test_missing_archive_member_is_rejected(self):
        stored = store_component(self.source, self.root / "cache")
        with tarfile.open(stored.directory / "payload.tar", "w"):
            pass
        with self.assertRaisesRegex(ValueError, "missing required"):
            unpack_component(stored.directory, None)

    def test_empty_component_still_requires_its_manifest(self):
        shutil.rmtree(self.source / "stage")
        stored = store_component(self.source, self.root / "cache")
        unpack_component(stored.directory, None)
        (self.source / "artifact_manifest.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            store_component(self.source, self.root / "cache")


@unittest.skipUnless(all(shutil.which(x) for x in ("cmake", "ninja", "git")),
                     "requires CMake, Ninja and Git")
class LocalCacheBuildTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.build = self.root / "source", self.root / "build"
        self.cache, self.dest = self.root / "cache", self.root / "restored"
        self.database, self.expected = self.root / "cache.db", self.root / "expected.json"
        self.source.mkdir()
        (self.source / "input.txt").write_text("version one\n")
        (self.source / "CMakeLists.txt").write_text(textwrap.dedent('''\
            cmake_minimum_required(VERSION 3.25)
            project(cache_build_fixture NONE)
            file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/input.txt" fingerprint)
            if(EXISTS "${CMAKE_BINARY_DIR}/flip")
              string(SHA256 fingerprint "changed during build")
            endif()
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/artifacts")
            file(WRITE "${CMAKE_BINARY_DIR}/artifacts/sample_lib_generic.fprint" "${fingerprint}")
            add_custom_target(exports ALL
              COMMAND "${TEST_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/export.py"
                      "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}"
              VERBATIM)
            '''), encoding="utf-8")
        (self.source / "export.py").write_text(textwrap.dedent('''\
            from pathlib import Path
            import os
            import sys
            source, build = map(Path, sys.argv[1:])
            count = build / "build-count"
            count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1))
            if (source / "fail").exists():
                raise SystemExit(17)
            if (source / "change-fingerprint").exists():
                (build / "flip").touch()
            if (source / "edit-source").exists():
                (source / "input.txt").write_text("changed while building")
            if (source / "omit-export").exists():
                raise SystemExit(0)
            export = build / "artifacts/sample_lib_generic"
            export.mkdir(exist_ok=True)
            (export / "artifact_manifest.txt").write_text("stage\\n")
            (export / "stage/bin").mkdir(parents=True, exist_ok=True)
            payload = export / "stage/bin/tool"
            payload.write_bytes((source / "input.txt").read_bytes())
            payload.chmod(0o755)
            if os.name != "nt":
                link = export / "stage/bin/tool-link"
                link.unlink(missing_ok=True)
                link.symlink_to("tool")
            '''), encoding="utf-8")
        self.run_command("git", "init", "-q", str(self.source))
        self.git("config", "user.email", "cache-test@example.invalid")
        self.git("config", "user.name", "Cache Test")
        self.commit()
        self.configure()
        self.expected.write_text(json.dumps({"schema_version": 1, "artifacts": [
            {"artifact_identity": IDENTITY, "fingerprint_path": IDENTITY + ".fprint"}
        ]}))
        with sqlite3.connect(self.database) as db:
            db.executescript(SCHEMA.read_text())

    def run_command(self, *command):
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def git(self, *args):
        return self.run_command("git", "-C", str(self.source), *args)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-qm", "fixture input")

    def configure(self):
        self.run_command("cmake", "-S", str(self.source), "-B", str(self.build),
                         "-G", "Ninja", f"-DTEST_PYTHON={sys.executable}")

    def cli(self, operation, *, success=True, extra=(), module=False):
        invocation = ["-m", "build_tools.github_actions.artifact_cache.collector"] if module else ["build_tools/github_actions/artifact_cache/collector.py"]
        command = [sys.executable, *invocation, operation,
                   "--db", str(self.database), "--expected", str(self.expected),
                   "--build-dir", str(self.build), "--cache-root", str(self.cache),
                   ]
        if operation == "build-publish":
            command += ["--label", "managed test", "--jobs", "2"]
        else:
            command += ["--destination", str(self.dest)]
        result = subprocess.run(command + list(extra), cwd=REPO, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def count(self, table="cache_entries"):
        with sqlite3.connect(self.database) as db:
            return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def publish(self):
        return self.cli("build-publish")

    def mark(self, name):
        (self.source / name).touch()
        self.commit()

    def test_real_build_publish_and_verified_restore(self):
        report = self.publish()
        self.assertTrue(report["build_executed"])
        self.assertEqual(self.count(), 1)
        self.assertEqual(self.count("artifact_files"), 2)
        restored = self.cli("restore", module=True)
        self.assertEqual(restored["restored"], 1)
        self.assertFalse(restored["build_skipped"])
        tool = self.dest / IDENTITY / "stage/bin/tool"
        self.assertEqual(tool.read_text(), "version one\n")
        if os.name != "nt":
            self.assertEqual(tool.stat().st_mode & 0o777, 0o755)
            self.assertEqual(os.readlink(tool.with_name("tool-link")), "tool")
        self.assertEqual((self.build / "build-count").read_text(), "1")

    def test_configure_only_change_cannot_publish_stale_payload(self):
        first = self.publish()["published"][0]
        (self.source / "input.txt").write_text("version two\n")
        self.commit()
        self.configure()
        # The old export still exists under the new configure-time fingerprint.
        export = self.build / "artifacts" / IDENTITY / "stage/bin/tool"
        self.assertEqual(export.read_text(), "version one\n")
        second = self.publish()["published"][0]
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual((self.build / "build-count").read_text(), "2")
        self.cli("restore")
        self.assertEqual((self.dest / IDENTITY / "stage/bin/tool").read_text(), "version two\n")

    def test_failed_build_with_old_payload_does_not_publish(self):
        self.publish()
        old_count = self.count()
        (self.source / "input.txt").write_text("version two\n")
        self.mark("fail")
        self.configure()
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), old_count)
        self.assertEqual(self.count("prototype_runs"), 1)

    def test_failed_configure_does_not_publish_old_fingerprints(self):
        self.publish()
        with (self.source / "CMakeLists.txt").open("a") as stream:
            stream.write('\nmessage(FATAL_ERROR "fixture failure")\n')
        self.commit()
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), 1)

    def test_fingerprint_change_during_build_is_rejected(self):
        self.mark("change-fingerprint")
        result = self.cli("build-publish", success=False)
        self.assertIn("changed during the build", result.stderr)
        self.assertEqual(self.count(), 0)

    def test_tracked_source_change_during_build_is_rejected(self):
        self.mark("edit-source")
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), 0)

    def test_dirty_source_is_rejected_before_build(self):
        (self.source / "input.txt").write_text("uncommitted")
        self.cli("build-publish", success=False)
        self.assertFalse((self.build / "build-count").exists())
        self.assertEqual(self.count(), 0)

    def test_successful_command_without_export_is_not_publishable(self):
        self.mark("omit-export")
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), 0)

    def test_prebuilt_imports_are_rejected(self):
        (self.build / "stage.prebuilt").touch()
        self.cli("build-publish", success=False)
        self.assertFalse((self.build / "build-count").exists())

    def test_managed_build_lock_is_respected(self):
        lock = self.build / ".therock-artifact-cache.lock"
        lock.write_text("another publisher")
        self.cli("build-publish", success=False)
        self.assertEqual(lock.read_text(), "another publisher")

    def test_cli_cannot_assert_build_success(self):
        result = self.cli("build-publish", success=False, extra=("--build-state", "succeeded"))
        self.assertIn("unrecognized arguments", result.stderr)
        self.assertEqual(self.count(), 0)

    def test_repeated_publication_keeps_distinct_immutable_generations(self):
        first = self.publish()["published"][0]
        path = Path(first["local_directory"]) / "payload.tar"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        second = self.publish()["published"][0]
        self.assertEqual(first["artifact_key_id"], second["artifact_key_id"])
        self.assertNotEqual(first["local_directory"], second["local_directory"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.count("artifact_keys"), 1)

    def test_missing_newest_candidate_falls_back_to_older_production(self):
        first = self.publish()["published"][0]
        second = self.publish()["published"][0]
        (Path(second["local_directory"]) / "payload.tar").unlink()
        report = self.cli("restore")
        self.assertEqual(report["artifacts"][0]["cache_entry_id"], first["cache_entry_id"])
        with sqlite3.connect(self.database) as db:
            states = db.execute("SELECT state FROM cache_entries ORDER BY cache_entry_id").fetchall()
        self.assertEqual(states, [("ready",), ("invalid",)])

    def test_corrupt_bytes_are_a_miss_and_never_exposed(self):
        stored = self.publish()["published"][0]
        archive = Path(stored["local_directory"]) / "payload.tar"
        data = bytearray(archive.read_bytes())
        data[1024] ^= 1
        archive.write_bytes(data)
        result = self.cli("restore", success=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["needs_build"], 1)
        self.assertFalse(self.dest.exists())

    def test_existing_destination_is_not_overwritten_or_cache_retired(self):
        self.publish()
        self.dest.mkdir()
        (self.dest / "mine").write_text("keep")
        self.cli("restore", success=False)
        self.assertEqual((self.dest / "mine").read_text(), "keep")
        self.assertEqual(self.count("reusable_cache_entries"), 1)

    def test_restore_reconfigures_before_lookup(self):
        self.publish()
        (self.source / "input.txt").write_text("new source version")
        self.commit()
        result = self.cli("restore", success=False)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(self.dest.exists())

    def test_missing_sidecar_does_not_publish(self):
        # The output contract includes an artifact that configure never emits.
        expected = json.loads(self.expected.read_text())
        expected["artifacts"].append({"artifact_identity": "absent_lib_generic",
                                      "fingerprint_path": "absent_lib_generic.fprint"})
        self.expected.write_text(json.dumps(expected))
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), 0)

    def test_ready_transition_failure_rolls_back_database_rows(self):
        with sqlite3.connect(self.database) as db:
            db.executescript('''CREATE TRIGGER fixture_reject_ready
                BEFORE UPDATE OF state ON cache_entries WHEN NEW.state = 'ready'
                BEGIN SELECT RAISE(ABORT, 'fixture publication failure'); END;''')
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), 0)
        self.assertEqual(self.count("artifact_files"), 0)

    @unittest.skipIf(os.name == "nt", "symlink fixture requires local symlink privileges")
    def test_export_link_escape_is_rejected(self):
        self.run_command("cmake", "--build", str(self.build))
        export = self.build / "artifacts" / IDENTITY
        (export / "bad-link").symlink_to("../outside")
        self.cli("build-publish", success=False)
        self.assertEqual(self.count(), 0)

    def test_inventory_digest_checks_members_even_with_valid_object_digest(self):
        good = self.publish()["published"][0]
        malicious = self.cache / "objects" / "malformed"
        shutil.copytree(good["local_directory"], malicious)
        inventory = json.loads((malicious / "inventory.json").read_text())
        for item in inventory["members"]:
            if item["type"] == "file":
                item["sha256"] = "0" * 64
        (malicious / "inventory.json").write_text(json.dumps(inventory))
        key = ArtifactKey(IDENTITY, good["fingerprint"])
        with CacheDB(self.database) as db:
            bad = db.register_local_outputs(producer_run_id=good["producer_run_id"], outputs=[
                (key, malicious, [_object_record(malicious / name)
                                  for name in ("inventory.json", "payload.tar")])
            ])[0]
        report = self.cli("restore")
        self.assertEqual(report["artifacts"][0]["cache_entry_id"], good["cache_entry_id"])
        self.assertEqual(report["artifacts"][0]["rejected_entries"][0]["cache_entry_id"], bad.cache_entry_id)

    def test_archive_traversal_is_rejected_before_extraction(self):
        good = self.publish()["published"][0]
        malicious = self.cache / "objects" / "traversal"
        shutil.copytree(good["local_directory"], malicious)
        with tarfile.open(malicious / "payload.tar", "a") as archive:
            member = tarfile.TarInfo("../outside")
            member.size = 3
            archive.addfile(member, io.BytesIO(b"bad"))
        key = ArtifactKey(IDENTITY, good["fingerprint"])
        with CacheDB(self.database) as db:
            db.register_local_outputs(producer_run_id=good["producer_run_id"], outputs=[
                (key, malicious, [_object_record(malicious / name)
                                  for name in ("inventory.json", "payload.tar")])
            ])
        report = self.cli("restore")
        self.assertEqual(report["artifacts"][0]["cache_entry_id"], good["cache_entry_id"])
        self.assertFalse((self.dest / "outside").exists())


if __name__ == "__main__":
    unittest.main()
