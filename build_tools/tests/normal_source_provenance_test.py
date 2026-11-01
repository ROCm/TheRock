# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Real inert Git and archive bytes; no compiler, package, or GPU execution."""

import gzip
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "source_identity", ROOT / "build_tools/normal_ci_input_identity.py"
)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
SHA = "a" * 40


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="normal-source-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.workspace = self.base / "checkout"
        self.workspace.mkdir()
        self.env = dict(
            os.environ,
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=str(self.base / "global-config"),
        )

    def git(self, *args, env=None, check=True):
        return subprocess.run(
            ["git", "-C", str(self.workspace), *args],
            env=env or self.env,
            capture_output=True,
            text=True,
            check=check,
        )

    def init(self):
        self.git("init")
        (self.workspace / "README.md").write_bytes(b"inert source fixture\n")
        self.git("add", "README.md")
        self.git(
            "-c",
            "user.name=Source Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "inert source",
        )
        return self.git("rev-parse", "HEAD").stdout.strip()

    def archive(self, *, entries=None, sha=SHA):
        entries = entries or [("README.md", b"inert source fixture\n", 0o644)]
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as archive:
            for name, body, mode in entries:
                member = tarfile.TarInfo(f"TheRock-{sha}/{name}")
                member.mode, member.size = mode, len(body)
                archive.addfile(member, io.BytesIO(body))
        return gzip.compress(raw.getvalue())

    def consumer(self):
        return dict(
            GITHUB_ACTIONS="true",
            GITHUB_REPOSITORY="ROCm/TheRock",
            GITHUB_EVENT_NAME="workflow_dispatch",
            GITHUB_JOB="test_native_linux_packages_install",
            GITHUB_SHA=SHA,
            GITHUB_RUN_ID="123",
            GITHUB_RUN_ATTEMPT="1",
            REPO_URL="https://therock-ci-artifacts.s3.amazonaws.com/123-linux/packages/rpm",
        )

    def test_real_dubious_owner_is_trusted_only_for_command_and_dirty_bytes_remain_measured(
        self,
    ):
        head = self.init()
        different = dict(self.env, GIT_TEST_ASSUME_DIFFERENT_OWNER="1", GITHUB_SHA=head)
        rejected = self.git("rev-parse", "HEAD", env=different, check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("dubious ownership", rejected.stderr)
        target = self.workspace / "README.md"
        target.write_bytes(b"reviewed dirty producer fixture\n")
        with patch.dict(os.environ, different, clear=True):
            source = SOURCE.capture_source(self.workspace, {}, digest, different)
        self.assertEqual(source["mode"], "measured-git")
        self.assertEqual(source["measured_git_head"], head)
        changed = source["repositories"][0]["changed_files"]
        self.assertEqual(
            changed,
            [
                dict(
                    path="README.md", sha256=digest(target), bytes=target.stat().st_size
                )
            ],
        )
        self.assertIn("README.md", source["working_changes"]["stdout"])
        self.assertFalse((self.base / "global-config").exists())
        self.assertNotEqual(
            self.git("rev-parse", "HEAD", env=different, check=False).returncode, 0
        )
        with self.assertRaises(ValueError):
            SOURCE.scoped_git(self.workspace, self.base, "rev-parse", "HEAD")

    def test_git_errors_and_ci_head_mismatch_never_fall_back_to_archive(self):
        self.init()
        with patch.dict(os.environ, self.env, clear=True), self.assertRaises(
            ValueError
        ):
            SOURCE.capture_source(self.workspace, {}, digest, {"GITHUB_SHA": SHA})
        with patch.dict(
            os.environ,
            dict(self.env, GIT_DIR=str(self.base / "missing-git")),
            clear=True,
        ), self.assertRaises(subprocess.CalledProcessError):
            SOURCE.capture_source(self.workspace, {}, digest, {})

    def test_nested_producer_clr_history_and_dirty_files_remain_measured(self):
        head = self.init()
        nested = self.workspace / "rocm-systems"
        nested.mkdir()
        target = nested / "projects/clr/CMakeLists.txt"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"inert CLR history fixture\n")
        for argv in [
            ["init"],
            ["add", "projects/clr/CMakeLists.txt"],
            [
                "-c",
                "user.name=Nested Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-m",
                "original CLR version directory",
            ],
        ]:
            subprocess.run(
                ["git", "-C", str(nested), *argv],
                env=self.env,
                check=True,
                capture_output=True,
            )
        original = subprocess.check_output(
            ["git", "-C", str(nested), "rev-parse", "HEAD"], env=self.env, text=True
        ).strip()
        target.write_bytes(b"reviewed dirty source, same version commit\n")
        different = dict(self.env, GIT_TEST_ASSUME_DIFFERENT_OWNER="1", GITHUB_SHA=head)
        with patch.dict(os.environ, different, clear=True):
            source = SOURCE.capture_source(
                self.workspace, {"rocm-systems": original}, digest, different
            )
        row = next(r for r in source["repositories"] if r["path"] == "rocm-systems")
        self.assertEqual(row["head"], original)
        self.assertEqual(row["clr_last_directory_commit"]["stdout"], original)
        self.assertEqual(row["changed_files"][0]["sha256"], digest(target))
        self.assertEqual(row["is_shallow"], "false")

    def test_gitless_exact_files_and_producer_url_are_verified_without_claiming_head(
        self,
    ):
        (self.workspace / "README.md").write_bytes(b"inert source fixture\n")
        archive = self.archive()
        with patch.object(SOURCE.shutil, "which", return_value=None), patch.object(
            SOURCE, "download_checkout_archive", return_value=archive
        ) as fetch:
            result = SOURCE.capture_source(self.workspace, {}, digest, self.consumer())
            fetch.assert_called_once_with(SHA)
            fetch.reset_mock()
            with self.assertRaises(ValueError):
                SOURCE.capture_source(
                    self.workspace,
                    {},
                    digest,
                    dict(self.consumer(), GITHUB_JOB="build_stage"),
                )
            fetch.assert_not_called()
        self.assertIsNone(result["measured_git_head"])
        self.assertEqual(result["ci_declared_sha"], SHA)
        self.assertTrue(result["checkout_archive"]["complete"])
        self.assertEqual(
            result["checkout_archive"]["files"][0]["sha256"],
            digest(self.workspace / "README.md"),
        )
        self.assertEqual(result["producer_references"]["run_id"], "123")
        self.assertIsNone(result["working_changes"])

    def test_gitless_tamper_missing_wrong_sha_archive_and_wrong_producer_fail(self):
        target = self.workspace / "README.md"
        target.write_bytes(b"tampered\n")
        with self.assertRaises(ValueError):
            SOURCE.compare_checkout_archive(self.workspace, SHA, self.archive(), digest)
        target.unlink()
        with self.assertRaises(ValueError):
            SOURCE.compare_checkout_archive(self.workspace, SHA, self.archive(), digest)
        target.write_bytes(b"inert source fixture\n")
        for archive in [
            self.archive(sha="b" * 40),
            self.archive(entries=[("../escape", b"x", 0o644)]),
        ]:
            with self.assertRaises(ValueError):
                SOURCE.compare_checkout_archive(self.workspace, SHA, archive, digest)
        with self.assertRaises((EOFError, OSError)):
            SOURCE.compare_checkout_archive(
                self.workspace, SHA, self.archive()[:-5], digest
            )
        with self.assertRaises(OSError):
            SOURCE.gitless_consumer_source(
                self.workspace,
                self.consumer(),
                digest,
                fetch=lambda sha: (_ for _ in ()).throw(OSError("archive unavailable")),
            )
        for updates in [
            dict(GITHUB_JOB="build_stage"),
            dict(
                REPO_URL="https://therock-ci-artifacts.s3.amazonaws.com/999-linux/packages/rpm"
            ),
            dict(GITHUB_SHA="main"),
        ]:
            with self.assertRaises(ValueError):
                SOURCE.gitless_consumer_source(
                    self.workspace,
                    dict(self.consumer(), **updates),
                    digest,
                    fetch=lambda sha: self.archive(),
                )

    @unittest.skipUnless(os.name == "posix", "Executable-bit and symlink semantics")
    def test_gitless_type_and_executable_mode_are_not_normalized(self):
        target = self.workspace / "README.md"
        target.write_bytes(b"inert source fixture\n")
        target.chmod(0o755)
        with self.assertRaises(ValueError):
            SOURCE.compare_checkout_archive(self.workspace, SHA, self.archive(), digest)
        target.unlink()
        target.symlink_to("elsewhere")
        with self.assertRaises(ValueError):
            SOURCE.compare_checkout_archive(self.workspace, SHA, self.archive(), digest)

    @unittest.skipUnless(os.name == "posix", "Executable-bit and symlink semantics")
    def test_gitless_exact_executable_and_symlink_inventory_passes(self):
        target = self.workspace / "README.md"
        target.write_bytes(b"inert source fixture\n")
        target.chmod(0o755)
        alias = self.workspace / "alias"
        alias.symlink_to("README.md")
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as archive:
            file = tarfile.TarInfo(f"TheRock-{SHA}/README.md")
            file.mode, file.size = 0o755, target.stat().st_size
            archive.addfile(file, io.BytesIO(target.read_bytes()))
            link = tarfile.TarInfo(f"TheRock-{SHA}/alias")
            link.type, link.linkname = tarfile.SYMTYPE, "README.md"
            archive.addfile(link)
        result = SOURCE.compare_checkout_archive(
            self.workspace, SHA, gzip.compress(raw.getvalue()), digest
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["verified_files"], 2)
        self.assertTrue(result["files"][0]["executable"])
        self.assertEqual(result["files"][1]["link_target"], "README.md")
        alias.unlink()
        alias.symlink_to("other")
        with self.assertRaises(ValueError):
            SOURCE.compare_checkout_archive(
                self.workspace, SHA, gzip.compress(raw.getvalue()), digest
            )


if __name__ == "__main__":
    unittest.main()
