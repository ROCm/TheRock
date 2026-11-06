#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for write_jax_bazel_cache_credentials.py"""

import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import write_jax_bazel_cache_credentials
from write_jax_bazel_cache_credentials import (
    TOKEN_NAME,
    WRITER_AWS_PROFILE,
    WRITER_ROLE,
    WRITER_SECRET_ID,
    fetch_writer_token,
    main,
    materialize_credentials,
    uses_writer_token,
)

MAIN_PUSH = ["--event-name", "push", "--ref", "refs/heads/main"]


def _outputs(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


class UsesWriterTokenTest(unittest.TestCase):
    """Tests which runs fetch the cache-writer token."""

    def test_push_to_main_in_therock_writes(self):
        self.assertTrue(uses_writer_token("push", "refs/heads/main", "ROCm/TheRock"))

    def test_everything_else_reads(self):
        for event_name, ref, repository in (
            ("pull_request", "refs/pull/1/merge", "ROCm/TheRock"),
            ("workflow_dispatch", "refs/heads/main", "ROCm/TheRock"),
            ("push", "refs/heads/users/someone/branch", "ROCm/TheRock"),
            # A fork's own main must not try the role: it would fail the job.
            ("push", "refs/heads/main", "someone/TheRock"),
        ):
            with self.subTest(event_name=event_name, ref=ref, repository=repository):
                self.assertFalse(uses_writer_token(event_name, ref, repository))


class FetchWriterTokenTest(unittest.TestCase):
    """Tests reading the writer JWT from Secrets Manager."""

    def _fetch(self, secret: dict, github_actions: bool = False):
        session = mock.MagicMock()
        client = session.return_value.client.return_value
        client.get_secret_value.return_value = {"SecretString": json.dumps(secret)}
        stdout = io.StringIO()
        env = {"GITHUB_ACTIONS": "true" if github_actions else "false"}
        with mock.patch.object(
            write_jax_bazel_cache_credentials.boto3, "Session", session
        ), mock.patch.dict(os.environ, env), mock.patch.object(sys, "stdout", stdout):
            token = fetch_writer_token()
        return token, session, client, stdout.getvalue()

    def test_reads_the_token_property_with_the_writer_profile(self):
        token, session, client, _ = self._fetch({"token": "test-writer\n"})
        self.assertEqual(token, "test-writer")
        self.assertEqual(session.call_args.kwargs["profile_name"], WRITER_AWS_PROFILE)
        client.get_secret_value.assert_called_once_with(SecretId=WRITER_SECRET_ID)

    def test_masks_the_token_in_github_actions(self):
        _, _, _, stdout = self._fetch({"token": "test-writer"}, github_actions=True)
        self.assertEqual(stdout.strip(), "::add-mask::test-writer")

    def test_raises_without_a_token(self):
        for secret in ({}, {"token": ""}):
            with self.subTest(secret=secret):
                with self.assertRaises(ValueError):
                    self._fetch(secret)


class MaterializeCredentialsTest(unittest.TestCase):
    """Tests that the runner-local token is copied privately, or skipped."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source_dir = Path(self.tmp.name) / "source"
        self.dest_dir = Path(self.tmp.name) / "dest"
        self.source_dir.mkdir()

    def _write_source(self) -> None:
        # Existence and size are what matter; a real JWT would trip scanners.
        (self.source_dir / TOKEN_NAME).write_text("test token")

    def test_copies_the_token_when_present(self):
        self._write_source()
        copied = materialize_credentials(self.source_dir, self.dest_dir)
        self.assertEqual(copied, self.dest_dir)
        self.assertEqual((self.dest_dir / TOKEN_NAME).read_text(), "test token")

    def test_returns_none_when_the_token_is_missing(self):
        self.assertIsNone(materialize_credentials(self.source_dir, self.dest_dir))
        self.assertFalse(self.dest_dir.exists())

    def test_raises_when_the_token_is_empty(self):
        # Present but unusable is a broken runner, not an unprovisioned one.
        (self.source_dir / TOKEN_NAME).write_text("")
        with self.assertRaises(ValueError):
            materialize_credentials(self.source_dir, self.dest_dir)

    def test_replaces_a_stale_destination(self):
        self._write_source()
        self.dest_dir.mkdir()
        (self.dest_dir / "stale").write_text("leftover")
        materialize_credentials(self.source_dir, self.dest_dir)
        self.assertFalse((self.dest_dir / "stale").exists())
        self.assertTrue((self.dest_dir / TOKEN_NAME).is_file())

    @unittest.skipIf(os.name == "nt", "POSIX mode bits")
    def test_restricts_destination_permissions(self):
        self._write_source()
        materialize_credentials(self.source_dir, self.dest_dir)
        dir_mode = stat.S_IMODE(self.dest_dir.stat().st_mode)
        token_mode = stat.S_IMODE((self.dest_dir / TOKEN_NAME).stat().st_mode)
        self.assertEqual(dir_mode, 0o700)
        self.assertEqual(token_mode, 0o600)


class MainTest(unittest.TestCase):
    """Tests that GITHUB_OUTPUT receives the destination path, or empty."""

    def _run(self, source_dir: Path, dest_dir: Path, extra=()) -> str:
        # Explicit event arguments, so a test run on a push to main cannot
        # take the writer path by accident.
        argv = ["--event-name", "pull_request", "--ref", "refs/pull/1/merge"]
        argv += ["--repository", "ROCm/TheRock", *extra]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "github_output"
            env = {"GITHUB_OUTPUT": str(output), "RUNNER_TEMP": tmp}
            with mock.patch.dict(os.environ, env, clear=False):
                main(
                    argv
                    + ["--source-dir", str(source_dir), "--dest-dir", str(dest_dir)]
                )
            return output.read_text()

    def test_plan_names_the_writer_role_only_for_main_pushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = _outputs(self._run(Path(tmp), Path(tmp), ["--plan", *MAIN_PUSH]))
            self.assertEqual(plan["writer_role"], WRITER_ROLE)
            self.assertEqual(plan["aws_profile"], WRITER_AWS_PROFILE)
            plan = _outputs(self._run(Path(tmp), Path(tmp), ["--plan"]))
            self.assertEqual(plan["writer_role"], "")

    def test_main_push_stages_the_writer_not_the_runner_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            dest_dir = Path(tmp) / "dest"
            source_dir.mkdir()
            (source_dir / TOKEN_NAME).write_text("test reader")
            with mock.patch.object(
                write_jax_bazel_cache_credentials,
                "fetch_writer_token",
                return_value="test writer",
            ):
                output = self._run(source_dir, dest_dir, MAIN_PUSH)
            self.assertEqual((dest_dir / TOKEN_NAME).read_text(), "test writer")
        self.assertIn(f"credentials_dir={dest_dir}", output)

    def test_writer_fetch_failure_fails_the_job(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            write_jax_bazel_cache_credentials,
            "fetch_writer_token",
            side_effect=RuntimeError("AccessDenied"),
        ):
            with self.assertRaises(RuntimeError):
                self._run(Path(tmp), Path(tmp) / "dest", MAIN_PUSH)

    def test_main_writes_the_destination_when_the_token_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            dest_dir = Path(tmp) / "dest"
            source_dir.mkdir()
            (source_dir / TOKEN_NAME).write_text("test token")
            output = self._run(source_dir, dest_dir)
        self.assertIn(f"credentials_dir={dest_dir}", output)

    def test_main_writes_empty_when_the_token_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            dest_dir = Path(tmp) / "dest"
            source_dir.mkdir()
            output = self._run(source_dir, dest_dir)
        self.assertRegex(output, r"credentials_dir=\s*$")


if __name__ == "__main__":
    unittest.main()
