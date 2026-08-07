# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from setup_ccache import _apply_remote_opts, _redact_userinfo

_BAZEL = (
    "http://bazelremote-svc.bazelremote-ns.svc.cluster.local:8080"
    "|layout=bazel|connect-timeout=50"
)


def _args(read_only=False):
    return argparse.Namespace(read_only=read_only)


class ApplyRemoteOptsTest(unittest.TestCase):
    """Access-tier transforms applied to a remote_storage string."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_unchanged(self):
        # No creds and no --read-only -> byte-identical to input.
        self.assertEqual(_apply_remote_opts(_BAZEL, _args()), _BAZEL)

    @patch.dict(os.environ, {}, clear=True)
    def test_read_only_appends_attribute(self):
        self.assertEqual(
            _apply_remote_opts(_BAZEL, _args(read_only=True)), _BAZEL + "|read-only"
        )

    @patch.dict(
        os.environ,
        {"CCACHE_REMOTE_USER": "ci", "CCACHE_REMOTE_PASSWORD": "p@ss:w/rd"},
        clear=True,
    )
    def test_creds_injected_url_encoded(self):
        out = _apply_remote_opts(_BAZEL, _args())
        # Userinfo is URL-encoded and placed after the scheme, before the host.
        self.assertTrue(out.startswith("http://ci:p%40ss%3Aw%2Frd@bazelremote-svc"))
        # Attributes after the first '|' are preserved.
        self.assertTrue(out.endswith("|layout=bazel|connect-timeout=50"))

    @patch.dict(
        os.environ,
        {"CCACHE_REMOTE_USER": "ci", "CCACHE_REMOTE_PASSWORD": "secret"},
        clear=True,
    )
    def test_creds_and_read_only_combined(self):
        out = _apply_remote_opts(_BAZEL, _args(read_only=True))
        self.assertIn("http://ci:secret@bazelremote-svc", out)
        self.assertTrue(out.endswith("|read-only"))

    @patch.dict(
        os.environ,
        {"CCACHE_REMOTE_USER": "ci", "CCACHE_REMOTE_PASSWORD": "secret"},
        clear=True,
    )
    def test_non_http_scheme_not_touched(self):
        # redis:// (or any non-http scheme) gets no userinfo injection.
        redis = (
            "redis://redis-ccache-svc.redis-ccache-ns.svc.cluster.local:6379"
            "|connect-timeout=50"
        )
        self.assertEqual(_apply_remote_opts(redis, _args()), redis)

    @patch.dict(os.environ, {"CCACHE_REMOTE_USER": "ci"}, clear=True)
    def test_partial_creds_ignored(self):
        # User without password -> no injection (both are required).
        self.assertEqual(_apply_remote_opts(_BAZEL, _args()), _BAZEL)

    @patch.dict(
        os.environ,
        {"CCACHE_REMOTE_USER": "ci", "CCACHE_REMOTE_PASSWORD": "s3cr3t"},
        clear=True,
    )
    def test_redact_masks_password(self):
        # redact=True builds a log-safe string: the real password never appears;
        # the userinfo is shown as ci:***.
        out = _apply_remote_opts(_BAZEL, _args(), redact=True)
        self.assertNotIn("s3cr3t", out)
        self.assertTrue(out.startswith("http://ci:***@bazelremote-svc"))
        self.assertTrue(out.endswith("|layout=bazel|connect-timeout=50"))

    @patch.dict(os.environ, {}, clear=True)
    def test_redact_no_creds_unchanged(self):
        # With no creds, redact=True leaves the string byte-identical.
        self.assertEqual(_apply_remote_opts(_BAZEL, _args(), redact=True), _BAZEL)


class RedactUserinfoTest(unittest.TestCase):
    """Password redaction so injected credentials are not echoed to CI logs."""

    def test_hides_password(self):
        line = "  remote_storage = http://ci:s3cr3t@bazelremote-svc:8080|layout=bazel"
        redacted = _redact_userinfo(line)
        self.assertNotIn("s3cr3t", redacted)
        self.assertIn("http://ci:***@bazelremote-svc:8080", redacted)

    def test_hides_url_encoded_password(self):
        line = "remote_storage = http://ci:p%40ss%3Aw%2Frd@host:8080|read-only"
        redacted = _redact_userinfo(line)
        self.assertNotIn("p%40ss", redacted)
        self.assertIn("http://ci:***@host:8080", redacted)

    def test_noop_without_userinfo(self):
        line = "remote_storage = " + _BAZEL
        self.assertEqual(_redact_userinfo(line), line)


if __name__ == "__main__":
    unittest.main()
