#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_jax_bazel_cache.py"""

import base64
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import configure_jax_bazel_cache
from configure_jax_bazel_cache import (
    HTTP2_CLIENT_PREFACE,
    REMOTE_CACHE_URL,
    CacheUnusableError,
    bazel_cache_options,
    check_token,
    credential_helper_response,
    endpoint_address,
    main,
    probe_cache,
    resolve_cache_url,
    upload_allowed,
)


def _segment(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")


def fake_jwt(expires: float) -> str:
    """A JWT-shaped string built at test time; literal tokens trip scanners."""
    return ".".join(
        [_segment({"alg": "none"}), _segment({"exp": expires}), "test-signature"]
    )


class ResolveCacheUrlTest(unittest.TestCase):
    """Tests which release types may read the shared cache."""

    def test_ci_uses_the_shared_cache(self):
        self.assertEqual(resolve_cache_url("", "ci"), REMOTE_CACHE_URL)

    def test_release_builds_get_no_shared_cache(self):
        # Release builds need a cache of their own, as nightly has for ccache.
        for release_type in ("dev", "dev-bkc", "nightly", "nightly-bkc", "prerelease"):
            with self.subTest(release_type=release_type):
                self.assertEqual(resolve_cache_url("", release_type), "")

    def test_unknown_release_type_gets_no_cache(self):
        self.assertEqual(resolve_cache_url("", "something-new"), "")

    def test_explicit_url_repoints_the_cache(self):
        self.assertEqual(
            resolve_cache_url("grpcs://cache.example", "ci"), "grpcs://cache.example"
        )

    def test_explicit_url_does_not_enable_the_cache(self):
        # The override only moves the cache; a release type that may not use
        # it stays isolated whatever the repository variable says.
        self.assertEqual(resolve_cache_url("grpcs://cache.example", "prerelease"), "")


class UploadAllowedTest(unittest.TestCase):
    """Tests which runs may write results back to the cache."""

    def test_push_to_main_uploads(self):
        self.assertTrue(upload_allowed("push", "refs/heads/main"))

    def test_everything_else_reads_only(self):
        for event_name, ref in (
            ("pull_request", "refs/pull/1/merge"),
            ("workflow_dispatch", "refs/heads/main"),
            ("workflow_dispatch", "refs/heads/users/someone/branch"),
            ("push", "refs/heads/multi_arch/experiment"),
            ("", ""),
        ):
            with self.subTest(event_name=event_name, ref=ref):
                self.assertFalse(upload_allowed(event_name, ref))


class EndpointAddressTest(unittest.TestCase):
    """Tests parsing the host and port out of a gRPC cache URL."""

    def test_default_port_is_tls(self):
        self.assertEqual(
            endpoint_address("grpcs://actinolite.cluster.engflow.com"),
            ("actinolite.cluster.engflow.com", 443),
        )

    def test_explicit_port_is_honored(self):
        self.assertEqual(
            endpoint_address("grpcs://cache.example:8980"), ("cache.example", 8980)
        )


class CheckTokenTest(unittest.TestCase):
    """Tests that only an unexpired JWT passes."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.token_file = Path(tmp.name) / "engflow-cache.jwt"

    def _check(self, contents: str, now: float = 1000.0):
        self.token_file.write_text(contents)
        check_token(self.token_file, now=now)

    def test_unexpired_token_passes(self):
        self._check(fake_jwt(expires=2000.0) + "\n")

    def test_expired_token_raises(self):
        with self.assertRaisesRegex(CacheUnusableError, "expired"):
            self._check(fake_jwt(expires=500.0))

    def test_malformed_tokens_raise(self):
        for contents in (
            "",
            "not-a-jwt",
            "a.b",
            "a.!!!.c",
            f"a.{_segment({'no_exp': 1})}.c",
            f"a.{base64.urlsafe_b64encode(b'[1]').decode()}.c",
        ):
            with self.subTest(contents=contents):
                with self.assertRaises(CacheUnusableError):
                    self._check(contents)

    def test_errors_never_include_the_token(self):
        token = fake_jwt(expires=500.0)
        with self.assertRaises(CacheUnusableError) as raised:
            self._check(token)
        self.assertNotIn(token.split(".")[1], str(raised.exception))


class ProbeCacheTest(unittest.TestCase):
    """Tests that the probe raises for any cache it cannot reach."""

    def setUp(self):
        self.context = mock.MagicMock()
        self.tls = self.context.wrap_socket.return_value.__enter__.return_value
        self.tls.recv.return_value = b"\x00"

    def _probe(self, url=REMOTE_CACHE_URL, connect_effect=None):
        with mock.patch.object(
            configure_jax_bazel_cache.ssl,
            "create_default_context",
            return_value=self.context,
        ), mock.patch.object(
            configure_jax_bazel_cache.socket,
            "create_connection",
            side_effect=connect_effect,
        ):
            probe_cache(url)

    def test_usable_when_the_server_answers_the_preface(self):
        self._probe()
        self.tls.sendall.assert_called_once_with(HTTP2_CLIENT_PREFACE)
        self.context.set_alpn_protocols.assert_called_once_with(["h2"])

    def test_raises_when_the_host_does_not_resolve(self):
        with self.assertRaises(CacheUnusableError):
            self._probe(connect_effect=OSError("name resolution failed"))

    def test_raises_when_the_connection_times_out(self):
        with self.assertRaises(CacheUnusableError):
            self._probe(connect_effect=TimeoutError())

    def test_raises_when_the_server_closes_the_connection(self):
        self.tls.recv.return_value = b""
        with self.assertRaises(CacheUnusableError):
            self._probe()

    def test_raises_when_the_url_has_no_host(self):
        with self.assertRaises(CacheUnusableError):
            self._probe(url="not-a-url")


class CredentialHelperResponseTest(unittest.TestCase):
    """Tests the headers handed to Bazel."""

    def test_sends_the_token_as_an_engflow_jwt(self):
        self.assertEqual(
            credential_helper_response("test-token"),
            {
                "headers": {
                    "x-engflow-auth-method": ["jwt-v0"],
                    "x-engflow-auth-token": ["test-token"],
                }
            },
        )


class BazelCacheOptionsTest(unittest.TestCase):
    """Tests the emitted build/build.py options."""

    # A path inside the build container, so it stays POSIX even when the test
    # itself runs on Windows.
    HELPER = PurePosixPath("/workspace/configure_jax_bazel_cache.py")

    def _options(self, allow_upload, url=REMOTE_CACHE_URL):
        return bazel_cache_options(url, self.HELPER, allow_upload)

    def test_points_bazel_at_the_cache(self):
        self.assertIn(
            f"--bazel_options=--remote_cache={REMOTE_CACHE_URL}",
            self._options(allow_upload=False),
        )

    def test_credential_helper_is_scoped_to_the_cache_host(self):
        # An unscoped helper would send the token to every host Bazel fetches.
        self.assertIn(
            f"--bazel_options=--credential_helper=cache.example={self.HELPER}",
            self._options(allow_upload=False, url="grpcs://cache.example:8980"),
        )

    def test_never_requests_remote_execution(self):
        # The build uses --config=rocm_release_wheel, so it must stay
        # cache-only: actions execute on the runner.
        self.assertFalse(
            [option for option in self._options(True) if "remote_executor" in option]
        )

    def test_upload_is_off_for_read_only_runs(self):
        self.assertIn(
            "--bazel_options=--remote_upload_local_results=false",
            self._options(allow_upload=False),
        )

    def test_upload_is_on_for_trusted_runs(self):
        self.assertIn(
            "--bazel_options=--remote_upload_local_results=true",
            self._options(allow_upload=True),
        )


class MainTest(unittest.TestCase):
    """Tests that stdout carries options only when the cache is usable."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Never the real default path: a machine that happens to hold a token
        # there would make the missing-token case pass for the wrong reason.
        self.token_file = Path(tmp.name) / "engflow-cache.jwt"
        self.token = fake_jwt(expires=time.time() + 3600)
        self.token_file.write_text(self.token + "\n")
        patcher = mock.patch.object(
            configure_jax_bazel_cache, "TOKEN_FILE", self.token_file
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _main(self, argv, reachable=True, stdin=""):
        probe = mock.patch.object(
            configure_jax_bazel_cache,
            "probe_cache",
            side_effect=None if reachable else CacheUnusableError("unreachable"),
        )
        # The defaults read the runner's own environment; keep it out so a test
        # run on a push to main cannot flip the upload decision.
        github_env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in ("GITHUB_EVENT_NAME", "GITHUB_REF", "JAX_BAZEL_REMOTE_CACHE_URL")
        }
        stdout = io.StringIO()
        with probe, mock.patch.dict(
            os.environ, github_env, clear=True
        ), mock.patch.object(sys, "stdout", stdout), mock.patch.object(
            sys, "stdin", io.StringIO(stdin)
        ):
            main(argv)
        return stdout.getvalue().strip()

    def test_prints_options_for_a_reachable_cache(self):
        output = self._main(
            ["--release-type", "ci", "--event-name", "push", "--ref", "refs/heads/main"]
        )
        self.assertIn(f"--bazel_options=--remote_cache={REMOTE_CACHE_URL}", output)
        self.assertIn("--bazel_options=--remote_upload_local_results=true", output)

    def test_options_never_carry_the_token(self):
        # build.py --verbose logs the full command line.
        self.assertNotIn(self.token, self._main(["--release-type", "ci"]))

    def test_prints_nothing_without_a_token(self):
        # Runners without the token leave the mount absent, so the build
        # command must come out unchanged.
        self.token_file.unlink()
        self.assertEqual(self._main(["--release-type", "ci"]), "")

    def test_fails_when_the_token_has_expired(self):
        self.token_file.write_text(fake_jwt(expires=time.time() - 60))
        with self.assertRaisesRegex(CacheUnusableError, "expired"):
            self._main(["--release-type", "ci"])

    def test_fails_when_the_cache_is_unusable(self):
        with self.assertRaises(CacheUnusableError):
            self._main(["--release-type", "ci"], reachable=False)

    def test_prints_nothing_when_the_release_type_has_no_cache(self):
        self.assertEqual(self._main(["--release-type", "prerelease"]), "")

    def test_probe_is_skipped_when_no_cache_is_configured(self):
        with mock.patch.object(
            configure_jax_bazel_cache,
            "probe_cache",
            side_effect=AssertionError("should not be called"),
        ):
            main(["--release-type", "prerelease"])

    def test_upload_defaults_to_read_only(self):
        output = self._main(["--release-type", "ci"])
        self.assertIn("--bazel_options=--remote_upload_local_results=false", output)

    def test_get_answers_the_credential_helper_protocol(self):
        output = self._main(
            ["get"], stdin='{"uri": "https://actinolite.cluster.engflow.com/x"}'
        )
        self.assertEqual(json.loads(output), credential_helper_response(self.token))


if __name__ == "__main__":
    unittest.main()
