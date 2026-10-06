#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_jax_bazel_cache.py"""

import io
import os
import ssl
import sys
import tempfile
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
    endpoint_address,
    main,
    probe_cache,
    resolve_cache_url,
    upload_allowed,
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
            endpoint_address("grpcs://wardite.cluster.engflow.com"),
            ("wardite.cluster.engflow.com", 443),
        )

    def test_explicit_port_is_honored(self):
        self.assertEqual(
            endpoint_address("grpcs://cache.example:8980"), ("cache.example", 8980)
        )


class ProbeCacheTest(unittest.TestCase):
    """Tests that the probe raises for any cache it cannot use."""

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
            probe_cache(url, Path("cert.crt"), Path("cert.key"))

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

    def test_raises_when_the_credentials_cannot_be_loaded(self):
        self.context.load_cert_chain.side_effect = ssl.SSLError("bad key")
        with self.assertRaises(CacheUnusableError):
            self._probe()

    def test_raises_when_tls13_rejects_the_certificate_after_the_handshake(self):
        # Under TLS 1.3 the handshake completes before the server checks the
        # client certificate, so the rejection only shows on the first read.
        self.tls.recv.side_effect = ssl.SSLError("TLSV1_ALERT_UNKNOWN_CA")
        with self.assertRaises(CacheUnusableError):
            self._probe()

    def test_raises_when_the_server_closes_the_connection(self):
        self.tls.recv.return_value = b""
        with self.assertRaises(CacheUnusableError):
            self._probe()

    def test_raises_when_the_url_has_no_host(self):
        with self.assertRaises(CacheUnusableError):
            self._probe(url="not-a-url")


class BazelCacheOptionsTest(unittest.TestCase):
    """Tests the emitted build/build.py options."""

    def _options(self, allow_upload):
        # Paths inside the build container, so they stay POSIX even when the
        # test itself runs on Windows.
        return bazel_cache_options(
            REMOTE_CACHE_URL,
            PurePosixPath("/data/ci-cert.crt"),
            PurePosixPath("/data/ci-cert.key"),
            allow_upload,
        )

    def test_points_bazel_at_the_cache_with_credentials(self):
        options = self._options(allow_upload=False)
        self.assertIn(f"--bazel_options=--remote_cache={REMOTE_CACHE_URL}", options)
        self.assertIn(
            "--bazel_options=--tls_client_certificate=/data/ci-cert.crt", options
        )
        self.assertIn("--bazel_options=--tls_client_key=/data/ci-cert.key", options)

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
        credentials = tempfile.TemporaryDirectory()
        self.addCleanup(credentials.cleanup)
        self.certificate = Path(credentials.name) / "ci-cert.crt"
        self.key = Path(credentials.name) / "ci-cert.key"
        # Only existence matters here; the handshake that would read these is
        # mocked out, and real PEM headers trip secret scanning.
        self.certificate.write_text("test certificate")
        self.key.write_text("test key")

    def _main(self, argv, reachable=True, with_credentials=True):
        # Always pass explicit paths: a developer machine may happen to have
        # credentials at the default location, which would make the
        # missing-credentials case pass for the wrong reason.
        certificate, key = self.certificate, self.key
        if not with_credentials:
            certificate = self.certificate.with_name("absent.crt")
            key = self.key.with_name("absent.key")
        argv = argv + [
            "--client-certificate",
            str(certificate),
            "--client-key",
            str(key),
        ]
        probe = mock.patch.object(
            configure_jax_bazel_cache,
            "probe_cache",
            side_effect=None if reachable else CacheUnusableError("rejected"),
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
        ), mock.patch.object(sys, "stdout", stdout):
            main(argv)
        return stdout.getvalue().strip()

    def test_prints_options_for_a_reachable_cache(self):
        output = self._main(
            ["--release-type", "ci", "--event-name", "push", "--ref", "refs/heads/main"]
        )
        self.assertIn(f"--bazel_options=--remote_cache={REMOTE_CACHE_URL}", output)
        self.assertIn("--bazel_options=--remote_upload_local_results=true", output)

    def test_prints_nothing_without_credentials(self):
        # Runners without `/data/ci-cert.*` leave the mount absent, so the
        # build command must come out unchanged.
        self.assertEqual(
            self._main(["--release-type", "ci"], with_credentials=False), ""
        )

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


if __name__ == "__main__":
    unittest.main()
