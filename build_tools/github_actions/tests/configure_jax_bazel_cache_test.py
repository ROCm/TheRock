#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for configure_jax_bazel_cache.py"""

import io
import ssl
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

import configure_jax_bazel_cache
from configure_jax_bazel_cache import (
    REMOTE_CACHE_URL,
    bazel_cache_options,
    endpoint_address,
    main,
    probe_cache,
    resolve_cache_url,
)


class ResolveCacheUrlTest(unittest.TestCase):
    """Tests which release types may read the shared cache."""

    def test_ci_and_dev_use_the_shared_cache(self):
        for release_type in ("ci", "dev", "dev-bkc", "nightly"):
            self.assertEqual(resolve_cache_url("", release_type), REMOTE_CACHE_URL)

    def test_release_builds_get_no_shared_cache(self):
        # Stable releases repackage prerelease artifacts, so they must not read
        # entries another build could have written.
        for release_type in ("nightly-bkc", "prerelease"):
            self.assertEqual(resolve_cache_url("", release_type), "")

    def test_unknown_release_type_gets_no_cache(self):
        self.assertEqual(resolve_cache_url("", "something-new"), "")

    def test_explicit_url_overrides_the_release_type_default(self):
        self.assertEqual(
            resolve_cache_url("grpcs://cache.example", "prerelease"),
            "grpcs://cache.example",
        )


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
    """Tests that the probe reports unreachable rather than raising."""

    def _probe(self, url=REMOTE_CACHE_URL, connect_effect=None, load_effect=None):
        context = mock.MagicMock()
        context.load_cert_chain.side_effect = load_effect
        with mock.patch.object(
            configure_jax_bazel_cache.ssl,
            "create_default_context",
            return_value=context,
        ), mock.patch.object(
            configure_jax_bazel_cache.socket,
            "create_connection",
            side_effect=connect_effect,
        ):
            return probe_cache(url, Path("cert.crt"), Path("cert.key"))

    def test_reachable_when_the_handshake_completes(self):
        self.assertTrue(self._probe())

    def test_unreachable_when_the_host_does_not_resolve(self):
        self.assertFalse(self._probe(connect_effect=OSError("name resolution failed")))

    def test_unreachable_when_the_connection_times_out(self):
        self.assertFalse(self._probe(connect_effect=TimeoutError()))

    def test_unreachable_when_the_credentials_are_rejected(self):
        # A stale or mismatched certificate is reported once here instead of as
        # a wall of Bazel cache warnings during the build.
        self.assertFalse(self._probe(load_effect=ssl.SSLError("bad key")))

    def test_unreachable_when_the_url_has_no_host(self):
        self.assertFalse(self._probe(url="not-a-url"))


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
        stdout = io.StringIO()
        with mock.patch.object(
            configure_jax_bazel_cache, "probe_cache", return_value=reachable
        ), mock.patch.object(sys, "stdout", stdout):
            main(argv)
        return stdout.getvalue().strip()

    def test_prints_options_for_a_reachable_cache(self):
        output = self._main(["--release-type", "ci", "--allow-upload", "true"])
        self.assertIn(f"--bazel_options=--remote_cache={REMOTE_CACHE_URL}", output)
        self.assertIn("--bazel_options=--remote_upload_local_results=true", output)

    def test_prints_nothing_without_credentials(self):
        # Fork pull requests get no secrets, so the mount is absent and the
        # build command must come out unchanged.
        self.assertEqual(
            self._main(["--release-type", "ci"], with_credentials=False), ""
        )

    def test_prints_nothing_when_the_cache_is_unreachable(self):
        self.assertEqual(self._main(["--release-type", "ci"], reachable=False), "")

    def test_prints_nothing_when_the_release_type_has_no_cache(self):
        self.assertEqual(self._main(["--release-type", "prerelease"]), "")

    def test_probe_is_skipped_when_no_cache_is_configured(self):
        with mock.patch.object(
            configure_jax_bazel_cache,
            "probe_cache",
            side_effect=AssertionError("should not be called"),
        ):
            main(["--release-type", "prerelease"])

    def test_allow_upload_defaults_to_read_only(self):
        output = self._main(["--release-type", "ci"])
        self.assertIn("--bazel_options=--remote_upload_local_results=false", output)


if __name__ == "__main__":
    unittest.main()
