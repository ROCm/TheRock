#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prints the `build/build.py` options that point the JAX wheel build at the
shared Bazel remote cache.

The JAX wheel job compiles jaxlib, the ROCm plugin and the XLA ROCm backend with
Bazel against an already-installed ROCm. Those compiles repeat across runs
whenever the JAX ref and the ROCm build are unchanged, which is the common case
for a pull request that only touches CI or test-selection files.

The cache is the EngFlow cluster ROCm's JAX CI already uses, reached with the
mTLS client credentials `rocm.bazelrc` expects at `/data`. Only the cache is
used: the build runs `--config=rocm_release_wheel`, never `rocm_rbe`, so it
never picks up that config's `--remote_executor` and every action still executes
locally.

This runs *inside* the manylinux build container rather than on the runner. The
container reaches the network through the Docker daemon and receives the
credentials through a mount, so it is the only place that can tell whether Bazel
will actually reach the cache. When the credentials are absent (fork pull
requests get no secrets), the release type may not read shared entries, or the
endpoint does not answer, this prints nothing and the build runs exactly as it
would without a cache.

Options are printed to stdout so the caller can expand them into the build
command; all logging goes to stderr.
"""

import argparse
import os
import socket
import ssl
import sys
from pathlib import Path
from urllib.parse import urlsplit

REMOTE_CACHE_URL = "grpcs://wardite.cluster.engflow.com"

# Paths `build/rocm/rocm.bazelrc` already expects, populated by mounting the
# runner's decoded credentials at /data.
CLIENT_CERTIFICATE = Path("/data/ci-cert.crt")
CLIENT_KEY = Path("/data/ci-cert.key")

# Release types that may read shared cache entries, mirroring the ccache policy
# in setup_ccache.py. Stable releases repackage prerelease artifacts, so
# "prerelease" and "nightly-bkc" are absent on purpose.
SHARED_CACHE_RELEASE_TYPES = frozenset({"ci", "dev", "dev-bkc", "nightly"})

DEFAULT_PROBE_TIMEOUT_SECONDS = 10
DEFAULT_REMOTE_TIMEOUT_SECONDS = 60
DEFAULT_TLS_PORT = 443


def _log(msg: str):
    print(f"[jax-bazel-cache] {msg}", file=sys.stderr)


def resolve_cache_url(cache_url: str, release_type: str) -> str:
    """Returns the cache URL to use, or "" when caching is disabled.

    An explicit `cache_url` wins so infrastructure can repoint the cache
    without a code change; otherwise the release type decides.
    """
    if cache_url:
        return cache_url
    if release_type in SHARED_CACHE_RELEASE_TYPES:
        return REMOTE_CACHE_URL
    return ""


def endpoint_address(url: str) -> tuple[str, int]:
    """Returns the (host, port) a `grpcs://host[:port]` cache URL points at."""
    split = urlsplit(url)
    return split.hostname or "", split.port or DEFAULT_TLS_PORT


def probe_cache(
    url: str,
    certificate: Path,
    key: Path,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> bool:
    """Returns whether the cache completes a TLS handshake from this process.

    This resolves the host, connects, and presents the client credentials, so a
    stale or mismatched certificate is reported here rather than as a wall of
    Bazel cache warnings during the build.
    """
    host, port = endpoint_address(url)
    if not host:
        _log(f"Cannot parse a host out of {url!r}; building without a remote cache")
        return False
    try:
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=certificate, keyfile=key)
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host):
                return True
    except (OSError, ssl.SSLError) as e:
        _log(f"Cache unreachable ({e}); building without a remote cache")
        return False


def bazel_cache_options(
    url: str, certificate: Path, key: Path, allow_upload: bool
) -> list[str]:
    """Returns the `--bazel_options=...` arguments for a usable cache."""
    return [
        f"--bazel_options=--remote_cache={url}",
        f"--bazel_options=--tls_client_certificate={certificate}",
        f"--bazel_options=--tls_client_key={key}",
        f"--bazel_options=--remote_timeout={DEFAULT_REMOTE_TIMEOUT_SECONDS}",
        f"--bazel_options=--remote_upload_local_results={str(allow_upload).lower()}",
    ]


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Print build/build.py options for the JAX Bazel remote cache."
    )
    parser.add_argument(
        "--cache-url",
        default=os.environ.get("JAX_BAZEL_REMOTE_CACHE_URL", ""),
        help="Remote cache URL. Overrides the release-type default.",
    )
    parser.add_argument(
        "--release-type",
        default=os.environ.get("RELEASE_TYPE", "ci"),
        help="Release types that must not read shared entries resolve to no cache.",
    )
    parser.add_argument(
        "--allow-upload",
        default=os.environ.get("JAX_BAZEL_CACHE_ALLOW_UPLOAD", "false"),
        help="Whether this run may write results back to the cache.",
    )
    parser.add_argument(
        "--client-certificate",
        type=Path,
        default=CLIENT_CERTIFICATE,
        help="mTLS client certificate for the cache.",
    )
    parser.add_argument(
        "--client-key",
        type=Path,
        default=CLIENT_KEY,
        help="mTLS client key for the cache.",
    )
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help="Seconds to wait for the cache to complete the handshake.",
    )
    args = parser.parse_args(argv)

    url = resolve_cache_url(args.cache_url, args.release_type)
    if not url:
        _log(f"No remote cache for release_type={args.release_type!r}")
        return

    missing = [p for p in (args.client_certificate, args.client_key) if not p.exists()]
    if missing:
        # Fork pull requests get no secrets, so this is a normal outcome and
        # not an error: the build simply runs without a cache.
        _log(f"No cache credentials at {', '.join(str(p) for p in missing)}")
        return

    allow_upload = args.allow_upload.strip().lower() == "true"
    _log(
        f"Cache mode: url={url} release_type={args.release_type} upload={allow_upload}"
    )
    if not probe_cache(
        url, args.client_certificate, args.client_key, args.probe_timeout
    ):
        return

    print(
        " ".join(
            bazel_cache_options(
                url, args.client_certificate, args.client_key, allow_upload
            )
        )
    )


if __name__ == "__main__":
    main()
