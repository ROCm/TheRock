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
credentials through a mount of the runner's `/data/ci-cert.*` files, so it is
the only place that can tell whether Bazel will actually reach the cache. When
those files are absent or the release type may not use the cache, this prints
nothing and the build runs exactly as it would without a cache. Credentials
that are present but rejected, or a cache that does not answer, raise instead.

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
# runner's `/data/ci-cert.*` files into the build container.
CLIENT_CERTIFICATE = Path("/data/ci-cert.crt")
CLIENT_KEY = Path("/data/ci-cert.key")

# Release types that may use the cache. Only CI for now: release builds need a
# cache of their own, as nightly has for ccache in setup_ccache.py.
SHARED_CACHE_RELEASE_TYPES = frozenset({"ci"})

DEFAULT_PROBE_TIMEOUT_SECONDS = 10
DEFAULT_REMOTE_TIMEOUT_SECONDS = 60
DEFAULT_TLS_PORT = 443

# The HTTP/2 connection preface and an empty SETTINGS frame, which every gRPC
# client sends first and a gRPC server answers with its own SETTINGS frame.
HTTP2_CLIENT_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n" + bytes.fromhex(
    "000000040000000000"
)


class CacheUnusableError(RuntimeError):
    """The cache is configured and credentialed but cannot be used."""


def _log(msg: str):
    print(f"[jax-bazel-cache] {msg}", file=sys.stderr)


def resolve_cache_url(cache_url: str, release_type: str) -> str:
    """Returns the cache URL to use, or "" when caching is disabled.

    The release type decides whether to cache at all; an explicit `cache_url`
    only repoints where the cache is, so infrastructure can move it without a
    code change.
    """
    if release_type not in SHARED_CACHE_RELEASE_TYPES:
        return ""
    return cache_url or REMOTE_CACHE_URL


def upload_allowed(event_name: str, ref: str) -> bool:
    """Only pushes to main write; pull requests and dispatches only read."""
    return event_name == "push" and ref == "refs/heads/main"


def endpoint_address(url: str) -> tuple[str, int]:
    """Returns the (host, port) a `grpcs://host[:port]` cache URL points at."""
    split = urlsplit(url)
    return split.hostname or "", split.port or DEFAULT_TLS_PORT


def probe_cache(
    url: str,
    certificate: Path,
    key: Path,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
):
    """Raises CacheUnusableError unless the cache accepts the client credentials.

    Under TLS 1.3 the server checks the client certificate after the handshake
    has completed on the client side, so a rejection only arrives on the first
    read. Sending the preface gRPC opens with and reading the reply surfaces it
    here rather than as a wall of Bazel cache warnings during the build.
    """
    host, port = endpoint_address(url)
    if not host:
        raise CacheUnusableError(f"Cannot parse a host out of {url!r}")
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_alpn_protocols(["h2"])
    try:
        context.load_cert_chain(certfile=certificate, keyfile=key)
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                tls.sendall(HTTP2_CLIENT_PREFACE)
                if not tls.recv(1):
                    raise CacheUnusableError(f"{url} closed the connection")
    except OSError as e:
        raise CacheUnusableError(
            f"Cannot use the cache at {url} with {certificate}: {e}"
        ) from e


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
        help="Remote cache URL, for release types that may use the cache.",
    )
    parser.add_argument(
        "--release-type",
        default=os.environ.get("RELEASE_TYPE", "ci"),
        help="Release types that must not read shared entries resolve to no cache.",
    )
    parser.add_argument(
        "--event-name",
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="GitHub event that started the run; only a push to main uploads.",
    )
    parser.add_argument(
        "--ref",
        default=os.environ.get("GITHUB_REF", ""),
        help="Git ref the run is for, e.g. refs/heads/main.",
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
        # Runners without `/data/ci-cert.*` skip the cache. That is a normal
        # outcome, not an error: the build simply compiles locally.
        _log(f"No cache credentials at {', '.join(str(p) for p in missing)}")
        return

    allow_upload = upload_allowed(args.event_name, args.ref)
    _log(
        f"Cache mode: url={url} release_type={args.release_type} upload={allow_upload}"
    )
    probe_cache(url, args.client_certificate, args.client_key, args.probe_timeout)

    print(
        " ".join(
            bazel_cache_options(
                url, args.client_certificate, args.client_key, allow_upload
            )
        )
    )


if __name__ == "__main__":
    main()
