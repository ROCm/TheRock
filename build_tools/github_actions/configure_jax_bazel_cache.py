#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prints the `build/build.py` options that point the JAX wheel build at the
shared Bazel remote cache.

The JAX wheel job compiles jaxlib, the ROCm plugin and the XLA ROCm backend with
Bazel against an already-installed ROCm. Those compiles repeat across runs
whenever the JAX ref and the ROCm build are unchanged, which is the common case
for a pull request that only touches CI or test-selection files.

The cache is the EngFlow cluster ROCm's JAX CI already uses, reached with an
EngFlow-issued JWT. Its role is fixed when EngFlow issues it: a `cache-reader`
token cannot write, whatever the client asks for. Only the cache is used: the
build runs `--config=rocm_release_wheel`, never `rocm_rbe`, so it never picks up
that config's `--remote_executor` and every action still executes locally.

This runs *inside* the manylinux build container rather than on the runner. The
container reaches the network through the Docker daemon and receives the token
through a read-only mount of a copy of the runner's file, so it is the only
place that can tell whether Bazel will actually reach the cache. When the token
is absent or the release type may not use the cache, this prints nothing and
the build runs exactly as it would without a cache. A token that is present but
malformed or expired, or a cache that does not answer, raises instead.

Options are printed to stdout so the caller can expand them into the build
command; all logging goes to stderr.

Run as `configure_jax_bazel_cache.py get`, this is also the Bazel credential
helper that hands the token to Bazel, so it never appears on a command line or
in a bazelrc that `--announce_rc` would print.
"""

import argparse
import base64
import binascii
import json
import os
import socket
import ssl
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

REMOTE_CACHE_URL = "grpcs://actinolite.cluster.engflow.com"

# Where the wheel job mounts the copy of the runner's token inside the build
# container.
TOKEN_FILE = Path("/data/engflow-cache.jwt")

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


def check_token(token_file: Path, now: float | None = None):
    """Raises CacheUnusableError unless token_file holds an unexpired JWT.

    EngFlow rejects a bad token per request, and Bazel then builds locally with
    only warnings, so an expired token would otherwise pass as a slow uncached
    build. Errors name the file, never its contents.
    """
    parts = token_file.read_text().strip().split(".")
    try:
        if len(parts) != 3:
            raise ValueError(f"{len(parts)} segments")
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        expires = float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except (binascii.Error, KeyError, TypeError, ValueError) as e:
        raise CacheUnusableError(f"{token_file} does not hold a JWT: {e}") from e
    if expires <= (time.time() if now is None else now):
        expired_at = datetime.fromtimestamp(expires, timezone.utc).isoformat()
        raise CacheUnusableError(f"The token in {token_file} expired at {expired_at}")


def probe_cache(url: str, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS):
    """Raises CacheUnusableError unless the cache answers a gRPC connection.

    Sending the preface gRPC opens with and reading the reply surfaces an
    unreachable cache here rather than as a wall of Bazel cache warnings during
    the build.
    """
    host, port = endpoint_address(url)
    if not host:
        raise CacheUnusableError(f"Cannot parse a host out of {url!r}")
    context = ssl.create_default_context()
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.set_alpn_protocols(["h2"])
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls:
                tls.sendall(HTTP2_CLIENT_PREFACE)
                if not tls.recv(1):
                    raise CacheUnusableError(f"{url} closed the connection")
    except OSError as e:
        raise CacheUnusableError(f"Cannot reach the cache at {url}: {e}") from e


def credential_helper_response(token: str) -> dict:
    """The Bazel credential helper reply carrying an EngFlow-issued JWT."""
    return {
        "headers": {
            "x-engflow-auth-method": ["jwt-v0"],
            "x-engflow-auth-token": [token],
        }
    }


def bazel_cache_options(url: str, helper: Path, allow_upload: bool) -> list[str]:
    """Returns the `--bazel_options=...` arguments for a usable cache."""
    host, _ = endpoint_address(url)
    return [
        f"--bazel_options=--remote_cache={url}",
        # Scoped to the cache host, so downloads from other hosts never see it.
        f"--bazel_options=--credential_helper={host}={helper}",
        f"--bazel_options=--remote_timeout={DEFAULT_REMOTE_TIMEOUT_SECONDS}",
        f"--bazel_options=--remote_upload_local_results={str(allow_upload).lower()}",
    ]


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["get"]:
        # Bazel sends the request URI on stdin; the token is the same for all.
        sys.stdin.read()
        token = TOKEN_FILE.read_text().strip()
        print(json.dumps(credential_helper_response(token)))
        return

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

    if not TOKEN_FILE.exists():
        # Runners without the token skip the cache. That is a normal outcome,
        # not an error: the build simply compiles locally.
        _log(f"No cache token at {TOKEN_FILE}")
        return

    allow_upload = upload_allowed(args.event_name, args.ref)
    _log(
        f"Cache mode: url={url} release_type={args.release_type} upload={allow_upload}"
    )
    check_token(TOKEN_FILE)
    probe_cache(url, args.probe_timeout)

    print(" ".join(bazel_cache_options(url, Path(__file__).resolve(), allow_upload)))


if __name__ == "__main__":
    main()
