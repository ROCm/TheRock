# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for fetch_repo_setup_package.py.

fetch_repo_setup_package.py lives in build_tools/packaging/linux/, which is not
on the path that conftest.py sets up (it only adds build_tools/). Add it
explicitly.

The bucket is a local HTTP server that answers S3's listing query and serves
objects, so listing, filtering and downloading run for real. Only bucket
resolution is stubbed, since that reads the CI environment.
"""

import sys
import threading
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_LINUX_DIR = Path(__file__).resolve().parents[1] / "packaging" / "linux"
sys.path.insert(0, str(_LINUX_DIR))

import fetch_repo_setup_package as frsp  # noqa: E402

RUN_ID = "12345"
BUCKET = "therock-dev-artifacts"
FOLDER = "12345-linux/amd-repo-setup/ubuntu2404/"
DEB = "amd-repo-setup_10.3.0~dev20261116-12345~ubuntu2404_all.deb"


class _FakeBucket:
    """Serves ``objects`` (key -> bytes) and S3-style listings of them."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.listing_status = 200
        self.requests: list[str] = []
        bucket = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                bucket.requests.append(self.path)
                url = urllib.parse.urlsplit(self.path)
                query = urllib.parse.parse_qs(url.query)
                if url.path == "/" and query.get("list-type") == ["2"]:
                    if bucket.listing_status != 200:
                        self.send_error(bucket.listing_status)
                        return
                    prefix = query.get("prefix", [""])[0]
                    keys = "".join(
                        f"<Contents><Key>{k}</Key></Contents>"
                        for k in sorted(bucket.objects)
                        if k.startswith(prefix)
                    )
                    self._send(f"<ListBucketResult>{keys}</ListBucketResult>".encode())
                    return
                key = urllib.parse.unquote(url.path.lstrip("/"))
                if key not in bucket.objects:
                    self.send_error(404)
                    return
                self._send(bucket.objects[key])

            def _send(self, body: bytes):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def bucket(monkeypatch):
    fake = _FakeBucket()
    thread = threading.Thread(target=fake.server.serve_forever, daemon=True)
    thread.start()
    resolved = {}

    def _retrieve_bucket_info(**kwargs):
        resolved.update(kwargs)
        return "", BUCKET

    monkeypatch.setattr(
        "_therock_utils.workflow_outputs._retrieve_bucket_info", _retrieve_bucket_info
    )
    monkeypatch.setattr(
        frsp, "_bucket_url", lambda name: fake.url if name == BUCKET else "bad"
    )
    fake.resolved = resolved
    yield fake
    fake.server.shutdown()
    fake.server.server_close()


def _argv(tmp_path, **overrides):
    return [
        "--run-id",
        overrides.get("run_id", RUN_ID),
        "--release-type",
        overrides.get("release_type", "dev"),
        "--os-profile",
        overrides.get("os_profile", "ubuntu2404"),
        "--pkg-type",
        overrides.get("pkg_type", "deb"),
        "--dest-dir",
        str(tmp_path / "pkg"),
    ]


def _fetch(tmp_path, **overrides):
    return frsp.fetch(frsp.parse_args(_argv(tmp_path, **overrides)))


# --- Downloading --------------------------------------------------------------


def test_downloads_the_one_package_beside_the_index_page(tmp_path, bucket):
    bucket.objects[FOLDER + DEB] = b"DEB-BYTES"
    bucket.objects[FOLDER + "index.html"] = b"<html/>"

    path = _fetch(tmp_path)

    assert path == tmp_path / "pkg" / DEB
    assert path.read_bytes() == b"DEB-BYTES"
    assert sorted(p.name for p in (tmp_path / "pkg").iterdir()) == [DEB]


def test_lists_this_runs_folder_in_the_release_types_bucket(tmp_path, bucket):
    bucket.objects[FOLDER + DEB] = b"x"

    _fetch(tmp_path, release_type="nightly")

    assert bucket.resolved["release_type"] == "nightly"
    listing = urllib.parse.parse_qs(urllib.parse.urlsplit(bucket.requests[0]).query)
    assert listing["prefix"] == [FOLDER]


def test_rpm_names_are_fetched_too(tmp_path, bucket):
    rpm = "amd-repo-setup-10.3.0~20261116g864846c8-12345.dev.el10.noarch.rpm"
    bucket.objects["12345-linux/amd-repo-setup/rhel10/" + rpm] = b"RPM"

    path = _fetch(tmp_path, os_profile="rhel10", pkg_type="rpm")

    assert path.read_bytes() == b"RPM"


# --- What doesn't count as the package ----------------------------------------


@pytest.mark.parametrize(
    "objects",
    [
        pytest.param({}, id="empty-folder"),
        pytest.param({FOLDER + "index.html": b""}, id="only-the-index-page"),
        pytest.param({FOLDER + "amd-repo-setup.rpm": b""}, id="other-type"),
        pytest.param(
            {"12345-linux/amd-repo-setup/ubuntu2204/" + DEB: b""}, id="other-distro"
        ),
    ],
)
def test_no_package_is_an_error(tmp_path, bucket, objects):
    bucket.objects.update(objects)

    with pytest.raises(FileNotFoundError, match="found 0"):
        _fetch(tmp_path)


def test_two_packages_is_an_error(tmp_path, bucket):
    bucket.objects[FOLDER + DEB] = b""
    bucket.objects[FOLDER + DEB.replace("12345", "12346")] = b""

    with pytest.raises(FileNotFoundError, match="found 2"):
        _fetch(tmp_path)


def test_an_unexpected_file_name_is_refused():
    listing = f"<Key>{FOLDER}../../x.deb</Key>"

    with pytest.raises(ValueError, match="unexpected file name"):
        frsp.find_package(listing, FOLDER, "deb")


def test_a_failed_listing_is_an_error(tmp_path, bucket):
    bucket.listing_status = 403

    with pytest.raises(urllib.error.HTTPError):
        _fetch(tmp_path)


# --- CLI validation -----------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"run_id": "../12345"}, id="run-id"),
        pytest.param({"os_profile": "../rhel10"}, id="os-profile"),
        pytest.param({"pkg_type": "tar"}, id="pkg-type"),
    ],
)
def test_bad_arguments_are_rejected(tmp_path, override):
    with pytest.raises(SystemExit):
        frsp.parse_args(_argv(tmp_path, **override))
