# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for downloading product-local Python index snapshots."""

import io
import os
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import download_rocm_python_index_snapshot as snapshot
from aggregate_index import (
    MANIFEST_SCHEMA_VERSION,
    OwnershipManifest,
    parse_ownership_manifest,
)


ORIGIN = "https://nightly.repo.amd.com"


def _manifest() -> OwnershipManifest:
    return parse_ownership_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "streams": {
                "known": ["nightly", "stable"],
                "groups": {"all": ["nightly", "stable"]},
                "default": "all",
            },
            "python_indexes": [
                {
                    "public_base": "/rocm/whl-next",
                    "packages": {
                        "rocm-sdk-core": {"owner_path": "core/whl-next"},
                        "jax": {
                            "owner_path": "jax/whl-next",
                            "streams": ["stable"],
                        },
                    },
                }
            ],
        }
    )


class _Response(io.BytesIO):
    def __init__(self, url: str, content: bytes) -> None:
        super().__init__(content)
        self._url = url

    def geturl(self) -> str:
        return self._url


def _root_html(*packages: str) -> bytes:
    links = "\n".join(f'<a href="{package}/">{package}</a>' for package in packages)
    return f"<!DOCTYPE html><html><body>{links}</body></html>".encode()


def _package_html(package: str) -> bytes:
    filename = f"{package}-1.0.0-py3-none-any.whl"
    return f'<html><body><a href="{filename}">{filename}</a></body></html>'.encode()


def _product_responses(*packages: str) -> dict[str, bytes]:
    owner_url = f"{ORIGIN}/rocm/core/whl-next"
    return {
        f"{owner_url}/index.html": _root_html(*packages),
        **{
            f"{owner_url}/{package}/index.html": _package_html(package)
            for package in packages
        },
    }


def _mock_responses(
    monkeypatch: pytest.MonkeyPatch, responses: dict[str, bytes]
) -> None:
    def fake_urlopen(url: str) -> _Response:
        try:
            return _Response(url, responses[url])
        except KeyError:
            raise HTTPError(url, 404, "Not Found", {}, None) from None

    monkeypatch.setattr(snapshot, "urlopen", fake_urlopen)


def test_downloads_every_linked_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_responses(monkeypatch, _product_responses("extra-core", "rocm-sdk-core"))
    output_dir = tmp_path / "snapshot"

    snapshot.download_snapshot(
        _manifest(), stream="nightly", origin=ORIGIN, output_dir=output_dir
    )

    assert (output_dir / "rocm/core/whl-next/extra-core/index.html").is_file()
    assert (output_dir / "rocm/core/whl-next/rocm-sdk-core/index.html").is_file()
    assert not (output_dir / "rocm/jax/whl-next/index.html").exists()


@pytest.mark.parametrize(
    "responses",
    [
        {},
        {f"{ORIGIN}/rocm/core/whl-next/index.html": _root_html("rocm-sdk-core")},
    ],
)
def test_missing_required_page_does_not_publish_partial_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[str, bytes],
) -> None:
    _mock_responses(monkeypatch, responses)
    output_dir = tmp_path / "snapshot"

    with pytest.raises(snapshot.SnapshotDownloadError, match="HTTP 404"):
        snapshot.download_snapshot(
            _manifest(), stream="nightly", origin=ORIGIN, output_dir=output_dir
        )

    assert not output_dir.exists()


def test_downloads_inactive_root_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = _product_responses("rocm-sdk-core") | {
        f"{ORIGIN}/rocm/jax/whl-next/index.html": _root_html("jax"),
        f"{ORIGIN}/rocm/jax/whl-next/jax/index.html": _package_html("jax"),
    }
    _mock_responses(monkeypatch, responses)
    output_dir = tmp_path / "snapshot"

    snapshot.download_snapshot(
        _manifest(), stream="nightly", origin=ORIGIN, output_dir=output_dir
    )

    assert (output_dir / "rocm/jax/whl-next/jax/index.html").is_file()


@pytest.mark.parametrize(
    "origin",
    [
        "http://nightly.repo.amd.com",
        "https://user@nightly.repo.amd.com",
        "https://nightly.repo.amd.com/prefix",
    ],
)
def test_rejects_non_origin_url(tmp_path: Path, origin: str) -> None:
    with pytest.raises(snapshot.SnapshotDownloadError, match="HTTPS origin"):
        snapshot.download_snapshot(
            _manifest(),
            stream="nightly",
            origin=origin,
            output_dir=tmp_path / "snapshot",
        )


def test_rejects_existing_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "snapshot"
    output_dir.mkdir()

    with pytest.raises(snapshot.SnapshotDownloadError, match="already exists"):
        snapshot.download_snapshot(
            _manifest(), stream="nightly", origin=ORIGIN, output_dir=output_dir
        )
