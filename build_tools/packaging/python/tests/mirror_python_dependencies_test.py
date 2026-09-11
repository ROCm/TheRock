# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import base64
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import os
from pathlib import Path
import sys

from botocore.exceptions import ClientError
import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import mirror_python_dependencies as mirror
from mirror_python_dependencies import (
    DependencyPolicy,
    DependencySnapshot,
    PypiWheel,
    SnapshotWheel,
)


_WHEEL_BYTES = b"wheel-content"
_WHEEL_SHA256 = hashlib.sha256(_WHEEL_BYTES).hexdigest()
_WHEEL_FILENAME = "demo_package-2.0.0-py3-none-any.whl"


def _pypi_file(
    filename: str,
    content: bytes,
    *,
    yanked: bool = False,
) -> dict[str, object]:
    return {
        "filename": filename,
        "url": f"https://files.pythonhosted.org/{filename}",
        "size": len(content),
        "digests": {"sha256": hashlib.sha256(content).hexdigest()},
        "packagetype": "bdist_wheel",
        "yanked": yanked,
    }


def _project_metadata() -> dict[str, object]:
    return {
        "releases": {
            "1.0.0": [
                _pypi_file(
                    "demo_package-1.0.0-py3-none-any.whl",
                    b"one",
                )
            ],
            "2.0.0rc1": [
                _pypi_file(
                    "demo_package-2.0.0rc1-py3-none-any.whl",
                    b"release-candidate",
                )
            ],
            "2.0.0": [_pypi_file(_WHEEL_FILENAME, _WHEEL_BYTES)],
            "3.0.0": [
                _pypi_file(
                    "demo_package-3.0.0-py3-none-any.whl",
                    b"yanked",
                    yanked=True,
                )
            ],
            "4.0.0rc1": [
                _pypi_file(
                    "demo_package-4.0.0rc1-py3-none-any.whl",
                    b"newer-release-candidate",
                )
            ],
        }
    }


def _snapshot_wheel() -> SnapshotWheel:
    return SnapshotWheel(
        package="demo-package",
        project="demo",
        requested_version="latest",
        selected_version="2.0.0",
        filename=_WHEEL_FILENAME,
        source_url=f"https://files.pythonhosted.org/{_WHEEL_FILENAME}",
        size=len(_WHEEL_BYTES),
        sha256=_WHEEL_SHA256,
        relative_path=f"wheels/demo-package/{_WHEEL_FILENAME}",
        destination_key=("v5/rocm/core/whl-next/demo-package/" + _WHEEL_FILENAME),
    )


def _write_snapshot(snapshot_dir: Path) -> DependencySnapshot:
    wheel = _snapshot_wheel()
    wheel_path = snapshot_dir / wheel.relative_path
    wheel_path.parent.mkdir(parents=True)
    wheel_path.write_bytes(_WHEEL_BYTES)
    snapshot = DependencySnapshot(
        schema_version=1,
        created_at="2026-08-25T00:00:00+00:00",
        source_revision="abc123",
        wheels=(wheel,),
    )
    mirror.write_snapshot_manifest(snapshot, snapshot_dir / "manifest.json")
    return snapshot


def _write_snapshot_with_filename(snapshot_dir: Path, filename: str) -> None:
    snapshot = _write_snapshot(snapshot_dir)
    wheel = snapshot.wheels[0]
    modified_wheel = replace(
        wheel,
        filename=filename,
        relative_path=f"wheels/demo-package/{filename}",
        destination_key=f"v5/rocm/core/whl-next/demo-package/{filename}",
    )
    mirror.write_snapshot_manifest(
        replace(snapshot, wheels=(modified_wheel,)),
        snapshot_dir / "manifest.json",
    )


def _mock_command_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mirror,
        "DEPENDENCIES",
        {
            "demo-package": DependencyPolicy(
                project="demo",
                versions=("latest",),
            )
        },
    )
    monkeypatch.setattr(
        mirror, "fetch_pypi_project", lambda package: _project_metadata()
    )

    def fake_download(wheel: PypiWheel, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_WHEEL_BYTES)

    monkeypatch.setattr(mirror, "download_wheel", fake_download)


def _mock_s3_client(
    monkeypatch: pytest.MonkeyPatch,
    client: "FakeS3Client",
) -> None:
    def fake_boto3_client(service_name: str) -> "FakeS3Client":
        assert service_name == "s3"
        return client

    monkeypatch.setattr(mirror.boto3, "client", fake_boto3_client)


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self.get_calls: list[dict[str, object]] = []
        self.put_calls: list[dict[str, object]] = []
        self.copy_calls: list[dict[str, object]] = []
        self.now = datetime(2026, 8, 25, tzinfo=timezone.utc)
        self.fail_copy_precondition = False
        self.corrupt_copy_checksum = False
        self.corrupt_put_checksum = False

    def seed(
        self,
        key: str,
        body: bytes = _WHEEL_BYTES,
        *,
        include_checksum: bool = True,
    ) -> None:
        self.objects[key] = self._object(body)
        if not include_checksum:
            del self.objects[key]["ChecksumSHA256"]

    def _object(self, body: bytes) -> dict[str, object]:
        return {
            "body": body,
            "ContentLength": len(body),
            "ContentType": "application/octet-stream",
            "Metadata": {"example": "value"},
            "ETag": f'"{hashlib.md5(body, usedforsecurity=False).hexdigest()}"',
            "ChecksumSHA256": base64.b64encode(hashlib.sha256(body).digest()).decode(
                "ascii"
            ),
            "LastModified": self.now,
        }

    def head_object(self, **kwargs: object) -> dict[str, object]:
        key = str(kwargs["Key"])
        if key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        return {
            name: value for name, value in self.objects[key].items() if name != "body"
        }

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(kwargs)
        key = str(kwargs["Key"])
        return {"Body": BytesIO(self.objects[key]["body"])}

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(kwargs)
        body = kwargs["Body"]
        data = body.read()
        key = str(kwargs["Key"])
        self.now += timedelta(seconds=1)
        self.objects[key] = self._object(data)
        if self.corrupt_put_checksum:
            self.objects[key]["ChecksumSHA256"] = "incorrect-checksum"
        return {}

    def copy_object(self, **kwargs: object) -> dict[str, object]:
        self.copy_calls.append(kwargs)
        if self.fail_copy_precondition:
            raise ClientError(
                {"Error": {"Code": "PreconditionFailed", "Message": "changed"}},
                "CopyObject",
            )
        key = str(kwargs["Key"])
        if kwargs["CopySourceIfMatch"] != self.objects[key]["ETag"]:
            raise AssertionError("copy did not use the current source ETag")
        body = self.objects[key]["body"]
        self.now += timedelta(seconds=1)
        copied = self._object(body)
        copied["Metadata"] = kwargs["Metadata"]
        copied["ContentType"] = kwargs["ContentType"]
        if self.corrupt_copy_checksum:
            copied["ChecksumSHA256"] = "incorrect-checksum"
        self.objects[key] = copied
        return {}


@pytest.mark.parametrize(
    "filename",
    [
        "numpy-2.0.0-cp310-cp310-linux_x86_64.whl",
        "numpy-2.0.0-cp314-cp314-manylinux_2_28_x86_64.whl",
        "numpy-2.0.0-cp315-cp315-manylinux_2_28_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-win_amd64.whl",
        "sympy-1.13.0-py3-none-any.whl",
    ],
)
def test_is_wheel_allowed(filename: str) -> None:
    assert mirror.is_wheel_allowed(filename)


@pytest.mark.parametrize(
    "filename",
    [
        "numpy-2.0.0-cp39-cp39-manylinux_2_28_x86_64.whl",
        "numpy-2.0.0-cp314-cp314t-manylinux_2_28_x86_64.whl",
        "numpy-2.0.0-cp315-cp315t-manylinux_2_28_x86_64.whl",
        "numpy-2.0.0-cp312-cp312-manylinux_2_28_aarch64.whl",
        "numpy-2.0.0-cp312-cp312-macosx_14_0_x86_64.whl",
        "six-1.0.0-py2.py3-none-any.whl",
        "numpy-2.0.0.tar.gz",
    ],
)
def test_is_wheel_rejected(filename: str) -> None:
    assert not mirror.is_wheel_allowed(filename)


def test_latest_selects_highest_non_yanked_stable_release() -> None:
    candidates = mirror.parse_pypi_wheels("demo-package", _project_metadata())
    selected = mirror.select_release_wheels(
        package="demo-package",
        requested_version="latest",
        candidates=candidates,
    )
    assert [wheel.filename for wheel in selected] == [_WHEEL_FILENAME]


@pytest.mark.parametrize("requested_version", ["1.0.0", "2.0.0rc1"])
def test_explicit_version_remains_explicit(requested_version: str) -> None:
    candidates = mirror.parse_pypi_wheels("demo-package", _project_metadata())
    selected = mirror.select_release_wheels(
        package="demo-package",
        requested_version=requested_version,
        candidates=candidates,
    )
    assert [str(wheel.version) for wheel in selected] == [requested_version]


def test_missing_explicit_version_fails() -> None:
    candidates = mirror.parse_pypi_wheels("demo-package", _project_metadata())
    with pytest.raises(ValueError, match="has no allowed non-yanked wheels"):
        mirror.select_release_wheels(
            package="demo-package",
            requested_version="9.0.0",
            candidates=candidates,
        )


def test_select_dependencies_filters_project_and_name() -> None:
    selected = mirror.select_dependencies(
        project="torch", dependency_names=frozenset({"typing_extensions"})
    )
    assert list(selected) == ["typing-extensions"]


def test_select_dependencies_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown dependency package"):
        mirror.select_dependencies(dependency_names=frozenset({"unknown"}))


def test_resolve_snapshot_downloads_once_and_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "snapshot"
    download_count = 0

    monkeypatch.setattr(
        mirror, "fetch_pypi_project", lambda package: _project_metadata()
    )

    def fake_download(wheel: PypiWheel, output_path: Path) -> None:
        nonlocal download_count
        download_count += 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_WHEEL_BYTES)

    monkeypatch.setattr(mirror, "download_wheel", fake_download)
    snapshot = mirror.resolve_snapshot(
        output_dir=output_dir,
        selected_dependencies={
            "demo-package": DependencyPolicy(
                project="demo", versions=("2.0.0", "latest")
            )
        },
        source_revision="abc123",
    )

    assert download_count == 1
    assert len(snapshot.wheels) == 1
    assert mirror.load_snapshot(output_dir) == snapshot


def test_resolve_failure_removes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "snapshot"
    monkeypatch.setattr(
        mirror, "fetch_pypi_project", lambda package: _project_metadata()
    )

    def fail_download(wheel: PypiWheel, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"partial")
        raise ValueError("download failed")

    monkeypatch.setattr(mirror, "download_wheel", fail_download)
    with pytest.raises(ValueError, match="download failed"):
        mirror.resolve_snapshot(
            output_dir=output_dir,
            selected_dependencies={
                "demo-package": DependencyPolicy(project="demo", versions=("latest",))
            },
        )
    assert not output_dir.exists()


def test_load_snapshot_rejects_corrupt_wheel(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    wheel = _snapshot_wheel()
    (tmp_path / wheel.relative_path).write_bytes(b"corrupt-wheel")
    with pytest.raises(ValueError, match="SHA256|size"):
        mirror.load_snapshot(tmp_path)


def test_load_snapshot_rejects_missing_wheel(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    (tmp_path / snapshot.wheels[0].relative_path).unlink()
    with pytest.raises(FileNotFoundError, match="Snapshot wheel not found"):
        mirror.load_snapshot(tmp_path)


@pytest.mark.parametrize(
    ("filename", "error_match"),
    [
        pytest.param(
            "../../../secret.whl",
            "must be a wheel basename",
            id="posix-separators",
        ),
        pytest.param(
            "..\\secret.whl",
            "must be a wheel basename",
            id="windows-separators",
        ),
        pytest.param(
            "invalid.whl",
            "not a valid wheel filename",
            id="invalid-wheel-filename",
        ),
        pytest.param(
            "other_package-2.0.0-py3-none-any.whl",
            "distribution.*does not match",
            id="mismatched-distribution",
        ),
        pytest.param(
            "demo_package-3.0.0-py3-none-any.whl",
            "version.*does not match",
            id="mismatched-version",
        ),
    ],
)
def test_load_snapshot_rejects_invalid_wheel_identity(
    tmp_path: Path,
    filename: str,
    error_match: str,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    _write_snapshot_with_filename(snapshot_dir, filename)
    with pytest.raises(ValueError, match=error_match):
        mirror.load_snapshot(snapshot_dir)


def test_load_snapshot_rejects_package_path(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot = _write_snapshot(snapshot_dir)
    wheel = snapshot.wheels[0]
    package = "../demo-package"
    modified_wheel = replace(
        wheel,
        package=package,
        relative_path=Path("wheels", package, wheel.filename).as_posix(),
        destination_key=mirror.dependency_destination_key(package, wheel.filename),
    )
    mirror.write_snapshot_manifest(
        replace(snapshot, wheels=(modified_wheel,)),
        snapshot_dir / "manifest.json",
    )

    with pytest.raises(ValueError, match="not a valid Python package name"):
        mirror.load_snapshot(snapshot_dir)


def test_load_snapshot_rejects_symlink_outside_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot = _write_snapshot(snapshot_dir)
    wheel_path = snapshot_dir / snapshot.wheels[0].relative_path
    wheel_path.unlink()
    external_wheel = tmp_path / "external.whl"
    external_wheel.write_bytes(_WHEEL_BYTES)
    wheel_path.symlink_to(external_wheel)

    with pytest.raises(ValueError, match="escapes snapshot directory"):
        mirror.load_snapshot(snapshot_dir)


def test_publish_uploads_missing_wheel(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    summary = mirror.publish_snapshot(
        snapshot_dir=tmp_path,
        bucket="test-bucket",
        refresh_existing=True,
        s3_client=client,
    )
    assert summary == mirror.PublishSummary(uploaded=1, refreshed=0, skipped=0)
    assert snapshot.wheels[0].destination_key in client.objects
    assert len(client.put_calls) == 1
    assert client.put_calls[0]["ChecksumSHA256"] == base64.b64encode(
        hashlib.sha256(_WHEEL_BYTES).digest()
    ).decode("ascii")


def test_publish_rejects_bad_post_upload_checksum(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.corrupt_put_checksum = True
    with pytest.raises(ValueError, match="SHA256 checksum"):
        mirror.publish_snapshot(
            snapshot_dir=tmp_path,
            bucket="test-bucket",
            refresh_existing=True,
            s3_client=client,
        )


def test_publish_refreshes_existing_wheel_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    snapshot = _write_snapshot(tmp_path)
    wheel = snapshot.wheels[0]
    client = FakeS3Client()
    client.seed(wheel.destination_key)
    original_modified = client.objects[wheel.destination_key]["LastModified"]

    summary = mirror.publish_snapshot(
        snapshot_dir=tmp_path,
        bucket="test-bucket",
        refresh_existing=True,
        s3_client=client,
    )

    assert summary == mirror.PublishSummary(uploaded=0, refreshed=1, skipped=0)
    assert len(client.copy_calls) == 1
    assert client.copy_calls[0]["MetadataDirective"] == "REPLACE"
    assert client.copy_calls[0]["Metadata"] == {"example": "value"}
    assert client.objects[wheel.destination_key]["LastModified"] > original_modified


def test_publish_rejects_same_size_wrong_existing_content(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(snapshot.wheels[0].destination_key, b"wrong-content")

    with pytest.raises(ValueError, match="SHA256 checksum"):
        mirror.publish_snapshot(
            snapshot_dir=tmp_path,
            bucket="test-bucket",
            refresh_existing=True,
            s3_client=client,
        )
    assert not client.copy_calls


def test_publish_hashes_legacy_object_without_checksum(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(snapshot.wheels[0].destination_key, include_checksum=False)

    summary = mirror.publish_snapshot(
        snapshot_dir=tmp_path,
        bucket="test-bucket",
        refresh_existing=False,
        s3_client=client,
    )

    assert summary == mirror.PublishSummary(uploaded=0, refreshed=0, skipped=1)
    assert len(client.get_calls) == 1


def test_publish_refreshes_legacy_object_with_sha256_checksum(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(snapshot.wheels[0].destination_key, include_checksum=False)

    summary = mirror.publish_snapshot(
        snapshot_dir=tmp_path,
        bucket="test-bucket",
        refresh_existing=True,
        s3_client=client,
    )

    assert summary == mirror.PublishSummary(uploaded=0, refreshed=1, skipped=0)
    assert len(client.get_calls) == 1
    assert client.copy_calls[0]["ChecksumAlgorithm"] == "SHA256"


def test_publish_rejects_wrong_legacy_object_without_checksum(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(
        snapshot.wheels[0].destination_key,
        b"wrong-content",
        include_checksum=False,
    )

    with pytest.raises(ValueError, match="has SHA256"):
        mirror.publish_snapshot(
            snapshot_dir=tmp_path,
            bucket="test-bucket",
            refresh_existing=False,
            s3_client=client,
        )


def test_publish_rejects_bad_post_refresh_checksum(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(snapshot.wheels[0].destination_key)
    client.corrupt_copy_checksum = True

    with pytest.raises(ValueError, match="SHA256 checksum"):
        mirror.publish_snapshot(
            snapshot_dir=tmp_path,
            bucket="test-bucket",
            refresh_existing=True,
            s3_client=client,
        )


def test_publish_propagates_conditional_copy_failure(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(snapshot.wheels[0].destination_key)
    client.fail_copy_precondition = True
    with pytest.raises(ClientError):
        mirror.publish_snapshot(
            snapshot_dir=tmp_path,
            bucket="test-bucket",
            refresh_existing=True,
            s3_client=client,
        )


def test_publish_skips_existing_wheel_when_refresh_disabled(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path)
    client = FakeS3Client()
    client.seed(snapshot.wheels[0].destination_key)
    summary = mirror.publish_snapshot(
        snapshot_dir=tmp_path,
        bucket="test-bucket",
        refresh_existing=False,
        s3_client=client,
    )
    assert summary == mirror.PublishSummary(uploaded=0, refreshed=0, skipped=1)
    assert not client.put_calls
    assert not client.copy_calls


def test_publish_dry_run_does_not_write(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    client = FakeS3Client()
    summary = mirror.publish_snapshot(
        snapshot_dir=tmp_path,
        bucket="test-bucket",
        refresh_existing=True,
        dry_run=True,
        s3_client=client,
    )
    assert summary.uploaded == 1
    assert not client.put_calls
    assert not client.copy_calls


def test_resolve_command_creates_snapshot_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_command_resolution(monkeypatch)
    snapshot_dir = tmp_path / "snapshot"
    summary_path = tmp_path / "summary.md"

    result = mirror.main(
        [
            "resolve",
            "--output-dir",
            os.fspath(snapshot_dir),
            "--source-revision",
            "command-revision",
            "--summary-file",
            os.fspath(summary_path),
        ]
    )

    assert result == 0
    snapshot = mirror.load_snapshot(snapshot_dir)
    assert snapshot.source_revision == "command-revision"
    assert [wheel.filename for wheel in snapshot.wheels] == [_WHEEL_FILENAME]
    assert (snapshot_dir / snapshot.wheels[0].relative_path).read_bytes() == (
        _WHEEL_BYTES
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "TheRock revision: `command-revision`" in summary_text
    assert "| `demo-package` | `2.0.0` |" in summary_text


def test_publish_command_uploads_snapshot_and_records_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot = _write_snapshot(snapshot_dir)
    summary_path = tmp_path / "summary.md"
    client = FakeS3Client()
    _mock_s3_client(monkeypatch, client)

    result = mirror.main(
        [
            "publish",
            "--snapshot-dir",
            os.fspath(snapshot_dir),
            "--bucket",
            "test-bucket",
            "--summary-file",
            os.fspath(summary_path),
        ]
    )

    assert result == 0
    assert client.objects[snapshot.wheels[0].destination_key]["body"] == _WHEEL_BYTES
    assert (
        "`test-bucket`: published; uploaded 1, refreshed 0, skipped 0"
        in summary_path.read_text(encoding="utf-8")
    )


def test_mirror_command_resolves_and_uploads_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_command_resolution(monkeypatch)
    snapshot_dir = tmp_path / "snapshot"
    summary_path = tmp_path / "summary.md"
    client = FakeS3Client()
    _mock_s3_client(monkeypatch, client)

    result = mirror.main(
        [
            "mirror",
            "--snapshot-dir",
            os.fspath(snapshot_dir),
            "--bucket",
            "test-bucket",
            "--summary-file",
            os.fspath(summary_path),
        ]
    )

    assert result == 0
    snapshot = mirror.load_snapshot(snapshot_dir)
    assert client.objects[snapshot.wheels[0].destination_key]["body"] == _WHEEL_BYTES
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "| `demo-package` | `2.0.0` |" in summary_text
    assert "`test-bucket`: published; uploaded 1, refreshed 0, skipped 0" in (
        summary_text
    )
