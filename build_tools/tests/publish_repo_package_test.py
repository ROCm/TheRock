# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for publish_repo_package.py.

publish_repo_package.py lives in build_tools/packaging/linux/, which is not on
the path that conftest.py sets up (it only adds build_tools/). Add it
explicitly. Importing the module is side-effect-free (boto3 is a deferred import
inside ``publish``), so key derivation is testable without boto3 installed.
"""

import sys
from pathlib import Path

import pytest

_LINUX_DIR = Path(__file__).resolve().parents[1] / "packaging" / "linux"
sys.path.insert(0, str(_LINUX_DIR))

import publish_repo_package as prp  # noqa: E402

PREFIX = "12345-linux/packages"


# --- object key derivation ----------------------------------------------------


def test_key_deb_profile():
    assert (
        prp.object_key(f"{PREFIX}/deb", "ubuntu2404", "deb")
        == "12345-linux/packages/deb/repo/ubuntu2404/amdrocm-repo.deb"
    )


@pytest.mark.parametrize("profile", ["rhel8", "rhel10", "sles16"])
def test_key_rpm_profiles(profile):
    assert (
        prp.object_key(f"{PREFIX}/rpm", profile, "rpm")
        == f"12345-linux/packages/rpm/repo/{profile}/amdrocm-repo.rpm"
    )


def test_key_is_sibling_not_in_repo_index():
    # The file must sit under repo/<profile>/, never inside the content index
    # (deb dists/, rpm x86_64/).
    deb = prp.object_key(f"{PREFIX}/deb", "ubuntu2404", "deb")
    rpm = prp.object_key(f"{PREFIX}/rpm", "rhel10", "rpm")
    for key in (deb, rpm):
        assert "/repo/" in key
    assert "/dists/" not in deb
    assert "/x86_64/" not in rpm


def test_key_tolerates_trailing_slash_prefix():
    assert (
        prp.object_key(f"{PREFIX}/rpm/", "rhel10", "rpm")
        == "12345-linux/packages/rpm/repo/rhel10/amdrocm-repo.rpm"
    )


# --- CLI validation -----------------------------------------------------------


def _args(tmp_path, **overrides):
    pkg = tmp_path / "amdrocm-repo-7.14.0-1.el10.noarch.rpm"
    pkg.write_bytes(b"PKG")
    argv = [
        "--file",
        str(overrides.get("file", pkg)),
        "--bucket",
        overrides.get("bucket", "therock-prerelease-artifacts"),
        "--prefix",
        overrides.get("prefix", f"{PREFIX}/rpm"),
        "--os-profile",
        overrides.get("os_profile", "rhel10"),
        "--pkg-type",
        overrides.get("pkg_type", "rpm"),
    ]
    return argv


def test_parse_args_ok(tmp_path):
    args = prp.parse_args(_args(tmp_path))
    assert args.bucket == "therock-prerelease-artifacts"
    assert args.pkg_type == "rpm"


def test_parse_args_rejects_empty_bucket(tmp_path):
    with pytest.raises(SystemExit):
        prp.parse_args(_args(tmp_path, bucket=""))


def test_parse_args_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        prp.parse_args(_args(tmp_path, file=str(tmp_path / "nope.rpm")))


def test_parse_args_rejects_bad_pkg_type(tmp_path):
    with pytest.raises(SystemExit):
        prp.parse_args(_args(tmp_path, pkg_type="tar"))


@pytest.mark.parametrize("bad", ["rhel10/../x", "a/b", "rhel 10", "rhel10\n", ""])
def test_parse_args_rejects_bad_os_profile(tmp_path, bad):
    # os_profile becomes an S3 key path segment; reject slashes/whitespace.
    with pytest.raises(SystemExit):
        prp.parse_args(_args(tmp_path, os_profile=bad))


# --- publish (boto3 mocked) ---------------------------------------------------


class _FakeS3:
    def __init__(self):
        self.uploaded = []
        self.headed = []
        self.head_should_fail = False

    def upload_file(self, filename, bucket, key):
        self.uploaded.append((filename, bucket, key))

    def head_object(self, Bucket, Key):
        self.headed.append((Bucket, Key))
        if self.head_should_fail:
            raise RuntimeError("object not found")


def _install_fake_boto3(monkeypatch, fake):
    import types

    fake_boto3 = types.SimpleNamespace(client=lambda service, **kw: fake)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)


def test_publish_uploads_then_confirms(tmp_path, monkeypatch):
    fake = _FakeS3()
    _install_fake_boto3(monkeypatch, fake)
    args = prp.parse_args(_args(tmp_path))
    key = prp.publish(args)
    assert key == "12345-linux/packages/rpm/repo/rhel10/amdrocm-repo.rpm"
    assert fake.uploaded == [(str(args.file), "therock-prerelease-artifacts", key)]
    assert fake.headed == [("therock-prerelease-artifacts", key)]


def test_publish_errors_when_head_fails(tmp_path, monkeypatch):
    fake = _FakeS3()
    fake.head_should_fail = True
    _install_fake_boto3(monkeypatch, fake)
    args = prp.parse_args(_args(tmp_path))
    with pytest.raises(RuntimeError):
        prp.publish(args)
