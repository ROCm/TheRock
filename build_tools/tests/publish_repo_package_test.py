# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for publish_repo_package.py.

publish_repo_package.py lives in build_tools/packaging/linux/, which is not on
the path that conftest.py sets up (it only adds build_tools/). Add it
explicitly.

These tests use a real ``LocalStorageBackend`` via ``--output-dir`` rather than
mocking an S3 client, so the assertions are about bytes on disk at a real path.
Only bucket resolution is stubbed, since that reads the CI environment.
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

_LINUX_DIR = Path(__file__).resolve().parents[1] / "packaging" / "linux"
sys.path.insert(0, str(_LINUX_DIR))

import publish_repo_package as prp  # noqa: E402

RUN_ID = "12345"
BUCKET = "therock-prerelease-artifacts"

# The key the destination method is expected to produce, spelled out here so a
# change to the published layout has to be made deliberately in two places.
EXPECTED_RPM_KEY = "12345-linux/packages/rpm/repo/rhel10/amdrocm-repo.rpm"


def _argv(tmp_path, **overrides):
    pkg = tmp_path / "amdrocm-repo-7.14.0-1.el10.noarch.rpm"
    if not pkg.exists():
        pkg.write_bytes(b"PKG-CONTENTS")
    argv = [
        "--file",
        str(overrides.get("file", pkg)),
        "--run-id",
        overrides.get("run_id", RUN_ID),
        "--os-profile",
        overrides.get("os_profile", "rhel10"),
        "--pkg-type",
        overrides.get("pkg_type", "rpm"),
    ]
    if "output_dir" in overrides:
        argv += ["--output-dir", str(overrides["output_dir"])]
    if overrides.get("dry_run"):
        argv.append("--dry-run")
    return argv


def _stub_bucket(monkeypatch, external_repo=""):
    """Stub bucket resolution, which otherwise reads the CI environment."""
    monkeypatch.setattr(
        "_therock_utils.workflow_outputs._retrieve_bucket_info",
        lambda **kwargs: (external_repo, BUCKET),
    )


# --- CLI validation -----------------------------------------------------------


def test_parse_args_ok(tmp_path):
    args = prp.parse_args(_argv(tmp_path))
    assert args.run_id == RUN_ID
    assert args.pkg_type == "rpm"
    assert args.output_dir is None
    assert args.dry_run is False


def test_parse_args_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        prp.parse_args(_argv(tmp_path, file=str(tmp_path / "nope.rpm")))


def test_parse_args_rejects_bad_pkg_type(tmp_path):
    with pytest.raises(SystemExit):
        prp.parse_args(_argv(tmp_path, pkg_type="tar"))


@pytest.mark.parametrize(
    "bad",
    [
        "rhel10/../x",
        "a/b",
        "rhel 10",
        "rhel10\n",
        "",
        # Dot-only and leading-dot forms pass a bare [A-Za-z0-9._-]+ character
        # class. ".." in particular escapes the intended directory once the key
        # is joined onto a local staging tree.
        ".",
        "..",
        "...",
        ".hidden",
    ],
)
def test_parse_args_rejects_bad_os_profile(tmp_path, bad):
    # os_profile becomes an object-key path segment; reject slashes, whitespace
    # and anything that is not anchored on an alphanumeric.
    with pytest.raises(SystemExit):
        prp.parse_args(_argv(tmp_path, os_profile=bad))


@pytest.mark.parametrize("good", ["ubuntu2404", "rhel8", "rhel10", "sles16"])
def test_parse_args_accepts_real_os_profiles(tmp_path, good):
    # Guards the tightened pattern against over-rejection.
    assert prp.parse_args(_argv(tmp_path, os_profile=good)).os_profile == good


@pytest.mark.parametrize(
    "bad",
    [
        # run_id is the leading path segment of the object key, so "../.."
        # escapes the destination once the key is joined onto a staging dir.
        "../../../../etc",
        "a/b",
        "12345\n",
        "12 345",
        "",
        "abc",
    ],
)
def test_parse_args_rejects_non_numeric_run_id(tmp_path, bad):
    with pytest.raises(SystemExit):
        prp.parse_args(_argv(tmp_path, run_id=bad))


def test_parse_args_accepts_a_numeric_run_id(tmp_path):
    assert prp.parse_args(_argv(tmp_path, run_id="12345678901")).run_id == "12345678901"


# --- publish via a real local backend -----------------------------------------


def test_publish_writes_file_at_expected_key(tmp_path, monkeypatch):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = prp.parse_args(_argv(tmp_path, output_dir=staging))

    key = prp.publish(args)

    assert key == EXPECTED_RPM_KEY
    # Build the expected path from the key string, not by joining Path
    # segments, so a separator bug on Windows is visible rather than masked.
    written = staging / EXPECTED_RPM_KEY
    assert written.is_file()
    assert written.read_bytes() == args.file.read_bytes()


@pytest.mark.parametrize(
    "pkg_type,os_profile",
    [("deb", "ubuntu2404"), ("rpm", "rhel8"), ("rpm", "rhel10"), ("rpm", "sles16")],
)
def test_publish_key_per_profile(tmp_path, monkeypatch, pkg_type, os_profile):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = prp.parse_args(
        _argv(tmp_path, output_dir=staging, pkg_type=pkg_type, os_profile=os_profile)
    )

    key = prp.publish(args)

    expected = (
        f"{RUN_ID}-linux/packages/{pkg_type}/repo/{os_profile}"
        f"/amdrocm-repo.{pkg_type}"
    )
    assert key == expected
    assert (staging / expected).is_file()


def test_publish_is_outside_the_repository_index(tmp_path, monkeypatch):
    # The bootstrap file is fetched by URL and must not be swept into the
    # content index, or the identically-named per-profile rpms would collide.
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = prp.parse_args(_argv(tmp_path, output_dir=staging))

    key = prp.publish(args)

    assert "/repo/" in key
    assert "/x86_64/" not in key
    assert "/dists/" not in key


def test_publish_carries_external_repo_prefix(tmp_path, monkeypatch):
    # external_repo is non-empty only for 'ci' from a fork or a non-TheRock
    # repository; the key must carry it so forks do not write to the shared
    # per-run prefix.
    _stub_bucket(monkeypatch, external_repo="Fork-TheRock/")
    staging = tmp_path / "staging"
    args = prp.parse_args(_argv(tmp_path, output_dir=staging))

    key = prp.publish(args)

    assert key == f"Fork-TheRock/{EXPECTED_RPM_KEY}"
    assert (staging / key).is_file()


def test_publish_dry_run_writes_nothing(tmp_path, monkeypatch):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = prp.parse_args(_argv(tmp_path, output_dir=staging, dry_run=True))

    key = prp.publish(args)

    assert key == EXPECTED_RPM_KEY
    assert not (staging / EXPECTED_RPM_KEY).exists()


def test_publish_dry_run_does_not_claim_to_have_published(
    tmp_path, monkeypatch, capsys
):
    # The publishing job runs under continue-on-error, so its log is the only
    # signal. A dry run that prints "Published:" reports an upload that never
    # happened.
    _stub_bucket(monkeypatch)
    args = prp.parse_args(
        _argv(tmp_path, output_dir=tmp_path / "staging", dry_run=True)
    )

    prp.publish(args)

    out = capsys.readouterr().out
    assert "Published:" not in out
    assert "DRY RUN" in out


def test_publish_reports_the_local_path_when_staging(tmp_path, monkeypatch, capsys):
    # With --output-dir nothing reaches S3, so an s3:// URI in the log would
    # describe an upload that did not occur.
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = prp.parse_args(_argv(tmp_path, output_dir=staging))

    prp.publish(args)

    out = capsys.readouterr().out
    assert "s3://" not in out
    assert str(staging / EXPECTED_RPM_KEY) in out


def test_publish_reports_the_s3_uri_when_uploading(tmp_path, monkeypatch, capsys):
    _stub_bucket(monkeypatch)
    args = prp.parse_args(_argv(tmp_path, dry_run=True))

    with mock.patch("publish_repo_package.create_storage_backend"):
        prp.publish(args)

    assert f"s3://{BUCKET}/{EXPECTED_RPM_KEY}" in capsys.readouterr().out


def test_publish_without_output_dir_selects_s3_backend(tmp_path, monkeypatch):
    # Omitting --output-dir means a real S3 upload. Pin that, so a test that
    # forgets the flag fails loudly here rather than reaching the network.
    _stub_bucket(monkeypatch)
    args = prp.parse_args(_argv(tmp_path, dry_run=True))
    assert args.output_dir is None

    with mock.patch("publish_repo_package.create_storage_backend") as factory:
        prp.publish(args)

    factory.assert_called_once()
    assert factory.call_args.kwargs["staging_dir"] is None
    assert factory.call_args.kwargs["dry_run"] is True


def test_publish_rejects_the_release_line(tmp_path, monkeypatch):
    # There is no artifacts bucket for the release line: every
    # therock-release-* bucket has iam_role=None because release upload is
    # external. The publishing job is gated off it; this is the backstop.
    monkeypatch.setenv("RELEASE_TYPE", "release")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ROCm/TheRock")
    args = prp.parse_args(_argv(tmp_path, output_dir=tmp_path / "staging"))

    with pytest.raises(ValueError, match="release_type='release' is invalid"):
        prp.publish(args)
