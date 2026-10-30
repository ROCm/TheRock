# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for upload_repo_setup_package.py.

upload_repo_setup_package.py lives in build_tools/packaging/linux/, which is not
on the path that conftest.py sets up (it only adds build_tools/). Add it
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

import upload_repo_setup_package as ursp  # noqa: E402

RUN_ID = "12345"
BUCKET = "therock-prerelease-artifacts"

# The key the upload is expected to write, spelled out here so a change to the
# uploaded layout has to be made deliberately in two places.
EXPECTED_RPM_KEY = "12345-linux/amd-repo-setup/rhel10/amd-repo-setup-stablerc.rpm"


def _built_dir(tmp_path, pkg_type="rpm"):
    """Return a builder output directory holding one built package."""
    dest = tmp_path / "repo-setup-package-out"
    dest.mkdir(exist_ok=True)
    pkg = dest / f"amd-repo-setup-7.14.0-1.stablerc.el10.noarch.{pkg_type}"
    if not pkg.exists():
        pkg.write_bytes(b"PKG-CONTENTS")
    return dest


def _argv(tmp_path, **overrides):
    pkg_type = overrides.get("pkg_type", "rpm")
    dest = overrides.get("dest_dir") or _built_dir(tmp_path, pkg_type)
    argv = [
        "--dest-dir",
        str(dest),
        "--tier",
        overrides.get("tier", "stablerc"),
        "--run-id",
        overrides.get("run_id", RUN_ID),
        "--os-profile",
        overrides.get("os_profile", "rhel10"),
        "--pkg-type",
        pkg_type,
    ]
    if "output_dir" in overrides:
        argv += ["--output-dir", str(overrides["output_dir"])]
    if overrides.get("dry_run"):
        argv.append("--dry-run")
    return argv


def _stub_bucket(monkeypatch, external_repo=""):
    """Stub bucket resolution, which otherwise reads the CI environment."""
    # CI sets GITHUB_OUTPUT for the test step itself; a test must not write the
    # upload's output into it.
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.setattr(
        "_therock_utils.workflow_outputs._retrieve_bucket_info",
        lambda **kwargs: (external_repo, BUCKET),
    )


# --- CLI validation -----------------------------------------------------------


def test_parse_args_ok(tmp_path):
    args = ursp.parse_args(_argv(tmp_path))
    assert args.run_id == RUN_ID
    assert args.tier == "stablerc"
    assert args.pkg_type == "rpm"
    assert args.output_dir is None
    assert args.dry_run is False


def test_parse_args_rejects_missing_dest_dir(tmp_path):
    with pytest.raises(SystemExit):
        ursp.parse_args(_argv(tmp_path, dest_dir=tmp_path / "nope"))


def test_parse_args_rejects_bad_pkg_type(tmp_path):
    with pytest.raises(SystemExit):
        ursp.parse_args(_argv(tmp_path, pkg_type="tar"))


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
        ursp.parse_args(_argv(tmp_path, os_profile=bad))


@pytest.mark.parametrize("good", ["ubuntu2404", "rhel8", "rhel10", "sles16"])
def test_parse_args_accepts_real_os_profiles(tmp_path, good):
    # Guards the tightened pattern against over-rejection.
    assert ursp.parse_args(_argv(tmp_path, os_profile=good)).os_profile == good


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
        ursp.parse_args(_argv(tmp_path, run_id=bad))


def test_parse_args_accepts_a_numeric_run_id(tmp_path):
    assert (
        ursp.parse_args(_argv(tmp_path, run_id="12345678901")).run_id == "12345678901"
    )


@pytest.mark.parametrize(
    "bad",
    # The tier goes into the file name; a slash or ".." would turn it into a
    # path. Empty is what get-repo-setup-params emits for a line with no stream.
    ["../x", "a/b", "nightly\n", "night ly", "", ".", "Nightly", "stable-rc"],
)
def test_parse_args_rejects_bad_tier(tmp_path, bad):
    with pytest.raises(SystemExit):
        ursp.parse_args(_argv(tmp_path, tier=bad))


@pytest.mark.parametrize("good", ["stable", "stablerc", "nightly"])
def test_parse_args_accepts_real_tiers(tmp_path, good):
    assert ursp.parse_args(_argv(tmp_path, tier=good)).tier == good


# --- finding the built package ------------------------------------------------


def test_find_package_returns_the_single_package(tmp_path):
    dest = _built_dir(tmp_path)
    assert ursp.find_package(dest, "rpm") == next(dest.glob("*.rpm"))


def test_find_package_ignores_the_builders_work_tree(tmp_path):
    # The builder leaves its work tree under <dest-dir>/<format>/, and rpmbuild
    # writes its own copy of the package there. Only the top level counts.
    dest = _built_dir(tmp_path)
    nested = dest / "rpm" / "amd-repo-setup" / "RPMS" / "noarch"
    nested.mkdir(parents=True)
    (nested / "amd-repo-setup-7.14.0-1.stablerc.el10.noarch.rpm").write_bytes(b"X")

    assert ursp.find_package(dest, "rpm").parent == dest


def test_find_package_rejects_an_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        ursp.find_package(tmp_path, "rpm")


def test_find_package_rejects_two_packages(tmp_path):
    dest = _built_dir(tmp_path)
    (dest / "amd-repo-setup-7.14.0-2.stablerc.el10.noarch.rpm").write_bytes(b"X")
    with pytest.raises(ValueError, match="found 2"):
        ursp.find_package(dest, "rpm")


def test_find_package_ignores_the_other_format(tmp_path):
    dest = _built_dir(tmp_path, pkg_type="deb")
    with pytest.raises(ValueError, match="found 0"):
        ursp.find_package(dest, "rpm")


def test_upload_fails_without_a_package(tmp_path, monkeypatch):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    empty = tmp_path / "empty"
    empty.mkdir()
    args = ursp.parse_args(_argv(tmp_path, dest_dir=empty, output_dir=staging))

    with pytest.raises(ValueError, match="expected exactly one .rpm"):
        ursp.upload(args)
    assert not staging.exists()


# --- upload via a real local backend ------------------------------------------


def test_upload_writes_file_at_expected_key(tmp_path, monkeypatch):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = ursp.parse_args(_argv(tmp_path, output_dir=staging))

    key = ursp.upload(args)

    assert key == EXPECTED_RPM_KEY
    # Build the expected path from the key string, not by joining Path
    # segments, so a separator bug on Windows is visible rather than masked.
    written = staging / EXPECTED_RPM_KEY
    assert written.is_file()
    assert written.read_bytes() == b"PKG-CONTENTS"


@pytest.mark.parametrize(
    "pkg_type,os_profile",
    [("deb", "ubuntu2404"), ("rpm", "rhel8"), ("rpm", "rhel10"), ("rpm", "sles16")],
)
@pytest.mark.parametrize("tier", ["stable", "stablerc", "nightly"])
def test_upload_key_per_profile_and_tier(
    tmp_path, monkeypatch, pkg_type, os_profile, tier
):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = ursp.parse_args(
        _argv(
            tmp_path,
            output_dir=staging,
            pkg_type=pkg_type,
            os_profile=os_profile,
            tier=tier,
        )
    )

    key = ursp.upload(args)

    expected = (
        f"{RUN_ID}-linux/amd-repo-setup/{os_profile}/amd-repo-setup-{tier}.{pkg_type}"
    )
    assert key == expected
    assert (staging / expected).is_file()


def test_upload_carries_external_repo_prefix(tmp_path, monkeypatch):
    # external_repo is non-empty only for 'ci' from a fork or a non-TheRock
    # repository; the key must carry it so forks do not write to the shared
    # per-run prefix.
    _stub_bucket(monkeypatch, external_repo="Fork-TheRock/")
    staging = tmp_path / "staging"
    args = ursp.parse_args(_argv(tmp_path, output_dir=staging))

    key = ursp.upload(args)

    assert key == f"Fork-TheRock/{EXPECTED_RPM_KEY}"
    assert (staging / key).is_file()


def test_upload_dry_run_writes_nothing(tmp_path, monkeypatch):
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = ursp.parse_args(_argv(tmp_path, output_dir=staging, dry_run=True))

    key = ursp.upload(args)

    assert key == EXPECTED_RPM_KEY
    assert not (staging / EXPECTED_RPM_KEY).exists()


def test_upload_dry_run_does_not_claim_to_have_uploaded(tmp_path, monkeypatch, capsys):
    # A dry run that prints "Uploaded:" reports an upload that never happened.
    _stub_bucket(monkeypatch)
    args = ursp.parse_args(
        _argv(tmp_path, output_dir=tmp_path / "staging", dry_run=True)
    )

    ursp.upload(args)

    out = capsys.readouterr().out
    assert "Uploaded:" not in out
    assert "DRY RUN" in out


def test_upload_reports_the_local_path_when_staging(tmp_path, monkeypatch, capsys):
    # With --output-dir nothing reaches S3, so an s3:// URI in the log would
    # describe an upload that did not occur.
    _stub_bucket(monkeypatch)
    staging = tmp_path / "staging"
    args = ursp.parse_args(_argv(tmp_path, output_dir=staging))

    ursp.upload(args)

    out = capsys.readouterr().out
    assert "s3://" not in out
    assert str(staging / EXPECTED_RPM_KEY) in out


def test_upload_reports_the_s3_uri_when_uploading(tmp_path, monkeypatch, capsys):
    _stub_bucket(monkeypatch)
    args = ursp.parse_args(_argv(tmp_path, dry_run=True))

    with mock.patch("upload_repo_setup_package.create_storage_backend"):
        ursp.upload(args)

    assert f"s3://{BUCKET}/{EXPECTED_RPM_KEY}" in capsys.readouterr().out


def test_upload_without_output_dir_selects_s3_backend(tmp_path, monkeypatch):
    # Omitting --output-dir means a real S3 upload. Pin that, so a test that
    # forgets the flag fails loudly here rather than reaching the network.
    _stub_bucket(monkeypatch)
    args = ursp.parse_args(_argv(tmp_path, dry_run=True))
    assert args.output_dir is None

    with mock.patch("upload_repo_setup_package.create_storage_backend") as factory:
        ursp.upload(args)

    factory.assert_called_once()
    assert factory.call_args.kwargs["staging_dir"] is None
    assert factory.call_args.kwargs["dry_run"] is True


def test_upload_writes_the_package_url_to_github_output(tmp_path, monkeypatch):
    # Later steps read the uploaded package's URL from the step output.
    _stub_bucket(monkeypatch)
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    args = ursp.parse_args(_argv(tmp_path))

    with mock.patch("upload_repo_setup_package.create_storage_backend"):
        ursp.upload(args)

    assert github_output.read_text().splitlines() == [
        f"repo_setup_package_url=https://{BUCKET}.s3.amazonaws.com/{EXPECTED_RPM_KEY}"
    ]


@pytest.mark.parametrize(
    "overrides",
    [{"dry_run": True}, {"output_dir": "staging"}],
    ids=["dry-run", "output-dir"],
)
def test_upload_writes_no_output_without_an_s3_upload(tmp_path, monkeypatch, overrides):
    # Neither a dry run nor local staging puts anything in S3, so a URL would
    # point at an object that does not exist.
    _stub_bucket(monkeypatch)
    github_output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    if "output_dir" in overrides:
        overrides = {"output_dir": tmp_path / overrides["output_dir"]}
    args = ursp.parse_args(_argv(tmp_path, **overrides))

    with mock.patch("upload_repo_setup_package.create_storage_backend"):
        ursp.upload(args)

    assert not github_output.exists()


def test_upload_rejects_the_release_line(tmp_path, monkeypatch):
    # There is no artifacts bucket for the release line: every
    # therock-release-* bucket has iam_role=None because release upload is
    # external. The upload job does not run for that line; this is the
    # backstop.
    monkeypatch.setenv("RELEASE_TYPE", "release")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ROCm/TheRock")
    args = ursp.parse_args(_argv(tmp_path, output_dir=tmp_path / "staging"))

    with pytest.raises(ValueError, match="release_type='release' is invalid"):
        ursp.upload(args)
