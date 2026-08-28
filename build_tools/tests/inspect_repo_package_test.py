# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for inspect_repo_package.py.

inspect_repo_package.py lives in build_tools/packaging/linux/, which is not on
the path that conftest.py sets up (it only adds build_tools/). Add it explicitly.
Importing the module is side-effect-free thanks to its ``__main__`` guard.

Only the pure functions are covered here: path derivation, listing parsers, and
the payload check. Building a package needs dpkg, rpm and gpg, none of which are
available on every leg this suite runs on, so that is left to the container
verification. The listings below are real output captured from a build.
"""

import sys
from pathlib import Path, PurePosixPath

import pytest

_LINUX_DIR = Path(__file__).resolve().parents[1] / "packaging" / "linux"
sys.path.insert(0, str(_LINUX_DIR))

import inspect_repo_package as irp  # noqa: E402

# Captured from `dpkg-deb -c` on ubuntu2404 builds. The repo filename carries
# the stream, so the signed and unsigned listings differ in it as well as in
# whether a keyring is present -- deriving one from the other would hide that.
_DEB_LISTING = """\
drwxr-xr-x root/root         0 2026-08-11 16:15 ./
drwxr-xr-x root/root         0 2026-08-11 16:15 ./etc/
drwxr-xr-x root/root         0 2026-08-11 16:15 ./etc/apt/
drwxr-xr-x root/root         0 2026-08-11 16:15 ./etc/apt/sources.list.d/
-rw-r--r-- root/root       150 2026-08-11 16:15 ./etc/apt/sources.list.d/{stem}.sources
drwxr-xr-x root/root         0 2026-08-11 16:15 ./usr/
drwxr-xr-x root/root         0 2026-08-11 16:15 ./usr/share/
drwxr-xr-x root/root         0 2026-08-11 16:15 ./usr/share/doc/
drwxr-xr-x root/root         0 2026-08-11 16:15 ./usr/share/doc/amdrocm-repo/
-rw-r--r-- root/root       177 2026-08-11 16:15 ./usr/share/doc/amdrocm-repo/changelog.gz
"""

DEB_UNSIGNED_LISTING = _DEB_LISTING.format(stem="amdrocm-nightly")

DEB_SIGNED_LISTING = (
    _DEB_LISTING.format(stem="amdrocm-stable")
    + """\
drwxr-xr-x root/root         0 2026-08-11 16:15 ./usr/share/keyrings/
-rw-r--r-- root/root       651 2026-08-11 16:15 ./usr/share/keyrings/amdrocm.gpg
"""
)

# Captured from `rpm -qp --qf` on a signed sles16 build.
RPM_SIGNED_LISTING = """\
/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm|root|root
/etc/zypp/repos.d/amdrocm-stable.repo|root|root
"""


class TestRepoFilePath:
    def test_deb_uses_sources_list_d(self):
        assert irp.repo_file_path("ubuntu2404", "stable") == PurePosixPath(
            "/etc/apt/sources.list.d/amdrocm-stable.sources"
        )

    @pytest.mark.parametrize("os_profile", ["rhel8", "rhel10"])
    def test_rhel_uses_yum_repos_d(self, os_profile):
        assert irp.repo_file_path(os_profile, "stable") == PurePosixPath(
            "/etc/yum.repos.d/amdrocm-stable.repo"
        )

    def test_sles_uses_zypp_repos_d(self):
        """SLES keeps repository files in /etc/zypp/repos.d, not /etc/yum.repos.d."""
        assert irp.repo_file_path("sles16", "stable") == PurePosixPath(
            "/etc/zypp/repos.d/amdrocm-stable.repo"
        )


class TestKeyPath:
    def test_deb_keyring_is_the_renamed_path(self):
        assert irp.key_path("ubuntu2404") == PurePosixPath(
            "/usr/share/keyrings/amdrocm.gpg"
        )

    def test_rpm_key_is_the_renamed_path(self):
        assert irp.key_path("rhel10") == PurePosixPath(
            "/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm"
        )

    @pytest.mark.parametrize("os_profile", sorted(irp.OS_PROFILES))
    def test_key_path_is_never_a_superseded_path(self, os_profile):
        assert irp.key_path(os_profile) not in irp._SUPERSEDED_KEY_PATHS


class TestExpectedPaths:
    def test_unsigned_expects_only_the_repo_file(self):
        # Unsigned means the per-build stream, so the filename says nightly.
        assert irp.expected_paths("ubuntu2404", signed=False) == {
            PurePosixPath("/etc/apt/sources.list.d/amdrocm-nightly.sources")
        }

    def test_signed_expects_the_key_too(self):
        assert irp.expected_paths("ubuntu2404", signed=True) == {
            PurePosixPath("/etc/apt/sources.list.d/amdrocm-stable.sources"),
            PurePosixPath("/usr/share/keyrings/amdrocm.gpg"),
        }

    def test_signed_rpm_expects_the_key_too(self):
        assert irp.expected_paths("sles16", signed=True) == {
            PurePosixPath("/etc/zypp/repos.d/amdrocm-stable.repo"),
            PurePosixPath("/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm"),
        }

    @pytest.mark.parametrize("os_profile", sorted(irp.OS_PROFILES))
    @pytest.mark.parametrize("signed", [False, True])
    def test_paths_are_posix_on_every_platform(self, os_profile, signed):
        """Target paths must not pick up the host's path flavour.

        These name locations inside a Linux package. Rendering them with the
        local flavour yields backslashes when the suite runs on Windows.
        """
        for path in irp.expected_paths(os_profile, signed):
            assert "\\" not in str(path)
            assert str(path).startswith("/")


class TestForbiddenPaths:
    def test_superseded_paths_are_forbidden_even_when_signed(self):
        forbidden = irp.forbidden_paths("ubuntu2404", signed=True)
        assert PurePosixPath("/usr/share/keyrings/rocm.gpg") in forbidden
        assert PurePosixPath("/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm") in forbidden

    def test_unsigned_also_forbids_the_current_key_path(self):
        assert irp.key_path("ubuntu2404") in irp.forbidden_paths(
            "ubuntu2404", signed=False
        )

    def test_signed_does_not_forbid_the_current_key_path(self):
        assert irp.key_path("ubuntu2404") not in irp.forbidden_paths(
            "ubuntu2404", signed=True
        )


class TestParseDebContents:
    def test_paths_are_absolute_and_owners_captured(self):
        entries = dict(irp.parse_deb_contents(DEB_UNSIGNED_LISTING))
        assert (
            entries[PurePosixPath("/etc/apt/sources.list.d/amdrocm-nightly.sources")]
            == "root/root"
        )

    def test_directories_are_included(self):
        present = {p for p, _ in irp.parse_deb_contents(DEB_UNSIGNED_LISTING)}
        assert PurePosixPath("/etc/apt/sources.list.d") in present

    def test_non_root_owner_is_preserved(self):
        listing = "-rw-r--r-- tester/tester 150 2026-08-11 16:15 ./etc/apt/x.sources\n"
        assert irp.parse_deb_contents(listing) == [
            (PurePosixPath("/etc/apt/x.sources"), "tester/tester")
        ]

    def test_symlink_target_is_used(self):
        listing = (
            "lrwxrwxrwx root/root 0 2026-08-11 16:15 ./etc/a.conf -> /etc/b.conf\n"
        )
        assert irp.parse_deb_contents(listing) == [
            (PurePosixPath("/etc/a.conf"), "root/root")
        ]

    def test_blank_and_short_lines_are_skipped(self):
        assert irp.parse_deb_contents("\n   \ngarbage\n") == []


class TestParseRpmContents:
    def test_paths_and_owners(self):
        assert irp.parse_rpm_contents(RPM_SIGNED_LISTING) == [
            (PurePosixPath("/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm"), "root/root"),
            (PurePosixPath("/etc/zypp/repos.d/amdrocm-stable.repo"), "root/root"),
        ]

    def test_non_root_owner_is_preserved(self):
        entries = irp.parse_rpm_contents(
            "/etc/zypp/repos.d/amdrocm-stable.repo|bin|wheel\n"
        )
        assert entries == [
            (PurePosixPath("/etc/zypp/repos.d/amdrocm-stable.repo"), "bin/wheel")
        ]

    def test_malformed_lines_are_skipped(self):
        assert irp.parse_rpm_contents("\nnot-a-listing\n") == []


class TestCheckPayload:
    def _check(self, listing, os_profile, signed, parser=None):
        parser = parser or irp.parse_deb_contents
        return irp.check_payload(
            parser(listing),
            irp.expected_paths(os_profile, signed),
            irp.forbidden_paths(os_profile, signed),
        )

    def test_real_unsigned_deb_passes(self):
        assert self._check(DEB_UNSIGNED_LISTING, "ubuntu2404", False) == []

    def test_real_signed_deb_passes(self):
        assert self._check(DEB_SIGNED_LISTING, "ubuntu2404", True) == []

    def test_real_signed_rpm_passes(self):
        assert (
            self._check(RPM_SIGNED_LISTING, "sles16", True, irp.parse_rpm_contents)
            == []
        )

    def test_missing_expected_file_is_reported_by_name(self):
        listing = "drwxr-xr-x root/root 0 2026-08-11 16:15 ./etc/apt/\n"
        problems = self._check(listing, "ubuntu2404", False)
        assert any(
            "/etc/apt/sources.list.d/amdrocm-nightly.sources" in p for p in problems
        )
        assert any(p.startswith("missing expected file") for p in problems)

    def test_wrong_owner_is_reported(self):
        listing = DEB_UNSIGNED_LISTING.replace(
            "root/root       150", "tester/tester   150"
        )
        problems = self._check(listing, "ubuntu2404", False)
        assert any("owned by tester/tester" in p for p in problems)

    def test_key_on_an_unsigned_build_is_rejected(self):
        problems = self._check(DEB_SIGNED_LISTING, "ubuntu2404", False)
        assert any("/usr/share/keyrings/amdrocm.gpg" in p for p in problems)

    def test_superseded_deb_keyring_name_is_rejected(self):
        """The rename is load-bearing: the old name collides with the driver.

        A suffix or substring comparison would pass this case, because
        "amdrocm.gpg" ends with "rocm.gpg". It must not.
        """
        listing = DEB_SIGNED_LISTING.replace(
            "./usr/share/keyrings/amdrocm.gpg", "./usr/share/keyrings/rocm.gpg"
        )
        problems = self._check(listing, "ubuntu2404", True)
        assert any("/usr/share/keyrings/rocm.gpg" in p for p in problems)
        assert any("must not ship" in p for p in problems)

    def test_superseded_rpm_key_name_is_rejected(self):
        listing = RPM_SIGNED_LISTING.replace(
            "/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm", "/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm"
        )
        problems = self._check(listing, "sles16", True, irp.parse_rpm_contents)
        assert any("RPM-GPG-KEY-rocm" in p for p in problems)

    def test_empty_payload_reports_every_expected_path(self):
        problems = self._check("", "sles16", True, irp.parse_rpm_contents)
        assert len(problems) == 2
