#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Build every amdrocm-repo profile of one package type and verify its payload.

This is a build-time check, not an install test. It builds the package for each
OS profile of the given package type and asserts that the archive contains the
files it is supposed to install, at the right paths, owned by root. The build
container has dpkg and rpm but no dnf or zypper, so the packages can be built
and inspected there but not installed; per-distro install coverage lives in
``test_native_linux_packages_install.yml``.

Each profile is built twice, because the two shapes install different files:

  unsigned (nightly)      the repository file only
  signed   (prerelease)   the repository file and the signing key

The signed build is what covers the key paths. Those were renamed so that
``amdrocm-repo`` does not claim a file already owned by the amdgpu driver
packages, and this is the only place CI checks that the renamed paths are the
ones that ship and that the previous paths are gone.

Neither build reaches the network:

  * The unsigned stream loads no key, and the
    repository URL check is skipped for it.
  * The signed stream reads its key from ``--gpg-key-file``, which returns before
    the build-time key fetch. The key is generated here, offline, and thrown
    away; it never signs anything and never leaves the build.
  * The repository URL check is opt-in and is not requested.

The repository URL is therefore a placeholder and the built packages are for
inspection only. They are written to temporary directories and discarded.

Usage:
  python build_tools/packaging/linux/inspect_repo_package.py \
      --pkg-type deb \
      --rocm-version 7.14.0~dev20260811 \
      --repo-base-url https://example.com/rocm/packages-multi-arch/ \
      --repo-sub-folder 20000101-inspect
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_repo_package import (
    DEB_KEYRING_PATH,
    OS_PROFILES,
    RPM_GPG_KEY_PATH,
    list_profiles,
    repo_id,
)

_BUILDER = _THIS_DIR / "build_repo_package.py"

# The two streams that produce the unsigned and signed payload shapes. The
# per-build stream requires a build sub-folder and the flat one rejects it, so
# they are passed different arguments. Within this module signedness and stream
# select each other, which is why the helpers below take only ``signed``.
_UNSIGNED_STREAM = "nightly"
_SIGNED_STREAM = "stable"

# Everything the package installs is owned by root, whatever user builds it.
_EXPECTED_OWNER = "root/root"

# Key paths used before the rename. The amdgpu driver setup installs its key at
# the deb path below, and neither dpkg nor rpm will unpack two packages that
# claim the same file, so shipping these names again would make amdrocm-repo and
# the driver mutually uninstallable. Asserted absent from every build.
_SUPERSEDED_KEY_PATHS = (
    PurePosixPath("/usr/share/keyrings/rocm.gpg"),
    PurePosixPath("/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm"),
)


def repo_file_path(os_profile: str, stream: str) -> PurePosixPath:
    """Return the repository configuration file this profile installs.

    The filename carries the stream, so it cannot be derived from the profile
    alone.
    """
    profile = OS_PROFILES[os_profile]
    stem = repo_id(stream)
    if profile["pkg_type"] == "deb":
        return PurePosixPath(f"/etc/apt/sources.list.d/{stem}.sources")
    return PurePosixPath(profile["rpm_repo_dir"]) / f"{stem}.repo"


def key_path(os_profile: str) -> PurePosixPath:
    """Return the signing key path this profile installs when signed."""
    if OS_PROFILES[os_profile]["pkg_type"] == "deb":
        return PurePosixPath(DEB_KEYRING_PATH)
    return PurePosixPath(RPM_GPG_KEY_PATH)


def stream_for(signed: bool) -> str:
    """Return the stream this inspection builds for a given signedness."""
    return _SIGNED_STREAM if signed else _UNSIGNED_STREAM


def expected_paths(os_profile: str, signed: bool) -> set[PurePosixPath]:
    """Return the paths a build of this profile must install."""
    paths = {repo_file_path(os_profile, stream_for(signed))}
    if signed:
        paths.add(key_path(os_profile))
    return paths


def forbidden_paths(os_profile: str, signed: bool) -> set[PurePosixPath]:
    """Return the paths a build of this profile must not install."""
    paths = set(_SUPERSEDED_KEY_PATHS)
    if not signed:
        # An unsigned stream ships no key at all: the repository it configures is
        # unsigned, so a key here would be inert at best and misleading at worst.
        paths.add(key_path(os_profile))
    return paths


def parse_deb_contents(text: str) -> list[tuple[PurePosixPath, str]]:
    """Parse ``dpkg-deb -c`` output into ``(path, owner)`` pairs.

    Entries are relative to the package root and are reported with a leading
    ``./``; they are normalised to absolute paths so both package types compare
    against the same values. Directory entries are included, since they carry
    ownership too.
    """
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # <mode> <owner>/<group> <size> <date> <time> <path>
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        owner, name = fields[1], fields[5]
        # Symlinks are reported as "target -> source"; keep the target.
        name = name.split(" -> ", 1)[0]
        entries.append((PurePosixPath("/") / name.removeprefix("./"), owner))
    return entries


def parse_rpm_contents(text: str) -> list[tuple[PurePosixPath, str]]:
    """Parse the ``rpm -qp --qf`` listing into ``(path, owner)`` pairs.

    The query emits ``<path>|<user>|<group>`` per file, already absolute. Owner
    is rejoined with a slash so it matches the deb representation.
    """
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != 3:
            continue
        name, user, group = fields
        entries.append((PurePosixPath(name), f"{user}/{group}"))
    return entries


def check_payload(
    entries: list[tuple[PurePosixPath, str]],
    expected: set[PurePosixPath],
    forbidden: set[PurePosixPath],
) -> list[str]:
    """Return a list of problems with a package's payload, empty if it is sound.

    Paths are compared in full. A substring or suffix comparison would be unsafe
    here: "amdrocm.gpg" ends with "rocm.gpg", so a loose check would accept the
    superseded name that the rename exists to eliminate.
    """
    problems = []
    present = {path for path, _ in entries}

    for path in sorted(expected):
        if path not in present:
            problems.append(f"missing expected file: {path}")

    for path in sorted(forbidden):
        if path in present:
            problems.append(f"installs a file it must not ship: {path}")

    for path, owner in entries:
        if owner != _EXPECTED_OWNER:
            problems.append(f"{path} is owned by {owner}, expected {_EXPECTED_OWNER}")

    return problems


def _run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, capturing output so a successful build stays quiet."""
    return subprocess.run(
        argv, check=True, text=True, capture_output=True, encoding="utf-8", **kwargs
    )


def generate_throwaway_key(dest: Path) -> Path:
    """Generate a signing key for the signed build and export it armored.

    The key exists only to exercise the code path that embeds a key in the
    package. It is generated into a private keyring directory, used once, and
    discarded with the temporary tree. A real key cannot be used: committing key
    material is forbidden, and fetching one would defeat the offline property.
    """
    home = dest / "gnupg"
    home.mkdir(mode=0o700)
    env = dict(os.environ, GNUPGHOME=str(home))
    _run(
        [
            "gpg",
            "--batch",
            "--pinentry-mode",
            "loopback",
            "--passphrase",
            "",
            "--quick-generate-key",
            "TheRock Package Inspection <noreply@example.com>",
            "rsa2048",
            "sign",
            "0",
        ],
        env=env,
    )
    armored = _run(["gpg", "--armor", "--export"], env=env).stdout
    key_file = dest / "inspect-key.asc"
    key_file.write_text(armored, encoding="utf-8")
    return key_file


def build_package(
    os_profile: str,
    signed: bool,
    dest_dir: Path,
    rocm_version: str,
    repo_base_url: str,
    repo_sub_folder: str,
    gpg_key_file: Path,
) -> Path:
    """Build one package and return its path."""
    argv = [
        sys.executable,
        str(_BUILDER),
        "--os-profile",
        os_profile,
        "--repo-base-url",
        repo_base_url,
        "--rocm-version",
        rocm_version,
        "--dest-dir",
        str(dest_dir),
    ]
    if signed:
        argv += [
            "--stream",
            _SIGNED_STREAM,
            "--gpg-key-file",
            str(gpg_key_file),
        ]
    else:
        # The build sub-folder is required on a per-build stream and rejected on
        # a flat one, so it is passed here and nowhere else.
        argv += [
            "--stream",
            _UNSIGNED_STREAM,
            "--repo-sub-folder",
            repo_sub_folder,
        ]

    try:
        _run(argv)
    except subprocess.CalledProcessError as e:
        # rpmbuild is verbose and only interesting when it fails.
        sys.stderr.write(e.stdout or "")
        sys.stderr.write(e.stderr or "")
        raise

    pkg_type = OS_PROFILES[os_profile]["pkg_type"]
    built = sorted(dest_dir.glob(f"*.{pkg_type}"))
    if len(built) != 1:
        raise RuntimeError(
            f"expected exactly one .{pkg_type} in {dest_dir}, found {len(built)}"
        )
    return built[0]


def read_payload(package: Path, pkg_type: str) -> list[tuple[PurePosixPath, str]]:
    """Return the ``(path, owner)`` payload of a built package."""
    if pkg_type == "deb":
        return parse_deb_contents(_run(["dpkg-deb", "-c", str(package)]).stdout)
    listing = _run(
        [
            "rpm",
            "-qp",
            "--qf",
            "[%{FILENAMES}|%{FILEUSERNAME}|%{FILEGROUPNAME}\n]",
            str(package),
        ]
    ).stdout
    return parse_rpm_contents(listing)


def inspect_profile(
    os_profile: str,
    signed: bool,
    work_dir: Path,
    args: argparse.Namespace,
    gpg_key_file: Path,
) -> list[str]:
    """Build one profile in one shape and return any problems with its payload."""
    dest_dir = work_dir / f"{os_profile}-{'signed' if signed else 'unsigned'}"
    dest_dir.mkdir()
    package = build_package(
        os_profile,
        signed,
        dest_dir,
        args.rocm_version,
        args.repo_base_url,
        args.repo_sub_folder,
        gpg_key_file,
    )
    entries = read_payload(package, OS_PROFILES[os_profile]["pkg_type"])
    return check_payload(
        entries,
        expected_paths(os_profile, signed),
        forbidden_paths(os_profile, signed),
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build and verify the payload of the amdrocm-repo packages.",
    )
    p.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Package type whose OS profiles are built",
    )
    p.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm version to build with",
    )
    p.add_argument(
        "--repo-base-url",
        required=True,
        help="Placeholder repository base URL; never contacted",
    )
    p.add_argument(
        "--repo-sub-folder",
        required=True,
        help="Placeholder dated sub-folder for the unsigned build",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    profiles = list_profiles(args.pkg_type)
    if not profiles:
        raise SystemExit(f"no OS profiles build {args.pkg_type} packages")

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        gpg_key_file = generate_throwaway_key(work_dir)
        for os_profile in profiles:
            for signed in (False, True):
                shape = "signed" if signed else "unsigned"
                problems = inspect_profile(
                    os_profile, signed, work_dir, args, gpg_key_file
                )
                if problems:
                    failures.append((os_profile, shape, problems))
                    for problem in problems:
                        print(f"FAIL {os_profile} ({shape}): {problem}")
                else:
                    print(f"ok   {os_profile} ({shape})")

    if failures:
        print(f"\n{len(failures)} package(s) have an unexpected payload")
        return 1
    print(
        f"\nall {len(profiles) * 2} {args.pkg_type} packages have the expected payload"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
