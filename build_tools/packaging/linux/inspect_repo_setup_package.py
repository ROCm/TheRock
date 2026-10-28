#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Verify the payload of a built amd-repo-setup package before it is uploaded.

This is a build-time check, not an install test. Given the directory a build
wrote its package to, it lists the archive and asserts that it contains the
files the package is supposed to install, at the right paths, owned by root,
and none of the files it must not ship.

What a package installs depends on its stream as well as its OS profile: the
installed repository filename carries the stream, and the two layout shapes
install different files:

  flat     (stable, rc)   the repository file and the signing key
  build_id (nightly)      the repository file only

Checking a package against the stream it was built for therefore also confirms
that it configures that stream: a package built for another stream installs a
differently named repository file, which is reported as missing. The name
itself (``rc`` installs ``amdrocm-stablerc``, not ``amdrocm-rc``) is pinned by
the unit tests.

The key paths were renamed away from generic ``rocm`` names as a precaution;
this checks that the package ships the renamed paths and not the previous ones.

Listing a package needs ``dpkg-deb`` for a deb and ``rpm`` for an rpm, which
building one already requires.

Usage:
  python build_tools/packaging/linux/inspect_repo_setup_package.py \
      --package-dir repo-setup-package-out \
      --os-profile ubuntu2404 \
      --stream nightly
"""

import argparse
import subprocess
import sys
from pathlib import Path, PurePosixPath

_THIS_DIR = Path(__file__).resolve().parent

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from build_repo_setup_package import (
    DEB_KEYRING_PATH,
    OS_PROFILES,
    RPM_GPG_KEY_PATH,
    STREAMS,
    repo_id,
    stream_shape,
)

# The layout shapes this inspector knows how to check. A flat stream is signed
# and ships its key; a per-build stream is unsigned and ships none. Streams come
# from the builder's own table, so one added there is accepted here without
# another edit -- but one whose shape is not handled here would be checked
# against the wrong files, so an unknown shape raises.
#
# These mirror build_repo_setup_package's private _FLAT / _BUILD_ID. They are compared
# against the public stream_shape() return value, and a test asserts the two
# spellings still agree.
_SHAPE_FLAT = "flat"
_SHAPE_BUILD_ID = "build_id"

# Everything the package installs is owned by root, whatever user builds it.
_EXPECTED_OWNER = "root/root"

# Key paths used before the rename. A generic rocm name is one another package
# could also claim, and neither dpkg nor rpm will unpack two packages that own
# the same file; see build_repo_setup_package's DEB_KEYRING_PATH for why the rename is
# precautionary. Asserted absent from every package.
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


def shape_of(stream: str) -> str:
    """Return a stream's layout shape, rejecting one this inspector cannot check."""
    shape = stream_shape(stream)
    if shape not in (_SHAPE_FLAT, _SHAPE_BUILD_ID):
        raise RuntimeError(
            f"stream {stream!r} has shape {shape!r}, which this inspector does "
            f"not know how to check"
        )
    return shape


def is_signed_stream(stream: str) -> bool:
    """Whether a stream's package ships a signing key. Follows the shape."""
    return shape_of(stream) == _SHAPE_FLAT


def expected_paths(os_profile: str, stream: str) -> set[PurePosixPath]:
    """Return the paths a build of this profile and stream must install."""
    paths = {repo_file_path(os_profile, stream)}
    if is_signed_stream(stream):
        paths.add(key_path(os_profile))
    return paths


def forbidden_paths(os_profile: str, stream: str) -> set[PurePosixPath]:
    """Return the paths a build of this profile and stream must not install."""
    paths = set(_SUPERSEDED_KEY_PATHS)
    if not is_signed_stream(stream):
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
        # Symlinks are reported as "link -> target"; keep the link's own path.
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
    """Run a command and capture its output."""
    return subprocess.run(
        argv, check=True, text=True, capture_output=True, encoding="utf-8", **kwargs
    )


def locate_built_package(dest_dir: Path, pkg_type: str) -> Path:
    """Return the single built package in ``dest_dir``.

    One build produces one package, so anything other than exactly one match is
    an error worth failing on rather than guessing about.
    """
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify the payload of a built amd-repo-setup package.",
    )
    p.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Directory holding the one package the build produced",
    )
    p.add_argument(
        "--os-profile",
        required=True,
        choices=sorted(OS_PROFILES),
        help="OS profile the package was built for; selects the package type",
    )
    p.add_argument(
        "--stream",
        required=True,
        choices=STREAMS,
        help="Stream the package was built for",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pkg_type = OS_PROFILES[args.os_profile]["pkg_type"]
    package = locate_built_package(args.package_dir, pkg_type)
    problems = check_payload(
        read_payload(package, pkg_type),
        expected_paths(args.os_profile, args.stream),
        forbidden_paths(args.os_profile, args.stream),
    )

    label = f"{package.name} ({args.os_profile}, {args.stream})"
    for problem in problems:
        print(f"FAIL {label}: {problem}")
    if problems:
        return 1
    print(f"ok   {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
