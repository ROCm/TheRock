#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Build the ``amdrocm-repo`` package that configures a system package manager
(apt/dnf/zypper) to install AMD ROCm from the public ROCm repositories.

One ``amdrocm-repo`` package is generated per target OS profile and RFC0012
stream. It ships a single repository definition and, for signed streams, the AMD
signing key, so that after installing the package a user can install ROCm with
their native package manager. The signing key is fetched at build time and
embedded in the package; it is never stored in the source tree.

Package naming
--------------

Every OS profile builds a *different* package, because each embeds its own
repository URL. The profile therefore has to reach the package name, or two of
them would be indistinguishable. Each ecosystem does that its own way: use the
distro token the ecosystem already defines where there is one, and the
``--os-profile`` name where there is not.

    deb   the profile goes in the version:   10.0.0-1~ubuntu2404
    rpm   the profile goes in the Release:   10.0.0-1.stable.el8

``-1`` is ``--version-suffix``, this package's own build number (default 1),
bumped to re-ship the repository configuration at an unchanged ROCm version.

Worked examples, all at ``--rocm-version 10.0.0`` with the default
``--version-suffix``. A build_id stream additionally takes
``--repo-sub-folder``; a flat stream rejects it:

    --os-profile  --stream  --repo-sub-folder  resulting package file
    ------------  --------  -----------------  -------------------------------------------
    ubuntu2404    stable    -                  amdrocm-repo_10.0.0-1~ubuntu2404_all.deb
    ubuntu2404    rc        -                  amdrocm-repo_10.0.0~pre-1~ubuntu2404_all.deb
    ubuntu2404    nightly   20260716-12345     amdrocm-repo_20260716.12345-1~ubuntu2404_all.deb
    rhel8         stable    -                  amdrocm-repo-10.0.0-1.stable.el8.noarch.rpm
    rhel8         rc        -                  amdrocm-repo-10.0.0-1.rc.el8.noarch.rpm
    rhel8         nightly   20260716-12345     amdrocm-repo-20260716-12345.1.nightly.el8.noarch.rpm
    rhel10        stable    -                  amdrocm-repo-10.0.0-1.stable.el10.noarch.rpm
    sles16        stable    -                  amdrocm-repo-10.0.0-1.stable.sles16.noarch.rpm

``rc`` configures the release-candidate repository for the next GA. Its marker
sorts below the plain version, so the stable package upgrades over it. Note that
``rc`` does not spell itself the same way everywhere -- its repo file is
``amdrocm-stablerc`` and its deb marker is ``~pre`` -- see STREAM_IDS.

``--rocm-version`` may carry a prerelease marker, which stays in the upstream
part of the version and sorts below the matching GA release:

    --rocm-version 7.14.0~dev20260811, --os-profile ubuntu2404
        -> amdrocm-repo_7.14.0~dev20260811-1~ubuntu2404_all.deb

```
python build_repo_package.py \
    --os-profile ubuntu2404 \
    --stream stable \
    --repo-base-url https://stable.repo.amd.com/rocm/core/packages \
    --rocm-version 10.0.0 \
    --dest-dir ./output
```

The same build for a per-build stream, which pins the repository to one build
folder and ships no signing key (those repositories are unsigned):

```
python build_repo_package.py \
    --os-profile rhel8 \
    --stream nightly \
    --repo-base-url https://nightly.repo.amd.com/rocm/core/packages \
    --repo-sub-folder 20260716-12345 \
    --rocm-version 10.0.0 \
    --dest-dir ./output
```
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path, PurePosixPath

from jinja2 import Environment, FileSystemLoader, select_autoescape

SCRIPT_DIR = Path(__file__).resolve().parent

# The package name. The repository identifier used for the repo file and its
# section header is derived per stream by ``repo_id()``. A single repository is
# configured per package.
PACKAGE_NAME = "amdrocm-repo"
REPO_ID_PREFIX = "amdrocm"
REPO_NAME = "AMD ROCm"

# The deb suite the repositories publish under (``dists/<suite>/``). Used to
# locate the index when verifying a repo URL. It is written literally in
# template/repo/deb/amdrocm.sources.j2; a test asserts the two agree.
#
# One suite for every stream, matching what the servers actually publish rather
# than naming the suite after the stream. Every stream serves ``dists/stable/``
# and declares ``Suite: stable`` / ``Codename: stable`` inside its own Release
# file; a per-stream suite such as ``dists/stablerc/`` returns 404. Pointing apt
# at a stream-named suite would therefore produce a package that installs
# cleanly and then fails on the user's first update, so this follows the server.
# Flagged to devops rather than encoded as a bug.
DEB_SUITE = "stable"

MAINTAINER = "ROCm Dev Support <rocm-dev.support@amd.com>"

# Where each package installs the signing key, and which the repo files
# reference. deb consumes a dearmored (binary) keyring via Signed-By; rpm
# references the (armored) key via gpgkey=file://.
#
# Both are named for this package (amdrocm) rather than the generic "rocm", so
# that no other package or hand-written setup step can claim the same path.
# dpkg and rpm both refuse to unpack two packages owning one path, so a generic
# name risks making this package uninstallable alongside something unrelated.
# This is precautionary: amdgpu-install 31.40 keeps its key elsewhere
# (/etc/apt/keyrings/rocm.gpg, /etc/amdgpu-install/rocm.gpg.key), so it is not
# a collision we have observed. The overlap that does exist with that package
# is the repository definition, and Conflicts handles it -- see
# LEGACY_INSTALLER_PACKAGE.
# These are installed names only; the key itself is fetched from the URL
# given by --gpg-key-url.
#
# /etc/pki/rpm-gpg is the RHEL-family convention. SUSE has no filesystem
# convention for third-party repo keys -- its trust store is the rpm database,
# and gpgkey= is only an import source -- so the same path is used there. No
# SUSE package provides /etc/pki/rpm-gpg, so nothing can collide with it.
DEB_KEYRING_PATH = "/usr/share/keyrings/amdrocm.gpg"
RPM_GPG_KEY_PATH = "/etc/pki/rpm-gpg/RPM-GPG-KEY-amdrocm"

# The legacy installer this package must not be co-installed with. It ships its
# own enabled ROCm repository as a packaged file -- /etc/apt/sources.list.d/
# rocm.list on deb, [rocm] in /etc/yum.repos.d/rocm.repo on rpm -- both pointing
# at repo.radeon.com. Its repo files use different names than ours, so nothing
# stops the two being installed together; the result is a system with two ROCm
# repositories configured. It also grants itself precedence (an apt pin at
# priority 600 against a 500 default, and priority=50 against dnf's 99 default),
# so it wins package resolution rather than reporting a conflict.
# Required by RFC0012, "Repository Package".
#
# What the declaration actually achieves differs by package manager, so do not
# describe it as closing the overlap outright:
#
#   dnf/zypper  refuse the pair with an explicit conflict error, and erasing
#               that package also removes its %config repo files. Clean.
#   apt         satisfies the conflict by removing the package to "rc" state.
#               Its repo file and priority pin are conffiles, so both survive.
#               The packages are no longer co-installed, but the repository
#               overlap remains until the user purges rather than removes.
#
# The Debian remedy is therefore "apt purge", which the documentation states.
LEGACY_INSTALLER_PACKAGE = "amdgpu-install"

GPG_FETCH_TIMEOUT_SEC = 60
# Retry the key fetch so a transient network/CDN blip does not fail a build.
GPG_FETCH_ATTEMPTS = 3
GPG_FETCH_BACKOFF_SEC = 3
# Upper bound on the fetched signing key. A real armored key is a few KB; this
# bounds memory use if a repository host returns an unexpectedly large body.
MAX_GPG_KEY_BYTES = 1 << 20  # 1 MiB

# Bounds for the optional repository reachability check (--verify-repo-url).
REPO_VERIFY_TIMEOUT_SEC = 30
REPO_VERIFY_ATTEMPTS = 3
REPO_VERIFY_BACKOFF_SEC = 3
# HTTP statuses worth retrying: request timeout and rate limiting. Any other 4xx
# means the repository is not at that URL, which retrying cannot change.
RETRYABLE_HTTP_STATUS = frozenset({408, 429})

# Primary-key fingerprint of the AMD ROCm signing key. The key is fetched over
# the network at build time and embedded as the package's trust anchor, so its
# identity is pinned here and verified after fetch. Rotating the repository
# signing key requires updating this value.
EXPECTED_KEY_FINGERPRINT = "D0F004A0025A1145C7807FCD0701EAC4D5E02107"

# Accepted input shapes. rocm_version and repo_sub_folder are rendered into the
# generated repo files, package version fields, and (for the deb version string)
# other metadata, so they are constrained to characters that cannot alter the
# surrounding syntax. repo_base_url is concatenated into the repo files as well.
# Use \Z (not $) so a trailing newline is not accepted.
#
# No "-": an rpm Version cannot contain one, since rpm uses it to separate the
# version from the release (see rpm_version_release). A pre-release marker
# belongs in the "~rc1" form, which both rpm and dpkg sort before the plain
# version.
#
# The same rule applies to the sub-folder's identifier: rpm_version_release
# splits on the first "-" and the identifier becomes the rpm Release, which
# cannot contain one either. Only the single "-" separating the date is
# allowed, so a further "-" is rejected here rather than failing later in
# rpmbuild -- where the deb would still have built, silently diverging.
_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+~]*\Z")
_SUB_FOLDER_RE = re.compile(r"^[0-9]{8}-[0-9A-Za-z._]+\Z")

# --version-suffix is the package's own build number, bumped to re-ship the
# repository configuration at an unchanged ROCm version. It is stricter than
# _VERSION_RE in two ways:
#
#   no "-"  deb_version joins it to the upstream version with the single "-"
#           that separates a debian revision, and rpm forbids one in a Release.
#   no "~"  deb_version appends "~<os-profile>" after it, and "~" sorts before
#           everything, so a "~" inside the suffix would reorder the two.
_VERSION_SUFFIX_RE = re.compile(r"^[0-9][0-9A-Za-z.+]*\Z")

# The RFC0012 streams, each served from <stream>.repo.amd.com.
#
# A stream has one of two shapes, and everything else follows from the shape:
#
#   flat      one repository per distro, holding every retained version.
#             Signed, so the package ships the key and enables gpgcheck.
#   build_id  the same tree with a <YYYYMMDD-runid> segment naming a single
#             build. Unsigned, so the package ships no key and trusts the repo.
#
# Signedness is not an independent axis: the flat streams are the ones that
# publish InRelease/Release.gpg and repomd.xml.asc, and the build_id streams
# publish none of them. Branch on the shape, never on the stream name.
_FLAT = "flat"
_BUILD_ID = "build_id"

STREAM_SHAPES = {
    "stable": _FLAT,
    "rc": _FLAT,
    "nightly": _BUILD_ID,
}
STREAMS = tuple(STREAM_SHAPES)

# Adding a *flat* stream is an entry here, any alternate names it carries (see
# STREAM_IDS), and its golden tests.
#
# Adding a second *build_id* stream needs more than that: deb_version() derives
# a build_id version from the sub-folder alone, so two build_id streams sharing
# a sub-folder would produce one deb version for two different packages. The rpm
# side already carries the stream in its Release. test_streams_never_share_a_
# package_version fails loudly if this is ever reached, but the fix is to give
# deb_version a stream token, not to relax the test.

# Alternate names a stream is known by, where they differ from the stream id.
# Streams absent here, or missing a key, use the stream id unchanged.
#
#   "file_id"     the installed repo file. RFC0012 names rc's tier package
#                 amdrocm-repo-stablerc, so the file is amdrocm-stablerc while
#                 the subdomain stays rc. Matching it now keeps the eventual
#                 per-tier split from having to rename a package and a file
#                 together -- the one case dpkg and rpm do not handle.
#   "deb_marker"  the prerelease marker. docs/packaging/versioning.md gives deb
#                 "~preN" against rpm's "~rcN", and the published packages
#                 agree, so this package follows suit alongside them. No
#                 candidate number: that would name one candidate, and this
#                 names the stream.
#
# rpm needs no marker: its Release carries the stream verbatim, and cannot
# contain a "-" anyway (see rpm_version_release).
STREAM_IDS = {
    "rc": {"file_id": "stablerc", "deb_marker": "pre"},
}


def _stream_id(stream: str, key: str) -> str:
    """Return a stream's alternate name for ``key``, or the stream name."""
    return STREAM_IDS.get(stream, {}).get(key, stream)


# "dist_tag" is appended to the rpm Release so two profiles cannot produce an
# identically named package. Stated per profile rather than taken from rpm's
# %{?dist}, which reads the *build host* rather than the target (a rhel8 build
# in a RHEL 9 container would stamp ".el9") and is empty on SUSE, so every SUSE
# profile would share one Release -- invisible while sles16 is the only one.
#
# ".el8"/".el10" reproduce what %{?dist} yields there, so RHEL names do not
# change. ".sles16" is a deliberate departure from SUSE's own convention, which
# leads the Release with a numeric code (160000.2.2 on SLE 16.0): that is a
# prefix where RHEL's is a suffix, so no single placement satisfies both.
#
# deb profiles carry no dist_tag -- deb_version() puts the profile in the
# version -- so read it with .get(); build_context() runs for both types.
OS_PROFILES = {
    "ubuntu2404": {
        "family": "debian",
        "pkg_type": "deb",
        "description": "Ubuntu 24.04",
    },
    "rhel8": {
        "family": "rhel",
        "pkg_type": "rpm",
        "rpm_repo_dir": "/etc/yum.repos.d",
        "rpm_refresh_cmd": "sudo dnf makecache",
        "dist_tag": ".el8",
        "description": "RHEL/Rocky/OL 8",
    },
    "rhel10": {
        "family": "rhel",
        "pkg_type": "rpm",
        "rpm_repo_dir": "/etc/yum.repos.d",
        "rpm_refresh_cmd": "sudo dnf makecache",
        "dist_tag": ".el10",
        "description": "RHEL/Rocky/OL 10",
    },
    "sles16": {
        "family": "sles",
        "pkg_type": "rpm",
        "rpm_repo_dir": "/etc/zypp/repos.d",
        "rpm_refresh_cmd": "sudo zypper refresh",
        "dist_tag": ".sles16",
        "description": "SLES 16",
    },
}


# --- URL / signing derivation ------------------------------------------------
#
# Every stream is served per distro under a common base:
#
#   flat      <base>/<os_profile>/                     (deb)
#             <base>/<os_profile>/x86_64/              (rpm)
#   build_id  <base>/<os_profile>/<YYYYMMDD-id>/       (deb)
#             <base>/<os_profile>/<YYYYMMDD-id>/x86_64/ (rpm)
#
# where <base> is https://<stream>.repo.amd.com/rocm/core/packages.
#
# The per-format trees (<base>/deb/, <base>/rpm/) are also served, and hold the
# same builds, so a package pointing at either one works today. They are not
# used here: the published install instructions configure the per-distro tree,
# and a stream that ever drops the per-format alias would break silently in the
# field rather than in CI. A dual-served layout cannot tell a right mapping from
# a wrong one, which is how an earlier GA-line break got through.
#
# The signing key is not under <base> at all -- it sits beside core/, two
# levels up -- so its URL is passed in whole rather than derived here. That
# also keeps the key's location a property of the publisher rather than an
# assumption baked into this tool.


def repo_id(stream: str) -> str:
    """Return the repo-file stem and section id for a stream.

    ``amdrocm-stable``, ``amdrocm-stablerc`` etc. The stem drives the installed
    filename in both families -- ``debian/install`` maps ``{repo_id}.sources``
    and the spec ``%files`` lists ``{repo_id}.repo`` -- so naming the stream
    here is what keeps two streams' repo files from overwriting one another on
    disk.

    The stem is not always the stream name: rc's file is ``amdrocm-stablerc``
    while its subdomain stays ``rc``. See STREAM_IDS.
    """
    stream_shape(stream)  # reject unknown streams before they reach a filename
    return f"{REPO_ID_PREFIX}-{_stream_id(stream, 'file_id')}"


def stream_shape(stream: str) -> str:
    """Return the layout shape for a stream.

    Raises:
        ValueError: If the stream is unknown.
    """
    try:
        return STREAM_SHAPES[stream]
    except KeyError as e:
        raise ValueError(
            f"stream={stream!r} is invalid, expected one of {STREAMS}"
        ) from e


def is_signed(stream: str) -> bool:
    """Whether this stream's public repository is signed.

    Signedness follows the shape: flat streams publish ``InRelease`` /
    ``Release.gpg`` (deb) and ``repomd.xml.asc`` (rpm); build_id streams publish
    none of them and are consumed with ``trusted=yes`` / ``gpgcheck=0``.
    """
    return stream_shape(stream) == _FLAT


def repo_baseurl(
    repo_base_url: str,
    pkg_type: str,
    os_profile: str,
    stream: str,
    repo_sub_folder: str,
) -> str:
    """Return the repository baseurl baked into the repo file.

    deb returns the repo root, beneath which apt resolves ``dists/stable/main/``;
    rpm returns the directory containing ``repodata/``. ``repo_sub_folder`` is
    the ``YYYYMMDD-runid`` segment naming a single build, required by the
    build_id shape and rejected for the flat one.

    Raises:
        ValueError: If the stream is unknown, or the sub-folder does not match
            the stream's shape.
    """
    base = repo_base_url.rstrip("/")
    shape = stream_shape(stream)
    # Silently dropping a missing sub-folder would emit a flat URL for a
    # build_id stream -- indistinguishable from a correct stable URL, and
    # serving nothing, so the package would install and fail on first refresh.
    if shape == _BUILD_ID and not repo_sub_folder:
        raise ValueError(f"stream={stream!r} requires a build sub-folder")
    if shape != _BUILD_ID and repo_sub_folder:
        raise ValueError(f"stream={stream!r} takes no build sub-folder")
    parts = [base, os_profile]
    if shape == _BUILD_ID:
        parts.append(repo_sub_folder)
    if pkg_type != "deb":
        parts.append("x86_64")
    return "/".join(parts) + "/"


def repo_metadata_url(baseurl: str, pkg_type: str) -> str:
    """Return a repository index file that must exist under ``baseurl``.

    apt reads ``dists/<suite>/Release``; dnf and zypper read
    ``repodata/repomd.xml``. Fetching one proves the baseurl serves a repository.
    """
    if pkg_type == "deb":
        return f"{baseurl}dists/{DEB_SUITE}/Release"
    return f"{baseurl}repodata/repomd.xml"


def verify_repo_url(baseurl: str, pkg_type: str) -> None:
    """Fail the build if ``baseurl`` does not serve a package repository.

    The baseurl is baked into the package and is only exercised the first time a
    user refreshes their package manager, so an unreachable one would otherwise
    ship undetected. Transient failures are retried; a repository that is simply
    not there fails immediately.

    The scheme is not re-checked here: this response is neither trusted nor
    retained, and the signing key (which is) enforces https separately.
    """
    url = repo_metadata_url(baseurl, pkg_type)
    last_error: Exception | None = None
    for attempt in range(1, REPO_VERIFY_ATTEMPTS + 1):
        try:
            print(f"Verifying repository (attempt {attempt}): {url}")
            with urllib.request.urlopen(url, timeout=REPO_VERIFY_TIMEOUT_SEC) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
            if status == 200:
                print(f"Repository verified: {baseurl}")
                return
            # urlopen already raised for the error codes, so anything else here
            # is a URL that answers but does not serve a repository index.
            # Retrying cannot change that, so fail now rather than sleeping
            # through the remaining attempts.
            raise RuntimeError(
                f"repository is not reachable at {baseurl} "
                f"(checked {url}): unexpected HTTP status {status}"
            )
        except urllib.error.HTTPError as e:
            # A missing or forbidden index is a wrong URL, not a blip. Some
            # object stores answer 403 rather than 404 for absent keys, so treat
            # both as final; only rate limiting and timeouts are worth retrying.
            if e.code not in RETRYABLE_HTTP_STATUS:
                raise RuntimeError(
                    f"repository is not reachable at {baseurl} "
                    f"(checked {url}): HTTP {e.code}"
                ) from e
            last_error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
        if attempt < REPO_VERIFY_ATTEMPTS:
            time.sleep(REPO_VERIFY_BACKOFF_SEC * attempt)
    raise RuntimeError(
        f"repository is not reachable at {baseurl} (checked {url}): {last_error}"
    )


def _require_https(url: str, what: str) -> None:
    """Reject a non-https URL used to fetch trusted key material."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"{what} must use https to protect the key in transit: {url}")


def _valid_repo_base_url(url: str) -> bool:
    """Whether a repo base URL is a well-formed http(s) URL with no control chars.

    The value is concatenated into the generated repo files, so whitespace or
    control characters (e.g. a newline) that could inject extra directives are
    rejected here.
    """
    if any(c.isspace() or ord(c) < 0x20 for c in url):
        return False
    parts = urllib.parse.urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def rpm_version_release(
    stream: str,
    rocm_version: str,
    repo_sub_folder: str,
    version_suffix: str,
    dist_tag: str,
) -> tuple[str, str]:
    """Return (Version, Release) for the rpm package.

    Flat streams track the (rolling) ROCm version. A build_id stream is pinned
    to its build: an rpm Version cannot contain ``-``, so the date becomes the
    Version and the id leads the Release.

    Release is ``[<id>.]<version_suffix>.<stream><dist_tag>``, e.g.
    ``1.stable.el8`` or ``12345.1.nightly.el8``. Each part exists to keep two
    packages from sharing a name:

    - ``stream`` -- without it, installing one stream's package over another at
      the same ROCm version is a silent no-op: the package manager reports
      success and leaves the old repository configured.
    - ``dist_tag`` -- one package is built per OS profile and their contents
      differ, so the profile must reach the name too (see OS_PROFILES).
    - ``version_suffix`` -- this package's own build number, so the repository
      configuration can be re-shipped at an unchanged ROCm version.

    Note that Version is compared before Release, and a build_id Version is a
    date (``20260827``) while a flat one is a semantic version (``10.0.0``), so
    a build_id package always outranks a flat one. Switching from nightly to
    stable is therefore a downgrade rather than an upgrade. That predates the
    streams and resolves itself if the streams ever become separate packages.
    """
    if stream_shape(stream) == _BUILD_ID:
        date, _, ident = repo_sub_folder.partition("-")
        lead = f"{ident}." if ident else ""
        return date, f"{lead}{version_suffix}.{stream}{dist_tag}"
    return rocm_version, f"{version_suffix}.{stream}{dist_tag}"


# An rpm %changelog date has to be English whatever LC_TIME the build machine
# happens to use. Both strftime("%a %b") and calendar.day_abbr read the process
# locale, so the names are pinned here instead. This mirrors what the standard
# library does for the deb changelog: email.utils.format_datetime carries its
# own English tables for the same reason.
_CHANGELOG_DAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_CHANGELOG_MONTH_ABBR = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def rpm_changelog_date(when: datetime) -> str:
    """Return ``when`` formatted as an rpm %changelog date, always in English."""
    return (
        f"{_CHANGELOG_DAY_ABBR[when.weekday()]} "
        f"{_CHANGELOG_MONTH_ABBR[when.month - 1]} "
        f"{when.day:02d} {when.year}"
    )


def deb_version(
    stream: str,
    rocm_version: str,
    repo_sub_folder: str,
    version_suffix: str,
    os_profile: str,
) -> str:
    """Return the deb package version, e.g. ``10.0.0-1~ubuntu2404``.

    The value is ``<upstream>-<debian revision>``. The upstream part names what
    the repository points at: the ROCm version for a flat stream, or the build
    folder for a build_id one. That folder's ``-`` becomes a ``.`` because only
    the *last* ``-`` in a deb version separates the revision, so leaving it
    would split the version in the wrong place. A flat stream other than
    ``stable`` also carries a ``~<marker>``, which sorts below the plain
    version so that stable upgrades over it.

    The marker is the deb spelling of the stream rather than the stream name:
    rc becomes ``~pre``, matching the ``~preN`` that the ROCm content packages
    in that same repository carry on deb (they use ``~rcN`` on rpm). It carries
    no candidate number -- see STREAM_IDS.

    The revision carries ``<version_suffix>~<os_profile>``:

    - ``version_suffix`` is this package's own build number, so the repository
      configuration can be re-shipped at an unchanged ROCm version.
    - ``os_profile`` is part of the version because one package is built per
      profile and their contents differ -- each embeds its own repository URL.
      Without it every debian-family profile would produce an identically named
      package holding different files.

    Profile names rather than Debian codenames: codenames are not
    alphabetically ordered, so ``10.0.0-1~bullseye1`` sorts *above*
    ``10.0.0-1~bookworm1`` and an in-place Debian 11 -> 12 upgrade would see the
    newer package as a downgrade. ``ubuntu2404``/``debian12`` are monotonic, and
    match both ``--os-profile`` and the ``repo/<os-profile>/`` publish path.

    Note that ``~`` sorts before everything, including nothing at all, so
    ``10.0.0-1~ubuntu2404`` ranks *below* a bare ``10.0.0-1``. Every package
    built here carries a profile, so that comparison never arises between two of
    ours -- but it is why the separator must stay ``~`` and not become ``.``.
    """
    if stream_shape(stream) == _BUILD_ID:
        upstream = repo_sub_folder.replace("-", ".")
    elif stream == "stable":
        upstream = rocm_version
    else:
        upstream = f"{rocm_version}~{_stream_id(stream, 'deb_marker')}"
    return f"{upstream}-{version_suffix}~{os_profile}"


# --- signing key -------------------------------------------------------------


def _fetch_signing_key(url: str) -> bytes:
    """Fetch the armored signing key over https, size-bounded, with retries.

    Only transient network errors are retried; a scheme downgrade or an
    over-size body fails immediately.
    """
    # The fetched key is embedded and trusted, so it must travel over https.
    _require_https(url, "signing key URL")
    last_error: Exception | None = None
    for attempt in range(1, GPG_FETCH_ATTEMPTS + 1):
        try:
            print(f"Fetching signing key (attempt {attempt}): {url}")
            with urllib.request.urlopen(url, timeout=GPG_FETCH_TIMEOUT_SEC) as resp:
                # urlopen follows redirects; re-check the scheme actually served
                # so an https URL cannot be silently downgraded to http.
                _require_https(resp.geturl(), "signing key URL (after redirects)")
                # Read one byte past the cap to detect an over-size body.
                key = resp.read(MAX_GPG_KEY_BYTES + 1)
            if len(key) > MAX_GPG_KEY_BYTES:
                raise ValueError(
                    f"signing key exceeds {MAX_GPG_KEY_BYTES} bytes: {url}"
                )
            return key
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_error = e
            if attempt < GPG_FETCH_ATTEMPTS:
                time.sleep(GPG_FETCH_BACKOFF_SEC * attempt)
    raise RuntimeError(
        f"failed to fetch signing key after {GPG_FETCH_ATTEMPTS} attempts: {url}"
    ) from last_error


def _run_gpg(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run gpg, reporting a missing binary by name instead of a traceback.

    A signed stream reaches gpg in two places: the deb path always, to dearmor
    the keyring, and either path when the key is fetched, to check its
    fingerprint.
    An rpm build given --gpg-key-file does neither, so it needs no gpg at all.
    """
    try:
        return subprocess.run(argv, check=True, **kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(
            "gpg is required to build a signed stream: it dearmors the deb "
            "keyring, and checks the fingerprint of a fetched key. Install "
            "gnupg/gnupg2. Note that --gpg-key-file skips the fetch, not the deb "
            "keyring step, so a signed deb build needs gpg either way"
        ) from e


def _key_fingerprint(armored: bytes) -> str:
    """Return the primary-key fingerprint of an armored public key."""
    with tempfile.TemporaryDirectory() as home:
        key_path = Path(home) / "key.asc"
        key_path.write_bytes(armored)
        result = _run_gpg(
            ["gpg", "--homedir", home, "--with-colons", "--show-keys", str(key_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    for line in result.stdout.decode(errors="replace").splitlines():
        if line.startswith("fpr:"):
            return line.split(":")[9]
    raise ValueError("no fingerprint found in the fetched signing key")


def _verify_key_fingerprint(armored: bytes) -> None:
    """Reject a fetched key whose fingerprint is not the pinned AMD ROCm key."""
    fingerprint = _key_fingerprint(armored)
    if fingerprint != EXPECTED_KEY_FINGERPRINT:
        raise ValueError(
            f"fetched signing key fingerprint {fingerprint} does not match the "
            f"expected {EXPECTED_KEY_FINGERPRINT}"
        )


def load_signing_key(args: argparse.Namespace) -> bytes:
    """Return the armored signing key bytes for a signed stream.

    Read from ``--gpg-key-file`` when given (offline/test builds), otherwise
    fetched from ``--gpg-key-url`` at build time. Never written into the
    source tree.

    Two guards apply to the fetch only: the https requirement
    (``_require_https``, including after redirects) and the pinned fingerprint.
    They exist to detect a tampered or misconfigured repository, which is a
    remote input. A key passed with ``--gpg-key-file`` is a local file the
    caller has already chosen, so it is embedded as provided and neither guard
    runs; callers supplying their own key are responsible for it being the
    right one.

    ``--gpg-key-url`` is still checked for http(s) shape by ``parse_args``
    whenever it is non-empty, including alongside ``--gpg-key-file``. That is a
    separate, earlier check -- it rejects a malformed value rather than
    vouching for the key.
    """
    if args.gpg_key_file:
        return args.gpg_key_file.read_bytes()
    key = _fetch_signing_key(args.gpg_key_url)
    _verify_key_fingerprint(key)
    return key


def dearmor_key(armored: bytes) -> bytes:
    """Convert an armored key to a binary keyring (for deb Signed-By)."""
    result = _run_gpg(
        ["gpg", "--dearmor"],
        input=armored,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


# --- templating --------------------------------------------------------------


def get_jinja_env() -> Environment:
    # keep_trailing_newline so rendered config files end with a newline (POSIX
    # text-file hygiene; Jinja strips the final newline by default).
    #
    # autoescape matches deb_package.py and rpm_package.py: escaping is enabled
    # for markup extensions only. Every template here renders a package-manager
    # config file, not markup, and escaping one would corrupt it -- an "&" in a
    # repository URL would become "&amp;". The values interpolated into these
    # files are constrained by the input patterns above, which is the control
    # that actually applies to this syntax.
    return Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        autoescape=select_autoescape(
            enabled_extensions=("html", "htm", "xml"),
            default_for_string=True,
            default=False,
        ),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render(
    env: Environment, template_path: str, output_path: Path, context: dict
) -> None:
    # Explicit encoding: write_text() would otherwise follow the build
    # machine's locale, and this module is deliberately locale-independent
    # elsewhere (see rpm_changelog_date).
    output_path.write_text(
        env.get_template(template_path).render(context), encoding="utf-8"
    )


def build_context(args: argparse.Namespace, profile: dict) -> dict:
    """Template fields shared by the repo file, spec, and deb metadata."""
    signed = is_signed(args.stream)
    return {
        "pkg_name": PACKAGE_NAME,
        "repo_id": repo_id(args.stream),
        "name": REPO_NAME,
        "baseurl": repo_baseurl(
            args.repo_base_url,
            profile["pkg_type"],
            args.os_profile,
            args.stream,
            args.repo_sub_folder,
        ),
        "signed": signed,
        "rocm_version": args.rocm_version,
        "maintainer": MAINTAINER,
        "description_short": "AMD ROCm repository configuration",
        "description_long": (
            "Configures the system package manager to install AMD ROCm packages."
        ),
        "deb_keyring_path": DEB_KEYRING_PATH,
        # Split out so debian/install maps the staged file to the same place
        # Signed-By points at, without repeating the name.
        #
        # PurePosixPath, not Path: these describe the layout of the package
        # being built, not of the machine building it. Path follows the local
        # flavour, so on Windows the parent would render as
        # "\usr\share\keyrings" and apt would never find the keyring.
        "deb_keyring_file": PurePosixPath(DEB_KEYRING_PATH).name,
        "deb_keyring_dir": str(PurePosixPath(DEB_KEYRING_PATH).parent),
        "rpm_gpg_key_path": RPM_GPG_KEY_PATH,
        # Same reason as deb: the spec stages and installs by basename, so
        # keep one source of truth for the name.
        "rpm_gpg_key_file": PurePosixPath(RPM_GPG_KEY_PATH).name,
        "rpm_repo_dir": profile.get("rpm_repo_dir", ""),
        # .get(), not a subscript: this function runs before main() dispatches
        # on pkg_type, so it is called for deb profiles too -- and those carry
        # no dist_tag.
        #
        # Consumed by build_rpm_package, which hands it to rpm_version_release.
        # The spec template renders the finished Release string and must not
        # append the tag a second time.
        "dist_tag": profile.get("dist_tag", ""),
        # Same .get() reasoning as dist_tag: deb profiles reach here too.
        "rpm_refresh_cmd": profile.get("rpm_refresh_cmd", ""),
        # Declared for every stream: amdgpu-install configures a ROCm
        # repository whichever stream this package points at.
        "legacy_installer_package": LEGACY_INSTALLER_PACKAGE,
    }


# --- rpm build ---------------------------------------------------------------


def build_rpm_package(
    args: argparse.Namespace, profile: dict, context: dict, dest_dir: Path
) -> None:
    env = get_jinja_env()
    build_dir = dest_dir / "rpm" / PACKAGE_NAME
    if build_dir.exists():
        shutil.rmtree(build_dir)
    for subdir in ("SOURCES", "SPECS", "BUILD", "RPMS", "SRPMS"):
        (build_dir / subdir).mkdir(parents=True, exist_ok=True)
    sources_dir = build_dir / "SOURCES"

    _render(
        env,
        "template/repo/rpm/amdrocm.repo.j2",
        sources_dir / f"{context['repo_id']}.repo",
        context,
    )

    if context["signed"]:
        # rpm consumes the armored key as-is (gpgkey=file://).
        (sources_dir / PurePosixPath(RPM_GPG_KEY_PATH).name).write_bytes(
            load_signing_key(args)
        )

    version, release = rpm_version_release(
        args.stream,
        args.rocm_version,
        args.repo_sub_folder,
        args.version_suffix,
        context["dist_tag"],
    )
    spec_context = {
        **context,
        "version": version,
        "release": release,
        "changelog_date": rpm_changelog_date(datetime.now(timezone.utc)),
    }
    spec_path = build_dir / "SPECS" / f"{PACKAGE_NAME}.spec"
    _render(env, "template/repo/rpm/amdrocm-repo.spec.j2", spec_path, spec_context)

    print(f"Building rpm package: {PACKAGE_NAME} for {profile['description']}")
    subprocess.run(
        ["rpmbuild", "--define", f"_topdir {build_dir}", "-bb", str(spec_path)],
        check=True,
    )
    for rpm_file in (build_dir / "RPMS" / "noarch").glob("*.rpm"):
        target = dest_dir / rpm_file.name
        shutil.move(str(rpm_file), str(target))
        print(f"Package created: {target}")

    # Leave only the built packages behind. The rpmbuild tree is scratch, and
    # CI publishes this directory, so anything left here ships with the
    # package. Failing to clean up must not fail a build that succeeded.
    shutil.rmtree(build_dir.parent, ignore_errors=True)


# --- deb build ---------------------------------------------------------------


def build_deb_package(
    args: argparse.Namespace, profile: dict, context: dict, dest_dir: Path
) -> None:
    env = get_jinja_env()
    package_dir = dest_dir / "deb" / PACKAGE_NAME
    deb_dir = package_dir / "debian"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    deb_dir.mkdir(parents=True, exist_ok=True)

    deb_context = {
        **context,
        "deb_version": deb_version(
            args.stream,
            args.rocm_version,
            args.repo_sub_folder,
            args.version_suffix,
            args.os_profile,
        ),
        "date": format_datetime(datetime.now(timezone.utc)),
    }

    # Repository definition (deb822) and, for a signed stream, the dearmored key,
    # placed at the package root and mapped into the filesystem by debian/install.
    _render(
        env,
        "template/repo/deb/amdrocm.sources.j2",
        package_dir / f"{deb_context['repo_id']}.sources",
        deb_context,
    )
    if context["signed"]:
        (package_dir / PurePosixPath(DEB_KEYRING_PATH).name).write_bytes(
            dearmor_key(load_signing_key(args))
        )

    for name in ("control", "changelog", "rules", "postinst", "install"):
        _render(
            env,
            f"template/repo/deb/{name}.j2",
            deb_dir / name,
            deb_context,
        )
    (deb_dir / "rules").chmod(0o755)
    (deb_dir / "postinst").chmod(0o755)
    # Native source format: this package has no upstream outside this
    # repository, so there is no separate tarball for a non-native format to
    # reference.
    #
    # A native package conventionally carries no debian revision, and this one
    # does (see deb_version). That convention is enforced by dpkg-source, which
    # a binary-only build never runs: "dpkg-buildpackage -b" below produces the
    # .deb straight from debian/. Checked against both dpkg-buildpackage and
    # lintian, neither of which reports a version or source-format tag for it.
    (deb_dir / "source").mkdir(exist_ok=True)
    (deb_dir / "source" / "format").write_text("3.0 (native)\n", encoding="utf-8")

    print(f"Building deb package: {PACKAGE_NAME} for {profile['description']}")
    subprocess.run(
        ["dpkg-buildpackage", "-uc", "-us", "-b"],
        cwd=package_dir,
        check=True,
    )
    # dpkg-buildpackage writes the .deb to the parent of the package directory.
    for deb_file in package_dir.parent.glob("*.deb"):
        target = dest_dir / deb_file.name
        shutil.move(str(deb_file), str(target))
        print(f"Package created: {target}")

    # As for rpm: drop the build tree, and with it the .buildinfo and .changes
    # dpkg-buildpackage leaves alongside the package.
    shutil.rmtree(package_dir.parent, ignore_errors=True)


# --- CLI ---------------------------------------------------------------------


def list_profiles(pkg_type: str) -> list[str]:
    """Return the OS profiles that build the given package type."""
    return [name for name, p in OS_PROFILES.items() if p["pkg_type"] == pkg_type]


def run_list_profiles(argv: list[str]) -> None:
    """Print the OS profiles for a package type as a JSON array (CI matrix)."""
    p = argparse.ArgumentParser(
        prog="build_repo_package.py list-profiles",
        description="Print the OS profiles for a package type as a JSON array.",
    )
    p.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    ns = p.parse_args(argv)
    print(json.dumps(list_profiles(ns.pkg_type)))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the amdrocm-repo package that configures ROCm repositories.",
    )
    p.add_argument(
        "--os-profile",
        required=True,
        choices=sorted(OS_PROFILES.keys()),
        help="Target distro profile",
    )
    p.add_argument(
        "--stream",
        required=True,
        choices=STREAMS,
        help="RFC0012 repository stream to configure",
    )
    p.add_argument(
        "--repo-base-url",
        required=True,
        help="Public repository base URL (the packages-multi-arch base)",
    )
    p.add_argument(
        "--repo-sub-folder",
        default="",
        help="Build sub-folder for a per-build stream (YYYYMMDD-<id>)",
    )
    p.add_argument(
        "--gpg-key-file",
        type=Path,
        default=None,
        help=(
            "Signing key file to embed (defaults to fetching --gpg-key-url). The "
            "pinned-fingerprint check applies to the fetched key only; a key "
            "given here is embedded as provided"
        ),
    )
    p.add_argument(
        "--rocm-version",
        required=True,
        help=(
            "ROCm version (e.g. 10.0.0). Required for every stream, but a "
            "per-build stream takes both of its version fields from "
            "--repo-sub-folder, so there this value reaches only the "
            "changelog entry"
        ),
    )
    p.add_argument(
        "--version-suffix",
        default="1",
        help=(
            "This package's own build number, separate from --rocm-version "
            "(default: 1). Bump it to re-ship the repository configuration "
            "at an unchanged ROCm version: without it there would be no "
            "higher version to publish, and the package manager would treat "
            "the corrected package as already installed"
        ),
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Output directory for built packages",
    )
    p.add_argument(
        "--gpg-key-url",
        default="",
        help=(
            "URL of the armored signing key to embed, e.g. https://"
            "stable.repo.amd.com/rocm/gpg/packages.gpg. Given whole rather "
            "than derived: the key is not under --repo-base-url (packages sit "
            "at <root>/core/packages/ and the key beside core/), and where it "
            "sits is the publisher's choice, not this tool's. The fetch is "
            "https-only and the key must match the pinned fingerprint. "
            "Required for a signed stream unless --gpg-key-file is given"
        ),
    )
    p.add_argument(
        "--verify-repo-url",
        action="store_true",
        help=(
            "Fail the build unless the configured repository is reachable. "
            "No-op for a build_id stream, whose build folder is published by "
            "the same run that builds the package"
        ),
    )
    args = p.parse_args(argv)
    if is_signed(args.stream) and not args.gpg_key_file and not args.gpg_key_url:
        p.error(
            f"--gpg-key-url or --gpg-key-file is required for --stream {args.stream}"
        )
    if args.gpg_key_url and not _valid_repo_base_url(args.gpg_key_url):
        p.error(f"--gpg-key-url is not a valid http(s) URL: {args.gpg_key_url!r}")
    build_id_shape = stream_shape(args.stream) == _BUILD_ID
    if build_id_shape and not args.repo_sub_folder:
        p.error(f"--repo-sub-folder is required for --stream {args.stream}")
    if args.repo_sub_folder and not build_id_shape:
        p.error(f"--repo-sub-folder is not valid for --stream {args.stream}")
    # Validate values that are rendered into repo files, version fields, and
    # maintainer metadata so they cannot alter the surrounding syntax.
    if not _VERSION_RE.match(args.rocm_version):
        p.error(f"--rocm-version has an unexpected format: {args.rocm_version!r}")
    if not _VERSION_SUFFIX_RE.match(args.version_suffix):
        p.error(f"--version-suffix has an unexpected format: {args.version_suffix!r}")
    if not _valid_repo_base_url(args.repo_base_url):
        p.error(f"--repo-base-url is not a valid http(s) URL: {args.repo_base_url!r}")
    if args.repo_sub_folder and not _SUB_FOLDER_RE.match(args.repo_sub_folder):
        p.error(f"--repo-sub-folder has an unexpected format: {args.repo_sub_folder!r}")
    # rpmbuild requires an absolute _topdir; normalize so a relative --dest-dir works.
    args.dest_dir = args.dest_dir.resolve()
    return args


def main(argv=None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "list-profiles":
        run_list_profiles(argv[1:])
        return
    args = parse_args(argv)
    profile = OS_PROFILES[args.os_profile]
    args.dest_dir.mkdir(parents=True, exist_ok=True)
    context = build_context(args, profile)
    # A build_id repository is published by the same run that builds this
    # package, so its build folder does not exist yet and cannot be checked.
    if args.verify_repo_url and stream_shape(args.stream) != _BUILD_ID:
        verify_repo_url(context["baseurl"], profile["pkg_type"])
    if profile["pkg_type"] == "deb":
        build_deb_package(args, profile, context, args.dest_dir)
    else:
        build_rpm_package(args, profile, context, args.dest_dir)


if __name__ == "__main__":
    main()
