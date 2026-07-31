#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

r"""Build the ``amdrocm-repo`` package that configures a system package manager
(apt/dnf/zypper) to install AMD ROCm from the public ROCm repositories.

One ``amdrocm-repo`` package is generated per target OS profile and release
line. It ships a single repository definition and, for signed lines, the AMD
signing key, so that after installing the package a user can install ROCm with
their native package manager. The signing key is fetched at build time and
embedded in the package; it is never stored in the source tree.

```
python build_repo_package.py \
    --os-profile ubuntu2404 \
    --release-type prerelease \
    --repo-base-url https://rocm.prereleases.amd.com/packages-multi-arch \
    --rocm-version 7.14.0 \
    --dest-dir ./output
```
"""

from __future__ import annotations

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
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

SCRIPT_DIR = Path(__file__).resolve().parent

# The package name and the repository identifier used for the repo file and its
# section header. A single repository is configured per package.
PACKAGE_NAME = "amdrocm-repo"
REPO_ID = "amdrocm"
REPO_NAME = "AMD ROCm"

# The deb suite the repositories publish under (``dists/<suite>/``). Used to
# locate the index when verifying a repo URL. It is written literally in
# template/repo/deb/amdrocm.sources.j2; a test asserts the two agree.
DEB_SUITE = "stable"

MAINTAINER = "ROCm Dev Support <rocm-dev.support@amd.com>"

# Where each package installs the signing key, and which the repo files
# reference. deb consumes a dearmored (binary) keyring via Signed-By; rpm
# references the (armored) key via gpgkey=file://.
DEB_KEYRING_PATH = "/usr/share/keyrings/rocm.gpg"
RPM_GPG_KEY_PATH = "/etc/pki/rpm-gpg/RPM-GPG-KEY-rocm"

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
_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+~-]*\Z")
_SUB_FOLDER_RE = re.compile(r"^[0-9]{8}-[0-9A-Za-z._-]+\Z")

# Release lines whose public repositories are signed. The nightly line is
# unsigned, so it ships no key and disables signature checking.
SIGNED_RELEASE_TYPES = ("prerelease", "release")
RELEASE_TYPES = ("prerelease", "release", "nightly")

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
        "description": "RHEL/Rocky/OL 8",
    },
    "rhel10": {
        "family": "rhel",
        "pkg_type": "rpm",
        "rpm_repo_dir": "/etc/yum.repos.d",
        "description": "RHEL/Rocky/OL 10",
    },
    "sles16": {
        "family": "sles",
        "pkg_type": "rpm",
        "rpm_repo_dir": "/etc/zypp/repos.d",
        "description": "SLES 16",
    },
}


# --- URL / signing derivation ------------------------------------------------
#
# The public ROCm repositories do not share a single layout, so the repository
# URL is derived per release line:
#
#   prerelease, release   one repository per distro:
#                           <base>/<os_profile>/             (deb)
#                           <base>/<os_profile>/x86_64/      (rpm)
#   nightly               one repository per package type, under a dated
#                         sub-folder:
#                           <base>/deb/<YYYYMMDD-id>/        (deb)
#                           <base>/rpm/<YYYYMMDD-id>/x86_64/ (rpm)
#
# The signing key is at <base>/gpg/rocm.gpg on every signed line.


def is_signed(release_type: str) -> bool:
    """Whether this release line's public repository is signed."""
    return release_type in SIGNED_RELEASE_TYPES


def repo_baseurl(
    repo_base_url: str,
    pkg_type: str,
    os_profile: str,
    release_type: str,
    repo_sub_folder: str = "",
) -> str:
    """Return the repository baseurl baked into the repo file.

    The layout differs per release line (see the section comment above), so both
    the target distro and the release line are required. deb returns the repo
    root, beneath which apt resolves ``dists/stable/main/``; rpm returns the
    directory containing ``repodata/``. ``repo_sub_folder`` is the dated segment
    used by the nightly line.
    """
    base = repo_base_url.rstrip("/")
    # The layout is a property of the release line, not of whether it is signed;
    # the two happen to correlate today but are independent.
    if release_type == "nightly":
        sub = f"{repo_sub_folder}/" if repo_sub_folder else ""
        if pkg_type == "deb":
            return f"{base}/deb/{sub}"
        return f"{base}/rpm/{sub}x86_64/"
    if pkg_type == "deb":
        return f"{base}/{os_profile}/"
    return f"{base}/{os_profile}/x86_64/"


def gpg_key_url(repo_base_url: str) -> str:
    """Return the signing-key URL for a repo base url."""
    return f"{repo_base_url.rstrip('/')}/gpg/rocm.gpg"


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
            last_error = RuntimeError(f"unexpected HTTP status {status}")
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
    release_type: str, rocm_version: str, repo_sub_folder: str
) -> tuple[str, str]:
    """Return (Version, Release) for the rpm package.

    Prerelease/release track the (rolling) ROCm version. Nightly is date-pinned
    from the ``YYYYMMDD-<id>`` sub-folder; an rpm Version cannot contain ``-``,
    so the date becomes the Version and the id becomes the Release.

    The release line is part of the Release field so that two lines can never
    produce an identically named package. Without it, installing the release
    package over the prerelease one at the same ROCm version is a silent no-op:
    the package manager reports success and leaves the old repository
    configured.
    """
    if release_type == "nightly":
        date, _, ident = repo_sub_folder.partition("-")
        return date, f"{ident or '1'}.nightly"
    return rocm_version, f"1.{release_type}"


def deb_version(release_type: str, rocm_version: str, repo_sub_folder: str) -> str:
    """Return the deb package version (nightly is date-pinned, no ``-``).

    Prerelease carries a ``~prerelease`` suffix. ``~`` sorts before the plain
    version, so the release package upgrades over the prerelease one and the two
    lines can never share a version (see ``rpm_version_release``).
    """
    if release_type == "nightly":
        return repo_sub_folder.replace("-", ".")
    if release_type == "prerelease":
        return f"{rocm_version}~prerelease"
    return rocm_version


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

    Building any signed line needs gpg: the deb path to dearmor the keyring, and
    both paths to check the key fingerprint.
    """
    try:
        return subprocess.run(argv, check=True, **kwargs)
    except FileNotFoundError as e:
        raise RuntimeError(
            "gpg is required to build a signed release line "
            "(install gnupg/gnupg2, or pass --gpg-key-file for an unsigned build)"
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
    """Return the armored signing key bytes for a signed line.

    Read from ``--gpg-key-file`` when given (offline/test builds), otherwise
    fetched from the repo's ``gpg/rocm.gpg`` at build time and verified against
    the pinned fingerprint. Never written into the source tree.
    """
    if args.gpg_key_file:
        return args.gpg_key_file.read_bytes()
    key = _fetch_signing_key(gpg_key_url(args.repo_base_url))
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
    return Environment(
        loader=FileSystemLoader(str(SCRIPT_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _render(
    env: Environment, template_path: str, output_path: Path, context: dict
) -> None:
    output_path.write_text(env.get_template(template_path).render(context))


def build_context(args: argparse.Namespace, profile: dict) -> dict:
    """Template fields shared by the repo file, spec, and deb metadata."""
    signed = is_signed(args.release_type)
    return {
        "pkg_name": PACKAGE_NAME,
        "repo_id": REPO_ID,
        "name": REPO_NAME,
        "baseurl": repo_baseurl(
            args.repo_base_url,
            profile["pkg_type"],
            args.os_profile,
            args.release_type,
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
        "rpm_gpg_key_path": RPM_GPG_KEY_PATH,
        "rpm_repo_dir": profile.get("rpm_repo_dir", ""),
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
        sources_dir / f"{REPO_ID}.repo",
        context,
    )

    if context["signed"]:
        # rpm consumes the armored key as-is (gpgkey=file://).
        (sources_dir / "RPM-GPG-KEY-rocm").write_bytes(load_signing_key(args))

    version, release = rpm_version_release(
        args.release_type, args.rocm_version, args.repo_sub_folder
    )
    spec_context = {
        **context,
        "version": version,
        "release": release,
        "changelog_date": datetime.now(timezone.utc).strftime("%a %b %d %Y"),
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
            args.release_type, args.rocm_version, args.repo_sub_folder
        ),
        "date": format_datetime(datetime.now(timezone.utc)),
    }

    # Repository definition (deb822) and, for signed lines, the dearmored key,
    # placed at the package root and mapped into the filesystem by debian/install.
    _render(
        env,
        "template/repo/deb/amdrocm.sources.j2",
        package_dir / f"{REPO_ID}.sources",
        deb_context,
    )
    if context["signed"]:
        (package_dir / "rocm.gpg").write_bytes(dearmor_key(load_signing_key(args)))

    for name in ("control", "changelog", "rules", "postinst", "install"):
        _render(
            env,
            f"template/repo/deb/{name}.j2",
            deb_dir / name,
            deb_context,
        )
    (deb_dir / "rules").chmod(0o755)
    (deb_dir / "postinst").chmod(0o755)
    # Native source format: the package version carries no debian revision, so
    # no separate upstream tarball is required.
    (deb_dir / "source").mkdir(exist_ok=True)
    (deb_dir / "source" / "format").write_text("3.0 (native)\n")

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
        "--release-type",
        required=True,
        choices=RELEASE_TYPES,
        help="Release line to configure",
    )
    p.add_argument(
        "--repo-base-url",
        required=True,
        help="Public repository base URL (the packages-multi-arch base)",
    )
    p.add_argument(
        "--repo-sub-folder",
        default="",
        help="Dated sub-folder for the nightly line (YYYYMMDD-<id>)",
    )
    p.add_argument(
        "--gpg-key-file",
        type=Path,
        default=None,
        help="Signing key file to embed (defaults to fetching gpg/rocm.gpg from the repo)",
    )
    p.add_argument(
        "--rocm-version",
        required=True,
        help="ROCm version (e.g. 7.14.0)",
    )
    p.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Output directory for built packages",
    )
    p.add_argument(
        "--verify-repo-url",
        action="store_true",
        help=(
            "Fail the build unless the configured repository is reachable. "
            "No-op for the nightly line, whose dated sub-folder is published by "
            "the same run that builds the package"
        ),
    )
    args = p.parse_args(argv)
    if args.release_type == "nightly" and not args.repo_sub_folder:
        p.error("--repo-sub-folder is required for --release-type nightly")
    if args.repo_sub_folder and args.release_type != "nightly":
        p.error("--repo-sub-folder is only valid for --release-type nightly")
    # Validate values that are rendered into repo files, version fields, and
    # maintainer metadata so they cannot alter the surrounding syntax.
    if not _VERSION_RE.match(args.rocm_version):
        p.error(f"--rocm-version has an unexpected format: {args.rocm_version!r}")
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
    # The nightly repository is published by the same run that builds this
    # package, so its dated sub-folder does not exist yet and cannot be checked.
    if args.verify_repo_url and args.release_type != "nightly":
        verify_repo_url(context["baseurl"], profile["pkg_type"])
    if profile["pkg_type"] == "deb":
        build_deb_package(args, profile, context, args.dest_dir)
    else:
        build_rpm_package(args, profile, context, args.dest_dir)


if __name__ == "__main__":
    main()
