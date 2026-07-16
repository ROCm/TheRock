#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Get URL/repo parameters: base URL from any URL, repo_sub_folder from an S3 prefix, or full repo URL from components.

Output is always KEY=value (suitable for GITHUB_OUTPUT).

Subcommands (get operations):

  get-base-url         Get base URL (scheme + netloc) from --from-url or from --release-type. Prints repo_base_url=<value>.
  get-repo-sub-folder  Get repo_sub_folder from an S3 prefix (last segment if YYYYMMDD-<id>, else empty). Prints repo_sub_folder=<value>.
  get-repo-url         Native install repo URL and gpg_key_url. Required: --release-type, --os-profile. Optional: --layout per_family|multi_arch (default per_family), --repo-sub-folder, --repo-base-url, --native-package-type, --from-url. Prints repo_url=<value> and gpg_key_url=<value>.
  extract-gfx-arch     Extract and normalize GPU architecture from artifact group. Prints gfx_arch=<value>.
  get-container-image  Get container image for a given OS profile. Prints container_image=<value>.

Usage:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url (--from-url <url> | --release-type <type>)
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder --from-s3-prefix <prefix>
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url ...
  python build_tools/packaging/linux/get_url_repo_params.py extract-gfx-arch --artifact-group <group>
  python build_tools/packaging/linux/get_url_repo_params.py get-container-image --os-profile <profile>

Examples:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url --from-url https://example.com/v2/whl
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url --release-type prerelease
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder --from-s3-prefix v3/packages/deb/20260204-12345
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url --release-type prerelease --os-profile ubuntu2404
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url --layout multi_arch --release-type nightly --os-profile ubuntu2404 --repo-sub-folder 20260501-25200531110
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url --release-type prerelease --native-package-type deb --repo-base-url https://repo_url.com --os-profile ubuntu2404 --repo-sub-folder ''
  python build_tools/packaging/linux/get_url_repo_params.py extract-gfx-arch --artifact-group gfx94X-dcgpu
  python build_tools/packaging/linux/get_url_repo_params.py get-container-image --os-profile ubuntu2404
"""

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
from github_actions.github_actions_api import gha_set_output

# Repo URL layout selectors (get-repo-url --layout).
LAYOUT_PER_FAMILY = "per_family"
LAYOUT_MULTI_ARCH = "multi_arch"
_PACKAGES_MULTI_ARCH = "packages-multi-arch"
_VALID_LAYOUTS = (LAYOUT_PER_FAMILY, LAYOUT_MULTI_ARCH, "legacy", "gfx", "multiarch")


def normalize_layout(layout: str | None) -> str:
    """Normalize --layout values to LAYOUT_PER_FAMILY or LAYOUT_MULTI_ARCH."""
    if not (layout or "").strip():
        return LAYOUT_PER_FAMILY
    key = layout.strip().lower().replace("-", "_")
    if key in (LAYOUT_PER_FAMILY, "legacy", "gfx"):
        return LAYOUT_PER_FAMILY
    if key in (LAYOUT_MULTI_ARCH, "multiarch"):
        return LAYOUT_MULTI_ARCH
    supported = ", ".join(_VALID_LAYOUTS)
    raise ValueError(f"Unknown layout {layout!r}; use one of: {supported}")


# --- base_url ---

# Canonical scheme+netloc for native-package distribution channels.
# See docs/packaging/versioning.md (Distribution channel / Base URL).
_RELEASE_TYPE_TO_REPO_BASE_URL: dict[str, str] = {
    "prerelease": "https://rocm.prereleases.amd.com",
    "prereleases": "https://rocm.prereleases.amd.com",
    "release": "https://repo.amd.com",
    "stable": "https://repo.amd.com",
    "nightly": "https://rocm.nightlies.amd.com",
    "nightlies": "https://rocm.nightlies.amd.com",
    "dev": "https://rocm.devreleases.amd.com",
}


def get_base_url_from_release_type(release_type: str) -> str:
    """Return repo base URL (scheme + netloc) for a known release line.

    Does not require a sample URL; aligns with public ROCm distribution endpoints.
    """
    if not (release_type or "").strip():
        raise ValueError("release_type cannot be empty")
    rt = release_type.strip().lower()
    base = _RELEASE_TYPE_TO_REPO_BASE_URL.get(rt)
    if base is None:
        supported = ", ".join(sorted(set(_RELEASE_TYPE_TO_REPO_BASE_URL)))
        raise ValueError(
            f"Unknown release_type {release_type!r}; use one of: {supported}"
        )
    return base


def get_base_url(url: str) -> str:
    """Return base URL (scheme + netloc only). No path, query, or fragment."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def cmd_base_url(args: argparse.Namespace) -> int:
    try:
        if args.release_type is not None:
            base_url = get_base_url_from_release_type(args.release_type)
        else:
            base_url = get_base_url(args.from_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    gha_set_output({"repo_base_url": base_url})
    return 0


# --- gpg_key_url ---


def get_gpg_key_url(package_url: str) -> str:
    """
    Get GPG key URL from package repository URL.

    ROCm hosts publish the key next to the package tree (…/gpg/rocm.gpg), not at the
    origin root. Supports per-family (…/packages/…) and multi-arch
    (…/packages-multi-arch/…) paths. See dockerfiles/install_rocm_packages.sh.

    Examples:
        https://rocm.prereleases.amd.com/packages/ubuntu2404
            -> https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg
        https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260501-123
            -> https://rocm.nightlies.amd.com/packages-multi-arch/gpg/rocm.gpg
        https://repo.amd.com/rocm/packages/rhel10/x86_64/
            -> https://repo.amd.com/rocm/packages/gpg/rocm.gpg
    """
    parsed = urlparse(package_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {package_url!r}")
    segments = [p for p in parsed.path.split("/") if p]
    for segment_name in (_PACKAGES_MULTI_ARCH, "packages"):
        try:
            idx = segments.index(segment_name)
        except ValueError:
            continue
        prefix = "/" + "/".join(segments[: idx + 1])
        return f"{parsed.scheme}://{parsed.netloc}{prefix}/gpg/rocm.gpg"
    # No known package-tree segment (e.g. bare base URL): match AMD defaults.
    if parsed.netloc.lower() == "repo.amd.com":
        return f"{parsed.scheme}://{parsed.netloc}/rocm/packages/gpg/rocm.gpg"
    return f"{parsed.scheme}://{parsed.netloc}/packages/gpg/rocm.gpg"


# Minimal package-repo URLs (path through …/packages/) so get_gpg_key_url() resolves the
# correct …/packages/gpg/rocm.gpg per channel; keys match _RELEASE_TYPE_TO_REPO_BASE_URL.
_RELEASE_TYPE_TO_MINIMAL_PACKAGE_URL_FOR_GPG: dict[str, str] = {
    "prerelease": "https://rocm.prereleases.amd.com/packages/",
    "prereleases": "https://rocm.prereleases.amd.com/packages/",
    "release": "https://repo.amd.com/rocm/packages/",
    "stable": "https://repo.amd.com/rocm/packages/",
    "nightly": "https://rocm.nightlies.amd.com/packages/",
    "nightlies": "https://rocm.nightlies.amd.com/packages/",
    "dev": "https://rocm.devreleases.amd.com/packages/",
}

_RELEASE_TYPE_TO_MINIMAL_MULTI_ARCH_URL_FOR_GPG: dict[str, str] = {
    "prerelease": "https://rocm.prereleases.amd.com/packages-multi-arch/",
    "prereleases": "https://rocm.prereleases.amd.com/packages-multi-arch/",
    "release": "https://repo.amd.com/packages-multi-arch/",
    "stable": "https://repo.amd.com/packages-multi-arch/",
    "nightly": "https://rocm.nightlies.amd.com/packages-multi-arch/",
    "nightlies": "https://rocm.nightlies.amd.com/packages-multi-arch/",
    "dev": "https://rocm.devreleases.amd.com/packages-multi-arch/",
    "ci": "https://rocm.nightlies.amd.com/packages-multi-arch/",
}


def get_gpg_key_url_from_release_type(
    release_type: str, layout: str = LAYOUT_PER_FAMILY
) -> str:
    """Return GPG key URL using only release line (no user-supplied package URL)."""
    if not (release_type or "").strip():
        raise ValueError("release_type cannot be empty")
    rt = release_type.strip().lower()
    normalized_layout = normalize_layout(layout)
    url_map = (
        _RELEASE_TYPE_TO_MINIMAL_MULTI_ARCH_URL_FOR_GPG
        if normalized_layout == LAYOUT_MULTI_ARCH
        else _RELEASE_TYPE_TO_MINIMAL_PACKAGE_URL_FOR_GPG
    )
    minimal = url_map.get(rt)
    if minimal is None:
        supported = ", ".join(sorted(set(url_map)))
        raise ValueError(
            f"Unknown release_type {release_type!r}; use one of: {supported}"
        )
    return get_gpg_key_url(minimal)


def derive_gpg_key_url_for_repo_outputs(
    release_type: str,
    from_url: str | None,
    layout: str = LAYOUT_PER_FAMILY,
) -> str:
    """Return gpg_key_url string for get-repo-url (empty when unsigned release lines)."""
    if not gpg_key_url_needed_for_release_type(release_type):
        return ""
    if from_url is not None:
        return get_gpg_key_url(from_url)
    return get_gpg_key_url_from_release_type(release_type, layout=layout)


def gpg_key_url_needed_for_release_type(release_type: str | None) -> bool:
    """
    Whether install workflows should use a repo GPG key URL for this release line.

    When release_type is None, callers treat this as "legacy / unspecified" and always
    derive the GPG URL from the package URL.

    When release_type is set (e.g. from GitHub Actions), only prerelease/prereleases,
    release, and stable lines use signed-repo GPG keys; dev/nightly/ci/etc. omit it
    (empty gpg_key_url).
    """
    if release_type is None:
        return True
    rt = release_type.strip().lower()
    return rt in ("prerelease", "prereleases", "release", "stable")


# --- repo_sub_folder ---

DATE_ARTIFACT_PATTERN = re.compile(r"^\d{8}-\d+$")


def get_repo_sub_folder(s3_prefix: str) -> str:
    """Return YYYYMMDD-<id> release folder from an S3 prefix, if present.

    Scans all path segments (newest last) so multi-arch prefixes such as
    ``{run_id}-linux/packages/deb/20260204-12345`` are handled.
    """
    segments = [p for p in s3_prefix.strip("/").split("/") if p]
    for seg in reversed(segments):
        if DATE_ARTIFACT_PATTERN.fullmatch(seg):
            return seg
    return ""


def cmd_repo_sub_folder(args: argparse.Namespace) -> int:
    repo_sub_folder = get_repo_sub_folder(args.from_s3_prefix)
    gha_set_output({"repo_sub_folder": repo_sub_folder})
    return 0


# --- repo_url ---


def _normalized_release_type_for_repo_url(release_type: str) -> str:
    """Map aliases so path rules match (e.g. prereleases → prerelease layout)."""
    rt = release_type.strip().lower()
    if rt == "prereleases":
        return "prerelease"
    return rt


def get_native_package_type_from_os_profile(os_profile: str) -> str:
    """Return deb or rpm from OS profile prefix (ubuntu/debian → deb; else rpm)."""
    if not (os_profile or "").strip():
        raise ValueError("os_profile cannot be empty")
    op = os_profile.strip().lower()
    if op.startswith("ubuntu") or op.startswith("debian"):
        return "deb"
    return "rpm"


def get_repo_url_per_family(
    release_type: str,
    native_package_type: str,
    repo_base_url: str,
    os_profile: str,
    repo_sub_folder: str,
) -> str:
    """
    Per-family native-package install repo URL (legacy layout).

    Matches docs/packaging/native_packaging.md (DEB/RPM install URL columns):
    - prerelease: .../packages/{os_profile} (deb) or .../packages/{os_profile}/x86_64/ (rpm)
    - release, stable: .../rocm/packages/{os_profile} (deb) or .../rocm/packages/{os_profile}/x86_64/ (rpm)
    - dev, nightly, ci, etc.: .../deb/{YYYYMMDD-id}/ or .../rpm/{id}/x86_64/
    """
    base = repo_base_url.rstrip("/")
    rt = _normalized_release_type_for_repo_url(release_type)
    if rt == "prerelease":
        if native_package_type == "deb":
            return f"{base}/packages/{os_profile}"
        return f"{base}/packages/{os_profile}/x86_64/"
    if rt in ("release", "stable"):
        if native_package_type == "deb":
            return f"{base}/rocm/packages/{os_profile}"
        return f"{base}/rocm/packages/{os_profile}/x86_64/"
    sub = (repo_sub_folder or "").strip().strip("/")
    if native_package_type == "deb":
        if sub:
            return f"{base}/deb/{sub}/"
        return f"{base}/deb/"
    if sub:
        return f"{base}/rpm/{sub}/x86_64/"
    return f"{base}/rpm/x86_64/"


def get_repo_url_multi_arch(
    repo_base_url: str,
    native_package_type: str,
    repo_sub_folder: str,
) -> str:
    """
    Multi-arch native-package install repo URL.

    Matches RELEASES.md — Installing multi-arch native Linux packages:
    - deb: {base}/packages-multi-arch/deb/{RELEASE_ID}
    - rpm: {base}/packages-multi-arch/rpm/{RELEASE_ID}/x86_64
    """
    base = repo_base_url.rstrip("/")
    sub = (repo_sub_folder or "").strip().strip("/")
    if native_package_type == "deb":
        if sub:
            return f"{base}/{_PACKAGES_MULTI_ARCH}/deb/{sub}"
        return f"{base}/{_PACKAGES_MULTI_ARCH}/deb"
    if sub:
        return f"{base}/{_PACKAGES_MULTI_ARCH}/rpm/{sub}/x86_64"
    return f"{base}/{_PACKAGES_MULTI_ARCH}/rpm/x86_64"


def get_repo_url(
    release_type: str,
    native_package_type: str,
    repo_base_url: str,
    os_profile: str,
    repo_sub_folder: str,
    layout: str = LAYOUT_PER_FAMILY,
) -> str:
    """Return the full native-package install repo URL for the requested layout."""
    normalized_layout = normalize_layout(layout)
    if normalized_layout == LAYOUT_MULTI_ARCH:
        return get_repo_url_multi_arch(
            repo_base_url=repo_base_url,
            native_package_type=native_package_type,
            repo_sub_folder=repo_sub_folder,
        )
    return get_repo_url_per_family(
        release_type=release_type,
        native_package_type=native_package_type,
        repo_base_url=repo_base_url,
        os_profile=os_profile,
        repo_sub_folder=repo_sub_folder,
    )


def cmd_repo_url(args: argparse.Namespace) -> int:
    try:
        native = args.native_package_type
        if native is None:
            native = get_native_package_type_from_os_profile(args.os_profile)
        repo_base = args.repo_base_url
        if repo_base is None:
            repo_base = get_base_url_from_release_type(args.release_type)
        layout = normalize_layout(getattr(args, "layout", None))
        url = get_repo_url(
            release_type=args.release_type,
            native_package_type=native,
            repo_base_url=repo_base,
            os_profile=args.os_profile,
            repo_sub_folder=args.repo_sub_folder or "",
            layout=layout,
        )
        gpg_url = derive_gpg_key_url_for_repo_outputs(
            args.release_type,
            getattr(args, "from_url", None),
            layout=layout,
        )
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    gha_set_output({"repo_url": url, "gpg_key_url": gpg_url})
    return 0


# --- extract-gfx-arch ---


def extract_gfx_arch(artifact_group: str) -> str:
    """
    Extract and normalize GPU architecture from artifact group(s).

    Supports both single and comma/semicolon-separated artifact groups.
    Output is always comma-separated.

    Examples:
        gfx94X-dcgpu -> gfx94x
        gfx1100-consumer -> gfx1100
        GFX942-server -> gfx942
        gfx94X-dcgpu,gfx1100-consumer -> gfx94x,gfx1100
        gfx94X-dcgpu;gfx1100-consumer -> gfx94x,gfx1100
    """
    if not artifact_group:
        raise ValueError("artifact_group cannot be empty")

    # Split on comma or semicolon to handle multiple groups
    # Replace semicolons with commas for consistent splitting
    normalized = artifact_group.replace(";", ",")
    groups = [g.strip() for g in normalized.split(",")]

    # Extract first segment (before dash) and lowercase each
    archs = [g.split("-")[0].lower() for g in groups if g]

    if not archs:
        raise ValueError("artifact_group cannot be empty after parsing")

    return ",".join(archs)


def cmd_extract_gfx_arch(args: argparse.Namespace) -> int:
    try:
        gfx_arch = extract_gfx_arch(args.artifact_group)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    gha_set_output({"gfx_arch": gfx_arch})
    return 0


# --- get-container-image ---

# Maps OS profile prefixes to container images (checked in order; first match wins).
# Single-profile entries (e.g. "rhel8") require an exact match so "rhel10" does not
# match the "rhel8" prefix via startswith.
_OS_PROFILE_TO_IMAGE: list[tuple[tuple[str, ...], str]] = [
    (("sles",), "registry.suse.com/bci/bci-base:16.0"),
    (("ubuntu", "debian"), "ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest"),
    (("rhel8",), "registry.access.redhat.com/ubi8/ubi:8.10"),
    ((), "registry.access.redhat.com/ubi10/ubi:10.1"),  # default (e.g. rhel10)
]

# Single-prefix entries that must match the full profile (not startswith).
_EXACT_PROFILE_PREFIXES = frozenset({"rhel8"})


def get_container_image(os_profile: str) -> str:
    """Return the container image for a given OS profile.

    Examples:
        ubuntu2404  -> ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest
        debian12    -> ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest
        sles16      -> registry.suse.com/bci/bci-base:16.0
        rhel8       -> registry.access.redhat.com/ubi8/ubi:8.10
        rhel10      -> registry.access.redhat.com/ubi10/ubi:10.1
    """
    profile = os_profile.lower()
    for prefixes, image in _OS_PROFILE_TO_IMAGE:
        if not prefixes:
            return image
        if len(prefixes) == 1 and prefixes[0] in _EXACT_PROFILE_PREFIXES:
            if profile == prefixes[0]:
                return image
            continue
        if any(profile.startswith(p) for p in prefixes):
            return image
    return _OS_PROFILE_TO_IMAGE[-1][1]  # unreachable but satisfies type checker


def cmd_container_image(args: argparse.Namespace) -> int:
    image = get_container_image(args.os_profile)
    gha_set_output({"container_image": image})
    return 0


# --- main ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Get URL/repo parameters: base URL (from any URL) or repo_sub_folder (from S3 prefix). Output is KEY=value for GITHUB_OUTPUT.",
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Get operation to run"
    )

    # get-base-url: get base URL from any input URL
    p_base = subparsers.add_parser(
        "get-base-url",
        help="Get base URL (scheme + netloc): from --from-url, or from --release-type using canonical ROCm hosts.",
    )
    g_base = p_base.add_mutually_exclusive_group(required=True)
    g_base.add_argument(
        "--from-url",
        type=str,
        default=None,
        metavar="URL",
        help="Any URL to derive base URL from (scheme + netloc only; path/query/fragment stripped).",
    )
    g_base.add_argument(
        "--release-type",
        type=str,
        default=None,
        metavar="TYPE",
        help="Release line only (no URL): prerelease, release, nightly, dev, stable; also prereleases, nightlies.",
    )
    p_base.set_defaults(func=cmd_base_url)

    # get-repo-sub-folder: get repo_sub_folder from S3 prefix
    p_repo = subparsers.add_parser(
        "get-repo-sub-folder",
        help="Get repo_sub_folder from an S3 prefix (last path segment if YYYYMMDD-<id>, else empty).",
    )
    p_repo.add_argument(
        "--from-s3-prefix",
        type=str,
        required=True,
        metavar="PREFIX",
        help="S3 key prefix to derive repo_sub_folder from (e.g. v3/packages/deb/20260204-12345 → 20260204-12345)",
    )
    p_repo.set_defaults(func=cmd_repo_sub_folder)

    # get-repo-url: full repo URL from components (replaces inline logic in workflows)
    p_url = subparsers.add_parser(
        "get-repo-url",
        help="Native repo URL and gpg_key_url. Requires --release-type and --os-profile; optional --layout, --repo-sub-folder, --repo-base-url, --native-package-type, --from-url.",
    )
    p_url.add_argument(
        "--layout",
        type=str,
        default=LAYOUT_PER_FAMILY,
        choices=list(_VALID_LAYOUTS),
        help="Repo URL layout: per_family (default, native_packaging.md) or multi_arch (RELEASES.md packages-multi-arch/…).",
    )
    p_url.add_argument(
        "--release-type", type=str, required=True, help="e.g. prerelease, dev, nightly"
    )
    p_url.add_argument(
        "--from-url",
        type=str,
        default=None,
        metavar="URL",
        help="Optional package repo URL to derive gpg_key_url when signed-repo applies; overrides canonical host from --release-type",
    )
    p_url.add_argument(
        "--native-package-type",
        type=str,
        default=None,
        choices=["deb", "rpm"],
        help="deb or rpm; if omitted, inferred from --os-profile (ubuntu/debian → deb, else rpm)",
    )
    p_url.add_argument(
        "--repo-base-url",
        type=str,
        default=None,
        metavar="URL",
        help="Base URL (scheme + netloc); if omitted, use canonical host for --release-type",
    )
    p_url.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g. ubuntu2404, rhel9)",
    )
    p_url.add_argument(
        "--repo-sub-folder",
        type=str,
        default="",
        help="YYYYMMDD-<run-id> (RELEASE_ID): per_family dev/nightly path, or multi_arch packages-multi-arch/{deb|rpm}/… segment.",
    )
    p_url.set_defaults(func=cmd_repo_url)

    # extract-gfx-arch: extract GPU architecture from artifact group
    p_gfx = subparsers.add_parser(
        "extract-gfx-arch",
        help="Extract and normalize GPU architecture from artifact group (e.g. gfx94X-dcgpu → gfx94x).",
    )
    p_gfx.add_argument(
        "--artifact-group",
        type=str,
        required=True,
        metavar="GROUP",
        help="Artifact group to extract gfx_arch from (e.g. gfx94X-dcgpu, gfx1100-consumer)",
    )
    p_gfx.set_defaults(func=cmd_extract_gfx_arch)

    # get-container-image: get container image for an OS profile
    p_img = subparsers.add_parser(
        "get-container-image",
        help="Get container image for a given OS profile (e.g. ubuntu2404 -> ubuntu:24.04).",
    )
    p_img.add_argument(
        "--os-profile",
        type=str,
        required=True,
        help="OS profile (e.g. ubuntu2404, sles16, rhel10)",
    )
    p_img.set_defaults(func=cmd_container_image)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
