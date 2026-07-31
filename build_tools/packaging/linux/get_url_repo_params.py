#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GitHub Actions helper: derive native Linux packaging URL parameters.

Used by CI workflows (see test_native_linux_packages_install.yml) to normalize
install-test inputs. Each subcommand prints KEY=value lines suitable for
$GITHUB_OUTPUT via gha_set_output().

Layout scope:
  - **per_family** (default): legacy GFX-specific tree from native_packaging.md
    (…/packages/{os}, …/rocm/packages/{os}, …/deb|rpm/{YYYYMMDD-id}/).
  - **multi_arch**: packages-multi-arch/… tree (install_rocm_packages.sh /
    s3_buckets.md). Signed stable uses …/rocm/packages-multi-arch/{os_profile}.

Subcommands:
  get-base-url         scheme + netloc from any URL → repo_base_url=
  get-gpg-url          GPG key URL from package repo URL → gpg_key_url=
                       (--release-type omits URL for dev/nightly/ci unsigned repos)
  get-repo-sub-folder  YYYYMMDD-<id> segment from S3 prefix → repo_sub_folder=
  get-repo-url         install repo URL from components → repo_url=
                       (--layout per_family default, or multi_arch)
  extract-gfx-arch     gfx94X-dcgpu → gfx94x (lists supported) → gfx_arch=
  get-container-image  os_profile → CI container image → container_image=

Workflow usage today:
  extract-gfx-arch and get-container-image are called from GHA.
  package_install_url and gpg_key_url are still passed as workflow inputs;
  get-repo-url / get-gpg-url / get-repo-sub-folder are available for wiring.

Examples:
  python build_tools/packaging/linux/get_url_repo_params.py get-base-url \\
      --from-url https://example.com/v2/whl
  python build_tools/packaging/linux/get_url_repo_params.py get-gpg-url \\
      --release-type prerelease \\
      --from-url https://rocm.prereleases.amd.com/packages/ubuntu2404
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-sub-folder \\
      --from-s3-prefix v3/packages/deb/20260204-12345
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url \\
      --release-type prerelease --native-package-type deb \\
      --repo-base-url https://rocm.prereleases.amd.com \\
      --os-profile ubuntu2404 --repo-sub-folder ''
  python build_tools/packaging/linux/get_url_repo_params.py get-repo-url \\
      --layout multi_arch --release-type stable --native-package-type deb \\
      --repo-base-url https://repo.amd.com --os-profile ubuntu2604 \\
      --repo-sub-folder ''
  python build_tools/packaging/linux/get_url_repo_params.py extract-gfx-arch \\
      --artifact-group gfx94X-dcgpu
  python build_tools/packaging/linux/get_url_repo_params.py get-container-image \\
      --os-profile ubuntu2404
"""

import argparse
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.fspath(Path(__file__).parent.parent.parent))
from github_actions.github_actions_api import gha_set_output

LAYOUT_PER_FAMILY = "per_family"
LAYOUT_MULTI_ARCH = "multi_arch"


def normalize_layout(layout: str | None) -> str:
    """Normalize a layout selector to :data:`LAYOUT_PER_FAMILY` or :data:`LAYOUT_MULTI_ARCH`.

    Args:
        layout: ``None``/empty → per-family; aliases ``legacy``, ``multiarch``.

    Raises:
        ValueError: If ``layout`` is not recognized.
    """
    if layout is None or not layout.strip():
        return LAYOUT_PER_FAMILY
    key = layout.strip().lower().replace("-", "_")
    aliases = {
        "legacy": LAYOUT_PER_FAMILY,
        "perfamily": LAYOUT_PER_FAMILY,
        "multiarch": LAYOUT_MULTI_ARCH,
    }
    normalized = aliases.get(key, key)
    if normalized not in (LAYOUT_PER_FAMILY, LAYOUT_MULTI_ARCH):
        raise ValueError(f"Unknown layout: {layout!r}")
    return normalized


def _normalize_release_type(release_type: str) -> str:
    """Normalize workflow release-type strings to canonical names.

    Maps ``prereleases`` → ``prerelease``, ``stable`` → ``release``,
    ``nightlies`` → ``nightly``; other values are lowercased as-is.
    """
    rt = release_type.strip().lower()
    aliases = {
        "prereleases": "prerelease",
        "stable": "release",
        "nightlies": "nightly",
    }
    return aliases.get(rt, rt)


# --- base_url ---


def get_base_url(url: str) -> str:
    """Return scheme and netloc only (strip path, query, and fragment).

    Args:
        url: Any HTTP(S) URL (e.g. package repo, S3 HTTPS, or CDN index URL).

    Returns:
        Base URL string ``{scheme}://{netloc}``.

    Raises:
        ValueError: If ``url`` has no scheme or netloc.
    """
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def cmd_base_url(args: argparse.Namespace) -> int:
    """CLI: ``get-base-url`` → writes ``repo_base_url=`` to ``$GITHUB_OUTPUT``."""
    try:
        base_url = get_base_url(args.from_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    gha_set_output({"repo_base_url": base_url})
    return 0


# --- gpg_key_url ---


def get_gpg_key_url(package_url: str) -> str:
    """Derive the AMD repo signing-key URL from a package repository URL.

    Keys sit beside the packages tree in the URL path:
    ``…/packages/gpg/``, ``…/rocm/packages/gpg/``, or
    ``…/packages-multi-arch/gpg/`` (stable: ``…/rocm/packages-multi-arch/gpg/``).

    Args:
        package_url: Full or partial native Linux package repo URL.

    Returns:
        HTTPS URL to ``rocm.gpg`` beside the matching packages tree.

    Raises:
        ValueError: If ``package_url`` is not a valid HTTP(S) URL.

    Examples:
        https://rocm.prereleases.amd.com/packages/ubuntu2404
            → https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg
        https://repo.amd.com/rocm/packages/rhel10/x86_64/
            → https://repo.amd.com/rocm/packages/gpg/rocm.gpg
        https://repo.amd.com/rocm/packages-multi-arch/ubuntu2604
            → https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
        https://rocm.nightlies.amd.com/packages-multi-arch/deb/20260204-12345/
            → https://rocm.nightlies.amd.com/packages-multi-arch/gpg/rocm.gpg
        https://repo.amd.com/
            → https://repo.amd.com/rocm/packages/gpg/rocm.gpg
    """
    parsed = urlparse(package_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {package_url!r}")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path or ""

    if "/rocm/packages-multi-arch/" in path or path.rstrip("/").endswith(
        "/rocm/packages-multi-arch"
    ):
        return f"{origin}/rocm/packages-multi-arch/gpg/rocm.gpg"
    if "/packages-multi-arch/" in path or path.rstrip("/").endswith(
        "/packages-multi-arch"
    ):
        return f"{origin}/packages-multi-arch/gpg/rocm.gpg"
    if "/rocm/packages/" in path or path.rstrip("/").endswith("/rocm/packages"):
        return f"{origin}/rocm/packages/gpg/rocm.gpg"
    if "/packages/" in path or path.rstrip("/").endswith("/packages"):
        return f"{origin}/packages/gpg/rocm.gpg"
    if parsed.netloc == "repo.amd.com":
        return f"{origin}/rocm/packages/gpg/rocm.gpg"
    return f"{origin}/packages/gpg/rocm.gpg"


def get_gpg_key_url_from_release_type(
    release_type: str,
    layout: str | None = None,
) -> str:
    """Return the canonical GPG key URL for a signed release line and layout.

    Args:
        release_type: ``prerelease`` / ``prereleases``, ``release`` / ``stable``.
        layout: ``multi_arch`` or per-family (default).

    Returns:
        HTTPS URL to ``rocm.gpg`` for the release host and layout.

    Raises:
        ValueError: If ``release_type`` is unsigned or unknown.

    Examples:
        prerelease + per_family
            → https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg
        stable + multi_arch
            → https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg
    """
    rt = _normalize_release_type(release_type)
    layout_norm = normalize_layout(layout)

    if layout_norm == LAYOUT_MULTI_ARCH:
        if rt == "prerelease":
            return "https://rocm.prereleases.amd.com/packages-multi-arch/gpg/rocm.gpg"
        if rt == "release":
            return "https://repo.amd.com/rocm/packages-multi-arch/gpg/rocm.gpg"
        raise ValueError(
            f"GPG key URL not defined for release_type={release_type!r} "
            f"with layout={layout!r}"
        )

    if rt == "prerelease":
        return "https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg"
    if rt == "release":
        return "https://repo.amd.com/rocm/packages/gpg/rocm.gpg"
    raise ValueError(f"Unknown or unsigned release_type: {release_type!r}")


def gpg_key_url_needed_for_release_type(release_type: str | None) -> bool:
    """Return whether a signed-repo GPG key URL applies for this release line.

    Args:
        release_type: Workflow release type, or ``None`` for legacy callers.

    Returns:
        ``True`` if a GPG URL should be emitted/derived; ``False`` for unsigned
        lines (``dev``, ``nightly``, ``ci``, empty string).

    Notes:
        ``None`` means always derive from the package URL (backward compatible).
        Signed lines: ``prerelease``, ``prereleases``, ``release``, ``stable``
        (case-insensitive, whitespace trimmed).
    """
    if release_type is None:
        return True
    rt = release_type.strip().lower()
    return rt in ("prerelease", "prereleases", "release", "stable")


def cmd_gpg_key_url(args: argparse.Namespace) -> int:
    """CLI: ``get-gpg-url`` → writes ``gpg_key_url=`` (or empty) to ``$GITHUB_OUTPUT``."""
    if not gpg_key_url_needed_for_release_type(args.release_type):
        gha_set_output({"gpg_key_url": ""})
        return 0
    try:
        gpg_url = get_gpg_key_url(args.from_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    gha_set_output({"gpg_key_url": gpg_url})
    return 0


# --- repo_sub_folder ---

DATE_ARTIFACT_PATTERN = re.compile(r"^\d{8}-\d+$")


def get_repo_sub_folder(s3_prefix: str) -> str:
    """Extract ``YYYYMMDD-<id>`` from an S3 key prefix when present.

    Used to build dev/nightly CDN URLs via :func:`get_repo_url` (per-family or
    multi-arch layout). Scans the **last** path segment only.

    Args:
        s3_prefix: S3 key prefix (with or without leading/trailing slashes).

    Returns:
        The last segment if it matches ``YYYYMMDD-<digits>``; otherwise ``""``.

    Examples:
        ``v3/packages/deb/20260204-12345`` → ``20260204-12345``
        ``12345678-linux/packages/deb/20260204-12345`` → ``20260204-12345``
        ``v3/packages/deb/`` → ``""``
    """
    segments = [p for p in s3_prefix.strip("/").split("/") if p]
    if not segments:
        return ""
    last = segments[-1]
    if DATE_ARTIFACT_PATTERN.fullmatch(last):
        return last
    return ""


def cmd_repo_sub_folder(args: argparse.Namespace) -> int:
    """CLI: ``get-repo-sub-folder`` → writes ``repo_sub_folder=`` to ``$GITHUB_OUTPUT``."""
    repo_sub_folder = get_repo_sub_folder(args.from_s3_prefix)
    gha_set_output({"repo_sub_folder": repo_sub_folder})
    return 0


# --- repo_url ---


def get_repo_url_per_family(
    release_type: str,
    native_package_type: str,
    repo_base_url: str,
    os_profile: str,
    repo_sub_folder: str,
) -> str:
    """Build a per-family native Linux package repo URL (``native_packaging.md``).

    Args:
        release_type: ``prerelease`` / ``prereleases``, ``release`` / ``stable``,
            or unsigned lines ``dev``, ``nightly``, ``ci`` (aliases normalized).
        native_package_type: ``deb`` or ``rpm``.
        repo_base_url: Scheme + host (e.g. ``https://rocm.prereleases.amd.com``).
        os_profile: OS profile slug (e.g. ``ubuntu2404``, ``rhel10``).
        repo_sub_folder: ``YYYYMMDD-<id>`` for dev/nightly; empty for signed lines.

    Returns:
        HTTPS URL pointing at the apt/dnf repo root (RPM URLs include ``/x86_64/``).
    """
    base = repo_base_url.rstrip("/")
    rt = _normalize_release_type(release_type)

    if rt == "prerelease":
        if native_package_type == "deb":
            return f"{base}/packages/{os_profile}"
        return f"{base}/packages/{os_profile}/x86_64/"

    if rt == "release":
        if native_package_type == "deb":
            return f"{base}/rocm/packages/{os_profile}"
        return f"{base}/rocm/packages/{os_profile}/x86_64/"

    if native_package_type == "deb":
        return f"{base}/deb/{repo_sub_folder}/"
    return f"{base}/rpm/{repo_sub_folder}/x86_64/"


def get_repo_url_multi_arch(
    release_type: str,
    native_package_type: str,
    repo_base_url: str,
    os_profile: str,
    repo_sub_folder: str,
) -> str:
    """Build a multi-arch native Linux package repo URL (``packages-multi-arch/``).

    Matches ``dockerfiles/install_rocm_packages.sh`` ``build_repo_url`` when
    ``multi_arch=1``.

    Args:
        release_type: ``prerelease``, ``release`` / ``stable``, or unsigned
            ``dev`` / ``nightly`` / ``ci`` (aliases normalized).
        native_package_type: ``deb`` or ``rpm``.
        repo_base_url: Scheme + host.
        os_profile: OS profile for signed lines (e.g. ``ubuntu2604``, ``rhel10``).
        repo_sub_folder: ``YYYYMMDD-<id>`` for unsigned lines; ignored for signed.

    Returns:
        HTTPS URL pointing at the apt/dnf repo root.

    Layout:
        - prerelease deb: ``{base}/packages-multi-arch/{os_profile}``
        - release deb: ``{base}/rocm/packages-multi-arch/{os_profile}``
        - nightly deb: ``{base}/packages-multi-arch/deb/{repo_sub_folder}``
        - nightly rpm: ``{base}/packages-multi-arch/rpm/{repo_sub_folder}/x86_64``
    """
    base = repo_base_url.rstrip("/")
    rt = _normalize_release_type(release_type)

    if rt == "prerelease":
        if native_package_type == "deb":
            return f"{base}/packages-multi-arch/{os_profile}"
        return f"{base}/packages-multi-arch/{os_profile}/x86_64/"

    if rt == "release":
        if native_package_type == "deb":
            return f"{base}/rocm/packages-multi-arch/{os_profile}"
        return f"{base}/rocm/packages-multi-arch/{os_profile}/x86_64/"

    if native_package_type == "deb":
        if repo_sub_folder:
            return f"{base}/packages-multi-arch/deb/{repo_sub_folder}"
        return f"{base}/packages-multi-arch/deb"
    if repo_sub_folder:
        return f"{base}/packages-multi-arch/rpm/{repo_sub_folder}/x86_64"
    return f"{base}/packages-multi-arch/rpm/x86_64"


def get_repo_url(
    release_type: str,
    native_package_type: str,
    repo_base_url: str,
    os_profile: str,
    repo_sub_folder: str,
    layout: str | None = None,
) -> str:
    """Build a native Linux package repo URL for the selected layout.

    Args:
        release_type: Workflow release type (aliases normalized).
        native_package_type: ``deb`` or ``rpm``.
        repo_base_url: Scheme + host.
        os_profile: OS profile slug (e.g. ``ubuntu2404``, ``rhel10``).
        repo_sub_folder: ``YYYYMMDD-<id>`` for unsigned lines; empty for signed.
        layout: ``per_family`` (default) or ``multi_arch`` (aliases accepted).

    Returns:
        HTTPS URL pointing at the apt/dnf repo root.

    See Also:
        :func:`get_repo_url_per_family`, :func:`get_repo_url_multi_arch`.
    """
    layout_norm = normalize_layout(layout)
    if layout_norm == LAYOUT_MULTI_ARCH:
        return get_repo_url_multi_arch(
            release_type=release_type,
            native_package_type=native_package_type,
            repo_base_url=repo_base_url,
            os_profile=os_profile,
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
    """CLI: ``get-repo-url`` → writes ``repo_url=`` to ``$GITHUB_OUTPUT``."""
    try:
        url = get_repo_url(
            release_type=args.release_type,
            native_package_type=args.native_package_type,
            repo_base_url=args.repo_base_url,
            os_profile=args.os_profile,
            repo_sub_folder=args.repo_sub_folder or "",
            layout=args.layout,
        )
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    gha_set_output({"repo_url": url})
    return 0


# --- extract-gfx-arch ---


def extract_gfx_arch(artifact_group: str) -> str:
    """Normalize GPU architecture token(s) from CI artifact group name(s).

    Takes the substring before the first ``-`` in each group, lowercases it,
    and joins multiple groups with commas.

    Args:
        artifact_group: One group or comma/semicolon-separated list
            (e.g. ``gfx94X-dcgpu`` or ``gfx94X-dcgpu;gfx1100-consumer``).

    Returns:
        Comma-separated gfx tokens (e.g. ``gfx94x,gfx1100``).

    Raises:
        ValueError: If ``artifact_group`` is empty or yields no tokens.

    Examples:
        ``gfx94X-dcgpu`` → ``gfx94x``
        ``GFX942-server`` → ``gfx942``
        ``gfx94X-dcgpu,gfx1100-consumer`` → ``gfx94x,gfx1100``
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
    """CLI: ``extract-gfx-arch`` → writes ``gfx_arch=`` to ``$GITHUB_OUTPUT``."""
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
    """Map an OS profile slug to the CI container image for install tests.

    Args:
        os_profile: Profile name (e.g. ``ubuntu2404``, ``rhel10``, ``sles16``).

    Returns:
        Container image reference used by ``test_native_linux_packages_install.yml``.

    Examples:
        ``ubuntu2404`` → ``ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest``
        ``debian12``   → ``ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest``
        ``sles16``     → ``registry.suse.com/bci/bci-base:16.0``
        ``rhel8``      → ``registry.access.redhat.com/ubi8/ubi:8.10``
        ``rhel10``     → ``registry.access.redhat.com/ubi10/ubi:10.1``
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
    """CLI: ``get-container-image`` → writes ``container_image=`` to ``$GITHUB_OUTPUT``."""
    image = get_container_image(args.os_profile)
    gha_set_output({"container_image": image})
    return 0


# --- main ---

_CLI_EPILOG = """
Testing (unit tests)
  Prerequisites:
    - Python 3.10 or newer
    - Run from the TheROCK repository root
    - No extra pip packages (tests mock $GITHUB_OUTPUT; stdlib only)

  From repo root:
    python3 -m unittest \\
      build_tools.packaging.linux.tests.get_url_repo_params_test -v

  Pass: per-family and multi_arch layout tests OK. Container-image tests expect
  get_container_image() output (align with #7004 if those three tests fail).

  Layout: get-repo-url accepts --layout per_family (default) or multi_arch.

  Optional: export GITHUB_OUTPUT=/tmp/out.txt to inspect CLI output locally.
"""


def main(argv: list[str] | None = None) -> int:
    """Parse CLI subcommands and dispatch to the matching ``cmd_*`` handler."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive native Linux packaging URL parameters for GitHub Actions "
            "(output is KEY=value for $GITHUB_OUTPUT)."
        ),
        epilog=_CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Get operation to run"
    )

    # get-base-url: get base URL from any input URL
    p_base = subparsers.add_parser(
        "get-base-url",
        help="Get base URL (scheme + netloc) from an input URL; path/query/fragment are stripped.",
    )
    p_base.add_argument(
        "--from-url",
        type=str,
        required=True,
        metavar="URL",
        help="Any URL to derive base URL from (scheme + netloc only; e.g. https://example.com/v2/whl → https://example.com)",
    )
    p_base.set_defaults(func=cmd_base_url)

    # get-gpg-url: get GPG key URL from package repository URL
    p_gpg = subparsers.add_parser(
        "get-gpg-url",
        help="Print gpg_key_url= for GITHUB_OUTPUT. With --release-type, only prerelease/release get a non-empty URL; otherwise gpg_key_url=. Omit --release-type to always derive from --from-url.",
    )
    p_gpg.add_argument(
        "--from-url",
        type=str,
        required=True,
        metavar="URL",
        help="Package repository URL to derive GPG key URL from when needed (per-family or packages-multi-arch paths; e.g. …/packages/ubuntu2404 → …/packages/gpg/rocm.gpg)",
    )
    p_gpg.add_argument(
        "--release-type",
        type=str,
        default=None,
        help="If set, emit non-empty GPG URL only for signed lines (prerelease/release/stable); for dev/nightly/ci print gpg_key_url=. If omitted, always derive from --from-url.",
    )
    p_gpg.set_defaults(func=cmd_gpg_key_url)

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
        help=(
            "Get full repo URL from release_type, native_package_type, "
            "repo_base_url, os_profile, repo_sub_folder; optional --layout "
            "(per_family or multi_arch)."
        ),
    )
    p_url.add_argument(
        "--release-type",
        type=str,
        required=True,
        help="e.g. prerelease, prereleases, release, stable, dev, nightly, ci",
    )
    p_url.add_argument(
        "--native-package-type",
        type=str,
        required=True,
        choices=["deb", "rpm"],
        help="Package type (deb or rpm)",
    )
    p_url.add_argument(
        "--repo-base-url",
        type=str,
        required=True,
        metavar="URL",
        help="Base URL (scheme + netloc, no trailing slash)",
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
        help="Repo subfolder (e.g. YYYYMMDD-<id> for dev/nightly; empty for prerelease)",
    )
    p_url.add_argument(
        "--layout",
        type=str,
        default=None,
        help="Repo layout: per_family (default) or multi_arch (packages-multi-arch/…)",
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
        help="Get container image for a given OS profile (e.g. ubuntu2404 → ghcr.io/rocm/no_rocm_image_ubuntu24_04:latest).",
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
