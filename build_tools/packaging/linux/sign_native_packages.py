#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sign native RPM packages in place using the remote GPG signing server.

Called from ``multi_arch_build_native_linux_packages.yml`` **after** Build
Packages and **before** ``build_package_repo.py``. This ordering matters:
``rpmsign`` changes each RPM's bytes (embeds the signature into the header),
and ``build_package_repo.py`` runs ``createrepo_c`` which embeds each RPM's
checksum/size into repodata — signing must happen first, or repodata would
describe the pre-signature (now-stale) file contents.

DEB packages are not signed individually here — only the repository metadata
(``Release``/``InRelease``, signed by ``sign_package_repo.py`` later) is
signed for DEB, matching ``apt``'s trust model. This script is a no-op for
``--pkg-type deb``.

Skips signing (exit 0, no error) rather than failing when:
  - ``--release-type`` resolves to no signing tier (release-type ``release``
    — that stays a manual, operator-driven process; see
    signing-server-design.md Section 4.1b), or
  - neither ``GPG_SIGNING_SERVER`` nor ``GPG_SIGNING_API_URL`` is configured
    (matches signing-server-requirements.md P1-CI-2's existing behavior).

Usage:
  python ./build_tools/packaging/linux/sign_native_packages.py \\
    --pkg-type rpm \\
    --package-dir /path/to/packages \\
    --release-type nightly
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _signing_client

_GPGSHIM_PATH = _THIS_DIR / "gpgshim"


def sign_rpm_packages(package_dir: Path, tier: str) -> int:
    """Sign every top-level .rpm file in ``package_dir`` in place via rpmsign.

    Args:
        package_dir: Directory containing built .rpm files (flat, before
            build_package_repo.py moves them into x86_64/).
        tier: GPG_TIER value (resolved from --release-type by the caller).

    Returns:
        Number of RPM files signed.

    Raises:
        subprocess.CalledProcessError: If rpmsign fails for any file — the
            workflow step should fail the build rather than publish an
            unsigned/partially-signed package set.
    """
    rpms = sorted(package_dir.glob("*.rpm"))
    if not rpms:
        print(f"[INFO] No .rpm files found in {package_dir} — nothing to sign")
        return 0

    env = os.environ.copy()
    env["GPG_TIER"] = tier

    print(f"Signing {len(rpms)} RPM package(s) with tier '{tier}'...")
    for rpm_file in rpms:
        print(f"  Signing: {rpm_file.name}")
        subprocess.run(
            [
                "rpmsign",
                "--define",
                f"%__gpg {_GPGSHIM_PATH}",
                # Forward-compat with RPM 6.x, which requires _gpg_name (or
                # _openpgp_sign_id) to be set even when the actual signing
                # command is redirected elsewhere — confirmed live against
                # RPM 6.0.1 (rpmsign fails with "you must set
                # %_openpgp_sign_id" without this). The value itself is
                # never used cryptographically: gpgshim's GPG_TIER-based
                # path (set above) always takes priority over any
                # --local-user value rpmsign might derive from this and
                # pass through. Current CI (Ubuntu 24.04) runs RPM 4.18.x,
                # where this define is inert but harmless — not separately
                # verified live, since 4.x predates the requirement entirely.
                "--define",
                "_gpg_name gpgshim-managed",
                "--addsign",
                str(rpm_file),
            ],
            check=True,
            env=env,
        )
    print(f"✅ Signed {len(rpms)} RPM package(s)")
    return len(rpms)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sign native RPM packages in place (no-op for deb).",
    )
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Directory containing built packages (modified in place for rpm).",
    )
    parser.add_argument(
        "--release-type",
        default="ci",
        help="Release channel (ci, dev, nightly, prerelease, release). "
        "Resolved to a signing tier via _signing_client.resolve_signing_tier().",
    )
    args = parser.parse_args(argv)

    package_dir = args.package_dir.resolve()
    if not package_dir.is_dir():
        parser.error(f"--package-dir is not a directory: {package_dir}")

    if args.pkg_type == "deb":
        print(
            "[INFO] DEB packages are not individually signed — skipping "
            "(only repository metadata is signed; see sign_package_repo.py)"
        )
        return 0

    tier = _signing_client.resolve_signing_tier(args.release_type)
    if tier is None:
        print(
            f"[INFO] release-type '{args.release_type}' is not auto-signed "
            "(release-tier signing stays manual — see signing-server-design.md "
            "Section 4.1b) — skipping"
        )
        return 0

    if not os.environ.get("GPG_SIGNING_API_URL") and not os.environ.get(
        "GPG_SIGNING_SERVER"
    ):
        # Matches P1-CI-2: if no signing endpoint is configured at all,
        # complete without signing rather than erroring.
        print(
            "[INFO] Neither GPG_SIGNING_API_URL nor GPG_SIGNING_SERVER is "
            "configured — skipping signing"
        )
        return 0

    sign_rpm_packages(package_dir, tier)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
