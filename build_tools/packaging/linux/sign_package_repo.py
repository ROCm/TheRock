#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Sign repository metadata (RPM repodata/repomd.xml, DEB Release) using the
remote GPG signing server.

Called from ``multi_arch_build_native_linux_packages.yml`` **after**
``build_package_repo.py`` (which must run first — metadata has to exist
before it can be signed) and **before** the Simulated install Test:

  build_package.py -> sign_native_packages.py -> build_package_repo.py
    -> this script -> native_linux_package_install_test.py -> upload_package_repo.py

Calls the signing server directly via ``_signing_client`` (not through
gpgshim) — metadata files are small, one-shot signing requests, unlike
gpgshim's high-volume ppid-cached RPM header signing.

Skips signing (exit 0, no error) under the same conditions as
sign_native_packages.py: release-type ``release`` (stays manual — see
signing-server-design.md Section 4.1b), or no signing endpoint configured.

Usage:
  python ./build_tools/packaging/linux/sign_package_repo.py \\
    --pkg-type rpm \\
    --package-dir /path/to/packages \\
    --release-type nightly
"""

import argparse
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import _signing_client


def sign_rpm_metadata(package_dir: Path, tier: str) -> None:
    """Sign x86_64/repodata/repomd.xml, writing repomd.xml.asc alongside it.

    Args:
        package_dir: Root of the package tree produced by build_package_repo.py.
        tier: GPG_TIER value (resolved from --release-type by the caller).
    """
    repomd = package_dir / "x86_64" / "repodata" / "repomd.xml"
    if not repomd.is_file():
        print(f"[INFO] No repomd.xml found at {repomd} — nothing to sign")
        return

    config = _signing_client.SigningClientConfig.from_env()
    data = repomd.read_bytes()

    print(f"Signing {repomd.relative_to(package_dir)} (detached, armored)...")
    signature = _signing_client.sign_data(config, data, tier=tier, artifact="repodata")
    sig_path = repomd.parent / (repomd.name + ".asc")
    sig_path.write_bytes(signature)
    print(f"✅ Wrote {sig_path.relative_to(package_dir)}")


def sign_deb_metadata(package_dir: Path, tier: str) -> None:
    """Sign dists/stable/Release, writing Release.gpg (detached) and
    InRelease (clearsigned, data+signature combined) alongside it.

    Args:
        package_dir: Root of the package tree produced by build_package_repo.py.
        tier: GPG_TIER value (resolved from --release-type by the caller).
    """
    release = package_dir / "dists" / "stable" / "Release"
    if not release.is_file():
        print(f"[INFO] No Release file found at {release} — nothing to sign")
        return

    config = _signing_client.SigningClientConfig.from_env()
    data = release.read_bytes()

    print(f"Signing {release.relative_to(package_dir)} (detached, armored)...")
    detached_sig = _signing_client.sign_data(
        config, data, tier=tier, artifact="deb-release"
    )
    release_gpg = release.parent / "Release.gpg"
    release_gpg.write_bytes(detached_sig)
    print(f"✅ Wrote {release_gpg.relative_to(package_dir)}")

    # Clearsign returns the complete data+signature document, not a
    # detached signature to append — write it as the entire InRelease file.
    print(f"Signing {release.relative_to(package_dir)} (clearsign -> InRelease)...")
    inrelease_content = _signing_client.sign_data(
        config, data, tier=tier, artifact="deb-inrelease"
    )
    inrelease_path = release.parent / "InRelease"
    inrelease_path.write_bytes(inrelease_content)
    print(f"✅ Wrote {inrelease_path.relative_to(package_dir)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sign repository metadata (repodata/Release).",
    )
    parser.add_argument("--pkg-type", required=True, choices=["deb", "rpm"])
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Directory containing repo metadata from build_package_repo.py.",
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
        print(
            "[INFO] Neither GPG_SIGNING_API_URL nor GPG_SIGNING_SERVER is "
            "configured — skipping signing"
        )
        return 0

    if args.pkg_type == "rpm":
        sign_rpm_metadata(package_dir, tier)
    else:
        sign_deb_metadata(package_dir, tier)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
