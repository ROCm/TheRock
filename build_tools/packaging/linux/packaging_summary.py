# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build-summary writers for native Linux packaging.

``write_build_manifest`` keeps the human ``built_packages.txt`` inventory
(filenames under Created Packages) and also writes ``built_packages.json``:
a structured claim of control-field metadata read from each Created
``.deb`` / ``.rpm`` on disk. The JSON claim is for a later pre-upload
verify step; presence-only verify continues to use the text manifest.
"""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from packaging_utils import *
from _therock_utils.log_utils import TheRockLogger

logger = TheRockLogger(__name__)

MANIFEST_TXT_NAME = "built_packages.txt"
MANIFEST_JSON_NAME = "built_packages.json"
MANIFEST_SCHEMA_VERSION = 1

# Control fields ROCm packaging writes / that pre-upload may later compare.
# Empty values are stored as "" so the key set stays stable for consumers.
DEB_CONTROL_FIELDS: tuple[str, ...] = (
    "Package",
    "Version",
    "Architecture",
    "Depends",
    "Pre-Depends",
    "Recommends",
    "Suggests",
    "Provides",
    "Replaces",
    "Conflicts",
    "Breaks",
    "Priority",
    "Section",
    "Maintainer",
    "Homepage",
)

# Scalar RPM tags queried via --queryformat (one line each, same order).
RPM_SCALAR_TAGS: tuple[tuple[str, str], ...] = (
    ("Name", "%{NAME}"),
    ("Version", "%{VERSION}"),
    ("Release", "%{RELEASE}"),
    ("Architecture", "%{ARCH}"),
    ("Group", "%{GROUP}"),
    ("License", "%{LICENSE}"),
    ("Vendor", "%{VENDOR}"),
)

# Multi-value RPM dependency / relation queries (one capability per line).
RPM_LIST_QUERIES: tuple[tuple[str, list[str]], ...] = (
    ("Requires", ["rpm", "-qpR"]),
    ("Provides", ["rpm", "-qp", "--provides"]),
    ("Conflicts", ["rpm", "-qp", "--conflicts"]),
    ("Obsoletes", ["rpm", "-qp", "--obsoletes"]),
    ("Recommends", ["rpm", "-qp", "--recommends"]),
    ("Suggests", ["rpm", "-qp", "--suggests"]),
)


@dataclass
class PackageList:
    # All base package names that were attempted
    total: list[str]
    # Packages that were successfully created (versioned + non-versioned)
    built: list[str]
    # Base packages that were skipped
    skipped: list[str]
    # Base packages that failed to produce any output
    failed: list[str] = field(default_factory=list)


@dataclass
class PackageClaim:
    """One Created package entry in ``built_packages.json``.

    Attributes:
        filename: Basename under the packages directory.
        control: Control/tag fields when the probe succeeded.
        error: Probe failure message when ``control`` is unavailable.
    """

    filename: str
    control: dict[str, str] | None = None
    error: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Serialize for ``json.dump`` (omit unset optional fields)."""
        out: dict[str, object] = {"filename": self.filename}
        if self.control is not None:
            out["control"] = self.control
        if self.error is not None:
            out["error"] = self.error
        return out


@dataclass
class ManifestDocument:
    """Structured ``built_packages.json`` claim document."""

    schema_version: int
    pkg_type: str
    rocm_version: str
    version_suffix: str
    gfx_arch: str
    build_date_utc: str
    packages: list[PackageClaim]
    failed: list[str]
    skipped: list[str]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pkg_type": self.pkg_type,
            "rocm_version": self.rocm_version,
            "version_suffix": self.version_suffix,
            "gfx_arch": self.gfx_arch,
            "build_date_utc": self.build_date_utc,
            "packages": [p.to_json_dict() for p in self.packages],
            "failed": self.failed,
            "skipped": self.skipped,
        }


def _empty_deb_control() -> dict[str, str]:
    return {name: "" for name in DEB_CONTROL_FIELDS}


def _empty_rpm_control() -> dict[str, str]:
    out = {name: "" for name, _ in RPM_SCALAR_TAGS}
    for name, _ in RPM_LIST_QUERIES:
        out[name] = ""
    return out


def read_deb_control_fields(package_path: Path) -> dict[str, str]:
    """Read DEB control fields from a ``.deb`` via ``dpkg-deb -f``.

    Returns a dict with every key in ``DEB_CONTROL_FIELDS`` (empty string if
    the field is absent). Raises ``FileNotFoundError`` if the path is missing,
    ``RuntimeError`` if ``dpkg-deb`` fails.
    """
    if not package_path.is_file():
        raise FileNotFoundError(f"package not found: {package_path}")
    cmd = ["dpkg-deb", "-f", str(package_path), *DEB_CONTROL_FIELDS]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"dpkg-deb -f failed for {package_path.name}: {err}")
    lines = (proc.stdout or "").splitlines()
    # dpkg-deb prints one line per requested field (blank line if unset).
    values = list(lines)
    if len(values) < len(DEB_CONTROL_FIELDS):
        values.extend([""] * (len(DEB_CONTROL_FIELDS) - len(values)))
    control = _empty_deb_control()
    for name, raw in zip(DEB_CONTROL_FIELDS, values):
        control[name] = raw.strip()
    return control


def _rpm_list_field(package_path: Path, argv_prefix: list[str]) -> str:
    """Run an ``rpm -qp…`` list query; join non-empty lines with ``, ``."""
    proc = subprocess.run(
        [*argv_prefix, str(package_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        # Weak-dep queries can fail on older rpm; treat as empty.
        return ""
    items = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    # Drop rpmlib() noise from Requires for a cleaner claim (keep others).
    if argv_prefix[-1] == "-qpR":
        items = [i for i in items if not i.startswith("rpmlib(")]
    return ", ".join(items)


def read_rpm_control_fields(package_path: Path) -> dict[str, str]:
    """Read RPM tags / relations from a ``.rpm`` via ``rpm -qp``.

    Returns a dict with scalar tags plus Requires/Provides/Conflicts/Obsoletes/
    Recommends/Suggests (comma-joined). Raises ``FileNotFoundError`` /
    ``RuntimeError`` on hard failures for scalar query.
    """
    if not package_path.is_file():
        raise FileNotFoundError(f"package not found: {package_path}")
    qf = "\\n".join(fmt for _, fmt in RPM_SCALAR_TAGS) + "\\n"
    proc = subprocess.run(
        ["rpm", "-qp", "--queryformat", qf, str(package_path)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"rpm -qp failed for {package_path.name}: {err}")
    lines = (proc.stdout or "").splitlines()
    control = _empty_rpm_control()
    for idx, (name, _) in enumerate(RPM_SCALAR_TAGS):
        control[name] = lines[idx].strip() if idx < len(lines) else ""
    for name, argv_prefix in RPM_LIST_QUERIES:
        control[name] = _rpm_list_field(package_path, argv_prefix)
    return control


def read_package_control_fields(package_path: Path) -> dict[str, str]:
    """Dispatch control-field read by package suffix (``.deb`` / ``.rpm``)."""
    suffix = package_path.suffix.lower()
    if suffix == ".deb":
        return read_deb_control_fields(package_path)
    if suffix == ".rpm":
        return read_rpm_control_fields(package_path)
    raise ValueError(f"unsupported package type for control read: {package_path.name}")


def collect_created_package_claims(
    packages_dir: Path, filenames: list[str]
) -> list[PackageClaim]:
    """Build claim records for Created package filenames under ``packages_dir``.

    Each record has ``filename`` and either ``control`` (field map) or ``error``.
    Per-package probe failures are recorded (not raised) so the text inventory
    and remaining claims still get written; presence verify catches missing files.
    """
    claims: list[PackageClaim] = []
    for name in sorted(filenames):
        path = Path(packages_dir) / name
        try:
            claims.append(
                PackageClaim(
                    filename=name,
                    control=read_package_control_fields(path),
                )
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning(f"control-field claim skipped for {name}: {exc}")
            claims.append(PackageClaim(filename=name, error=str(exc)))
    return claims


def build_manifest_document(
    config: PackageConfig,
    pkg_list: PackageList,
    *,
    build_date_utc: str | None = None,
) -> ManifestDocument:
    """Assemble the ``built_packages.json`` document (does not write)."""
    if build_date_utc is None:
        build_date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return ManifestDocument(
        schema_version=MANIFEST_SCHEMA_VERSION,
        pkg_type=str(config.pkg_type).lower(),
        rocm_version=config.rocm_version,
        version_suffix=config.version_suffix or "",
        gfx_arch=config.gfx_arch,
        build_date_utc=build_date_utc,
        packages=collect_created_package_claims(config.dest_dir, pkg_list.built),
        failed=sorted(pkg_list.failed),
        skipped=sorted(pkg_list.skipped),
    )


def write_build_manifest_json(
    config: PackageConfig, pkg_list: PackageList
) -> Path | None:
    """Write ``built_packages.json`` next to the text manifest. Returns path or None."""
    manifest_json = Path(config.dest_dir) / MANIFEST_JSON_NAME
    try:
        doc = build_manifest_document(config, pkg_list)
        with open(manifest_json, "w", encoding="utf-8") as f:
            json.dump(doc.to_json_dict(), f, indent=2, sort_keys=False)
            f.write("\n")
        print(f"✅ Built packages JSON claim written to: {manifest_json}")
        return manifest_json
    except (OSError, TypeError, ValueError) as e:
        # Match text-manifest soft-fail: do not abort the package build.
        print(f"⚠️  WARNING: Failed to write built packages JSON claim: {e}")
        logger.warning(f"Failed to write {manifest_json}: {e}")
        return None


def write_build_manifest(config: PackageConfig, pkg_list: PackageList) -> None:
    """Write ``built_packages.txt`` and enriched ``built_packages.json`` claim.

    The text file remains filename-only (presence inventory). The JSON file
    records control fields (Package/Name, Version, Depends/Requires, and other
    relation/metadata fields) read from each Created artifact for later
    pre-upload verification.

    Parameters:
    config: Configuration object containing package metadata
    pkg_list: List of all packages attempted/built/skipped

    Returns: None
    """
    logger.debug("write_build_manifest")

    # Write successful packages manifest
    manifest_file = Path(config.dest_dir) / MANIFEST_TXT_NAME

    total_basepkg = len(pkg_list.total) + len(pkg_list.skipped)
    built = len(pkg_list.built)
    failed = len(pkg_list.failed)

    try:
        with open(manifest_file, "w", encoding="utf-8") as f:
            f.write(f"# Built Packages Manifest\n")
            f.write(f"# Package Type: {config.pkg_type.upper()}\n")
            f.write(f"# ROCm Version: {config.rocm_version}\n")
            f.write(f"# Graphics Architecture: {config.gfx_arch}\n")
            f.write(
                f"# Build Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            )
            f.write(f"# Total base packages: {total_basepkg}\n")
            f.write(f"# Skipped base packages: {len(pkg_list.skipped)}\n")
            f.write(f"# Successfully built: {built}\n")
            f.write(f"# Failed to build: {failed}\n")
            f.write(f"\n")

            if pkg_list.built:
                f.write(f"# Created Packages:\n")
                for pkg in sorted(pkg_list.built):
                    f.write(f"{pkg}\n")

            if pkg_list.failed:
                f.write(f"\n# Failed Packages:\n")
                for pkg in sorted(pkg_list.failed):
                    f.write(f"{pkg}\n")

            if pkg_list.skipped:
                f.write(f"\n# Skipped Packages:\n")
                f.write(
                    f"# Note: Package names shown are base names from package.json\n"
                )
                for pkg in sorted(pkg_list.skipped):
                    f.write(f"{pkg}\n")

        print(f"✅ Built packages manifest written to: {manifest_file}")
    except OSError as e:
        print(f"⚠️  WARNING: Failed to write built packages manifest: {e}")

    write_build_manifest_json(config, pkg_list)


def print_build_status(config: PackageConfig, pkg_list: PackageList) -> None:
    """Print a summary of the build process.

    Parameters:
    config: Configuration object containing package metadata
    pkg_list: List of all packages attempted/built/skipped

    Returns: None
    """
    print("\n" + "=" * 80)
    print("BUILD SUMMARY")
    print("=" * 80)

    total_basepkg = len(pkg_list.total) + len(pkg_list.skipped)
    built = len(pkg_list.built)
    failed = len(pkg_list.failed)

    print(f"\nTotal base packages: {total_basepkg} ")
    print(f"⏭️ Skipped base packages: {len(pkg_list.skipped)}")
    print(f"✅ Successfully built: {built}")
    print(f"❌ Failed to build: {failed}")

    print(f"\nCreated packages")
    for pkg in sorted(pkg_list.built):
        print(f"   - {pkg}")

    if pkg_list.failed:
        print(f"\n❌ Failed packages")
        for pkg in sorted(pkg_list.failed):
            print(f"   - {pkg}")

    if pkg_list.skipped:
        print(f"\n⏭️   Skipped packages")
        print(f"   (Base package names from package.json)")
        for pkg in sorted(pkg_list.skipped):
            print(f"   - {pkg}")

    print("\n" + "=" * 80)
    print(f"Package type: {config.pkg_type.upper()}")
    print(f"ROCm version: {config.rocm_version}")
    print(f"Output directory: {config.dest_dir}")
    print("=" * 80 + "\n")


def print_build_summary(config: PackageConfig, pkg_list: PackageList) -> None:
    write_build_manifest(config, pkg_list)
    print_build_status(config, pkg_list)
