#!/usr/bin/env python3
"""Generate WiX v4 .wxs files for ROCm Windows MSI installers.

Uses artifact archives (or a local build's artifact directories) as the source
of truth for which files each MSI package should include.  The artifact
manifest embedded in each archive already encodes the correct file set, so no
separate TOML descriptor or glob patterns are needed at generation time.

Usage
-----
List available packages:
    python generate_msi_wxs.py --list

Generate from CI artifacts (recommended):
    python generate_msi_wxs.py --package runtime \\
        --artifacts-url https://therock-nightly-artifacts.s3.amazonaws.com/<run-id>-windows

Generate from a local build:
    python generate_msi_wxs.py --package runtime --build build/

The generated .wxs is compiled into an MSI by running:
    wix build <name>.wxs -o <name>.msi

The resulting MSI requires no user interaction (no UI sequences are defined),
making it suitable for scripted or enterprise deployments:
    msiexec /i <name>.msi /qn

Installation location
---------------------
The default install path is:
    C:\\Program Files\\AMD\\ROCm\\{package-subdir}-{version}\\

Override at build time via --install-root, --product-dir, --version-dir.
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import pyzstd
import yaml

# Add build_tools/ to sys.path so _therock_utils can be imported.
BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent.parent
if str(BUILD_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_TOOLS_DIR))

from _therock_utils.artifacts import ArtifactCatalog, ArtifactName, ArtifactPopulator


# ---------------------------------------------------------------------------
# Package definitions
# ---------------------------------------------------------------------------


@dataclass
class PackageDef:
    """Definition of one distributable MSI package."""

    # Display name shown in Add/Remove Programs.
    product_name: str

    # Artifact names to include (matched against ArtifactName.name, e.g. "core-hip").
    artifacts: list[str]

    # Default output filename stem (without extension).
    output_stem: str

    # Versioned install subdirectory name. Formatted with the placeholders
    # {version} (X.Y.Z), {major}, and {minor}.
    install_subdir: str

    # Fixed GUID for MajorUpgrade matching across versions.  Must never change.
    upgrade_code: str

    # WiX Feature Id and Title for the primary feature.
    feature_id: str
    feature_title: str

    # Registry key written under HKLM to record the install location.
    registry_key: str

    # One-line description for --list output.
    description: str

    # Per-artifact file include overrides. Keys are artifact names; values are
    # glob patterns passed to ArtifactCatalog as includes, restricting which
    # files are collected from that artifact. Artifacts not listed are
    # collected without restriction.
    per_artifact_includes: dict[str, list[str]] = None

    # DLL filenames to install into the Windows System32 directory as a
    # default-enabled "legacy install" feature (for applications that load
    # ROCm DLLs from System32 rather than PATH). Each name is resolved against
    # the extracted artifacts and the _legacy/bin cache fallback. Empty means
    # the package emits no legacy feature.
    legacy_system32_dlls: list[str] = None

    def __post_init__(self):
        if self.per_artifact_includes is None:
            self.per_artifact_includes = {}
        if self.legacy_system32_dlls is None:
            self.legacy_system32_dlls = []


# Keys are the --package selector values.
PACKAGES: dict[str, PackageDef] = {
    "runtime": PackageDef(
        description="ROCm runtime redistributable (HIP runtime + amd_comgr.dll)",
        product_name="AMD ROCm Runtime",
        artifacts=[
            "core-hip",  # HIP runtime DLLs (amdhip64_7, hiprtc*) — lib component only
            "core-kpack",  # Kernel package support (rocm_kpack.dll)
            "amd-llvm",  # comgr only — see per_artifact_includes
        ],
        output_stem="amdrocm-runtime",
        install_subdir="core-{major}.{minor}",
        upgrade_code="C3D4E5F6-A7B8-9012-CDEF-123456789012",
        feature_id="ROCmRuntime",
        feature_title="AMD ROCm Runtime",
        registry_key="Software\\AMD\\ROCm\\runtime\\{major}.{minor}",
        per_artifact_includes={
            "amd-llvm": ["bin/amd_comgr.dll"],
        },
        legacy_system32_dlls=[
            "amdhip64_6.dll",  # ROCm 6.x HIP compat runtime (from AMD driver)
            "amdhip64_7.dll",  # current HIP runtime
            "amd_comgr.dll",  # code object manager
            "amd_comgr_2.dll",  # comgr v2 compat (from AMD driver)
            "rocm_kpack.dll",  # kernel package support
        ],
    ),
    "core": PackageDef(
        description="ROCm core runtime redistributable (ROCR, HIP, AMDsmi, OpenCL)",
        product_name="AMD ROCm Core Runtime",
        artifacts=[
            "core-runtime",  # ROCR-Runtime + rocminfo
            "core-hip",  # HIP runtime DLLs (amdhip64_7, hiprtc*) — lib component only
            "core-kpack",  # Kernel package support (rocm_kpack.dll)
            "core-hipinfo",  # Windows-only: bin/hipInfo*
            "core-amdsmi",  # AMD System Management Interface
            "core-ocl-icd",  # OpenCL ICD loader (bin/OpenCL.dll on Windows)
        ],
        output_stem="amdrocm-core",
        install_subdir="core-{major}.{minor}",
        upgrade_code="A1B2C3D4-E5F6-7890-ABCD-EF1234567890",
        feature_id="ROCmCore",
        feature_title="AMD ROCm Core Runtime",
        registry_key="Software\\AMD\\ROCm\\core\\{major}.{minor}",
    ),
}


# Only collect the lib component. On Windows, DLLs live in bin/ which the
# artifact TOML classifies as "run", but the run component also contains dev
# tools (hipcc_cmake_linker_helper, hipdemangleatp, hrr-playback.exe) that
# don't belong in a runtime redistributable. The lib component contains the
# actual DLLs (amdhip64_7.dll, hiprtc*.dll) plus lib/ and share/ metadata.
PACKAGE_COMPONENTS: set[str] = {"lib"}

STANDARD_DIR_TOKENS: set[str] = {
    "ProgramFilesFolder",
    "ProgramFiles64Folder",
    "SystemFolder",
    "System64Folder",
    "WindowsFolder",
    "TempFolder",
    "DesktopFolder",
    "AppDataFolder",
    "LocalAppDataFolder",
    "CommonAppDataFolder",
}


# ---------------------------------------------------------------------------
# Artifact download, extraction, and file collection
# ---------------------------------------------------------------------------


def fetch_artifacts(
    artifacts_url: str,
    artifact_names: list[str],
    components: set[str],
    dest_dir: Path,
) -> Path:
    """Download and extract artifact tarballs from a remote URL into dest_dir.

    For each (artifact, component) pair, downloads:
        {artifacts_url}/{artifact}_{component}_generic.tar.zst
    and extracts it into dest_dir/_extracted/{artifact}_{component}_generic/,
    preserving the internal layout (basedir paths from artifact_manifest.txt).
    The artifact_manifest.txt is also written to disk so ArtifactCatalog can
    read it.

    Returns the _extracted/ directory.
    """

    def _open_zst(path: Path):
        return tarfile.TarFile(fileobj=pyzstd.ZstdFile(path, "rb"), mode="r")

    artifacts_url = artifacts_url.rstrip("/")
    download_dir = dest_dir / "_downloads"
    extract_dir = dest_dir / "_extracted"
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    for artifact_name in artifact_names:
        for component in sorted(components):
            filename = f"{artifact_name}_{component}_generic.tar.zst"
            url = f"{artifacts_url}/{filename}"
            local_path = download_dir / filename

            if local_path.exists():
                print(f"  Cached:    {filename}")
            else:
                print(f"  Fetching:  {filename}")
                tmp_path = local_path.with_suffix(".tmp")
                try:
                    urllib.request.urlretrieve(url, tmp_path)
                except urllib.error.HTTPError as e:
                    tmp_path.unlink(missing_ok=True)
                    # 404 is expected: not every artifact publishes every
                    # component (e.g. Windows-only components have no Linux
                    # tarball). Any other HTTP error is a real failure.
                    if e.code == 404:
                        print(f"  Skipped:   {filename} (not found)")
                        continue
                    sys.exit(f"Error fetching {url}: HTTP {e.code} {e.reason}")
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
                tmp_path.rename(local_path)

            artifact_out = extract_dir / f"{artifact_name}_{component}_generic"
            manifest_path = artifact_out / "artifact_manifest.txt"
            already_extracted = artifact_out.exists()
            if already_extracted and manifest_path.exists():
                print(f"  Extracted: {filename} (cached)")
                continue
            artifact_out.mkdir(parents=True, exist_ok=True)
            # Read manifest from first tar member and write it to disk so
            # ArtifactCatalog can find it, then extract the rest of the files.
            with _open_zst(local_path) as tf:
                manifest_member = tf.next()
                if (
                    manifest_member is None
                    or manifest_member.name != "artifact_manifest.txt"
                ):
                    sys.exit(
                        f"Archive {filename} missing artifact_manifest.txt as first member"
                    )
                manifest_text = tf.extractfile(manifest_member).read().decode()
                manifest_path.write_text(manifest_text)
            if already_extracted:
                print(f"  Manifest:  {filename} (files already present)")
            else:
                print(f"  Extracting {filename}...")
                populator = ArtifactPopulator(output_path=artifact_out, flatten=False)
                populator(local_path)

    return extract_dir


def collect_files_from_catalog(
    artifact_dir: Path,
    pkg_def: "PackageDef",
) -> list[tuple[Path, Path]]:
    """Return a sorted, deduplicated list of (install_rel, source) pairs.

    Reads artifact_manifest.txt files from extracted artifact subdirectories
    in artifact_dir, filters to the package's artifact names and runtime
    components (run, lib), and enumerates files directly.

    install_rel: flat path relative to the install root (e.g. bin/amdhip64.dll)
    source:      absolute path to the file on disk (for WiX Source=)
    """
    if not artifact_dir.is_dir():
        print(
            f"Warning: artifact directory not found: {artifact_dir}\n"
            "Run a Windows build or provide --artifacts-url.",
            file=sys.stderr,
        )
        return []

    artifact_set = set(pkg_def.artifacts)
    overrides = pkg_def.per_artifact_includes  # {artifact_name: [glob, ...]}

    # Artifacts with per-artifact include overrides get their own catalog so
    # the include patterns are scoped to just those artifacts.
    # All other artifacts share a single unrestricted catalog.
    restricted = {name for name in artifact_set if name in overrides}
    unrestricted = artifact_set - restricted

    seen: dict[str, Path] = {}

    def _collect_catalog(catalog: "ArtifactCatalog") -> None:
        for relpath, direntry in catalog.pm.matches():
            if direntry.is_dir():
                continue
            if relpath not in seen:
                seen[relpath] = Path(direntry.path)

    if unrestricted:

        def _filter_unrestricted(name: ArtifactName) -> bool:
            return name.name in unrestricted and name.component in PACKAGE_COMPONENTS

        catalog = ArtifactCatalog(artifact_dir, filter=_filter_unrestricted)
        _collect_catalog(catalog)

    for artifact_name in restricted:
        includes = overrides[artifact_name]

        def _filter_restricted(name: ArtifactName, _a=artifact_name) -> bool:
            return name.name == _a and name.component in PACKAGE_COMPONENTS

        catalog = ArtifactCatalog(
            artifact_dir, filter=_filter_restricted, includes=includes
        )
        _collect_catalog(catalog)

    if not seen:
        print(
            f"Warning: no matching artifacts found in {artifact_dir}",
            file=sys.stderr,
        )
        return []

    return sorted((Path(r), s) for r, s in seen.items())


def fetch_legacy_dlls_from_dvc(
    dest_dir: Path,
    repo_root: Path,
) -> None:
    """Download legacy DLLs tracked by DVC in rocm-systems into dest_dir.

    Reads *.dvc pointer files from
    rocm-systems/shared/amdgpu-windows-interop/legacy/ and fetches each DLL
    from the DVC S3 remote (s3://therock-dvc/rocm-systems, anonymous) into
    dest_dir. The S3 object key is derived from the md5 hash in the pointer
    file: <md5[:2]>/<md5[2:]>.

    dest_dir is the _legacy/bin/ path that resolve_legacy_dlls() searches as
    its fallback. Missing DLLs are warned and skipped; a DVC fetch failure
    does not abort the generator.
    """
    dvc_dir = (
        repo_root / "rocm-systems" / "shared" / "amdgpu-windows-interop" / "legacy"
    )
    if not dvc_dir.is_dir():
        print(
            f"Warning: rocm-systems legacy DVC dir not found: {dvc_dir}\n"
            "  (submodule not initialized? run: git submodule update --init rocm-systems)",
            file=sys.stderr,
        )
        return

    dvc_remote = "https://therock-dvc.s3.us-east-2.amazonaws.com/rocm-systems/files/md5"
    dest_dir.mkdir(parents=True, exist_ok=True)

    for dvc_file in sorted(dvc_dir.glob("*.dvc")):
        dll_name = dvc_file.stem  # e.g. amdhip64_6.dll
        dest = dest_dir / dll_name
        if dest.exists():
            print(f"  Cached:    {dll_name} (legacy DVC)")
            continue

        try:
            data = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))
            md5 = data["outs"][0]["md5"]
        except Exception as e:
            print(
                f"Warning: could not parse {dvc_file.name}: {e}",
                file=sys.stderr,
            )
            continue

        s3_key = f"{md5[:2]}/{md5[2:]}"
        url = f"{dvc_remote}/{s3_key}"
        print(f"  Fetching:  {dll_name} (legacy DVC, md5={md5[:8]}...)")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(
                f"Warning: failed to fetch legacy DLL {dll_name} from {url}: {e}",
                file=sys.stderr,
            )
            dest.unlink(missing_ok=True)


def resolve_legacy_dlls(
    artifact_dir: Path,
    dll_names: list[str],
) -> list[tuple[str, Path]]:
    """Resolve legacy System32 DLLs to concrete source paths.

    Each name is searched for (by basename) first among the extracted
    artifacts in artifact_dir, then in the _legacy/bin cache fallback
    (a sibling of artifact_dir, i.e. <cache>/_legacy/bin). Driver-sourced
    DLLs like amdhip64_6.dll and amd_comgr_2.dll live only in the fallback.

    Returns a list of (dll_name, source_path). Missing DLLs are reported as
    a warning and skipped so the generator never fails on an absent legacy
    DLL.
    """
    if not dll_names:
        return []

    legacy_fallback = artifact_dir.parent / "_legacy" / "bin"
    resolved: list[tuple[str, Path]] = []
    for name in dll_names:
        found: Path | None = None
        if artifact_dir.is_dir():
            for candidate in artifact_dir.rglob(name):
                if candidate.is_file():
                    found = candidate
                    break
        if found is None:
            fallback = legacy_fallback / name
            if fallback.is_file():
                found = fallback
        if found is None:
            print(
                f"Warning: legacy DLL not found, skipping: {name} "
                f"(searched {artifact_dir} and {legacy_fallback})",
                file=sys.stderr,
            )
            continue
        resolved.append((name, found))
    return resolved


# ---------------------------------------------------------------------------
# Stable WiX element ID generation
# ---------------------------------------------------------------------------


def make_id(path: Path, prefix: str) -> str:
    """Return a stable, WiX-legal element ID derived from a relative file path.

    WiX v4 requires alphanumeric or underscore characters only, max 72 chars.
    Only ASCII alphanumerics are kept; every other character (dashes, spaces,
    dots, non-ASCII/unicode letters) is replaced with an underscore.  An
    8-hex-digit SHA-256 digest of the normalized path is appended to prevent
    collisions after sanitization (e.g. foo-bar vs foo_bar, or café vs cafe,
    would otherwise collide).  The digest is deterministic across runs.
    """
    safe = "".join(
        c if (c.isascii() and c.isalnum()) or c == "_" else "_"
        for c in str(path).replace("\\", "/")
    )
    h = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    return f"{prefix}_{safe}"[:55] + f"_{h}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_rocm_version(repo_root: Path) -> str:
    version_file = repo_root / "version.json"
    try:
        data = json.loads(version_file.read_text())
        return data["rocm-version"]
    except (OSError, KeyError, json.JSONDecodeError):
        return "7.0.0"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent
    default_build = repo_root / "build"
    default_version = _read_rocm_version(repo_root)

    parser = argparse.ArgumentParser(
        description="Generate WiX v4 .wxs files for ROCm Windows MSI installers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--package",
        metavar="NAME",
        choices=list(PACKAGES),
        help="Package to generate. Use --list to see available packages.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available package names and exit.",
    )
    parser.add_argument(
        "--artifacts-url",
        default=None,
        metavar="URL",
        help=(
            "Base URL of a TheRock artifact storage directory containing "
            "{name}_{component}_generic.tar.zst files. When set, artifacts "
            "are downloaded and extracted into --artifacts-cache-dir and used "
            "as stage trees instead of --build-root. "
            "Example: https://therock-nightly-artifacts.s3.amazonaws.com/27315369389-windows"
        ),
    )
    parser.add_argument(
        "--fetch-legacy-dlls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Fetch legacy DLLs from DVC (rocm-systems submodule) into the "
            "_legacy/bin cache used by the legacy System32 install feature. "
            "Defaults to True when --artifacts-url is set, False otherwise."
        ),
    )
    parser.add_argument(
        "--artifacts-cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Directory for downloaded and extracted artifacts when using "
            "--artifacts-url. Defaults to <script-dir>/artifact-cache. "
            "Reuse this dir across runs to avoid re-downloading."
        ),
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=default_build,
        metavar="PATH",
        help=(
            "CMake build directory. Artifacts are read from <build-root>/artifacts/. "
            f"Default: {default_build}. Ignored when --artifacts-url is set."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Destination path for the generated .wxs file. "
            "Default: <script-dir>/<output-stem>.wxs"
        ),
    )
    parser.add_argument(
        "--install-root",
        default="ProgramFilesFolder",
        metavar="ROOT",
        help=(
            "Where to root the install tree. Accepts a Windows Installer "
            "standard-directory token (e.g. ProgramFilesFolder) or an "
            "absolute path (e.g. C:\\AMD). Default: ProgramFilesFolder"
        ),
    )
    parser.add_argument(
        "--product-dir",
        default="AMD",
        metavar="NAME",
        help="First subdirectory under --install-root. Default: AMD",
    )
    parser.add_argument(
        "--version-dir",
        default="ROCm",
        metavar="NAME",
        help="Second subdirectory under --product-dir. Default: ROCm",
    )
    parser.add_argument(
        "--package-version",
        default=default_version,
        metavar="X.Y.Z",
        help=(f"MSI version string. Default: {default_version} (from version.json)"),
    )

    args = parser.parse_args()

    if args.list:
        print("Available packages:")
        for name, pkg in PACKAGES.items():
            print(f"  {name:<20} {pkg.description}")
        sys.exit(0)

    if not args.package:
        parser.error("--package is required (use --list to see options)")

    if args.output is None:
        pkg = PACKAGES[args.package]
        args.output = script_dir / f"{pkg.output_stem}.wxs"

    if args.artifacts_cache_dir is None:
        args.artifacts_cache_dir = script_dir / "artifact-cache"

    return args


# ---------------------------------------------------------------------------
# WXS builder
# ---------------------------------------------------------------------------

# WiX v4 schema namespace. Every generated element is Clark-notation qualified
# with it (ElementTree's convention for namespaced tags).
WXS_NS = "http://wixtoolset.org/schemas/v4/wxs"


def _tag(name: str) -> str:
    """Return the namespace-qualified form of a WiX element name."""
    return f"{{{WXS_NS}}}{name}"


def _stable_guid(*parts: str) -> str:
    """Return a deterministic upper-case GUID derived from the given parts.

    Using uuid5 over a fixed namespace keeps component GUIDs stable across runs
    for the same logical identity, which Windows Installer requires for correct
    upgrade and repair behavior.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "/".join(parts))).upper()


@dataclass
class PackageInputs:
    """Everything gathered from disk before any WiX XML is emitted.

    files:       (install_rel, source) payload pairs for the primary feature.
    legacy_dlls: (dll_name, source) pairs for the legacy System32 feature.
    """

    package: PackageDef
    version: str
    files: list[tuple[Path, Path]]
    legacy_dlls: list[tuple[str, Path]]


@dataclass
class InstallLayout:
    """Resolved install location, split into the pieces WiX needs.

    subdir_name is the final versioned path segment (e.g. "core-7.15"); major
    and minor are the parsed version components used to expand registry keys.
    """

    install_root: str
    product_dir: str
    version_dir: str
    version: str
    subdir_name: str
    major: str
    minor: str

    @property
    def uses_standard_dir(self) -> bool:
        """True when install_root is a Windows Installer standard-directory token."""
        return self.install_root in STANDARD_DIR_TOKENS

    @property
    def display_path(self) -> str:
        """Human-readable install path for logging."""
        return (
            f"[{self.install_root}]\\{self.product_dir}"
            f"\\{self.version_dir}\\{self.subdir_name}\\"
        )


@dataclass
class WixDocument:
    """A WiX document under construction.

    Holds the <Wix> root, the <Package> element, and the leaf <Directory
    Id="InstallDir"> that payload components hang off of. `directory_cache`
    deduplicates on-demand payload subdirectories by their POSIX path.
    """

    root: ET.Element
    package: ET.Element
    install_dir: ET.Element = None
    directory_cache: dict[str, ET.Element] = field(default_factory=dict)


def resolve_package_inputs(args: argparse.Namespace) -> PackageInputs:
    """Gather artifacts and payload/legacy file lists for the selected package.

    Downloads artifacts when --artifacts-url is set and fetches legacy DLLs from
    DVC when applicable, then enumerates the concrete files to install.
    """
    package = PACKAGES[args.package]

    if args.artifacts_url:
        print(f"Fetching artifacts from {args.artifacts_url} ...")
        artifact_dir = fetch_artifacts(
            artifacts_url=args.artifacts_url,
            artifact_names=package.artifacts,
            components=PACKAGE_COMPONENTS,
            dest_dir=args.artifacts_cache_dir,
        )
        legacy_bin = args.artifacts_cache_dir / "_legacy" / "bin"
    else:
        # Local build: artifacts live at build/artifacts/{name}_{component}_generic/
        artifact_dir = args.build_root / "artifacts"
        legacy_bin = args.build_root / "_legacy" / "bin"

    # --fetch-legacy-dlls defaults to True in --artifacts-url mode, False for a
    # local build. Skip entirely when the package declares no legacy DLLs.
    fetch_legacy = args.fetch_legacy_dlls
    if fetch_legacy is None:
        fetch_legacy = bool(args.artifacts_url)
    if fetch_legacy and package.legacy_system32_dlls:
        print("Fetching legacy DLLs from DVC ...")
        fetch_legacy_dlls_from_dvc(
            dest_dir=legacy_bin,
            repo_root=Path(__file__).parent.parent.parent.parent,
        )

    return PackageInputs(
        package=package,
        version=args.package_version,
        files=collect_files_from_catalog(artifact_dir, package),
        legacy_dlls=resolve_legacy_dlls(artifact_dir, package.legacy_system32_dlls),
    )


def resolve_install_layout(args: argparse.Namespace, version: str) -> InstallLayout:
    """Resolve the install location and versioned subdirectory name."""
    parts = version.split(".")
    major = parts[0] if len(parts) > 0 else ""
    minor = parts[1] if len(parts) > 1 else ""
    package = PACKAGES[args.package]
    subdir_name = package.install_subdir.format(
        version=version, major=major, minor=minor
    )
    return InstallLayout(
        install_root=args.install_root,
        product_dir=args.product_dir,
        version_dir=args.version_dir,
        version=version,
        subdir_name=subdir_name,
        major=major,
        minor=minor,
    )


def create_wix_document(package: PackageDef, version: str) -> WixDocument:
    """Create the <Wix>/<Package> skeleton with summary, upgrade, and properties.

    Emits everything that precedes the directory tree: SummaryInformation,
    MajorUpgrade, MediaTemplate, the three control Properties, and the
    INSTALLFOLDER redirect.
    """
    root = ET.Element(_tag("Wix"))
    pkg = ET.SubElement(
        root,
        _tag("Package"),
        Name=package.product_name,
        Version=version,
        Manufacturer="Advanced Micro Devices, Inc.",
        UpgradeCode=package.upgrade_code,
        Language="1033",
        Codepage="1252",
        InstallerVersion="500",
        Compressed="yes",
    )
    # NOTE: package architecture is set at compile time via `wix build -arch
    # x64`, not here (WiX v4 has no Package/@Platform attribute). x64 is
    # required so SystemFolder resolves to C:\Windows\System32 rather than the
    # WOW64-redirected SysWOW64, and so the 64-bit DLLs install correctly.
    ET.SubElement(
        pkg,
        _tag("SummaryInformation"),
        Keywords="ROCm AMD GPU",
        Description=f"{package.product_name} {version}",
    )
    ET.SubElement(
        pkg,
        _tag("MajorUpgrade"),
        DowngradeErrorMessage=f"A newer version of {package.product_name} is already installed.",
    )
    ET.SubElement(pkg, _tag("MediaTemplate"), EmbedCab="yes")
    # Long path support is on by default; set ENABLE_LONG_PATHS=0 to disable.
    ET.SubElement(
        pkg, _tag("Property"), Id="ENABLE_LONG_PATHS", Value="1", Secure="yes"
    )
    ET.SubElement(pkg, _tag("Property"), Id="INSTALLFOLDER", Secure="yes")
    # Legacy System32 install is on by default; set LEGACY_INSTALL=0 to disable.
    ET.SubElement(pkg, _tag("Property"), Id="LEGACY_INSTALL", Value="1", Secure="yes")
    # When INSTALLFOLDER is set on the command line, redirect InstallDir to it.
    # Runs in both UI and execute sequences so repair/modify picks it up too.
    ET.SubElement(
        pkg,
        _tag("SetDirectory"),
        Id="InstallDir",
        Value="[INSTALLFOLDER]",
        Sequence="both",
        Condition="INSTALLFOLDER",
    )
    return WixDocument(root=root, package=pkg)


def add_install_directory_tree(doc: WixDocument, layout: InstallLayout) -> None:
    """Build the install directory tree and record its leaf on the document.

    Sets doc.install_dir to the <Directory Id="InstallDir"> that payload
    components attach to.
    """
    if layout.uses_standard_dir:
        parent = ET.SubElement(
            doc.package, _tag("StandardDirectory"), Id=layout.install_root
        )
    else:
        target_dir = ET.SubElement(
            doc.package, _tag("Directory"), Id="TARGETDIR", Name="SourceDir"
        )
        parent = ET.SubElement(
            target_dir,
            _tag("Directory"),
            Id="CustomInstallRoot",
            Name=layout.install_root,
        )

    rocm_dir = ET.SubElement(
        parent, _tag("Directory"), Id="ROCmDir", Name=layout.product_dir
    )
    version_dir = ET.SubElement(
        rocm_dir, _tag("Directory"), Id="ROCmVerDir", Name=layout.version_dir
    )
    doc.install_dir = ET.SubElement(
        version_dir, _tag("Directory"), Id="InstallDir", Name=layout.subdir_name
    )


def add_primary_feature(doc: WixDocument, package: PackageDef) -> ET.Element:
    """Create and return the package's primary <Feature>."""
    return ET.SubElement(
        doc.package,
        _tag("Feature"),
        Id=package.feature_id,
        Title=package.feature_title,
        Level="1",
    )


def _get_or_create_directory(doc: WixDocument, install_rel: Path) -> ET.Element:
    """Return the <Directory> for a payload file's parent, creating ancestors.

    Intermediate directories are created once and cached by their POSIX path so
    that repeated file paths reuse the same directory elements.
    """
    parent = doc.install_dir
    accumulated = Path()
    for part in install_rel.parent.parts:
        accumulated = accumulated / part
        dir_key = accumulated.as_posix()
        cached = doc.directory_cache.get(dir_key)
        if cached is None:
            dir_id = "Dir_" + "".join(
                c if c.isascii() and c.isalnum() else "_" for c in dir_key
            )
            cached = ET.SubElement(parent, _tag("Directory"), Id=dir_id, Name=part)
            doc.directory_cache[dir_key] = cached
        parent = cached
    return parent


def add_payload_components(
    doc: WixDocument, feature: ET.Element, files: list[tuple[Path, Path]]
) -> None:
    """Emit one <Component>/<File> per payload file under the install tree."""
    for install_rel, source in files:
        parent = _get_or_create_directory(doc, install_rel)
        comp_id = make_id(install_rel, "c")
        component = ET.SubElement(
            parent,
            _tag("Component"),
            Id=comp_id,
            Guid=_stable_guid(str(install_rel)),
        )
        ET.SubElement(
            component,
            _tag("File"),
            Id=make_id(install_rel, "f"),
            Source=str(source.resolve()),
            Name=source.name,
            KeyPath="yes",
        )
        ET.SubElement(feature, _tag("ComponentRef"), Id=comp_id)


def add_path_registration(
    doc: WixDocument,
    feature: ET.Element,
    package: PackageDef,
    layout: InstallLayout,
    files: list[tuple[Path, Path]],
) -> None:
    """Add the machine PATH entry and install-dir registry marker.

    No-op when the package installs nothing under bin/, since there would be
    nothing worth adding to PATH.
    """
    installs_to_bin = any(install_rel.parts[0] == "bin" for install_rel, _ in files)
    if not installs_to_bin:
        return

    component = ET.SubElement(
        doc.install_dir,
        _tag("Component"),
        Id="EnvPath",
        Guid=_stable_guid(f"ROCm_{package.feature_id}_PATH_component"),
    )
    ET.SubElement(
        component,
        _tag("Environment"),
        Id="ROCmBinPath",
        Name="PATH",
        Value="[InstallDir]bin",
        Permanent="no",
        Part="last",
        Action="set",
        System="yes",
    )
    registry_key = package.registry_key.format(
        version=layout.version, major=layout.major, minor=layout.minor
    )
    ET.SubElement(
        component,
        _tag("RegistryValue"),
        Root="HKLM",
        Key=registry_key,
        Name="InstallDir",
        Value="[InstallDir]",
        Type="string",
        KeyPath="yes",
    )
    ET.SubElement(feature, _tag("ComponentRef"), Id="EnvPath")


def add_long_paths_feature(doc: WixDocument, package: PackageDef) -> None:
    """Add the optional LongPathsEnabled feature (on unless ENABLE_LONG_PATHS=0).

    A pre-existing value is preserved: the component installs (and therefore
    uninstalls) only when the key was not already set to 1 by the user or
    another installer.
    """
    # Read the pre-existing LongPathsEnabled value before install. WiX raw
    # integer registry searches return values as "#<decimal>", so a pre-set
    # value of 1 appears as "#1".
    ET.SubElement(doc.package, _tag("Property"), Id="LONGPATHS_PREEXISTING", Value="0")
    ET.SubElement(
        doc.package,
        _tag("RegistrySearch"),
        Id="LongPathsPreExistingSearch",
        Root="HKLM",
        Key="SYSTEM\\CurrentControlSet\\Control\\FileSystem",
        Name="LongPathsEnabled",
        Type="raw",
        Result="value",
        Property="LONGPATHS_PREEXISTING",
    )
    component = ET.SubElement(
        doc.install_dir,
        _tag("Component"),
        Id="LongPathsEnable",
        Guid=_stable_guid(f"ROCm_{package.feature_id}_LongPaths_component"),
    )
    ET.SubElement(
        component,
        _tag("RegistryValue"),
        Root="HKLM",
        Key="SYSTEM\\CurrentControlSet\\Control\\FileSystem",
        Name="LongPathsEnabled",
        Value="1",
        Type="integer",
        KeyPath="yes",
    )
    ET.SubElement(component, _tag("Condition")).text = (
        'NOT LONGPATHS_PREEXISTING = "#1"'
    )
    feature = ET.SubElement(
        doc.package,
        _tag("Feature"),
        Id="LongPaths",
        Title="Enable Long Paths",
        Level="1",
    )
    # Turn the feature off when ENABLE_LONG_PATHS=0. Default property value
    # is "1", so the feature is enabled unless explicitly disabled.
    ET.SubElement(
        feature, _tag("Level"), Value="0", Condition='ENABLE_LONG_PATHS = "0"'
    )
    ET.SubElement(feature, _tag("ComponentRef"), Id="LongPathsEnable")


def add_legacy_system32_feature(
    doc: WixDocument, legacy_dlls: list[tuple[str, Path]]
) -> None:
    """Add the legacy System32 DLL feature (on unless LEGACY_INSTALL=0).

    No-op when the package has no legacy DLLs to install.
    """
    if not legacy_dlls:
        return

    # System64Folder is the real C:\Windows\System32 in an x64 package;
    # SystemFolder resolves to the WOW64-redirected SysWOW64 regardless of
    # package architecture, which is not what we want for 64-bit DLLs.
    system_dir = ET.SubElement(
        doc.package, _tag("StandardDirectory"), Id="System64Folder"
    )
    feature = ET.SubElement(
        doc.package,
        _tag("Feature"),
        Id="LegacyInstall",
        Title="Legacy System32 DLLs",
        Level="1",
    )
    # Turn the feature off when LEGACY_INSTALL=0. Default property value
    # is "1", so the feature is enabled unless explicitly disabled.
    ET.SubElement(feature, _tag("Level"), Value="0", Condition='LEGACY_INSTALL = "0"')
    for dll_name, source in legacy_dlls:
        install_rel = Path("System32") / dll_name
        comp_id = make_id(install_rel, "c")
        component = ET.SubElement(
            system_dir,
            _tag("Component"),
            Id=comp_id,
            Guid=_stable_guid("System32", dll_name),
        )
        ET.SubElement(
            component,
            _tag("File"),
            Id=make_id(install_rel, "f"),
            Source=str(source.resolve()),
            Name=dll_name,
            KeyPath="yes",
        )
        ET.SubElement(feature, _tag("ComponentRef"), Id=comp_id)


def write_wxs(root: ET.Element, output_path: Path) -> None:
    """Serialize the WiX document to output_path with a stable XML declaration."""
    ET.register_namespace("", WXS_NS)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)


def build_wxs(args: argparse.Namespace) -> None:
    """Generate a WiX v4 .wxs source file for the selected package."""
    inputs = resolve_package_inputs(args)
    layout = resolve_install_layout(args, inputs.version)

    doc = create_wix_document(inputs.package, inputs.version)
    add_install_directory_tree(doc, layout)
    feature = add_primary_feature(doc, inputs.package)
    add_payload_components(doc, feature, inputs.files)
    add_path_registration(doc, feature, inputs.package, layout, inputs.files)
    add_long_paths_feature(doc, inputs.package)
    add_legacy_system32_feature(doc, inputs.legacy_dlls)

    write_wxs(doc.root, args.output)

    print(f"Written:  {args.output}")
    print(f"Files:    {len(inputs.files)}")
    print(f"Install:  {layout.display_path}")


if __name__ == "__main__":
    build_wxs(parse_args())
