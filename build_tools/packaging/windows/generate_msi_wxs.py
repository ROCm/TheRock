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
    python generate_msi_wxs.py --package hip-runtime \\
        --artifacts-url https://therock-nightly-artifacts.s3.amazonaws.com/<run-id>-windows

Generate from a local build:
    python generate_msi_wxs.py --package hip-runtime --build build/

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
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


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

    # Versioned install subdirectory name (formatted with version=X.Y.Z).
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

    def __post_init__(self):
        if self.per_artifact_includes is None:
            self.per_artifact_includes = {}


# Keys are the --package selector values.
PACKAGES: dict[str, PackageDef] = {
    "hip-runtime": PackageDef(
        description="HIP runtime DLLs, hipcc, hipconfig, kernel package support",
        product_name="AMD ROCm HIP Runtime",
        artifacts=[
            "core-hip",      # HIP runtime DLLs, hipcc, hipconfig, roc-obj, etc.
            "core-kpack",    # Kernel package support (rocm_kpack.dll)
            "core-hipinfo",  # Windows-only: bin/hipInfo*
        ],
        output_stem="amdrocm-hip-runtime",
        install_subdir="hip-runtime-{version}",
        upgrade_code="B2C3D4E5-F6A7-8901-BCDE-F12345678901",
        feature_id="HIPRuntime",
        feature_title="ROCm HIP Runtime",
        registry_key="Software\\AMD\\ROCm\\hip-runtime\\{version}",
    ),
    "runtimes": PackageDef(
        description="ROCm runtime redistributable (HIP runtime + amd_comgr.dll)",
        product_name="AMD ROCm Runtimes",
        artifacts=[
            "core-hip",      # HIP runtime DLLs, hipcc, hipconfig, roc-obj, etc.
            "core-kpack",    # Kernel package support (rocm_kpack.dll)
            "core-hipinfo",  # Windows-only: bin/hipInfo*
            "amd-llvm",      # comgr only — see per_artifact_includes
        ],
        output_stem="amdrocm-runtimes",
        install_subdir="runtimes-{version}",
        upgrade_code="C3D4E5F6-A7B8-9012-CDEF-123456789012",
        feature_id="ROCmRuntimes",
        feature_title="AMD ROCm Runtimes",
        registry_key="Software\\AMD\\ROCm\\runtimes\\{version}",
        per_artifact_includes={
            "amd-llvm": ["bin/amd_comgr.dll"],
        },
    ),
    "core": PackageDef(
        description="ROCm core runtime redistributable (ROCR, HIP, AMDsmi, OpenCL)",
        product_name="AMD ROCm Core Runtime",
        artifacts=[
            "core-runtime",  # ROCR-Runtime + rocminfo
            "core-hip",      # HIP runtime DLLs, hipcc, hipconfig, roc-obj, etc.
            "core-kpack",    # Kernel package support (rocm_kpack.dll)
            "core-hipinfo",  # Windows-only: bin/hipInfo*
            "core-amdsmi",   # AMD System Management Interface
            "core-ocl-icd",  # OpenCL ICD loader (bin/OpenCL.dll on Windows)
        ],
        output_stem="amdrocm-core",
        install_subdir="core-{version}",
        upgrade_code="A1B2C3D4-E5F6-7890-ABCD-EF1234567890",
        feature_id="ROCmCore",
        feature_title="AMD ROCm Core Runtime",
        registry_key="Software\\AMD\\ROCm\\core\\{version}",
    ),
}


# Only package runtime-relevant components; exclude dev headers, debug
# symbols, and docs from the MSI.
PACKAGE_COMPONENTS: set[str] = {"run", "lib"}

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


def _import_therock_utils() -> None:
    """Add build_tools/ to sys.path so _therock_utils can be imported."""
    build_tools_dir = Path(__file__).parent.parent.parent
    if str(build_tools_dir) not in sys.path:
        sys.path.insert(0, str(build_tools_dir))


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
    _import_therock_utils()
    try:
        from _therock_utils.artifacts import ArtifactPopulator
    except ImportError as e:
        sys.exit(f"Error: could not import _therock_utils: {e}")

    import tarfile

    def _open_zst(path: Path):
        try:
            import pyzstd
        except ModuleNotFoundError:
            sys.exit("pyzstd is required for --artifacts-url: pip install pyzstd")
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
                try:
                    urllib.request.urlretrieve(url, local_path)
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        print(f"  Skipped:   {filename} (not found)")
                        continue
                    sys.exit(f"Error fetching {url}: {e}")

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
                if manifest_member is None or manifest_member.name != "artifact_manifest.txt":
                    sys.exit(f"Archive {filename} missing artifact_manifest.txt as first member")
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

    _import_therock_utils()
    try:
        from _therock_utils.artifacts import ArtifactCatalog, ArtifactName
    except ImportError as e:
        sys.exit(f"Error: could not import _therock_utils: {e}")

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


# ---------------------------------------------------------------------------
# Stable WiX element ID generation
# ---------------------------------------------------------------------------


def make_id(path: Path, prefix: str) -> str:
    """Return a stable, WiX-legal element ID derived from a relative file path.

    WiX v4 requires alphanumeric or underscore characters only, max 72 chars.
    An 8-hex-digit SHA-256 digest of the normalized path is appended to prevent
    collisions after sanitization (e.g. foo-bar vs foo_bar would otherwise
    collide).  The digest is deterministic across interpreter runs.
    """
    safe = "".join(
        c if c.isalnum() or c == "_" else "_"
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
        "--artifacts-cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Directory for downloaded and extracted artifacts when using "
            "--artifacts-url. Defaults to <script-dir>/.artifact-cache. "
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
        help=(
            f"MSI version string. Default: {default_version} (from version.json)"
        ),
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
        args.artifacts_cache_dir = script_dir / ".artifact-cache"

    return args


# ---------------------------------------------------------------------------
# WXS builder
# ---------------------------------------------------------------------------


def build_wxs(args: argparse.Namespace) -> None:
    pkg_def = PACKAGES[args.package]
    version = args.package_version
    use_standard_dir = args.install_root in STANDARD_DIR_TOKENS

    if getattr(args, "artifacts_url", None):
        print(f"Fetching artifacts from {args.artifacts_url} ...")
        artifact_dir = fetch_artifacts(
            artifacts_url=args.artifacts_url,
            artifact_names=pkg_def.artifacts,
            components=PACKAGE_COMPONENTS,
            dest_dir=args.artifacts_cache_dir,
        )
    else:
        # Local build: artifacts live at build/artifacts/{name}_{component}_generic/
        artifact_dir = args.build_root / "artifacts"

    files = collect_files_from_catalog(artifact_dir, pkg_def)

    ET.register_namespace("", "http://wixtoolset.org/schemas/v4/wxs")
    ns = "http://wixtoolset.org/schemas/v4/wxs"

    # -----------------------------------------------------------------------
    # Root and Package
    # -----------------------------------------------------------------------
    root = ET.Element(f"{{{ns}}}Wix")

    pkg = ET.SubElement(
        root,
        f"{{{ns}}}Package",
        Name=pkg_def.product_name,
        Version=version,
        Manufacturer="Advanced Micro Devices, Inc.",
        UpgradeCode=pkg_def.upgrade_code,
        Language="1033",
        Codepage="1252",
        InstallerVersion="500",
        Compressed="yes",
    )

    ET.SubElement(
        pkg,
        f"{{{ns}}}SummaryInformation",
        Keywords="ROCm AMD GPU",
        Description=f"{pkg_def.product_name} {version}",
    )

    ET.SubElement(
        pkg,
        f"{{{ns}}}MajorUpgrade",
        DowngradeErrorMessage=f"A newer version of {pkg_def.product_name} is already installed.",
    )

    ET.SubElement(pkg, f"{{{ns}}}MediaTemplate", EmbedCab="yes")
    ET.SubElement(pkg, f"{{{ns}}}Property", Id="ENABLE_LONG_PATHS", Secure="yes")
    ET.SubElement(pkg, f"{{{ns}}}Property", Id="INSTALLFOLDER", Secure="yes")
    # When INSTALLFOLDER is set on the command line, redirect InstallDir to it.
    # Runs in both UI and execute sequences so repair/modify picks it up too.
    ET.SubElement(
        pkg,
        f"{{{ns}}}SetDirectory",
        Id="InstallDir",
        Value="[INSTALLFOLDER]",
        Sequence="both",
        Condition="INSTALLFOLDER",
    )

    # -----------------------------------------------------------------------
    # Directory tree
    # -----------------------------------------------------------------------
    if use_standard_dir:
        install_dir = ET.SubElement(
            pkg, f"{{{ns}}}StandardDirectory", Id=args.install_root
        )
        rocm_dir = ET.SubElement(
            install_dir, f"{{{ns}}}Directory", Id="ROCmDir", Name=args.product_dir
        )
    else:
        targetdir = ET.SubElement(
            pkg, f"{{{ns}}}Directory", Id="TARGETDIR", Name="SourceDir"
        )
        custom_root = ET.SubElement(
            targetdir,
            f"{{{ns}}}Directory",
            Id="CustomInstallRoot",
            Name=args.install_root,
        )
        rocm_dir = ET.SubElement(
            custom_root, f"{{{ns}}}Directory", Id="ROCmDir", Name=args.product_dir
        )

    ver_dir = ET.SubElement(
        rocm_dir, f"{{{ns}}}Directory", Id="ROCmVerDir", Name=args.version_dir
    )
    install_subdir_name = pkg_def.install_subdir.format(version=version)
    install_dir_el = ET.SubElement(
        ver_dir, f"{{{ns}}}Directory", Id="InstallDir", Name=install_subdir_name
    )

    # -----------------------------------------------------------------------
    # Feature
    # -----------------------------------------------------------------------
    feature = ET.SubElement(
        pkg,
        f"{{{ns}}}Feature",
        Id=pkg_def.feature_id,
        Title=pkg_def.feature_title,
        Level="1",
    )

    # -----------------------------------------------------------------------
    # Components — one per file, grouped by parent directory
    # -----------------------------------------------------------------------
    dir_elements: dict[str, ET.Element] = {}

    for install_rel, source in files:
        parent_rel = install_rel.parent

        current_parent = install_dir_el
        accumulated = Path()
        for part in parent_rel.parts:
            accumulated = accumulated / part
            dir_key = str(accumulated)
            if dir_key not in dir_elements:
                dir_id = "Dir_" + (
                    dir_key
                    .replace("\\", "_")
                    .replace("/", "_")
                    .replace("-", "_")
                    .replace(".", "_")
                )
                dir_elements[dir_key] = ET.SubElement(
                    current_parent,
                    f"{{{ns}}}Directory",
                    Id=dir_id,
                    Name=part,
                )
            current_parent = dir_elements[dir_key]

        file_id = make_id(install_rel, "f")
        comp_id = make_id(install_rel, "c")
        guid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(install_rel))).upper()

        comp_el = ET.SubElement(
            current_parent, f"{{{ns}}}Component", Id=comp_id, Guid=guid
        )
        ET.SubElement(
            comp_el,
            f"{{{ns}}}File",
            Id=file_id,
            Source=str(source.resolve()),
            Name=source.name,
            KeyPath="yes",
        )
        ET.SubElement(feature, f"{{{ns}}}ComponentRef", Id=comp_id)

    # -----------------------------------------------------------------------
    # PATH environment variable + registry install-dir marker
    # -----------------------------------------------------------------------
    has_bin = any(install_rel.parts[0] == "bin" for install_rel, _ in files)
    if has_bin:
        env_comp = ET.SubElement(
            install_dir_el,
            f"{{{ns}}}Component",
            Id="EnvPath",
            Guid=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"ROCm_{pkg_def.feature_id}_PATH_component",
                )
            ).upper(),
        )
        ET.SubElement(
            env_comp,
            f"{{{ns}}}Environment",
            Id="ROCmBinPath",
            Name="PATH",
            Value="[InstallDir]bin",
            Permanent="no",
            Part="last",
            Action="set",
            System="yes",
        )
        reg_key = pkg_def.registry_key.format(version=version)
        ET.SubElement(
            env_comp,
            f"{{{ns}}}RegistryValue",
            Root="HKLM",
            Key=reg_key,
            Name="InstallDir",
            Value="[InstallDir]",
            Type="string",
            KeyPath="yes",
        )
        ET.SubElement(feature, f"{{{ns}}}ComponentRef", Id="EnvPath")

    # -----------------------------------------------------------------------
    # Optional long-path support
    # -----------------------------------------------------------------------
    lp_comp = ET.SubElement(
        install_dir_el,
        f"{{{ns}}}Component",
        Id="LongPathsEnable",
        Guid=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"ROCm_{pkg_def.feature_id}_LongPaths_component",
            )
        ).upper(),
    )
    ET.SubElement(
        lp_comp,
        f"{{{ns}}}RegistryValue",
        Root="HKLM",
        Key="SYSTEM\\CurrentControlSet\\Control\\FileSystem",
        Name="LongPathsEnabled",
        Value="1",
        Type="integer",
        KeyPath="yes",
    )
    lp_feature = ET.SubElement(
        pkg, f"{{{ns}}}Feature", Id="LongPaths", Title="Enable Long Paths", Level="0"
    )
    ET.SubElement(lp_feature, f"{{{ns}}}Level", Value="1", Condition="ENABLE_LONG_PATHS")
    ET.SubElement(lp_feature, f"{{{ns}}}ComponentRef", Id="LongPathsEnable")

    # -----------------------------------------------------------------------
    # Serialize
    # -----------------------------------------------------------------------
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)

    print(f"Written:  {args.output}")
    print(f"Files:    {len(files)}")
    print(
        f"Install:  [{args.install_root}]\\{args.product_dir}"
        f"\\{args.version_dir}\\{install_subdir_name}\\"
    )


if __name__ == "__main__":
    build_wxs(parse_args())
