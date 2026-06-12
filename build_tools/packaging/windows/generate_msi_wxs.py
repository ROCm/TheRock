#!/usr/bin/env python3
"""Generate WiX v4 .wxs files for ROCm Windows MSI installers.

Reads artifact TOML descriptors to determine which files each MSI should
include, then produces a WiX v4 XML source file (.wxs).  Using artifact
TOMLs as the source of truth means the MSI file lists automatically track
the build system's own packaging rules with no separate manifest to maintain.

Usage
-----
List available packages:
    python generate_msi_wxs.py --list

Generate a specific package:
    python generate_msi_wxs.py --package hip-runtime
    python generate_msi_wxs.py --package runtimes

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
import sys
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit(
            "Python < 3.11 requires the 'tomli' package: pip install tomli"
        )


# ---------------------------------------------------------------------------
# Package definitions
# ---------------------------------------------------------------------------

@dataclass
class PackageDef:
    """Definition of one distributable MSI package."""

    # Display name shown in Add/Remove Programs.
    product_name: str

    # Artifact descriptor names to include (searched as artifact-{name}.toml).
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

    # Glob patterns excluded only in dist fallback mode (when stage dirs are
    # absent).  Stage scoping would naturally prevent these files from being
    # included; this list approximates that filtering without stage dirs.
    fallback_excludes: list[str] = field(default_factory=list)


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
        fallback_excludes=[
            # Kernel databases from math libs (rocBLAS, hipBLASLt, MIOpen, etc.)
            "**/*.dat",
            "**/*.co",
            "**/*.hsaco",
            "**/*.model",
            # Math/ML lib subdirectories under bin/ and lib/
            "bin/rocblas/**",
            "bin/hipblaslt/**",
            "bin/hipblaslt_plugin/**",
            "bin/MIOpen/**",
            "bin/miopen_plugin/**",
            "bin/hipdnn/**",
            "bin/hipdnn_plugins/**",
            "bin/hipdnn_integration_tests_ctest/**",
            "bin/hipdnn_samples/**",
            "bin/hip_kernel_provider/**",
            "bin/hipkernelprovider/**",
            "bin/test_plugins/**",
            "bin/rocwmma/**",
            "bin/hipcub/**",
            "bin/rocprim/**",
            "bin/rocthrust/**",
            "bin/hipblas/**",
            "bin/hipRAND/**",
            "bin/rocRAND/**",
            "bin/hipsparse/**",
            "bin/rocsparse/**",
            # Test assets under share/
            "share/hip/catch_tests/**",
        ],
    ),
    "runtimes": PackageDef(
        description="HIP runtime + AMD LLVM compiler runtime (hipcc, comgr, device libs)",
        product_name="AMD ROCm Runtimes",
        artifacts=[
            "core-hip",      # HIP runtime DLLs, hipcc, hipconfig, roc-obj, etc.
            "core-kpack",    # Kernel package support (rocm_kpack.dll)
            "core-hipinfo",  # Windows-only: bin/hipInfo*
            "amd-llvm",      # LLVM/Clang, comgr, hipcc, device libs
        ],
        output_stem="amdrocm-runtimes",
        install_subdir="runtimes-{version}",
        upgrade_code="C3D4E5F6-A7B8-9012-CDEF-123456789012",
        feature_id="ROCmRuntimes",
        feature_title="AMD ROCm Runtimes",
        registry_key="Software\\AMD\\ROCm\\runtimes\\{version}",
        fallback_excludes=[
            # Kernel databases from math libs (rocBLAS, hipBLASLt, MIOpen, etc.)
            "**/*.dat",
            "**/*.co",
            "**/*.hsaco",
            "**/*.model",
            # Math/ML lib subdirectories under bin/ and lib/
            "bin/rocblas/**",
            "bin/hipblaslt/**",
            "bin/hipblaslt_plugin/**",
            "bin/MIOpen/**",
            "bin/miopen_plugin/**",
            "bin/hipdnn/**",
            "bin/hipdnn_plugins/**",
            "bin/hipdnn_integration_tests_ctest/**",
            "bin/hipdnn_samples/**",
            "bin/hip_kernel_provider/**",
            "bin/hipkernelprovider/**",
            "bin/test_plugins/**",
            "bin/rocwmma/**",
            "bin/hipcub/**",
            "bin/rocprim/**",
            "bin/rocthrust/**",
            "bin/hipblas/**",
            "bin/hipRAND/**",
            "bin/rocRAND/**",
            "bin/hipsparse/**",
            "bin/rocsparse/**",
            # Test assets under share/
            "share/hip/catch_tests/**",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Component default include patterns (Windows-only subset)
# ---------------------------------------------------------------------------

# Mirrors the Windows-relevant subset of artifact_builder.py ComponentDefaults.
# Applied when a component section in an artifact TOML has no explicit "include"
# list and default_patterns is not disabled.
COMPONENT_DEFAULT_INCLUDES: dict[str, list[str]] = {
    "lib": ["**/*.dll"],
    "run": [],  # no default patterns; a bare run entry claims all unclaimed files
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
# Artifact download and extraction
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
    and extracts it into dest_dir using ArtifactPopulator, producing a
    layout that mirrors what a local build's stage trees would look like.

    Returns dest_dir (usable as build_root for collect_files_from_artifacts).
    """
    # Import here so the rest of the script works without _therock_utils on path.
    script_dir = Path(__file__).parent
    build_tools_dir = script_dir.parent.parent
    if str(build_tools_dir) not in sys.path:
        sys.path.insert(0, str(build_tools_dir))

    try:
        from _therock_utils.artifacts import ArtifactPopulator
        from _therock_utils.archive_util import open_archive_for_read
    except ImportError as e:
        sys.exit(f"Error: could not import _therock_utils: {e}")

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
            if artifact_out.exists():
                print(f"  Extracted: {filename} (cached)")
                continue
            print(f"  Extracting {filename}...")
            artifact_out.mkdir(parents=True, exist_ok=True)
            populator = ArtifactPopulator(output_path=artifact_out, flatten=False)
            populator(local_path)

    return extract_dir


# ---------------------------------------------------------------------------
# TOML descriptor parsing
# ---------------------------------------------------------------------------


def find_descriptor(artifact_name: str, repo_root: Path) -> Path:
    """Return the path to artifact-{name}.toml by searching under repo_root."""
    matches = list(repo_root.rglob(f"artifact-{artifact_name}.toml"))
    if not matches:
        sys.exit(
            f"Error: could not find artifact-{artifact_name}.toml under {repo_root}"
        )
    if len(matches) > 1:
        sys.exit(
            f"Error: found multiple descriptors for '{artifact_name}':\n"
            + "\n".join(f"  {p}" for p in matches)
        )
    return matches[0]


def collect_files_from_artifacts(
    dist_root: Path,
    artifact_names: list[str],
    repo_root: Path,
    components: set[str],
    build_root: Path,
    fallback_excludes: list[str] = (),
) -> list[tuple[Path, Path]]:
    """Return a sorted, deduplicated list of (install_rel, source) pairs.

    install_rel: path relative to the install root (e.g. bin/amdhip64.dll)
    source:      absolute path to the file on disk (for WiX Source=)

    Patterns are globbed against the artifact's own stage directory
    (build_root / basedir) so that "bin/**" only sees that artifact's bin/,
    not the entire merged dist tree.

    When stage dirs are absent, falls back to dist_root with fallback_excludes
    applied to compensate for the lack of scoping.

    force_include patterns bypass excludes and are always added if matched.
    When a component has no explicit include list and default_patterns is not
    disabled, COMPONENT_DEFAULT_INCLUDES provides the fallback patterns.
    """
    # Keyed by install_rel to deduplicate across artifacts.
    seen: dict[Path, Path] = {}

    def _add(install_rel: Path, source: Path) -> None:
        if install_rel not in seen:
            seen[install_rel] = source

    for artifact_name in artifact_names:
        descriptor = find_descriptor(artifact_name, repo_root)
        with open(descriptor, "rb") as f:
            data = tomllib.load(f)

        components_data: dict = data.get("components", {})

        for comp_type, basedirs in components_data.items():
            if comp_type not in components:
                continue
            if not isinstance(basedirs, dict):
                continue

            for basedir, spec in basedirs.items():
                if not isinstance(spec, dict):
                    continue

                stage_dir = build_root / basedir

                use_defaults = spec.get("default_patterns", True)
                explicit_includes: list[str] = spec.get("include", [])
                force_includes: list[str] = spec.get("force_include", [])
                excludes: list[str] = spec.get("exclude", [])

                if explicit_includes:
                    includes = explicit_includes
                elif use_defaults:
                    includes = COMPONENT_DEFAULT_INCLUDES.get(comp_type, [])
                else:
                    includes = []

                # Glob against the stage dir when available (precise scoping),
                # falling back to dist_root when the stage dir wasn't built.
                if stage_dir.is_dir():
                    search_root = stage_dir
                    extra_excludes: list[str] = []
                else:
                    print(
                        f"Warning: stage dir not found, falling back to dist root: {stage_dir}",
                        file=sys.stderr,
                    )
                    search_root = dist_root
                    extra_excludes = list(fallback_excludes)

                def _collect(pattern: str, bypass_excludes: bool = False) -> None:
                    for match in sorted(search_root.glob(pattern)):
                        if not match.is_file():
                            continue
                        # install_rel is relative to the search root (stage or
                        # dist), giving the flattened install layout.
                        install_rel = match.relative_to(search_root)
                        rel_posix = install_rel.as_posix()
                        if not bypass_excludes:
                            all_excludes = excludes + extra_excludes
                            if any(
                                match.match(ex) or rel_posix == ex
                                for ex in all_excludes
                            ):
                                continue
                        # Resolve the source path: prefer the dist_root copy
                        # when available (hard-linked, same inode), otherwise
                        # use the stage path directly.
                        dist_path = dist_root / install_rel
                        source = dist_path if dist_path.is_file() else match
                        _add(install_rel, source)

                for pattern in includes:
                    _collect(pattern)
                for pattern in force_includes:
                    _collect(pattern, bypass_excludes=True)

    return sorted(seen.items())


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
    default_dist = default_build / "dist" / "rocm"
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
            "CMake build directory (contains per-component stage trees). "
            f"Default: {default_build}. Ignored when --artifacts-url is set."
        ),
    )
    parser.add_argument(
        "--dist-root",
        type=Path,
        default=default_dist,
        metavar="PATH",
        help=(
            "Root of the merged ROCm distribution tree (build/dist/rocm/). "
            f"Default: {default_dist}"
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
        "--repo-root",
        type=Path,
        default=repo_root,
        metavar="PATH",
        help=(
            "Repo root used to locate artifact-*.toml descriptor files. "
            f"Default: {repo_root}"
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

    build_root = args.build_root
    if getattr(args, "artifacts_url", None):
        print(f"Fetching artifacts from {args.artifacts_url} ...")
        extracted = fetch_artifacts(
            artifacts_url=args.artifacts_url,
            artifact_names=pkg_def.artifacts,
            components=PACKAGE_COMPONENTS,
            dest_dir=args.artifacts_cache_dir,
        )
        # ArtifactPopulator extracts into {artifact}_{component}_generic/
        # Each archive contains paths like {basedir}/... matching the TOML keys.
        # We need build_root such that build_root / basedir resolves correctly.
        # ArtifactPopulator with flatten=False preserves the manifest relpaths,
        # so extracted/{artifact}_{component}_generic/{basedir}/... exists.
        # Merge all extracted artifact dirs into a single stage tree that
        # mirrors the layout the TOML basedir keys expect.  Each extracted
        # archive already contains paths like {basedir}/... so we can use
        # the merged dir directly as build_root.  We also use it as dist_root
        # so WiX Source= paths resolve correctly without a separate build.
        build_root = args.artifacts_cache_dir / "_stage"
        build_root.mkdir(parents=True, exist_ok=True)
        for artifact_dir in extracted.iterdir():
            if not artifact_dir.is_dir():
                continue
            for src in artifact_dir.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(artifact_dir)
                dst = build_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    dst.hardlink_to(src)
        # When using downloaded artifacts, stage and dist are the same tree.
        args = argparse.Namespace(**{**vars(args), "dist_root": build_root})

    files = collect_files_from_artifacts(
        dist_root=args.dist_root,
        artifact_names=pkg_def.artifacts,
        repo_root=args.repo_root,
        components=PACKAGE_COMPONENTS,
        build_root=build_root,
        fallback_excludes=pkg_def.fallback_excludes,
    )
    if not files:
        print(
            "Warning: no files collected. "
            "Run a Windows build or provide --artifacts-url.",
            file=sys.stderr,
        )

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
