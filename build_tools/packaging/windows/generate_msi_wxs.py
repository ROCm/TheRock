#!/usr/bin/env python3
"""Generate a WiX v4 .wxs file for the ROCm runtime MSI installer.

This script inspects the built ROCm distribution tree (build/dist/rocm/) and
produces a WiX v4 XML source file (.wxs) that describes every file, component,
directory, and feature needed to build a silent MSI installer.

The generated .wxs is then compiled into an MSI by running:
    wix build amdrocm-runtimes.wxs -o amdrocm-runtimes.msi

The resulting MSI requires no user interaction (no UI sequences are defined),
making it suitable for scripted or enterprise deployments:
    msiexec /i amdrocm-runtimes.msi /qn

Installation location
---------------------
The default install path is controlled at build time via this script's flags
(--install-root, --product-dir, --version-dir).  See --help for details.
"""

import argparse
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Package scope — which subdirectories of dist/rocm/ to include
# ---------------------------------------------------------------------------

# "bin" contains runtime DLLs (.dll) and executables (.exe, .bat, scripts).
# "lib" contains import libraries (.lib) needed to link against the DLLs at
# build time. Both are considered part of the "runtime" package; dev headers
# and CMake config files (include/, lib/cmake/) are excluded.
RUNTIME_DIRS = ["bin", "lib"]

# Some files in the dist tree are internal build bookkeeping artifacts that
# have no meaning outside the source tree and should not be shipped.
SKIP_FILES: set[str] = set()

# Wildcard patterns for DLLs that legacy ROCm/HIP installers placed directly
# into C:\Windows\System32\.  They shadow the versioned DLLs installed by this
# package into Program Files, so they must be removed before InstallFiles runs.
LEGACY_SYSTEM32_PATTERNS = [
    "amdhip64_*.dll",
    "amd_comgr_*.dll",
]

# The UpgradeCode GUID identifies this product family across all versions.
# Windows Installer matches it during upgrades to find prior installations.
# It must NEVER change between releases of the same product line.
UPGRADE_CODE = "A1B2C3D4-E5F6-7890-ABCD-EF1234567890"


# ---------------------------------------------------------------------------
# Helper: stable WiX element IDs
# ---------------------------------------------------------------------------

def make_id(path: Path, prefix: str) -> str:
    """Return a stable, WiX-legal element ID derived from a relative file path.

    WiX v4 imposes two constraints on element IDs:
      - Characters must be alphanumeric or underscores (no dots, hyphens, etc.)
      - Maximum length is 72 characters

    Strategy:
      1. Replace every non-alphanumeric character in the path with an underscore.
      2. Prepend a short prefix ("f" for File elements, "c" for Component
         elements) so file and component IDs for the same path are distinct.
      3. Append an 8-hex-digit hash of the original path string so that even
         after truncation to 55 characters the ID remains unique across files
         whose sanitised names would otherwise collide (e.g. foo-bar vs foo_bar).

    The hash uses Python's built-in hash() masked to 32 bits. This is NOT
    cryptographic — it only needs to be collision-resistant across the ~40 files
    in this package.
    """
    # Replace every character that WiX does not allow in identifiers
    safe = str(path).replace("\\", "_").replace("/", "_").replace(".", "_").replace("-", "_")

    # Build the full candidate identifier with its prefix
    ident = f"{prefix}_{safe}"

    # Compute a short hash of the original (unsanitised) path for uniqueness
    h = format(hash(str(path)) & 0xFFFFFFFF, "08x")

    # Keep the total length at or below 72 chars: 55 body chars + "_" + 8 hash
    return f"{ident[:55]}_{h}"


# ---------------------------------------------------------------------------
# Helper: load the explicit file allowlist
# ---------------------------------------------------------------------------

def load_file_list(path: Path) -> set[str]:
    """Return the set of allowed relative paths from a file-list text file.

    Each non-blank, non-comment line is a forward-slash relative path from the
    dist root (e.g. 'bin/amdhip64_7.dll').  Lines starting with '#' are ignored.
    Backslashes are normalised to forward slashes so the file is
    platform-neutral.
    """
    allowed: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                allowed.add(line.replace("\\", "/"))
    return allowed


# ---------------------------------------------------------------------------
# Helper: enumerate installable files in a dist subdirectory
# ---------------------------------------------------------------------------

def collect_files(dist_root: Path, subdir: str, allowed: set[str]) -> list[Path]:
    """Return a sorted list of allowed files in dist_root/<subdir>.

    Only files whose forward-slash path relative to dist_root appears in
    `allowed` are returned.  Returns an empty list if the directory does not
    exist.
    """
    d = dist_root / subdir
    if not d.is_dir():
        return []

    files = []
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        rel = f.relative_to(dist_root)
        if str(rel).replace("\\", "/") in allowed:
            files.append(f)
    return files


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments.

    Installation location flags
    ---------------------------
    The default install path is assembled from three independent pieces so that
    each can be overridden without affecting the others:

        [install-root] \\ [product-dir] \\ [version-dir] \\

    Examples:
        Default (no flags):
            C:\\Program Files\\ROCm\\7.13\\

        --product-dir ROCm-HIP --version-dir 7.2:
            C:\\Program Files\\ROCm-HIP\\7.2\\

        --install-root "C:\\AMD":
            C:\\AMD\\ROCm\\7.13\\

        --install-root "C:\\AMD" --product-dir HIP --version-dir 7:
            C:\\AMD\\HIP\\7\\

    The install-root flag accepts either a Windows Installer standard-directory
    token (e.g. "ProgramFilesFolder", "ProgramFiles64Folder") or an absolute
    path (e.g. "C:\\AMD").  When an absolute path is given, the MSI uses it as
    a fixed default baked into the directory tree.
    """
    # Script-relative defaults so the script is location-independent.
    # Script lives at build_tools/packaging/windows/ — go up three levels to
    # reach the repo root.
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent.parent
    default_dist = repo_root / "build" / "dist" / "rocm"
    default_output = script_dir / "amdrocm-runtimes.wxs"
    default_file_list = script_dir / "amdrocm-runtimes-artifacts.txt"

    parser = argparse.ArgumentParser(
        description="Generate a WiX v4 .wxs for the ROCm runtime MSI installer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # -- Source / output paths -----------------------------------------------

    parser.add_argument(
        "--dist-root",
        type=Path,
        default=default_dist,
        metavar="PATH",
        help=(
            "Root of the built ROCm distribution tree to harvest files from. "
            "Must contain bin/ and lib/ subdirectories. "
            f"Default: {default_dist}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        metavar="PATH",
        help=(
            "Destination path for the generated .wxs file. "
            f"Default: {default_output}"
        ),
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=default_file_list,
        metavar="PATH",
        help=(
            "Text file listing the dist-root-relative paths of files to include "
            "in the MSI, one per line (forward slashes, '#' comments ignored). "
            "Only files present in both this list and the dist tree are packaged. "
            f"Default: {default_file_list}"
        ),
    )

    # -- Install location ----------------------------------------------------

    parser.add_argument(
        "--install-root",
        default="ProgramFilesFolder",
        metavar="ROOT",
        help=(
            "Where to root the install tree. Accepts either a Windows Installer "
            "standard-directory token or an absolute path.\n\n"
            "Standard-directory tokens resolve at install time to system "
            "directories, making the MSI portable across machines:\n"
            "  ProgramFilesFolder   -> C:\\Program Files\\ (default)\n"
            "  ProgramFiles64Folder -> C:\\Program Files\\ (always 64-bit)\n"
            "  SystemFolder         -> C:\\Windows\\System32\\\n\n"
            "An absolute path (e.g. C:\\AMD) bakes a fixed default into the MSI.\n\n"
            "Default: ProgramFilesFolder"
        ),
    )
    parser.add_argument(
        "--product-dir",
        default="AMD",
        metavar="NAME",
        help=(
            "Name of the first subdirectory created under --install-root. "
            "Default: AMD  ->  [install-root]\\AMD\\"
        ),
    )
    parser.add_argument(
        "--version-dir",
        default="ROCm",
        metavar="NAME",
        help=(
            "Name of the second subdirectory created under --product-dir. "
            "A runtimes-{package-version} subdirectory is always appended beneath it. "
            "Default: ROCm  ->  [install-root]\\AMD\\ROCm\\runtimes-{package-version}\\"
        ),
    )

    # -- Package metadata ----------------------------------------------------

    parser.add_argument(
        "--package-version",
        default="7.13.0",
        metavar="X.Y.Z",
        help=(
            "Four-part MSI version string (only the first three parts are used "
            "by Windows Installer for upgrade comparisons). "
            "Default: 7.13.0"
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main WXS builder
# ---------------------------------------------------------------------------

def build_wxs(args: argparse.Namespace) -> None:
    """Build the complete WiX XML document and write it to args.output.

    WiX v4 document structure overview
    ------------------------------------
    <Wix>                        — root namespace declaration
      <Package>                  — MSI package metadata (name, version, upgrade
                                   code, compression, etc.)
        <SummaryInformation>     — populates the MSI property table shown in
                                   Programs and Features / Add-Remove Programs
        <MajorUpgrade>           — automatic upgrade/downgrade policy
        <MediaTemplate>          — how files are stored inside the MSI
                                   (EmbedCab="yes" -> single self-contained file)
        <StandardDirectory|      — anchors the install tree; either a well-known
         Directory (TARGETDIR)>    token or a custom absolute-path root
          <Directory> ...        — nested directory structure under that root
            <Component>          — atomic install unit; each file gets its own
              <File>             — the actual file to copy to disk
        <Feature>                — top-level install feature referencing all
          <ComponentRef>           components; Windows Installer requires every
                                   component to belong to at least one feature
    """

    # Determine whether --install-root is a known WI standard-directory token
    # or an arbitrary absolute path the user wants as the fixed default root.
    # Known tokens are passed verbatim as StandardDirectory/@Id; absolute paths
    # require a TARGETDIR -> custom-root -> product -> version directory chain.
    STANDARD_DIR_TOKENS = {
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
    use_standard_dir = args.install_root in STANDARD_DIR_TOKENS

    # Load the explicit file allowlist — only files listed here are packaged.
    allowed = load_file_list(args.file_list)

    # Register the WiX v4 namespace as the default so ET does not emit "ns0:"
    # prefixes throughout the generated XML.
    ET.register_namespace("", "http://wixtoolset.org/schemas/v4/wxs")
    ns = "http://wixtoolset.org/schemas/v4/wxs"

    # -----------------------------------------------------------------------
    # Root element
    # -----------------------------------------------------------------------
    root = ET.Element(f"{{{ns}}}Wix")

    # -----------------------------------------------------------------------
    # Package element — core MSI metadata
    # -----------------------------------------------------------------------
    # UpgradeCode: a fixed GUID that identifies this product across versions.
    # Windows Installer uses it to detect prior installations during upgrades.
    # It must NEVER change between releases of the same product line.
    #
    # InstallerVersion="500" requires Windows Installer 5.0 (Windows 7+), which
    # enables per-machine installation and 64-bit support without extra flags.
    #
    # Compressed="yes" instructs WiX to embed all files in a cabinet (.cab)
    # stored inside the MSI itself rather than as loose files alongside it.
    pkg = ET.SubElement(root, f"{{{ns}}}Package",
        Name="ROCm Runtime",
        Version=args.package_version,
        Manufacturer="Advanced Micro Devices, Inc.",
        UpgradeCode=UPGRADE_CODE,
        Language="1033",        # English (United States)
        Codepage="1252",        # Western European / Windows-1252
        InstallerVersion="500",
        Compressed="yes",
    )

    # -----------------------------------------------------------------------
    # SummaryInformation — MSI property stream metadata
    # -----------------------------------------------------------------------
    # This populates the summary property stream embedded in every MSI.  The
    # Description field appears in Programs and Features under "Comment".
    # No UI sequences are referenced anywhere in this file — the MSI is
    # intentionally headless and must be invoked with /qn for silent install.
    ET.SubElement(pkg, f"{{{ns}}}SummaryInformation",
        Keywords="ROCm HIP AMD GPU runtime",
        Description=f"ROCm {args.package_version} Runtime (bin + lib)",
    )

    # -----------------------------------------------------------------------
    # MajorUpgrade — upgrade / downgrade policy
    # -----------------------------------------------------------------------
    # Tells Windows Installer to automatically uninstall any previously
    # installed version of this product (matched by UpgradeCode) before
    # installing the new one.  This covers both in-place upgrades (higher
    # version) and the downgrade guard (DowngradeErrorMessage is surfaced by
    # msiexec if a user tries to install an older version over a newer one).
    ET.SubElement(pkg, f"{{{ns}}}MajorUpgrade",
        DowngradeErrorMessage="A newer version of ROCm Runtime is already installed.",
    )

    # -----------------------------------------------------------------------
    # MediaTemplate — cabinet (CAB) embedding strategy
    # -----------------------------------------------------------------------
    # EmbedCab="yes" merges the cabinet into the MSI stream so the installer
    # is a single portable file.  WiX will automatically split into multiple
    # cabinets if the content exceeds the MSI stream size limit (~2 GB).
    ET.SubElement(pkg, f"{{{ns}}}MediaTemplate", EmbedCab="yes")

    # -----------------------------------------------------------------------
    # Public properties
    # -----------------------------------------------------------------------
    # ENABLE_LONG_PATHS: pass on the msiexec command line to opt in to setting
    # HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1.
    # Secure="yes" marks it as a public property so Windows Installer allows
    # it to be passed on the command line:
    #     msiexec /i amdrocm-runtimes.msi /qn ENABLE_LONG_PATHS=1
    ET.SubElement(pkg, f"{{{ns}}}Property",
        Id="ENABLE_LONG_PATHS",
        Secure="yes",
    )

    # -----------------------------------------------------------------------
    # Legacy System32 cleanup — remove old ROCm DLLs before installing
    # -----------------------------------------------------------------------
    # Older ROCm/HIP Windows installers dropped versioned DLLs (e.g.
    # amdhip64_6.dll, amd_comgr_2.dll) directly into C:\Windows\System32\.
    # Those files take DLL search-order priority over Program Files and will
    # shadow this package's DLLs if left in place.
    #
    # Two-action pattern to reliably delete files from System32 under elevation:
    #
    # 1. SetProperty (immediate, execute sequence):
    #    Runs during the script-generation phase when session properties ARE
    #    available.  Captures [System64Folder]cmd.exe into CMD_EXE so the
    #    resolved path is ready for the deferred action below.
    #
    # 2. CustomAction Type 50 (Property + ExeCommand, deferred):
    #    Property    — CMD_EXE, read from the session at script-generation time
    #                  and embedded in the deferred script as the executable path.
    #    ExeCommand  — del arguments with fully-qualified [System64Folder] paths.
    #                  Property references are formatted at script-generation time
    #                  (immediate phase) so the absolute paths are baked into the
    #                  script before the deferred action runs.
    #    Execute     — "deferred" runs in the elevated Windows Installer service.
    #    Impersonate — "no" retains the SYSTEM token; required for System32 writes.
    #    Return      — "ignore" tolerates "no matching files" exit codes from del.
    #
    # Using Type 34 (Directory + ExeCommand) was previously attempted but the
    # working directory property is not reliably resolved at deferred-execution
    # time, so relative del patterns ran against the wrong directory.
    ET.SubElement(pkg, f"{{{ns}}}SetProperty",
        Id="CMD_EXE",
        Value="[System64Folder]cmd.exe",
        Before="RemoveLegacyROCmDlls",
        Sequence="execute",
    )

    del_args = " ".join(
        f'"[System64Folder]{p}"' for p in LEGACY_SYSTEM32_PATTERNS
    )
    ET.SubElement(pkg, f"{{{ns}}}CustomAction",
        Id="RemoveLegacyROCmDlls",
        Property="CMD_EXE",
        ExeCommand=f"/c del /f /q {del_args} 2>nul",
        Execute="deferred",
        Impersonate="no",
        Return="ignore",
    )

    # Schedule RemoveLegacyROCmDlls after InstallInitialize (elevated execute
    # sequence) and before InstallFiles so System32 is clean before new DLLs
    # are written.  SetProperty is ordered Before="RemoveLegacyROCmDlls" above.
    seq = ET.SubElement(pkg, f"{{{ns}}}InstallExecuteSequence")
    ET.SubElement(seq, f"{{{ns}}}Custom",
        Action="RemoveLegacyROCmDlls",
        After="InstallInitialize",
    )

    # -----------------------------------------------------------------------
    # Directory tree — install location
    # -----------------------------------------------------------------------
    # Two cases depending on whether --install-root is a standard-directory
    # token or an absolute path:
    #
    # Case A — standard-directory token (e.g. ProgramFilesFolder):
    #   <StandardDirectory Id="ProgramFilesFolder">
    #     <Directory Id="ROCmDir"       Name="AMD">
    #       <Directory Id="ROCmVerDir"  Name="ROCm">
    #         <Directory Id="RuntimeDir" Name="runtimes-7.13.0">
    #
    # Case B — absolute path (e.g. C:\AMD):
    #   <Directory Id="TARGETDIR" Name="SourceDir">   ← WI root anchor (required)
    #     <Directory Id="CustomRoot" Name="C:\AMD">   ← fixed absolute root
    #       <Directory Id="ROCmDir"       Name="AMD">
    #         <Directory Id="ROCmVerDir"  Name="ROCm">
    #           <Directory Id="RuntimeDir" Name="runtimes-7.13.0">
    #
    # In both cases RuntimeDir is the deepest directory — the one that bin/ and
    # lib/ live under — and its Id is referenced by the PATH component below.
    if use_standard_dir:
        # Anchor to a well-known system directory resolved by Windows Installer.
        install_dir = ET.SubElement(pkg, f"{{{ns}}}StandardDirectory",
            Id=args.install_root)
        rocm_dir = ET.SubElement(install_dir, f"{{{ns}}}Directory",
            Id="ROCmDir", Name=args.product_dir)
    else:
        # Anchor to TARGETDIR (mandatory WI root), then add the absolute-path
        # custom root as a child directory.  The Name attribute on a TARGETDIR
        # child that starts with a drive letter is treated as an absolute path
        # by Windows Installer.
        targetdir = ET.SubElement(pkg, f"{{{ns}}}Directory",
            Id="TARGETDIR", Name="SourceDir")
        custom_root = ET.SubElement(targetdir, f"{{{ns}}}Directory",
            Id="CustomInstallRoot", Name=args.install_root)
        rocm_dir = ET.SubElement(custom_root, f"{{{ns}}}Directory",
            Id="ROCmDir", Name=args.product_dir)

    # Product subdirectory — common to both cases
    ver_dir = ET.SubElement(rocm_dir, f"{{{ns}}}Directory",
        Id="ROCmVerDir", Name=args.version_dir)

    # Runtimes subdirectory — appended beneath the product dir, named
    # "runtimes-{package_version}" so that multiple package types (e.g. dev,
    # runtimes) can coexist under the same ROCm product directory side by side.
    runtime_dir_name = f"runtimes-{args.package_version}"
    runtime_dir = ET.SubElement(ver_dir, f"{{{ns}}}Directory",
        Id="RuntimeDir", Name=runtime_dir_name)

    # -----------------------------------------------------------------------
    # Feature — top-level install feature
    # -----------------------------------------------------------------------
    # Windows Installer requires all components to be attached to at least one
    # feature.  Level="1" means the feature is selected by default.  Because
    # there is no UI, the feature selection state is never shown to the user;
    # all Level=1 features are always installed.
    feature = ET.SubElement(pkg, f"{{{ns}}}Feature",
        Id="Runtime", Title="ROCm Runtime", Level="1")

    # -----------------------------------------------------------------------
    # Components — one per file
    # -----------------------------------------------------------------------
    # Windows Installer's reference-counting and repair mechanisms operate at
    # the Component granularity.  Best practice (and the WiX recommendation)
    # is one file per component so that each file can be independently tracked,
    # repaired, or removed.
    #
    # Each Component needs:
    #   Guid  — a stable per-component UUID.  We derive it deterministically
    #           from the relative install path using uuid5 (SHA-1 in the URL
    #           namespace) so it remains the same across script runs, which is
    #           required for upgrades to work correctly.
    #   KeyPath="yes" on the File — tells Windows Installer to use this file's
    #           presence/version as the authoritative indicator of whether the
    #           component is installed.
    for subdir in RUNTIME_DIRS:
        files = collect_files(args.dist_root, subdir, allowed)
        if not files:
            continue  # skip silently if a subdir doesn't exist in this build

        # Create the <Directory> node for this subdir under the runtimes dir
        dir_id = f"Dir_{subdir.replace('-', '_')}"
        sub_dir_el = ET.SubElement(runtime_dir, f"{{{ns}}}Directory",
            Id=dir_id, Name=subdir)

        for fpath in files:
            # Paths relative to dist_root are used as the stable identity for
            # ID and GUID generation (e.g. "bin\amdhip64_7.dll")
            rel = fpath.relative_to(args.dist_root)

            file_id = make_id(rel, "f")   # unique ID for the <File> element
            comp_id = make_id(rel, "c")   # unique ID for the <Component> element

            # uuid5 produces a deterministic UUID from a name string.
            # Using NAMESPACE_URL is conventional; the actual namespace URI
            # does not matter as long as it is consistent.
            guid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(rel))).upper()

            comp_el = ET.SubElement(sub_dir_el, f"{{{ns}}}Component",
                Id=comp_id,
                Guid=guid,
            )
            ET.SubElement(comp_el, f"{{{ns}}}File",
                Id=file_id,
                # Source is the path on the build machine where WiX reads the
                # file from; backslashes are required by the WiX compiler.
                Source=str(fpath).replace("/", "\\"),
                Name=fpath.name,
                KeyPath="yes",
            )

            # Wire this component into the feature so Windows Installer knows
            # to install it when the "Runtime" feature is selected.
            ET.SubElement(feature, f"{{{ns}}}ComponentRef", Id=comp_id)

    # -----------------------------------------------------------------------
    # PATH environment variable + registry install-dir marker
    # -----------------------------------------------------------------------
    # A dedicated lightweight component handles two post-install side effects:
    #
    #   1. Environment / PATH:
    #      Appends [ROCmVerDir]bin to the system PATH so that hipcc.exe,
    #      hipconfig.exe, clinfo.exe, etc. are discoverable from any command
    #      prompt after installation.
    #        Part="last"    -> append rather than prepend
    #        Permanent="no" -> remove the entry on uninstall
    #        System="yes"   -> modify the machine-wide PATH (HKLM), not per-user
    #
    #   2. Registry key:
    #      Records the install directory under HKLM\Software\AMD\ROCm\<version>
    #      so that other software (e.g. IDEs, build systems) can discover the
    #      ROCm root without parsing PATH.  This registry value also serves as
    #      the KeyPath for the component, which is the recommended pattern when
    #      a component does not contain a File element.
    bin_files = collect_files(args.dist_root, "bin", allowed)
    if bin_files:
        env_comp_id = "EnvPath"
        env_comp = ET.SubElement(runtime_dir, f"{{{ns}}}Component",
            Id=env_comp_id,
            # Stable GUID derived from a fixed string — must not change between
            # versions so Windows Installer can track this component correctly.
            Guid=str(uuid.uuid5(uuid.NAMESPACE_URL, "ROCm_PATH_component")).upper(),
        )

        # Append [RuntimeDir]\bin to the system PATH.
        # [RuntimeDir] is a Windows Installer directory property that resolves
        # to the actual install path at runtime (e.g. C:\Program Files\AMD\ROCm\runtimes-7.13.0\).
        ET.SubElement(env_comp, f"{{{ns}}}Environment",
            Id="ROCmBinPath",
            Name="PATH",
            Value="[RuntimeDir]bin",
            Permanent="no",   # undo on uninstall
            Part="last",      # append to existing PATH entries
            Action="set",
            System="yes",     # system-wide (HKLM), not per-user (HKCU)
        )

        # Registry value: install directory marker + component KeyPath.
        # KeyPath="yes" on a RegistryValue means Windows Installer checks for
        # this key's existence (not a file) to decide if the component is
        # already installed.  The registry key path includes the version subdir
        # name so that multiple ROCm versions can coexist in the registry.
        ET.SubElement(env_comp, f"{{{ns}}}RegistryValue",
            Root="HKLM",
            Key=f"Software\\AMD\\ROCm\\{args.package_version}",
            Name="InstallDir",
            Value="[RuntimeDir]",
            Type="string",
            KeyPath="yes",
        )

        ET.SubElement(feature, f"{{{ns}}}ComponentRef", Id=env_comp_id)

    # -----------------------------------------------------------------------
    # Long-paths enablement (optional)
    # -----------------------------------------------------------------------
    # Windows 10 1607+ supports paths longer than MAX_PATH (260 characters)
    # via a single DWORD registry value.  ROCm build trees and Python paths
    # frequently exceed 260 characters, so this is strongly recommended.
    #
    # The value is written under HKLM and requires elevation, which this
    # per-machine MSI already runs with.  A reboot is needed for the setting
    # to take effect in all processes, but most new processes launched after
    # the installer exits will already see it.
    #
    # The component lives in a dedicated Feature with Level="0" (off by default).
    # A Condition child raises the level to 1 (on) when ENABLE_LONG_PATHS=1 is
    # passed on the msiexec command line:
    #     msiexec /i amdrocm-runtimes.msi /qn ENABLE_LONG_PATHS=1
    # This is the WiX v4 idiomatic way to make a component install-time optional
    # without using the removed <Condition> child of <Component>.
    lp_comp_id = "LongPathsEnable"
    lp_comp = ET.SubElement(runtime_dir, f"{{{ns}}}Component",
        Id=lp_comp_id,
        Guid=str(uuid.uuid5(uuid.NAMESPACE_URL, "ROCm_LongPaths_component")).upper(),
    )
    # KeyPath="yes" — Windows Installer uses this registry value's presence
    # to determine whether the component is installed or needs repair.
    ET.SubElement(lp_comp, f"{{{ns}}}RegistryValue",
        Root="HKLM",
        Key="SYSTEM\\CurrentControlSet\\Control\\FileSystem",
        Name="LongPathsEnabled",
        Value="1",
        Type="integer",
        KeyPath="yes",
    )

    # Separate feature, off by default (Level="0"), turned on by the Level child.
    # In WiX v4 the old <Condition Level="N"> child is replaced by
    # <Level Value="N" Condition="..."/> — when the condition is true at install
    # time Windows Installer raises the feature level from 0 (absent) to 1 (install).
    lp_feature = ET.SubElement(pkg, f"{{{ns}}}Feature",
        Id="LongPaths", Title="Enable Long Paths", Level="0")
    ET.SubElement(lp_feature, f"{{{ns}}}Level",
        Value="1", Condition="ENABLE_LONG_PATHS")
    ET.SubElement(lp_feature, f"{{{ns}}}ComponentRef", Id=lp_comp_id)

    # -----------------------------------------------------------------------
    # Serialise the XML document to disk
    # -----------------------------------------------------------------------
    # ET.indent() (Python 3.9+) adds whitespace text nodes for human-readable
    # formatting.  The indent() stub below is a no-op kept for compatibility
    # documentation; the actual formatting is done by ET.indent().
    indent(root)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write a proper XML declaration separately because ElementTree's built-in
    # xml_declaration=True uses single quotes around the encoding value, which
    # some older MSI tooling rejects.  Writing it manually guarantees standard
    # double-quoted form: <?xml version="1.0" encoding="UTF-8"?>
    with open(args.output, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding="utf-8", xml_declaration=False)

    # Report the effective default install path so it's visible in CI logs
    print(f"Written: {args.output}")
    print(f"Default install root : {args.install_root}")
    print(f"Default install path : [install-root]\\{args.product_dir}\\{args.version_dir}\\runtimes-{args.package_version}\\")


def indent(elem: ET.Element, level: int = 0) -> None:
    """No-op stub retained for documentation purposes.

    Python 3.9 introduced ET.indent() which is called directly in build_wxs().
    This function exists only to make it explicit that pretty-printing is
    intentionally delegated to the standard library rather than implemented
    here.  On Python < 3.9 the output would be a single long line, but WiX
    does not require formatted input so the MSI build would still succeed.
    """
    pass  # ET.indent() in build_wxs() handles all formatting


if __name__ == "__main__":
    build_wxs(parse_args())
