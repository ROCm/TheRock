# generate_msi_wxs.py — Usage Guide

`build_tools/packaging/windows/generate_msi_wxs.py` reads artifact TOML
descriptors to determine which files each MSI should include, then produces a
WiX v4 `.wxs` source file.  Using artifact TOMLs as the source of truth means
the MSI file lists automatically track the build system's own packaging rules
with no separate manifest to maintain.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Or Python 3.9+ with `pip install tomli` |
| [WiX Toolset v4](https://wixtoolset.org/) | `dotnet tool install --global wix --version "4.*"` |
| `pyzstd` | Required for `--artifacts-url`: `pip install pyzstd` |
| Built TheRock or artifact URL | Local build or remote `.tar.zst` artifacts |

## Quick Start

### From a remote artifact URL (recommended)

```bat
:: Generate the WiX source from nightly artifacts
python build_tools\packaging\windows\generate_msi_wxs.py ^
    --package hip-runtime ^
    --artifacts-url https://therock-nightly-artifacts.s3.amazonaws.com/<run-id>-windows

:: Compile to MSI
wix build build_tools\packaging\windows\amdrocm-hip-runtime.wxs ^
    -o build_tools\packaging\windows\amdrocm-hip-runtime.msi
```

### From a local build

```bat
:: Generate (uses build/dist/rocm and build/<component>/stage/ automatically)
python build_tools\packaging\windows\generate_msi_wxs.py --package hip-runtime

:: Compile to MSI
wix build build_tools\packaging\windows\amdrocm-hip-runtime.wxs ^
    -o build_tools\packaging\windows\amdrocm-hip-runtime.msi
```

## Available Packages

```bat
python build_tools\packaging\windows\generate_msi_wxs.py --list
```

| Package | Output stem | Contents |
|---|---|---|
| `hip-runtime` | `amdrocm-hip-runtime` | HIP runtime DLLs, hipcc, hipconfig, kernel package support |
| `runtimes` | `amdrocm-runtimes` | HIP runtime + AMD LLVM compiler runtime (hipcc, comgr, device libs) |

## Options

### Package selection

| Flag | Description |
|---|---|
| `--package NAME` | Package to generate (required). Use `--list` to see options. |
| `--list` | Print available package names and descriptions, then exit. |

### Artifact source

| Flag | Default | Description |
|---|---|---|
| `--artifacts-url URL` | *(none)* | Base URL of a TheRock artifact storage directory containing `{name}_{component}_generic.tar.zst` files. When set, artifacts are downloaded, extracted, and used as precise stage trees. |
| `--artifacts-cache-dir PATH` | `<script-dir>/.artifact-cache` | Cache directory for downloaded and extracted artifacts. Reuse across runs to avoid re-downloading. |
| `--build-root PATH` | `build/` | CMake build directory containing per-component stage trees (`build/<basedir>/stage/`). Ignored when `--artifacts-url` is set. |
| `--dist-root PATH` | `build/dist/rocm` | Merged ROCm distribution tree. Used as the fallback search root when stage dirs are absent, and for resolving `Source=` paths in the generated WXS. |

### Output

| Flag | Default | Description |
|---|---|---|
| `--output PATH` | `<script-dir>/<output-stem>.wxs` | Destination path for the generated `.wxs` file. |

### Install location

The default install path is assembled as:

```
[install-root] \ [product-dir] \ [version-dir] \ <package-subdir>-<version> \
```

For example: `C:\Program Files\AMD\ROCm\hip-runtime-7.14.0\`

| Flag | Default | Description |
|---|---|---|
| `--install-root ROOT` | `ProgramFilesFolder` | Root of the install tree. Accepts a Windows Installer standard-directory token or an absolute path (e.g. `C:\AMD`). |
| `--product-dir NAME` | `AMD` | First subdirectory under `--install-root`. |
| `--version-dir NAME` | `ROCm` | Second subdirectory under `--product-dir`. |
| `--package-version X.Y.Z` | From `version.json` | MSI version string. Auto-detected from the repo's `version.json`. |

**Standard-directory tokens** resolve at install time on the target machine:

| Token | Resolves to |
|---|---|
| `ProgramFilesFolder` *(default)* | `C:\Program Files\` |
| `ProgramFiles64Folder` | `C:\Program Files\` (always 64-bit view) |

**Absolute paths** (e.g. `C:\AMD`) bake a fixed default into the MSI.

### `--repo-root PATH`

Used to locate `artifact-{name}.toml` descriptor files. Defaults to the repo
root inferred from the script location. Override only if running from an
unusual directory.

## How File Collection Works

For each artifact in a package, the generator:

1. Locates `artifact-{name}.toml` under `--repo-root`.
2. For each `run` and `lib` component entry, reads `include`, `exclude`, and
   `force_include` glob patterns.
3. Globs patterns against the artifact's **stage directory**
   (`build_root / basedir`) when available — this is the precise scope used by
   the build system's own artifact builder.
4. Falls back to `--dist-root` when stage dirs are absent (e.g. dist-only
   builds), applying `fallback_excludes` to suppress known noise from other
   artifacts present in the merged tree.
5. When `--artifacts-url` is set, downloads and extracts `.tar.zst` archives
   into `--artifacts-cache-dir` and uses those as the stage trees, bypassing
   both `--build-root` and the fallback entirely.

Files are installed **flat** — `bin/`, `lib/`, and `share/` are direct
children of `InstallDir`, regardless of the `basedir` path in the build tree.

## MSI Install Options

```bat
:: Silent install to default location
msiexec /i amdrocm-hip-runtime.msi /qn

:: Override install directory at install time
msiexec /i amdrocm-hip-runtime.msi /qn INSTALLFOLDER="C:\MyROCm"

:: Enable Windows long-path support
msiexec /i amdrocm-hip-runtime.msi /qn ENABLE_LONG_PATHS=1

:: Silent install with a log file
msiexec /i amdrocm-hip-runtime.msi /qn /l*v "%TEMP%\rocm-install.log"
```

## Registry Entries

| Key | Name | Value | Condition |
|---|---|---|---|
| `HKLM\Software\AMD\ROCm\<package>\<version>` | `InstallDir` | Install path | Always |
| `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem` | `LongPathsEnabled` | `1` | Only if `ENABLE_LONG_PATHS=1` |

Standard MSI uninstall entries are also written under
`HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\`.

## Upgrade Policy

| Scenario | Behaviour |
|---|---|
| No prior installation | Installs normally |
| Older version present | Old version removed automatically, new version installs |
| Same version present | Repairs in place |
| Newer version present | Blocked with an error message |

## Adding a New Package

Add an entry to the `PACKAGES` dict in `generate_msi_wxs.py`:

```python
"my-package": PackageDef(
    description="One-line description for --list",
    product_name="AMD ROCm My Package",
    artifacts=["artifact-name-1", "artifact-name-2"],
    output_stem="amdrocm-my-package",
    install_subdir="my-package-{version}",
    upgrade_code="<new unique GUID>",        # never reuse an existing GUID
    feature_id="MyPackage",
    feature_title="AMD ROCm My Package",
    registry_key="Software\\AMD\\ROCm\\my-package\\{version}",
),
```

Each package must have a unique `upgrade_code` GUID, `feature_id`, and
`output_stem`.

## Running the Tests

```bat
python -m unittest discover -s build_tools\packaging\windows -p "*_test.py" -v
```
