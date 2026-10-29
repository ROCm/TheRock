# ROCm Windows MSI Packaging via TheRock

Given ROCm artifact directories, selects a file set and generates WiX v4 source
(`.wxs`) that compiles into machine-scoped Windows MSI installers.

This process reuses the build system's own artifact manifests as the source of
truth for file selection, then emits the MSI-specific structure (WiX XML,
upgrade codes, PATH/registry configuration) on top.

## General Design

The generator (`build_tools/packaging/windows/generate_msi_wxs.py`) uses
`ArtifactCatalog` from `_therock_utils.artifacts` to read the
`artifact_manifest.txt` embedded in each artifact archive, and emits a WiX v4
`.wxs` file listing exactly those files. The artifact archives are the
authoritative packaging output; the MSI generator consumes that output rather
than reimplementing the packaging rules.

Each MSI package is defined by a `PackageDef` entry in the `PACKAGES` table,
which names the artifacts to include (and, optionally, component groups and
per-artifact file scoping). The set of packages is expected to grow over time —
adding one is a single `PackageDef` entry with no WiX source authored and no
file lists maintained. See
[Adding a New Package](/build_tools/packaging/windows/msi-generator-usage.md#adding-a-new-package).

Consequences of this approach:

- **No hand-maintained file lists.** File classification lives in the
  `artifact-{name}.toml` descriptors owned by each component team. When a
  component adds, removes, or renames a file, the descriptor is updated as part
  of that component's development and every MSI that includes that artifact
  picks up the change automatically — there is no second place to update. This
  is the same source of truth that drives the Linux `.deb`/`.rpm` and Python
  packaging, so file sets stay consistent across platforms.
- **Stage-scoped selection.** `ArtifactCatalog` enumerates files per artifact
  stage directory, so an artifact's `bin/**` only sees that artifact's `bin/`,
  not the merged distribution tree. Cross-artifact bleed is prevented by
  construction, with no explicit exclusion lists.
- **Component-group filtering.** A package selects which component groups
  (`run`, `lib`, `dev`, `doc`, `test`, ...) it wants from its artifacts, so a
  runtime package can take just `run`/`lib` while a future development package
  could additionally take `dev`.
- **Per-artifact scoping when needed.** `per_artifact_includes` on `PackageDef`
  narrows a single artifact to a subset of its files, for cases where a package
  wants only a specific DLL out of an otherwise-large artifact.

Each package installs to a versioned subdirectory under
`C:\Program Files\AMD\ROCm\`, enabling side-by-side installation of multiple
ROCm versions and clean upgrades via WiX `MajorUpgrade`. Each package carries a
fixed `upgrade_code` GUID that must never change after first release — Windows
Installer uses it to locate and remove prior installs of the same product line.

The currently defined packages are listed by `--package`/`--list`; see the
[usage guide](/build_tools/packaging/windows/msi-generator-usage.md#available-packages)
for the up-to-date set.

## Building Packages

The generator runs from either remote CI artifacts or a local build, then WiX
compiles the emitted `.wxs` into an MSI:

```bat
:: Generate WiX source from nightly CI artifacts
python build_tools\packaging\windows\generate_msi_wxs.py ^
    --package <name> ^
    --artifacts-url https://therock-nightly-artifacts.s3.amazonaws.com/<run-id>-windows

:: Compile to MSI (requires WiX v4: dotnet tool install --global wix --version "4.*")
wix build build_tools\packaging\windows\<output-stem>.wxs -arch x64 ^
    -o <output-stem>.msi
```

For the local-build workflow, all generator options, the file-collection
details, and package-authoring instructions, see the
[MSI generator usage guide](/build_tools/packaging/windows/msi-generator-usage.md).

## Using Packages

```bat
:: Silent install to the default location
msiexec /i <package>.msi /qn

:: Silent uninstall
msiexec /x <package>.msi /qn
```

Installation adds `InstallDir\bin` to the machine `PATH` (removed on uninstall)
so ROCm binaries and DLLs are discoverable without manual configuration. For
install/upgrade/uninstall/troubleshooting details, install-location overrides,
and the long-path and legacy-System32 options, see the
[Windows installer README](/build_tools/packaging/windows/README.md).

## Implementation Status

The core generator and package definitions are functional and tested against CI
artifacts. The following are scoped additions requiring no architectural
changes, to be addressed before public distribution:

- **`dep:Requires` declarations** — inter-package dependencies (e.g. a `-dev`
  package depending on a runtime package) are not yet declared. The
  WixToolset.Dependency extension is already included in the WiX build command.
- **CI workflow integration** — a GitHub Actions job to run the generator and
  `wix build` on CI artifacts is not yet merged into the release pipeline.
- **Upgrade code finalization** — current upgrade codes are placeholder GUIDs
  that must be replaced with random UUIDs before first public release, and never
  changed afterward.
- **RFC alignment** — once the Windows packaging RFC (PR #3973) lands, install
  paths, registry keys, and discovery mechanisms will be reconciled with it.
