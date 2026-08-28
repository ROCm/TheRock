# Windows MSI Packaging Design

This document explains the design decisions behind TheRock's Windows MSI
packaging infrastructure and why the chosen approach best fits the project's
architecture.

## Background

ROCm on Windows has historically required users to manually copy DLLs and
configure PATH — a process that is error-prone and unsuitable for enterprise
deployment. The goal of the MSI packaging work is to produce machine-scoped
Windows installers that support silent installation, automatic upgrades, and
clean uninstall via standard Windows tooling (`msiexec`).

## Core Design Principle: Artifact Manifests as Source of Truth

TheRock's build system already defines which files belong to which distribution
group through `artifact-{name}.toml` descriptor files. These descriptors are
processed at build time by `artifact_builder.py` to produce artifact archives
(`.tar.zst`) each containing an embedded `artifact_manifest.txt`. The manifest
records precisely which files belong to the artifact — this work has already
been done.

The MSI generator (`generate_msi_wxs.py`) exploits this by using
`ArtifactCatalog` from `_therock_utils.artifacts` to read those manifests
directly. This means:

- **No TOML parsing at generation time** — the generator does not re-read or
  re-evaluate the descriptor files that were already used to build the archives
- **No manual file lists** — there are no hand-maintained lists of DLLs, headers,
  or paths to keep in sync with the build system
- **No glob re-evaluation** — the generator does not re-run the same include/exclude
  patterns that `artifact_builder.py` already applied

The artifact archives are the authoritative packaging output. The MSI generator
is a consumer of that output, not a reimplementation of the packaging rules.

## Why Not Hand-Authored WiX Files?

An alternative approach (explored in PR #5712) involves hand-authoring WiX
source files (`.wxs`, `.wxi`) that list every file to be installed. This was
rejected for several reasons:

**Maintenance burden.** A 108-file WiX source tree must be manually updated
every time a component adds, removes, or renames a file. The build system's
TOML descriptors already track this — duplicating the information in hand-authored
WiX files creates two sources of truth that will diverge.

**Platform drift.** The Linux packaging (`.deb`, `.rpm`, Python wheels) is
driven by the same artifact TOML descriptors. Hand-authored Windows file lists
have no connection to those descriptors, making it difficult to ensure Windows
and Linux packages contain equivalent file sets.

**Hardcoded paths.** Static WiX files require hardcoded build tree paths (e.g.
`B:\build\math-libs\BLAS\hipBLAS\dist\bin\hipblas.dll`). These are
machine-specific and break on any other build environment.

**Version hardcoding.** Static WiX files require hardcoded version strings that
must be manually updated each release and break upgrade detection if forgotten.

## Why ArtifactCatalog?

`ArtifactCatalog` (`_therock_utils/artifacts.py`) was designed for exactly this
use case: enumerating files from a set of extracted artifact directories,
filtered by artifact name and component type. It:

- Reads `artifact_manifest.txt` from each extracted artifact directory to
  discover which stage directories are present
- Uses `PatternMatcher` to efficiently enumerate files with proper include/
  exclude glob support
- Accepts a `filter` callable on `ArtifactName` to select specific artifacts
  and component groups (`run`, `lib`, `dev`, etc.)
- Is already used by the Python packaging pipeline, ensuring it is well-tested
  and maintained

The MSI generator requests only `run` and `lib` component groups — the runtime
files needed to execute ROCm programs — automatically excluding development
headers, debug symbols, documentation, and test binaries without any explicit
exclusion lists.

## Stage-Dir Scoping

When artifacts are downloaded from CI (`--artifacts-url`), each artifact is
extracted into its own subdirectory (`_extracted/{name}_{component}_generic/`).
`ArtifactCatalog` scopes file enumeration to each artifact's own stage
directory, so `bin/**` from `core-hip` only sees CLR's `bin/` — not the merged
distribution tree containing files from every other component. This prevents
cross-artifact file bleed without requiring any explicit exclusion lists.

This contrasts with the dist-tree fallback approach (operating against
`build/dist/rocm/`) which requires extensive exclusion lists to filter out
files from other artifacts. The stage-scoped approach produces correct results
by construction.

## Per-Artifact Includes

Some packages need only a subset of a particular artifact's files. The
`per_artifact_includes` field on `PackageDef` passes include glob patterns to
`ArtifactCatalog` for specific artifacts, while leaving all others unrestricted.

For example, the `runtime` package includes `amd-llvm` solely for
`amd_comgr.dll` (the code object manager, required at runtime for GPU code
object loading). The `amd-llvm` artifact also contains compiler binaries,
LLVM bitcode, libclang, and clang resource headers — none of which are needed
to run a pre-compiled HIP program. Using `per_artifact_includes`:

```python
per_artifact_includes = {
    "amd-llvm": ["bin/amd_comgr.dll"],
}
```

This produces a clean runtime redistributable without requiring a new TOML
descriptor or build system change. The pattern is stable because `amd_comgr.dll`
is an unversioned name on Windows (the versioned `amd_comgr_{major}.dll`
convention is deprecated).

## Package Structure

Three packages are defined, covering different deployment scenarios:

| Package               | Artifacts                                                                   | Intent                                                                                   |
| --------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `amdrocm-runtime`     | core-hip, core-kpack, amd-llvm (comgr only)                                 | HIP runtime + code object manager for applications that load GPU code objects at runtime |
| `amdrocm-core`        | core-runtime, core-hip, core-kpack, core-hipinfo, core-amdsmi, core-ocl-icd | Full core runtime: HIP + OpenCL + AMD SMI + ROCR-Runtime                                 |

Each package installs to a versioned subdirectory under
`C:\Program Files\AMD\ROCm\`, enabling side-by-side installation of multiple
ROCm versions and clean upgrades via `MajorUpgrade`.

## CI Integration

The generator supports two modes:

**From CI artifacts (`--artifacts-url`):** Downloads and extracts
`{name}_{component}_generic.tar.zst` archives from an S3 artifact store. Each
archive's embedded `artifact_manifest.txt` is written to disk so
`ArtifactCatalog` can discover it. No local ROCm build is required — any CI
run's artifact output can be packaged directly.

**From a local build (`--build`):** Points `ArtifactCatalog` at
`build/artifacts/`, where the CMake build system deposits exploded artifact
directories during `ninja artifacts`. The same code path handles both modes.

## WiX Toolset v4

WiX v4 is used rather than the legacy WiX v3 because:

- It is the current supported release with active development
- WiX v4 schema (`http://wixtoolset.org/schemas/v4/wxs`) is cleaner and
  removes several deprecated patterns
- `MajorUpgrade` handles upgrade and downgrade detection declaratively
- `dep:Requires` (WixToolset.Dependency.wixext) enables inter-package
  dependency declarations for future dev/runtime package splits

## Upgrade Codes

Each package has a fixed upgrade code GUID that must never change after first
release. Windows Installer uses this GUID to locate and remove previous
installations of the same package when upgrading. The value has no semantic
meaning — it is a stable unique identifier per product line.

## Scalability and Low Maintenance Cost

### Adding new packages

Defining a new MSI package requires only a new `PackageDef` entry in
`PACKAGES` — specifying which existing artifact names to include, an upgrade
code, and an install subdirectory. No WiX source files are authored, no file
lists are maintained, and no build system changes are needed. The generator
derives the complete file set from the artifact manifests automatically.

For example, adding an `amdrocm-math` package covering the BLAS, FFT, and
sparse math libraries would be a ~15-line addition to `generate_msi_wxs.py`
and nothing else — the TOML descriptors for those artifacts already exist and
already classify files into `run`/`lib`/`dev`/`test` groups.

### Accommodating new components

When a new ROCm component is added to the build system, it gets its own
`artifact-{name}.toml`. Any existing MSI package that lists that artifact
automatically picks up its files on the next generator run. No changes to
`generate_msi_wxs.py` are required unless the new package warrants its own
MSI.

When an existing component adds, removes, or renames a file, the TOML
descriptor is updated as part of that component's development. The MSI
generator picks up the change automatically — there is no second place to
update.

### Low maintenance cost

The maintenance cost is low because ownership of file classification stays
where it belongs — with the component developers who know what their build
produces. The `artifact-{name}.toml` files are maintained by the teams
responsible for each component (BLAS, LLVM, HIP, etc.) as part of normal
development. The MSI generator has no knowledge of individual files; it
delegates all classification decisions to those descriptors.

This is in direct contrast to hand-authored WiX files, where every file must
be explicitly listed. When a component's output changes — a common occurrence
during active development — hand-authored lists silently become stale: new
files are missing from the installer, removed files cause build errors. The
artifact manifest approach fails loudly if an artifact is missing but silently
stays correct as file sets evolve.

The generator itself (`generate_msi_wxs.py`) is intentionally thin — under
500 lines — covering only the concerns that are specific to MSI packaging:
WiX XML structure, upgrade code management, PATH/registry configuration, and
artifact download. Everything else is delegated to existing, well-tested
infrastructure.

## Work in Progress

This is an active work-in-progress PR. The core generator and package
definitions are functional and tested against CI artifacts, but the following
gaps remain and will be addressed in follow-up commits before the packages are
publicly distributed:

- **`dep:Requires` declarations** — inter-package dependencies (e.g.
  `amdrocm-runtime-dev` depending on `amdrocm-runtime`) are not yet
  declared. The WixToolset.Dependency extension is already included in the WiX
  build command; this is a generator-side addition.
- **CI workflow integration** — a GitHub Actions job to call the generator and
  run `wix build` on CI-produced artifacts has not yet been merged into the
  release pipeline.
- **Upgrade code finalization** — the current upgrade codes are placeholder
  GUIDs. These must be replaced with properly random UUIDs before first public
  release and must never change afterward.
- **RFC alignment** — once the Windows packaging RFC (PR #3973) lands, this
  implementation will be reviewed against it to ensure install paths, registry
  keys, and discovery mechanisms are consistent with the approved design.
- **`core-amdsmi` and `core-runtime` Windows artifacts** — these are not yet
  published in nightly artifact runs, so the `amdrocm-core` package currently
  omits AMD SMI and ROCR-Runtime. They will be included automatically once the
  artifacts are available.

All of the above are scoped additions requiring no architectural changes — the
generator design and ArtifactCatalog integration are considered stable.
