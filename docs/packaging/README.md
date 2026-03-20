# ROCm Packaging Overview

This document provides a high-level overview of how ROCm build artifacts are
packaged and distributed through TheRock's CI/CD pipelines.

Table of contents:

- [Packaging Pipeline](#packaging-pipeline)
- [Distribution Channels](#distribution-channels)
- [Packaging Formats](#packaging-formats)
  - [Tarballs](#tarballs)
  - [Python Wheels](#python-wheels)
  - [Native Linux Packages (Deb/RPM)](#native-linux-packages-debrpm)
- [Artifact System Foundation](#artifact-system-foundation)
- [Version Management](#version-management)
- [Detailed Documentation](#detailed-documentation)

## Packaging Pipeline

The overall packaging flow follows a linear dependency chain where each stage
consumes the output of the previous one:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CMake Build System                                 │
│                                                                             │
│  cmake --build build --target therock-archives therock-dist                 │
│                                                                             │
│  Produces:                                                                  │
│    build/artifacts/    → per-component sliced directories                   │
│    build/dist/rocm/    → unified flattened install tree                     │
└──────────────┬──────────────────────────────────┬───────────────────────────┘
               │                                  │
               ▼                                  ▼
┌──────────────────────────┐    ┌─────────────────────────────────────────────┐
│     Tarball Generation   │    │         Artifact Archives (.tar.xz)         │
│                          │    │                                             │
│  tar cfz on dist/rocm/   │    │  Per-component: base_lib_generic.tar.xz    │
│                          │    │                 core-hip_dev_gfx942.tar.xz  │
│  Single unified SDK      │    │                 blas_run_gfx1100.tar.xz    │
│  tarball for end users   │    │                 ...                         │
└──────────────────────────┘    └──────────────┬──────────────────────────────┘
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        │                      │                      │
                        ▼                      ▼                      ▼
          ┌─────────────────────┐ ┌──────────────────────┐ ┌─────────────────┐
          │   Python Wheels     │ │   Debian Packages    │ │  RPM Packages   │
          │                     │ │                      │ │                 │
          │ build_python_       │ │ build_package.py     │ │ build_package.py│
          │   packages.py       │ │   --pkg-type deb     │ │   --pkg-type rpm│
          │                     │ │                      │ │                 │
          │ Consumes artifact   │ │ Consumes artifact    │ │ Consumes same   │
          │ directories from    │ │ directories from     │ │ artifact dirs   │
          │ CI run              │ │ CI run               │ │                 │
          └─────────────────────┘ └──────────────────────┘ └─────────────────┘
```

### CI Workflow Orchestration

The primary release workflow is
[`release_portable_linux_packages.yml`](/.github/workflows/release_portable_linux_packages.yml),
which orchestrates the full pipeline:

1. **Build phase** — CMake builds all components, producing artifacts and
   the unified `dist/rocm/` tree.
2. **Tarball phase** — The unified tree is archived into a single SDK tarball
   (e.g. `therock-dist-linux-gfx94X-dcgpu-7.12.0a20260310.tar.gz`) and uploaded
   to S3.
3. **Python packaging phase** — `build_python_packages.py` consumes the artifact
   directories and produces `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries-<gfx>`,
   and `rocm-sdk-devel` packages. These are uploaded to S3 with PEP 503
   index generation.
4. **Native packaging phase** — The workflow dispatches
   `build_native_linux_packages.yml` twice (once for DEB, once for RPM),
   passing the artifact run ID. Each invocation downloads the artifact archives,
   extracts them, and runs `build_package.py` to produce native packages.

Native packaging (DEB and RPM) runs **in parallel** with Python packaging,
since both consume the same artifact directories independently.

### Pre-submit vs Nightly Builds

| Aspect | Pre-submit (PR CI) | Nightly Release |
| --- | --- | --- |
| **Trigger** | Pull request events | Scheduled cron (`0 04 * * *` UTC) |
| **Scope** | Per-component artifact archives | Full ROCm SDK tarball + all packages |
| **Artifacts** | Individual `{name}_{component}_{target}.tar.xz` | Unified `therock-dist-linux-*.tar.gz` |
| **Python pkgs** | Optional (via `build_portable_linux_python_packages.yml`) | Always built and uploaded |
| **Native pkgs** | Not built | Built for both DEB and RPM |
| **S3 upload** | GitHub Actions artifacts only | S3 buckets + CDN |

## Distribution Channels

| Channel | Tarball URL | Python Index | Native Packages |
| --- | --- | --- | --- |
| Stable | `https://repo.amd.com/rocm/tarball/` | `https://repo.amd.com/rocm/whl/` | `https://repo.amd.com/rocm/` |
| Nightly | `https://rocm.nightlies.amd.com/tarball/` | `https://rocm.nightlies.amd.com/v2/` | S3 hosted |
| Prerelease | `https://rocm.prereleases.amd.com/tarball/` | `https://rocm.prereleases.amd.com/whl` | S3 hosted |
| Dev | `https://rocm.devreleases.amd.com/tarball/` | `https://rocm.devreleases.amd.com/v2/` | S3 hosted |

See [versioning.md](versioning.md) for version format details per channel.

## Packaging Formats

### Tarballs

Tarballs provide a flat, self-contained ROCm SDK directory that mirrors the
traditional `/opt/rocm/` layout. They are the simplest distribution format and
the foundation that other packaging formats build upon.

```
therock-dist-linux-gfx94X-dcgpu-7.12.0a20260310.tar.gz
└── bin/           # CLI tools (rocminfo, hipcc, amdclang, etc.)
    include/       # Development headers
    lib/           # Shared libraries, CMake configs
    libexec/       # Internal executables
    share/         # Data files, manifests, module files
    .info/         # Build metadata
```

See [tarball_packaging.md](tarball_packaging.md) for the deep dive.

### Python Wheels

Python wheels provide a `pip`-installable distribution of ROCm. The package
hierarchy uses a selector pattern:

| Package | Type | Contents |
| --- | --- | --- |
| `rocm` | sdist (meta) | Selector that detects GPU family, declares deps |
| `rocm-sdk-core` | wheel | LLVM, HIP runtime, base tools (target-neutral) |
| `rocm-sdk-libraries-<gfx>` | wheel | rocBLAS, rocFFT, MIOpen, etc. (GPU-specific) |
| `rocm-sdk-devel` | wheel | Headers, CMake configs, dev tools (as embedded tarball) |

Wheels are built from the **same artifact directories** as tarballs, with
additional file surgery (RPATH patching, symlink removal, layout
reorganization).

See [whl_packaging.md](whl_packaging.md) for the deep dive.

### Native Linux Packages (Deb/RPM)

Native packages integrate with system package managers (`apt`, `dnf`/`yum`).
Each ROCm component maps to a named package (e.g. `amdrocm-blas`,
`amdrocm-llvm`) with proper dependency declarations.

Both versioned and non-versioned variants are produced:
- **Versioned** (e.g. `amdrocm-blas-7.12.0`): Contains actual files, installed
  to `/opt/rocm/core/<version>/`.
- **Non-versioned** (e.g. `amdrocm-blas`): Meta-package depending on the
  versioned package.

See [deb_rpm_packaging.md](deb_rpm_packaging.md) for the deep dive.

## Artifact System Foundation

All packaging formats are built on top of TheRock's artifact subsystem. The
CMake function `therock_provide_artifact()` defines how sub-project install
trees are sliced into artifact components:

| Component | Role | Typical Contents |
| --- | --- | --- |
| `lib` | Runtime libraries | `.so` files, DLLs, runtime data |
| `dev` | Build-time dependencies | Headers, static libs, CMake configs |
| `run` | Executable tools | CLI binaries, utility scripts |
| `dbg` | Debug symbols | `.debug` files, symbol tables |
| `doc` | Documentation | Man pages, license files |
| `test` | Test infrastructure | Test binaries, test data |

Artifact contents are defined by TOML descriptors (`artifact.toml`) using
ant-style glob patterns. See
[docs/development/artifacts.md](/docs/development/artifacts.md) for the full
artifact system design.

### Key Tools

| Tool | Purpose |
| --- | --- |
| [`fileset_tool.py`](/build_tools/fileset_tool.py) | Artifact population, flattening, and archiving |
| [`build_python_packages.py`](/build_tools/build_python_packages.py) | Python wheel/sdist generation |
| [`build_package.py`](/build_tools/packaging/linux/build_package.py) | Native DEB/RPM generation |
| [`install_rocm_from_artifacts.py`](/build_tools/install_rocm_from_artifacts.py) | Tarball/artifact installation utility |
| [`compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py) | Python package version computation |
| [`compute_rocm_native_package_version.py`](/build_tools/compute_rocm_native_package_version.py) | Native package version computation |

## Version Management

Package versions are derived from the base version in
[`version.json`](/version.json) and vary by distribution channel:

| Channel | Python | DEB | RPM |
| --- | --- | --- | --- |
| Stable | `7.12.0` | `7.12.0` | `7.12.0` |
| Prerelease | `7.12.0rcN` | `7.12.0~preN` | `7.12.0~rcN` |
| Nightly | `7.12.0aYYYYMMDD` | `7.12.0~YYYYMMDD` | `7.12.0~YYYYMMDD` |
| Dev | `7.12.0.dev0+hash` | `7.12.0~devYYYYMMDD` | `7.12.0~YYYYMMDDg<hash>` |

See [versioning.md](versioning.md) for the full versioning specification.

## Detailed Documentation

| Document | Description |
| --- | --- |
| [tarball_packaging.md](tarball_packaging.md) | Tarball generation, structure, and installation |
| [whl_packaging.md](whl_packaging.md) | Python wheel packaging pipeline |
| [deb_rpm_packaging.md](deb_rpm_packaging.md) | Native Debian and RPM packaging |
| [python_packaging.md](python_packaging.md) | Python packaging design (general) |
| [rocm_python_packaging.md](rocm_python_packaging.md) | Python wheel build/publish/dependency resolution |
| [native_packaging.md](native_packaging.md) | Native packaging design (general) |
| [versioning.md](versioning.md) | Version schemes across all packaging formats |
| [artifacts.md](/docs/development/artifacts.md) | Build artifact system design |
