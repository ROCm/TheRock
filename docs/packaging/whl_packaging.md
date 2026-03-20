# Python Wheel (WHL) Packaging

This document provides a deep dive into how ROCm Python wheels are generated,
what metadata they contain, how they relate to tarballs, and what the installed
directory structure looks like.

Table of contents:

- [Overview](#overview)
- [Package Hierarchy](#package-hierarchy)
  - [rocm (Selector sdist)](#rocm-selector-sdist)
  - [rocm-sdk-core (Runtime Wheel)](#rocm-sdk-core-runtime-wheel)
  - [rocm-sdk-libraries (Runtime Wheel)](#rocm-sdk-libraries-runtime-wheel)
  - [rocm-sdk-devel (Development Wheel)](#rocm-sdk-devel-development-wheel)
- [Generation Pipeline](#generation-pipeline)
  - [Step 1: Artifact Consumption](#step-1-artifact-consumption)
  - [Step 2: File Surgery](#step-2-file-surgery)
  - [Step 3: Package Population](#step-3-package-population)
  - [Step 4: Wheel Building](#step-4-wheel-building)
  - [Step 5: Upload and Index Generation](#step-5-upload-and-index-generation)
- [Artifact Reuse from Tarballs](#artifact-reuse-from-tarballs)
- [Critical Wheel Metadata](#critical-wheel-metadata)
- [Installed Directory Structure](#installed-directory-structure)
  - [Site-Packages Layout](#site-packages-layout)
  - [Component Placement Map](#component-placement-map)
- [Linked vs On-Demand Packages](#linked-vs-on-demand-packages)
- [Tarball vs WHL Installation Comparison](#tarball-vs-whl-installation-comparison)
- [Installation](#installation)
  - [From Nightly Index](#from-nightly-index)
  - [From Local Artifacts](#from-local-artifacts)
  - [Verification](#verification)
- [Relevant Source Files](#relevant-source-files)

## Overview

ROCm Python wheels provide a `pip`-installable distribution of the ROCm SDK.
Unlike tarballs (which are flat archives), wheels integrate with the Python
ecosystem, supporting dependency resolution, virtual environments, and
version management.

The wheel packaging system consumes the **same artifact directories** produced
by the CMake build that tarballs are generated from. No separate build is
required — the Python packaging scripts perform file surgery (RPATH patching,
symlink transformation, layout reorganization) on existing artifacts to produce
wheels.

## Package Hierarchy

ROCm is distributed as four Python packages with a selector pattern:

```
pip install rocm[libraries,devel]
       │
       ▼
   ┌───────┐
   │ rocm  │  ← sdist (meta-package, selector logic)
   └───┬───┘
       │ declares dependencies based on GPU detection
       ▼
   ┌────────────────┐     ┌──────────────────────────────┐
   │ rocm-sdk-core  │     │ rocm-sdk-libraries-<gfx>     │
   │ (target-neutral│     │ (GPU-family specific)         │
   │  runtime)      │     │                               │
   └────────────────┘     └──────────────────────────────┘
       │                      │
       │     ┌────────────────┘
       ▼     ▼
   ┌────────────────┐
   │ rocm-sdk-devel │  ← only with [devel] extra
   │ (dev tools as  │
   │  embedded tar) │
   └────────────────┘
```

### rocm (Selector sdist)

| Property | Value |
| --- | --- |
| **Package type** | Source distribution (sdist) |
| **Format** | `rocm-<version>.tar.gz` |
| **Contents** | `rocm-sdk` CLI tool, `rocm_sdk` Python module |
| **Purpose** | GPU family detection, dependency declaration |

The `rocm` package is intentionally built as an **sdist** (not a wheel) so that
`setup.py` executes at install time. This allows it to:

1. Detect the GPU family on the target system.
2. Dynamically declare the correct `rocm-sdk-libraries-<gfx>` dependency.
3. Provide the `rocm-sdk` CLI for path queries, testing, and initialization.

Extras supported:
- `rocm[libraries]` — installs GPU-specific math libraries.
- `rocm[devel]` — installs development headers, CMake configs, and tools.
- `rocm[libraries,devel]` — full SDK.

### rocm-sdk-core (Runtime Wheel)

| Property | Value |
| --- | --- |
| **Package type** | Platform wheel |
| **Format** | `rocm_sdk_core-<version>-py3-none-manylinux_2_28_x86_64.whl` |
| **Contents** | LLVM, HIP runtime, base tools, system deps |
| **GPU-specific?** | No (target-neutral) |

Sourced from these artifact filters:
- `amd-llvm` (lib, run)
- `base` (lib, run)
- `core-hip` (lib, run, dev — for HIP headers)
- `core-runtime` (lib, run)
- `core-amdsmi` (lib, run)
- `core-ocl` (lib, run, dev)
- `sysdeps` (lib, run)
- `rocprofiler-sdk` (lib, run)
- `hipify` (lib, run)

The core wheel excludes CMake configs (`**/cmake/**`) to keep the runtime
package lean — those go into `rocm-sdk-devel`.

### rocm-sdk-libraries (Runtime Wheel)

| Property | Value |
| --- | --- |
| **Package type** | Platform wheel |
| **Format** | `rocm_sdk_libraries_<gfx>-<version>-py3-none-manylinux_2_28_x86_64.whl` |
| **Contents** | rocBLAS, rocFFT, MIOpen, RCCL, and other math/ML libs |
| **GPU-specific?** | Yes (one wheel per GPU family) |

One library wheel is produced per GPU family (e.g.
`rocm_sdk_libraries_gfx94x_dcgpu`, `rocm_sdk_libraries_gfx110x_all`). Sourced
from artifact filters:
- `blas` (lib) — rocBLAS, rocSOLVER, rocSPARSE, hipBLAS, hipBLASLt, hipSOLVER, hipSPARSE, hipSPARSELt
- `fft` (lib) — rocFFT, hipFFT
- `rand` (lib) — rocRAND, hipRAND
- `rccl` (lib) — RCCL
- `prim` (lib) — rocPRIM, hipCUB, rocThrust
- `MIOpen` (lib) — MIOpen
- `hipdnn` (lib) — hipDNN
- `rocdecode` (lib) — Video decode
- `rocjpeg` (lib) — JPEG decode

Only artifacts matching the specific GPU family (or `generic`) are included.
RPATHs are patched to reference the core package's library paths.

### rocm-sdk-devel (Development Wheel)

| Property | Value |
| --- | --- |
| **Package type** | Platform wheel |
| **Format** | `rocm_sdk_devel-<version>-py3-none-manylinux_2_28_x86_64.whl` |
| **Contents** | Headers, CMake configs, static libs, dev tools |
| **GPU-specific?** | No |

The devel package is a catch-all for development files. Since symlinks and
non-standard file attributes cannot be stored in a wheel, the platform contents
are embedded as a `_devel.tar.xz` (or `_devel.tar`) inside the wheel.

On first use (or explicit `rocm-sdk init`), the tarball is extracted to
`_rocm_sdk_devel/` in the site-packages directory. Files already present in
runtime packages are represented as relative symlinks or hardlinks.

## Generation Pipeline

### Step 1: Artifact Consumption

`build_python_packages.py` takes `--artifact-dir` pointing to the
`build/artifacts/` tree from a CMake build (or downloaded CI artifacts).
It creates an `ArtifactCatalog` that indexes all available artifact
directories by name, component, and target.

```bash
python build_tools/build_python_packages.py \
    --artifact-dir ./build/artifacts \
    --dest-dir ./packages
```

### Step 2: File Surgery

For each artifact being included in a wheel, `py_packaging.py` performs:

1. **Symlink removal** — Wheels cannot contain symlinks. SONAME symlinks
   (e.g. `libfoo.so → libfoo.so.1`) are resolved to keep only the SONAME
   library. Executable symlinks are replaced with compiled shim binaries that
   `execle()` the real binary.
2. **RPATH patching** — `patchelf --set-rpath` is used on Linux to set
   correct relative RPATHs so libraries can find each other across package
   boundaries within the same site-packages tree.
3. **Layout reorganization** — Files are moved from the build-tree layout
   (`core/clr/build/stage/lib/...`) into a flat install-tree layout
   (`lib/libamdhip64.so`).

### Step 3: Package Population

The `PopulatedDistPackage` class populates each package's directory tree
using templates from `build_tools/packaging/python/templates/`:

```
packages/
├── rocm/                          # sdist template
│   ├── pyproject.toml
│   ├── setup.py
│   └── src/rocm_sdk/
│       ├── __init__.py
│       ├── _dist_info.py          # Generated: version, target family
│       └── ...
├── rocm-sdk-core/                 # core wheel tree
│   ├── pyproject.toml
│   ├── setup.py
│   └── _rocm_sdk_core/
│       ├── bin/
│       ├── lib/
│       └── ...
├── rocm-sdk-libraries-gfx94x-dcgpu/
│   ├── pyproject.toml
│   ├── setup.py
│   └── _rocm_sdk_libraries_gfx94x_dcgpu/
│       └── lib/
└── rocm-sdk-devel/
    ├── pyproject.toml
    ├── setup.py
    └── _rocm_sdk_devel/
        └── _devel.tar.xz          # Compressed dev content
```

### Step 4: Wheel Building

`py_packaging.build_packages()` runs `python setup.py bdist_wheel` (for
wheels) or `python setup.py sdist` (for the `rocm` meta-package) in each
populated package directory. The resulting `.whl` and `.tar.gz` files are
placed in `packages/dist/`.

### Step 5: Upload and Index Generation

In CI, the upload pipeline:

1. `upload_python_packages.py` pushes wheels to S3 buckets
   (`therock-nightly-python`, `therock-dev-python`, etc.).
2. `manage.py` generates PEP 503-compliant `index.html` files for each
   GPU-family subdirectory (e.g. `v2/gfx94X-dcgpu/`).
3. CDN fronts the S3 buckets at user-facing URLs like
   `https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/`.

External PyPI dependencies (numpy, filelock, etc.) are pre-mirrored into the
same S3 buckets by `update_dependencies.py`, making the index fully
self-contained — no PyPI access is needed at install time.

## Artifact Reuse from Tarballs

Wheels do **not** require a separate build from tarballs. The generation
pipeline is:

```
CMake Build → build/artifacts/ → tarball (tar cfz dist/rocm/)
                    │
                    └──────────→ wheels (build_python_packages.py)
```

Both formats consume the same `build/artifacts/` directories. The key
differences are:

| Aspect | Tarball | Wheel |
| --- | --- | --- |
| **Input** | `build/dist/rocm/` (flattened) | `build/artifacts/` (structured) |
| **Symlinks** | Preserved | Removed (SONAME-only or shim binaries) |
| **RPATHs** | Build-time paths | Patched to site-packages relative |
| **CMake configs** | Included in flat tree | In `_devel.tar.xz` inside devel wheel |
| **GPU selection** | One tarball per family | Dynamic via `rocm` selector |

## Critical Wheel Metadata

Each wheel contains metadata that is essential for correct installation and
dependency resolution:

### pyproject.toml / setup.py

- **`name`**: Package name (e.g. `rocm-sdk-core`).
- **`version`**: Computed by `compute_rocm_package_version.py`.
- **`install_requires`**: Inter-package dependencies.
- **`extras_require`**: Optional dependency groups (`libraries`, `devel`).
- **`entry_points`**: CLI tools (e.g. `rocm-sdk` console script).

### _dist_info.py (Generated)

This file is generated at build time into the `rocm` sdist and contains:

- `ROCM_SDK_VERSION`: The exact ROCm version string.
- `DEFAULT_TARGET_FAMILY`: Fallback GPU family if detection fails.
- `TARGET_FAMILIES`: List of all GPU families this build supports.

### Wheel Tags

- **Python tag**: `py3` (pure Python launcher, native content in data dirs).
- **ABI tag**: `none`.
- **Platform tag**: `manylinux_2_28_x86_64` (Linux) or `win_amd64` (Windows).

## Installed Directory Structure

### Site-Packages Layout

After `pip install rocm[libraries,devel]`, the virtual environment looks like:

```
.venv/
├── bin/
│   ├── rocm-sdk              # Console script entry point
│   ├── hipcc                 # Shim → _rocm_sdk_core/bin/hipcc
│   ├── amdclang              # Shim → _rocm_sdk_core/llvm/bin/clang
│   ├── rocminfo              # Shim → _rocm_sdk_core/bin/rocminfo
│   └── ...
└── lib/python3.12/site-packages/
    ├── rocm_sdk/                         # From 'rocm' sdist
    │   ├── __init__.py                   # Process initialization
    │   ├── _dist_info.py                 # Version, target family info
    │   ├── _cli.py                       # rocm-sdk CLI implementation
    │   └── ...
    ├── _rocm_sdk_core/                   # From 'rocm-sdk-core' wheel
    │   ├── bin/
    │   │   ├── hipcc
    │   │   ├── rocminfo
    │   │   └── ...
    │   ├── lib/
    │   │   ├── libamdhip64.so.6
    │   │   ├── libhsa-runtime64.so.1
    │   │   ├── libamd_comgr.so.2
    │   │   ├── rocm_sysdeps/
    │   │   │   └── lib/
    │   │   │       ├── librocm_sysdeps_elf.so.1
    │   │   │       └── ...
    │   │   └── ...
    │   ├── llvm/
    │   │   ├── bin/clang
    │   │   └── lib/
    │   ├── include/
    │   │   └── hip/
    │   │       └── hip_runtime.h
    │   ├── libexec/
    │   ├── share/
    │   └── .info/
    ├── _rocm_sdk_libraries_gfx94x_dcgpu/ # From 'rocm-sdk-libraries' wheel
    │   └── lib/
    │       ├── librocblas.so.4
    │       ├── librocfft.so.0
    │       ├── libMIOpen.so.1
    │       ├── librccl.so.1
    │       ├── librocrand.so.1
    │       └── ...
    └── _rocm_sdk_devel/                  # From 'rocm-sdk-devel' (after init)
        ├── include/
        │   ├── rocblas/
        │   ├── rocfft/
        │   └── ...
        ├── lib/
        │   ├── cmake/
        │   │   ├── hip/
        │   │   ├── rocblas/
        │   │   └── ...
        │   ├── librocblas.so → ../_rocm_sdk_libraries_gfx94x_dcgpu/lib/librocblas.so.4
        │   └── ...
        └── share/
```

### Component Placement Map

| Component | Package | Install Path (relative to site-packages) | Why Here? |
| --- | --- | --- | --- |
| HIP runtime (`libamdhip64.so`) | `rocm-sdk-core` | `_rocm_sdk_core/lib/` | Target-neutral host runtime |
| LLVM/Clang toolchain | `rocm-sdk-core` | `_rocm_sdk_core/llvm/` | Compiler needed for HIP compilation |
| HSA runtime (`libhsa-runtime64.so`) | `rocm-sdk-core` | `_rocm_sdk_core/lib/` | Low-level GPU driver interface |
| System deps (`librocm_sysdeps_*.so`) | `rocm-sdk-core` | `_rocm_sdk_core/lib/rocm_sysdeps/lib/` | Private deps avoid system conflicts |
| rocBLAS (`librocblas.so`) | `rocm-sdk-libraries` | `_rocm_sdk_libraries_<gfx>/lib/` | GPU-family-specific device code |
| rocFFT (`librocfft.so`) | `rocm-sdk-libraries` | `_rocm_sdk_libraries_<gfx>/lib/` | GPU-family-specific device code |
| MIOpen (`libMIOpen.so`) | `rocm-sdk-libraries` | `_rocm_sdk_libraries_<gfx>/lib/` | GPU-family-specific device code |
| RCCL (`librccl.so`) | `rocm-sdk-libraries` | `_rocm_sdk_libraries_<gfx>/lib/` | GPU-family-specific device code |
| HIP headers (`hip_runtime.h`) | `rocm-sdk-core` | `_rocm_sdk_core/include/hip/` | Needed at compile time for HIP code |
| CMake configs | `rocm-sdk-devel` | `_rocm_sdk_devel/lib/cmake/` | Build system integration |
| Development headers | `rocm-sdk-devel` | `_rocm_sdk_devel/include/` | Build-time only |
| `rocm-sdk` CLI | `rocm` (sdist) | `rocm_sdk/` | Python entry point for SDK management |

## Linked vs On-Demand Packages

| Package | Install Behavior | When Installed |
| --- | --- | --- |
| `rocm` | Always installed | Base `pip install rocm` |
| `rocm-sdk-core` | Auto-installed as dependency of `rocm` | Always (core dependency) |
| `rocm-sdk-libraries-<gfx>` | Installed when `[libraries]` extra requested | `pip install "rocm[libraries]"` |
| `rocm-sdk-devel` | Installed when `[devel]` extra requested | `pip install "rocm[devel]"` |

The `rocm-sdk-devel` package requires an additional **initialization step**
after installation:

```bash
rocm-sdk init
# Or it initializes lazily on first use from Python
```

This extracts the `_devel.tar.xz` embedded in the wheel into the
`_rocm_sdk_devel/` directory in site-packages.

## Tarball vs WHL Installation Comparison

| Aspect | Tarball | Python Wheel |
| --- | --- | --- |
| **Install method** | `tar -xf` or `install_rocm_from_artifacts.py` | `pip install rocm[libraries,devel]` |
| **Install location** | User-chosen directory | Python virtual environment `site-packages/` |
| **Environment setup** | Manual (`export ROCM_HOME=...`, `export PATH=...`) | Automatic (shims in `.venv/bin/`) |
| **GPU family selection** | Download correct tarball | Automatic detection by `rocm` sdist |
| **Dependency management** | None (standalone) | pip handles versions and conflicts |
| **Symlinks** | Preserved (SONAME links, convenience links) | Removed (SONAME-only, shim binaries) |
| **Development files** | Included inline | In compressed `_devel.tar.xz`, expanded on demand |
| **CMake integration** | `cmake -DCMAKE_PREFIX_PATH=<install>/` | `cmake -DCMAKE_PREFIX_PATH=$(rocm-sdk path --cmake)` |
| **Version management** | Manual (track which tarball you have) | `pip freeze`, `rocm-sdk version` |
| **Multiple versions** | Multiple directories | Multiple virtual environments |
| **Disk space** | Single flat tree | Runtime packages + compressed devel |
| **Update mechanism** | Re-download and extract | `pip install --upgrade` |
| **Offline install** | Copy tarball file | `pip install --find-links=<dir>` |
| **Build reproducibility** | `share/therock/therock_manifest.json` | `rocm-sdk version` + pip metadata |
| **Ideal for** | CI/CD, Docker images, system-wide installs | Python development, venv isolation, framework builds |

## Installation

### From Nightly Index

```bash
python -m venv .venv && source .venv/bin/activate

# Core + libraries + development tools
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ \
    "rocm[libraries,devel]"
```

### From Local Artifacts

```bash
# Build packages from local build artifacts
python build_tools/build_python_packages.py \
    --artifact-dir ./build/artifacts \
    --dest-dir ./packages

# Install from local packages
pip install "rocm[libraries,devel]" --pre \
    --find-links=./packages/dist
```

### Verification

```bash
# Check installed packages
pip freeze | grep rocm

# Run self-tests
rocm-sdk test

# Check paths
rocm-sdk path --root
rocm-sdk path --cmake
rocm-sdk path --bin

# Check GPU targets
rocm-sdk targets
```

## Relevant Source Files

| File | Purpose |
| --- | --- |
| [`build_tools/build_python_packages.py`](/build_tools/build_python_packages.py) | Main wheel/sdist build entrypoint |
| [`build_tools/_therock_utils/py_packaging.py`](/build_tools/_therock_utils/py_packaging.py) | Layout, RPATH patching, build orchestration |
| [`build_tools/packaging/python/templates/`](/build_tools/packaging/python/templates/) | Package templates directory |
| [`build_tools/packaging/python/templates/rocm/`](/build_tools/packaging/python/templates/rocm/) | `rocm` sdist template |
| [`build_tools/packaging/python/templates/rocm-sdk-core/`](/build_tools/packaging/python/templates/rocm-sdk-core/) | Core wheel template |
| [`build_tools/packaging/python/templates/rocm-sdk-libraries/`](/build_tools/packaging/python/templates/rocm-sdk-libraries/) | Libraries wheel template |
| [`build_tools/packaging/python/templates/rocm-sdk-devel/`](/build_tools/packaging/python/templates/rocm-sdk-devel/) | Devel wheel template |
| [`build_tools/compute_rocm_package_version.py`](/build_tools/compute_rocm_package_version.py) | Version computation |
| [`build_tools/github_actions/upload_python_packages.py`](/build_tools/github_actions/upload_python_packages.py) | S3 upload |
| [`build_tools/third_party/s3_management/manage.py`](/build_tools/third_party/s3_management/manage.py) | PEP 503 index generation |
| [`build_tools/third_party/s3_management/update_dependencies.py`](/build_tools/third_party/s3_management/update_dependencies.py) | PyPI dependency mirroring |
| [`docs/packaging/python_packaging.md`](/docs/packaging/python_packaging.md) | Python packaging general design |
| [`docs/packaging/rocm_python_packaging.md`](/docs/packaging/rocm_python_packaging.md) | Build/publish/dependency resolution details |
