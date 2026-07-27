# ROCm Package Dependency Trees

This document shows the dependency tree structure for the three types of packages in the ROCm packaging system:

1. **Non-GfxArch Package** - Architecture-independent packages
1. **GfxArch Package** - Architecture-specific packages (host-device split)
1. **Meta Package** - Aggregator packages with no content

______________________________________________________________________

## 1. Non-GfxArch Package

**Example**: `amdrocm-sysdeps`

**Characteristics**:

- `Gfxarch: False` (or not specified)
- Same content for all GPU architectures
- No host-device split in kpack mode
- Creates versioned + non-versioned packages only

### Package Configuration

```json
{
  "Package": "amdrocm-sysdeps",
  "DEBDepends": ["libc6"],
  "Gfxarch": "False"
}
```

### Dependency Tree (Kpack Mode)

```
amdrocm-sysdeps (non-versioned)
│
│   [No files - dependency pointer only]
│
└─► amdrocm-sysdeps8.2 (versioned)
    │
    │   [Files: actual package content]
    │   /opt/rocm/core/lib/...
    │   /opt/rocm/core/share/...
    │
    └─► libc6 (system dependency)
```

### Package Generation

| Package Name         | Type          | Files       | Direct Dependencies |
| -------------------- | ------------- | ----------- | ------------------- |
| `amdrocm-sysdeps8.2` | Versioned     | All content | libc6               |
| `amdrocm-sysdeps`    | Non-versioned | None        | amdrocm-sysdeps8.2  |

### Visual Diagram

```
┌─────────────────────────────┐
│     amdrocm-sysdeps         │  ◄── User installs this
│     (non-versioned)         │
│         [No files]          │
└─────────────┬───────────────┘
              │ depends on
              ▼
┌─────────────────────────────┐
│    amdrocm-sysdeps8.2       │  ◄── Contains actual files
│       (versioned)           │
│                             │
│  Files:                     │
│  - /opt/rocm/core/lib/...   │
│  - /opt/rocm/core/share/... │
└─────────────┬───────────────┘
              │ depends on
              ▼
┌─────────────────────────────┐
│          libc6              │  ◄── System package
│    (system dependency)      │
└─────────────────────────────┘
```

______________________________________________________________________

## 2. GfxArch Package

**Example**: `amdrocm-blas`

**Characteristics**:

- `Gfxarch: True`
- Architecture-specific content (GPU binaries)
- Host-device split in kpack mode
- Creates: host + devices + meta + non-versioned packages

### Package Configuration

```json
{
  "Package": "amdrocm-blas",
  "DEBDepends": [
    "libc6",
    "amdrocm-runtime",
    "amdrocm-solver",
    "amdrocm-profiler-base"
  ],
  "Gfxarch": "True"
}
```

### Dependency Tree (Kpack Mode)

**For gfxarch_list = ["gfx1100", "gfx942"]**

**Dependencies from package.json**:

- `libc6` - system package
- `amdrocm-runtime` - non-gfxarch package
- `amdrocm-solver` - **gfxarch package**
- `amdrocm-profiler-base` - non-gfxarch package

```
amdrocm-blas (non-versioned)
│
│   [No files - user-facing package]
│
└─► amdrocm-blas8.2 (meta)
    │
    │   [No files - aggregator only]
    │
    ├─► amdrocm-blas-host8.2 (host)
    │   │
    │   │   [Files: libraries, docs]
    │   │   /opt/rocm/core/lib/librocblas.so
    │   │   /opt/rocm/core/lib/libhipblas.so
    │   │   /opt/rocm/core/share/doc/...
    │   │
    │   ├─► libc6
    │   ├─► amdrocm-runtime8.2 (non-gfxarch)
    │   ├─► amdrocm-solver-host8.2 (gfxarch → host variant)
    │   └─► amdrocm-profiler-base8.2 (non-gfxarch)
    │
    ├─► amdrocm-blas8.2-gfx1100 (device)
    │   │
    │   │   [Files: gfx1100 kpack files]
    │   │   /opt/rocm/core/.kpack/blas_lib_gfx1100.kpack
    │   │
    │   ├─► amdrocm-blas-host8.2 (own host package)
    │   └─► amdrocm-solver8.2-gfx1100 (gfxarch → same arch)
    │
    └─► amdrocm-blas8.2-gfx942 (device)
        │
        │   [Files: gfx942 kpack files]
        │   /opt/rocm/core/.kpack/blas_lib_gfx942.kpack
        │
        ├─► amdrocm-blas-host8.2 (own host package)
        └─► amdrocm-solver8.2-gfx942 (gfxarch → same arch)
```

### Package Generation

| Package Name              | Type          | Files                      | Direct Dependencies                                                         |
| ------------------------- | ------------- | -------------------------- | --------------------------------------------------------------------------- |
| `amdrocm-blas-host8.2`    | Host          | Libraries (.so), docs      | libc6, amdrocm-runtime8.2, amdrocm-solver-host8.2, amdrocm-profiler-base8.2 |
| `amdrocm-blas8.2-gfx1100` | Device        | .kpack files (GPU kernels) | amdrocm-blas-host8.2, amdrocm-solver8.2-gfx1100                             |
| `amdrocm-blas8.2-gfx942`  | Device        | .kpack files (GPU kernels) | amdrocm-blas-host8.2, amdrocm-solver8.2-gfx942                              |
| `amdrocm-blas8.2`         | Meta          | None                       | host + all devices                                                          |
| `amdrocm-blas`            | Non-versioned | None                       | amdrocm-blas8.2                                                             |

### Visual Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         amdrocm-blas                                  │
│                       (non-versioned)                                 │
│                         [No files]                                    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ depends on
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        amdrocm-blas8.2                                │
│                        (meta package)                                 │
│                         [No files]                                    │
└────────┬──────────────────────┬───────────────────────┬──────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ amdrocm-blas   │    │  amdrocm-blas   │    │  amdrocm-blas   │
│  -host8.2      │    │ 8.2-gfx1100     │    │ 8.2-gfx942      │
│                │    │                 │    │                 │
│ [Libs/Docs]    │    │ [.kpack files]  │    │ [.kpack files]  │
└───────┬────────┘    └────────┬────────┘    └────────┬────────┘
        │                      │                      │
        │                      │                      │
        ▼                      ▼                      ▼
┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Host Deps:     │    │ Device Deps:    │    │ Device Deps:    │
│ - libc6        │    │ - blas-host8.2  │    │ - blas-host8.2  │
│ - runtime8.2   │    │ - solver8.2     │    │ - solver8.2     │
│ - solver-host  │    │   -gfx1100      │    │   -gfx942       │
│   8.2          │    │                 │    │                 │
│ - profiler8.2  │    │                 │    │                 │
└────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Points

1. **Host packages depend on HOST variants** of gfxarch dependencies (for headers)
1. **Device packages depend on SAME-ARCH variants** of gfxarch dependencies (for binaries)
1. **Non-gfxarch dependencies** go to host package only (inherited by devices)
1. **No cross-architecture dependencies** - gfx1100 device never depends on gfx942 packages

______________________________________________________________________

## 3. Meta Package

There are two types of metapackages based on their `Gfxarch` setting:

### 3.1 GfxArch=True Metapackage

**Example**: `amdrocm-core`

**Characteristics**:

- `Metapackage: True` + `Gfxarch: True`
- Creates architecture-specific variants
- Generic variant depends on all arch-specific variants
- Arch-specific variants depend on actual packages

### Package Configuration

```json
{
  "Package": "amdrocm-core",
  "Metapackage": "True",
  "Gfxarch": "True",
  "DEBDepends": [
    "amdrocm-base",
    "amdrocm-sysdeps",
    "amdrocm-llvm",
    "amdrocm-runtime",
    "amdrocm-blas",
    "amdrocm-fft",
    "amdrocm-amdsmi",
    ...
  ]
}
```

### Dependency Tree (Kpack Mode)

**For gfxarch_list = ["gfx1100", "gfx942"]**

```
amdrocm-core (non-versioned)
│
│   [No files]
│
└─► amdrocm-core8.2 (generic meta)
    │
    │   [No files]
    │   This is the GENERIC variant that aggregates all arch-specific variants
    │
    ├─► amdrocm-core8.2-gfx1100 (arch-specific meta)
    │   │
    │   │   [No files]
    │   │   Depends on actual packages with gfx1100 architecture
    │   │
    │   ├─► amdrocm-base8.2 (non-gfxarch)
    │   ├─► amdrocm-sysdeps8.2 (non-gfxarch)
    │   ├─► amdrocm-llvm8.2 (non-gfxarch)
    │   ├─► amdrocm-runtime8.2 (non-gfxarch)
    │   ├─► amdrocm-blas8.2-gfx1100 (gfxarch → same arch)
    │   ├─► amdrocm-fft8.2-gfx1100 (gfxarch → same arch)
    │   ├─► amdrocm-amdsmi8.2 (non-gfxarch)
    │   └─► ...
    │
    └─► amdrocm-core8.2-gfx942 (arch-specific meta)
        │
        │   [No files]
        │   Depends on actual packages with gfx942 architecture
        │
        ├─► amdrocm-base8.2 (non-gfxarch)
        ├─► amdrocm-sysdeps8.2 (non-gfxarch)
        ├─► amdrocm-llvm8.2 (non-gfxarch)
        ├─► amdrocm-runtime8.2 (non-gfxarch)
        ├─► amdrocm-blas8.2-gfx942 (gfxarch → same arch)
        ├─► amdrocm-fft8.2-gfx942 (gfxarch → same arch)
        ├─► amdrocm-amdsmi8.2 (non-gfxarch)
        └─► ...
```

### Package Generation (GfxArch=True Metapackage)

| Package Name              | Type               | Files | Direct Dependencies             |
| ------------------------- | ------------------ | ----- | ------------------------------- |
| `amdrocm-core8.2`         | Generic Meta       | None  | All arch-specific meta variants |
| `amdrocm-core8.2-gfx1100` | Arch-specific Meta | None  | Actual packages with gfx1100    |
| `amdrocm-core8.2-gfx942`  | Arch-specific Meta | None  | Actual packages with gfx942     |
| `amdrocm-core`            | Non-versioned      | None  | amdrocm-core8.2                 |

### Visual Diagram (GfxArch=True Metapackage)

```
┌─────────────────────────────────────────────────────────────────┐
│                        amdrocm-core                              │
│                      (non-versioned)                             │
│                         [No files]                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ depends on
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      amdrocm-core8.2                             │
│                      (generic meta)                              │
│                        [No files]                                │
│                                                                  │
│   Aggregates all architecture-specific metapackage variants      │
└───────────────┬─────────────────────────────┬───────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│   amdrocm-core8.2         │   │   amdrocm-core8.2         │
│        -gfx1100           │   │        -gfx942            │
│   (arch-specific meta)    │   │   (arch-specific meta)    │
│       [No files]          │   │       [No files]          │
└─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ Depends on:             │   │ Depends on:             │
│ - base8.2 (non-gfxarch) │   │ - base8.2 (non-gfxarch) │
│ - sysdeps8.2            │   │ - sysdeps8.2            │
│ - llvm8.2               │   │ - llvm8.2               │
│ - blas8.2-gfx1100       │   │ - blas8.2-gfx942        │
│ - fft8.2-gfx1100        │   │ - fft8.2-gfx942         │
│ - amdsmi8.2             │   │ - amdsmi8.2             │
└─────────────────────────┘   └─────────────────────────┘
```

______________________________________________________________________

### 3.2 GfxArch=False Metapackage

**Example**: `amdrocm-developer-tools`

**Characteristics**:

- `Metapackage: True` + `Gfxarch: False`
- NO architecture-specific variants
- Simple versioned + non-versioned structure
- Depends on versioned packages directly

### Package Configuration

```json
{
  "Package": "amdrocm-developer-tools",
  "Metapackage": "True",
  "Gfxarch": "False",
  "DEBDepends": [
    "amdrocm-base",
    "amdrocm-amdsmi",
    "amdrocm-profiler-base",
    "amdrocm-profiler"
  ]
}
```

### Dependency Tree (Kpack Mode)

```
amdrocm-developer-tools (non-versioned)
│
│   [No files]
│
└─► amdrocm-developer-tools8.2 (versioned meta)
    │
    │   [No files]
    │   Depends on versioned packages (no arch-specific variants)
    │
    ├─► amdrocm-base8.2
    ├─► amdrocm-amdsmi8.2
    ├─► amdrocm-profiler-base8.2
    └─► amdrocm-profiler8.2
```

### Package Generation (GfxArch=False Metapackage)

| Package Name                 | Type           | Files | Direct Dependencies                 |
| ---------------------------- | -------------- | ----- | ----------------------------------- |
| `amdrocm-developer-tools8.2` | Versioned Meta | None  | All deps versioned (no arch suffix) |
| `amdrocm-developer-tools`    | Non-versioned  | None  | amdrocm-developer-tools8.2          |

### Visual Diagram (GfxArch=False Metapackage)

```
┌─────────────────────────────────────────────────────────────────┐
│                   amdrocm-developer-tools                        │
│                      (non-versioned)                             │
│                         [No files]                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ depends on
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                amdrocm-developer-tools8.2                        │
│                    (versioned meta)                              │
│                       [No files]                                 │
└───────┬──────────┬──────────────┬──────────────┬────────────────┘
        │          │              │              │
        ▼          ▼              ▼              ▼
  ┌──────────┐┌──────────┐┌──────────────┐┌──────────┐
  │  base    ││ amdsmi   ││profiler-base ││ profiler │
  │   8.2    ││   8.2    ││     8.2      ││   8.2    │
  └──────────┘└──────────┘└──────────────┘└──────────┘
```

______________________________________________________________________

### Key Points

1. **GfxArch=True Metapackages**:

   - Generic meta → depends on all arch-specific metas
   - Arch-specific meta → depends on actual packages with matching arch
   - Creates N+2 packages (generic + N arch-specific + non-versioned)

1. **GfxArch=False Metapackages**:

   - No arch-specific variants
   - Versioned meta → depends on versioned packages directly
   - Creates 2 packages (versioned + non-versioned)

1. **All metapackages have NO files** - pure dependency aggregators

______________________________________________________________________

## Summary Comparison

| Aspect                | Non-GfxArch                   | GfxArch                                      | Meta Package                  |
| --------------------- | ----------------------------- | -------------------------------------------- | ----------------------------- |
| **Gfxarch**           | False                         | True                                         | N/A                           |
| **Host-Device Split** | No                            | Yes                                          | No                            |
| **Has Files**         | Yes (versioned)               | Yes (host + devices)                         | No                            |
| **Packages Created**  | 2 (versioned + non-versioned) | 4+ (host + N devices + meta + non-versioned) | 2 (versioned + non-versioned) |
| **Example**           | amdrocm-sysdeps               | amdrocm-blas                                 | amdrocm-core                  |

______________________________________________________________________

## Dependency Resolution Rules

### Host Package

```
For each dependency in package.json:
  - If non-gfxarch: add versioned dep (e.g., amdrocm-runtime8.2)
  - If gfxarch: add HOST variant (e.g., amdrocm-solver-host8.2)

Example for amdrocm-blas-host8.2:
  - libc6 (system)
  - amdrocm-runtime8.2 (non-gfxarch → versioned)
  - amdrocm-solver-host8.2 (gfxarch → host variant)
  - amdrocm-profiler-base8.2 (non-gfxarch → versioned)
```

### Device Package

```
Dependencies:
  - Own host package (e.g., amdrocm-blas-host8.2)
  - Gfxarch deps with SAME architecture suffix

Example for amdrocm-blas8.2-gfx1100:
  - amdrocm-blas-host8.2 (own host)
  - amdrocm-solver8.2-gfx1100 (gfxarch → same arch)
```

### Meta Package (versioned, e.g., amdrocm-blas8.2)

```
Dependencies = [host + all devices]
             = [amdrocm-blas-host8.2,
                amdrocm-blas8.2-gfx1100,
                amdrocm-blas8.2-gfx942, ...]
```

### Metapackage (e.g., amdrocm-core8.2)

```
Dependencies = [all deps from package.json, versioned]
             = [amdrocm-base8.2, amdrocm-sysdeps8.2,
                amdrocm-blas8.2, amdrocm-fft8.2, ...]
```

______________________________________________________________________

**Document Version**: 1.0
**Date**: 2026-05-29
**Architectures Used in Examples**: gfx1100, gfx942
**ROCm Version Used in Examples**: 8.2
