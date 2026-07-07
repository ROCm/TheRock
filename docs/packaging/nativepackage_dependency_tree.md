# ROCm Package Dependency Trees

This document shows the dependency tree structure for the package types in the ROCm packaging system:

1. **Non-GfxArch Package** - Architecture-independent packages
1. **GfxArch Package** - Architecture-specific packages (host-device split)
1. **Meta Package** - Aggregator packages with no content
1. **Devel Package** - Development packages (headers, cmake configs, pkgconfig)

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

## 4. Devel Packages

This section shows the dependency tree structure for development packages in the ROCm packaging system.

**Key Design Principles:**

- All devel packages are `Gfxarch: False` (architecture-independent)
- Devel packages depend on **full meta** (host + all devices) of GfxArch=True runtime packages
- Devel packages depend on **versioned** variant of GfxArch=False runtime packages
- Devel meta packages depend on runtime meta (all arch variants)

______________________________________________________________________

### 4.1 Devel Non-Meta with GfxArch=False Runtime Dependency

**Example**: `amdrocm-llvm-devel` depends on `amdrocm-llvm` (GfxArch=False)

**Characteristics**:

- Devel package for a non-gfxarch runtime package
- Simple versioned + non-versioned structure
- Runtime dependency resolves to versioned package

#### Package Configuration

```json
{
  "Package": "amdrocm-llvm-devel",
  "Gfxarch": "False",
  "DEBDepends": [
    "amdrocm-llvm"
  ]
}
```

#### Dependency Tree

```
amdrocm-llvm-devel (non-versioned)
│
│   [No files]
│
└─► amdrocm-llvm-devel8.2 (versioned)
    │
    │   [Files: development files]
    │   /opt/rocm/core/include/llvm/...
    │   /opt/rocm/core/lib/cmake/llvm/...
    │
    └─► amdrocm-llvm8.2 (GfxArch=False → VERSIONED)
            │
            │   [Files: libraries]
            │   /opt/rocm/core/lib/libLLVM.so
            │
            └─► libc6 (system dependency)
```

#### Package Generation

| Package Name            | Type          | Files          | Direct Dependencies   |
| ----------------------- | ------------- | -------------- | --------------------- |
| `amdrocm-llvm-devel8.2` | Versioned     | Headers, cmake | amdrocm-llvm8.2       |
| `amdrocm-llvm-devel`    | Non-versioned | None           | amdrocm-llvm-devel8.2 |

#### Visual Diagram

```
┌─────────────────────────────┐
│    amdrocm-llvm-devel       │  ◄── User installs this
│     (non-versioned)         │
│        [No files]           │
└─────────────┬───────────────┘
              │ depends on
              ▼
┌─────────────────────────────┐
│   amdrocm-llvm-devel8.2     │  ◄── Contains dev files
│       (versioned)           │
│                             │
│  Files:                     │
│  - /opt/rocm/include/llvm/  │
│  - /opt/rocm/lib/cmake/llvm/│
└─────────────┬───────────────┘
              │ depends on
              ▼
┌─────────────────────────────┐
│     amdrocm-llvm8.2         │  ◄── Runtime (VERSIONED)
│       (versioned)           │
│                             │
│  [Libraries: .so files]     │
└─────────────────────────────┘
```

______________________________________________________________________

### 4.2 Devel Non-Meta with GfxArch=True Runtime Dependency

**Example**: `amdrocm-blas-devel` depends on `amdrocm-blas` (GfxArch=True)

**Characteristics**:

- Devel package for a gfxarch runtime package
- Runtime dependency resolves to **full meta** (host + all device packages)
- Ensures complete runtime environment for development and testing

#### Package Configuration

```json
{
  "Package": "amdrocm-blas-devel",
  "Gfxarch": "False",
  "DEBDepends": [
    "amdrocm-blas"
  ]
}
```

#### Dependency Tree

**For gfxarch_list = ["gfx1100", "gfx942"]**

```
amdrocm-blas-devel (non-versioned)
│
│   [No files]
│
└─► amdrocm-blas-devel8.2 (versioned)
    │
    │   [Files: development files]
    │   /opt/rocm/core/include/rocblas/...
    │   /opt/rocm/core/include/hipblas/...
    │   /opt/rocm/core/lib/cmake/rocblas/...
    │   /opt/rocm/core/lib/pkgconfig/rocblas.pc
    │
    ├─► amdrocm-blas8.2 (GfxArch=True → FULL META)
    │   │
    │   │   [No files - meta aggregator]
    │   │
    │   ├─► amdrocm-blas-host8.2
    │   │   │   [Files: libraries]
    │   │   │   /opt/rocm/core/lib/librocblas.so
    │   │   │   /opt/rocm/core/lib/libhipblas.so
    │   │   └─► (host dependencies...)
    │   │
    │   ├─► amdrocm-blas8.2-gfx1100
    │   │   │   [Files: device kernels]
    │   │   │   /opt/rocm/core/.kpack/blas_lib_gfx1100.kpack
    │   │   └─► (device dependencies...)
    │   │
    │   └─► amdrocm-blas8.2-gfx942
    │       │   [Files: device kernels]
    │       │   /opt/rocm/core/.kpack/blas_lib_gfx942.kpack
    │       └─► (device dependencies...)
```

#### Package Generation

| Package Name            | Type          | Files                     | Direct Dependencies   |
| ----------------------- | ------------- | ------------------------- | --------------------- |
| `amdrocm-blas-devel8.2` | Versioned     | Headers, cmake, pkgconfig | amdrocm-blas8.2       |
| `amdrocm-blas-devel`    | Non-versioned | None                      | amdrocm-blas-devel8.2 |

#### Visual Diagram

```
┌─────────────────────────────┐
│    amdrocm-blas-devel       │  ◄── User installs this
│     (non-versioned)         │
│        [No files]           │
└─────────────┬───────────────┘
              │ depends on
              ▼
┌─────────────────────────────┐
│   amdrocm-blas-devel8.2     │  ◄── Contains dev files
│       (versioned)           │
│                             │
│  Files:                     │
│  - /opt/rocm/include/...    │
│  - /opt/rocm/lib/cmake/...  │
│  - /opt/rocm/lib/pkgconfig/ │
└──────┬──────────────────────┘
       │ depends on
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      amdrocm-blas8.2                              │
│                        (FULL META)                                │
│                         [No files]                                │
└────────┬──────────────────────┬───────────────────────┬──────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ amdrocm-blas   │    │  amdrocm-blas   │    │  amdrocm-blas   │
│  -host8.2      │    │ 8.2-gfx1100     │    │ 8.2-gfx942      │
│                │    │                 │    │                 │
│ [.so libs]     │    │ [.kpack files]  │    │ [.kpack files]  │
└────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Why Full Meta?

1. **Complete environment**: Developers get both headers AND full runtime for testing
1. **Compile and run**: Can build and immediately test on any supported GPU
1. **No manual installs**: Single devel install provides everything needed
1. **Consistency**: Runtime behavior matches what end users will have

______________________________________________________________________

### 4.3 Devel Meta with GfxArch=False Devel Dependencies

**Example**: `amdrocm-developer-tools-devel` depends on devel packages for GfxArch=False components

**Characteristics**:

- Meta package aggregating devel packages
- All child devel packages have GfxArch=False runtime dependencies
- No architecture-specific variants

#### Package Configuration

```json
{
  "Package": "amdrocm-developer-tools-devel",
  "Metapackage": "True",
  "Gfxarch": "False",
  "DEBDepends": [
    "amdrocm-llvm-devel",
    "amdrocm-amdsmi-devel",
    "amdrocm-profiler-base-devel"
  ]
}
```

#### Dependency Tree

```
amdrocm-developer-tools-devel (non-versioned)
│
│   [No files]
│
└─► amdrocm-developer-tools-devel8.2 (versioned meta)
    │
    │   [No files]
    │
    ├─► amdrocm-llvm-devel8.2
    │   │   [Files: headers, cmake]
    │   └─► amdrocm-llvm8.2 (GfxArch=False → VERSIONED)
    │
    ├─► amdrocm-amdsmi-devel8.2
    │   │   [Files: headers, cmake]
    │   └─► amdrocm-amdsmi8.2 (GfxArch=False → VERSIONED)
    │
    └─► amdrocm-profiler-base-devel8.2
        │   [Files: headers, cmake]
        └─► amdrocm-profiler-base8.2 (GfxArch=False → VERSIONED)
```

#### Package Generation

| Package Name                       | Type           | Files | Direct Dependencies                                    |
| ---------------------------------- | -------------- | ----- | ------------------------------------------------------ |
| `amdrocm-developer-tools-devel8.2` | Versioned Meta | None  | llvm-devel8.2, amdsmi-devel8.2, profiler-base-devel8.2 |
| `amdrocm-developer-tools-devel`    | Non-versioned  | None  | amdrocm-developer-tools-devel8.2                       |

#### Visual Diagram

```
┌───────────────────────────────────────────────────────┐
│          amdrocm-developer-tools-devel                 │  ◄── User installs this
│                 (non-versioned)                        │
│                    [No files]                          │
└───────────────────────┬───────────────────────────────┘
                        │ depends on
                        ▼
┌───────────────────────────────────────────────────────┐
│        amdrocm-developer-tools-devel8.2                │  ◄── Versioned meta
│               (versioned meta)                         │
│                    [No files]                          │
└───────────┬───────────────┬───────────────┬───────────┘
            │               │               │
            ▼               ▼               ▼
      ┌───────────┐   ┌───────────┐   ┌────────────────┐
      │ llvm-     │   │ amdsmi-   │   │ profiler-base- │
      │ devel8.2  │   │ devel8.2  │   │ devel8.2       │
      │ [headers] │   │ [headers] │   │ [headers]      │
      └─────┬─────┘   └─────┬─────┘   └───────┬────────┘
            │               │                 │
            ▼               ▼                 ▼
      ┌───────────┐   ┌───────────┐   ┌────────────────┐
      │ llvm8.2   │   │ amdsmi8.2 │   │ profiler-base  │
      │(VERSIONED)│   │(VERSIONED)│   │ 8.2 (VERSIONED)│
      │ [.so libs]│   │ [.so libs]│   │ [.so libs]     │
      └───────────┘   └───────────┘   └────────────────┘
```

______________________________________________________________________

### 4.4 Devel Meta with GfxArch=True Devel Dependencies

**Example**: `amdrocm-core-devel` depends on devel packages for GfxArch=True components

**Characteristics**:

- Meta package aggregating devel packages
- Child devel packages have GfxArch=True runtime dependencies
- Depends on runtime meta (all architecture variants)
- Provides complete SDK with full runtime for all GPUs

#### Package Configuration

```json
{
  "Package": "amdrocm-core-devel",
  "Metapackage": "True",
  "Gfxarch": "False",
  "DEBDepends": [
    "amdrocm-core",
    "amdrocm-blas-devel",
    "amdrocm-fft-devel",
    "amdrocm-solver-devel",
    ...
  ]
}
```

#### Dependency Tree

**For gfxarch_list = ["gfx1100", "gfx942"]**

```
amdrocm-core-devel (non-versioned)
│
│   [No files]
│
└─► amdrocm-core-devel8.2 (versioned meta)
    │
    │   [No files]
    │
    ├─► amdrocm-core8.2 (RUNTIME META - all arch variants)
    │   │
    │   │   [No files - aggregates all arch-specific metas]
    │   │
    │   ├─► amdrocm-core8.2-gfx1100 (arch-specific meta)
    │   │   ├─► amdrocm-blas8.2-gfx1100
    │   │   ├─► amdrocm-fft8.2-gfx1100
    │   │   ├─► amdrocm-solver8.2-gfx1100
    │   │   └─► ...
    │   │
    │   └─► amdrocm-core8.2-gfx942 (arch-specific meta)
    │       ├─► amdrocm-blas8.2-gfx942
    │       ├─► amdrocm-fft8.2-gfx942
    │       ├─► amdrocm-solver8.2-gfx942
    │       └─► ...
    │
    ├─► amdrocm-blas-devel8.2
    │   │   [Files: headers, cmake]
    │   └─► amdrocm-blas8.2 (full meta via runtime dep)
    │
    ├─► amdrocm-fft-devel8.2
    │   │   [Files: headers, cmake]
    │   └─► amdrocm-fft8.2 (full meta via runtime dep)
    │
    └─► amdrocm-solver-devel8.2
        │   [Files: headers, cmake]
        └─► amdrocm-solver8.2 (full meta via runtime dep)
```

#### Package Generation

| Package Name            | Type           | Files | Direct Dependencies                                   |
| ----------------------- | -------------- | ----- | ----------------------------------------------------- |
| `amdrocm-core-devel8.2` | Versioned Meta | None  | core8.2, blas-devel8.2, fft-devel8.2, solver-devel8.2 |
| `amdrocm-core-devel`    | Non-versioned  | None  | amdrocm-core-devel8.2                                 |

#### Visual Diagram

```
┌───────────────────────────────────────────────────────┐
│               amdrocm-core-devel                       │  ◄── User installs this
│                 (non-versioned)                        │
│                    [No files]                          │
└───────────────────────┬───────────────────────────────┘
                        │ depends on
                        ▼
┌───────────────────────────────────────────────────────┐
│             amdrocm-core-devel8.2                      │  ◄── Versioned meta
│               (versioned meta)                         │
│                    [No files]                          │
└───────┬───────────────┬───────────────┬───────────────┘
        │               │               │
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌───────────┐   ┌───────────┐
│ amdrocm-core  │ │ blas-     │   │ fft-      │  ...
│    8.2        │ │ devel8.2  │   │ devel8.2  │
│ (RUNTIME META)│ │ [headers] │   │ [headers] │
└───────┬───────┘ └─────┬─────┘   └─────┬─────┘
        │               │               │
        ▼               ▼               ▼
┌───────────────────────────────────────────────────────┐
│                  FULL RUNTIME                          │
│                                                        │
│  ┌─────────────────────┐    ┌─────────────────────┐   │
│  │ core8.2-gfx1100     │    │ core8.2-gfx942      │   │
│  │ ├─ blas8.2-gfx1100  │    │ ├─ blas8.2-gfx942   │   │
│  │ ├─ fft8.2-gfx1100   │    │ ├─ fft8.2-gfx942    │   │
│  │ └─ solver8.2-gfx1100│    │ └─ solver8.2-gfx942 │   │
│  └─────────────────────┘    └─────────────────────┘   │
│                                                        │
│  + All host packages (blas-host8.2, fft-host8.2, ...) │
└───────────────────────────────────────────────────────┘
```

#### Why Include Runtime Meta?

1. **Complete SDK**: Single install provides headers + full runtime for all GPUs
1. **Immediate testing**: Developers can compile and run on any supported GPU
1. **Consistent environment**: Matches production deployment configuration
1. **Simplified workflow**: No need to separately install runtime packages

______________________________________________________________________

### Devel Package Summary

#### Dependency Resolution Rules

| Devel Package Type | Runtime Dep GfxArch | Resolves To             |
| ------------------ | ------------------- | ----------------------- |
| Non-Meta           | False               | Versioned               |
| Non-Meta           | True                | Full Meta (host+devs)   |
| Meta               | False               | Versioned               |
| Meta               | True                | Runtime Meta (all arch) |

#### Package Generation Summary

| Use Case | Example                       | Packages Created        |
| -------- | ----------------------------- | ----------------------- |
| 4.1      | amdrocm-llvm-devel            | 2 (versioned + non-ver) |
| 4.2      | amdrocm-blas-devel            | 2 (versioned + non-ver) |
| 4.3      | amdrocm-developer-tools-devel | 2 (versioned + non-ver) |
| 4.4      | amdrocm-core-devel            | 2 (versioned + non-ver) |

#### Key Design Decisions

1. **All devel packages are GfxArch=False** — no host-device split for devel
1. **GfxArch=True runtime deps resolve to full meta** — provides complete runtime (host + all devices)
1. **Devel meta depends on runtime meta** — ensures all arch variants are available
1. **Complete environment** — developers can compile AND run immediately
1. **Naming convention**: `-devel` (RPM) / `-dev` (DEB, auto-transformed)

______________________________________________________________________

**Document Version**: 1.1
**Date**: 2026-07-07
**Architectures Used in Examples**: gfx1100, gfx942
**ROCm Version Used in Examples**: 8.2
