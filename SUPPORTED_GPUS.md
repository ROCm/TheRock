# ROCm Support on GPUs and APUs

> [!WARNING]
> This project is still under active development and is not yet stable for
> production use.

> ⚠️ **Note:** This document covers the development-status of GPU support. To download official development builds for various configurations, please see the release page: [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

TheRock represents the development branch of ROCm and this page documents the progress that each AMD GPU architecture is making towards being supported in a future released version of ROCm. The official [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) should be consulted for AMD GPU support in released ROCm software, whereas the content on this page serves as a leading indicator for what will be referenced there. Please check back as development progresses to support each GPU architecture.

Note that some fully supported GPU architectures may show a more limited state of readiness on this page because they have been qualified through a different, pre-existing release mechanism and are still in the process of being fully onboarded to TheRock.

The tables below also serve as our **prioritized roadmap** for the architectures we plan to test and eventually support as part of TheRock. They are a list of prioritized roadmaps divided by OS (Linux/Windows) and architecture. Each individual section is its own roadmap and we will be in parallel trying to support at least one *new* architecture per section in parallel working top-to-bottom. Current focus areas are in **bold**. There will be occasional exceptions to the top-to-bottom ordering based on test device availability.

See also the [ROCm Device Support Wishlist GitHub Discussion](https://github.com/ROCm/ROCm/discussions/4276).

> [!NOTE]
> For the purposes of the table below:
>
> - *Sanity-Tested* means "either in CI or some light form of manual QA has been performed".
> - *Release-Ready* means "it is supported and tested as part of our overall release process".

> [!WARNING]
> A ✅ in the **Build Passing** column only indicates that a wheel or tarball
> is produced and published. It does **not** imply the runtime is functional
> on target hardware. Until an architecture also has ✅ in **Sanity Tested**,
> treat its packages as unverified: `pip install` will succeed, but device
> enumeration, kernel launch, or library loads may fail at runtime. Please
> file an issue if you hit one so we can prioritize coverage.

## ROCm on Linux

### AMD Instinct - Linux

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **CDNA4**    | **gfx950**  | ✅            |               |               |
| **CDNA3**    | **gfx942**  | ✅            | ✅            | ✅            |
| CDNA2        | gfx90a      | ✅            |               |               |
| CDNA         | gfx908      | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |
| GCN5.0       | gfx900      | ✅            |               |               |

### AMD Radeon - Linux

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **RDNA4**    | **gfx1201** | ✅            | ✅            | ✅            |
| **RDNA4**    | **gfx1200** | ✅            | ✅            | ✅            |
| **RDNA3.5**  | **gfx1153** | ✅            |               |               |
| **RDNA3.5**  | **gfx1152** | ✅            |               |               |
| **RDNA3.5**  | **gfx1151** | ✅            | ✅            |               |
| **RDNA3.5**  | **gfx1150** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1103** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1102** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1101** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1100** | ✅            | ✅            |               |
| RDNA2        | gfx1036     | ✅            |               |               |
| RDNA2        | gfx1035     | ✅            |               |               |
| RDNA2        | gfx1034     | ✅            |               |               |
| RDNA2        | gfx1033     | ✅            |               |               |
| RDNA2        | gfx1032     | ✅            |               |               |
| RDNA2        | gfx1031     | ✅            |               |               |
| RDNA2        | gfx1030     | ✅            |               |               |
| RDNA1        | gfx1012     | ✅            |               |               |
| RDNA1        | gfx1011     | ✅            |               |               |
| RDNA1        | gfx1010     | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |
| GCN5.0       | gfx90c      |               |               |               |
| GCN5.0       | gfx900      | ✅            |               |               |

## ROCm on Windows

Check [windows_support.md](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md) on current status of development.

### AMD Radeon - Windows

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **RDNA4**    | **gfx1201** | ✅            |               |               |
| **RDNA4**    | **gfx1200** | ✅            |               |               |
| **RDNA3.5**  | **gfx1153** | ✅            |               |               |
| **RDNA3.5**  | **gfx1152** | ✅            |               |               |
| **RDNA3.5**  | **gfx1151** | ✅            | ✅            | ✅            |
| **RDNA3.5**  | **gfx1150** | ✅            |               |               |
| **RDNA3**    | **gfx1103** | ✅            |               |               |
| **RDNA3**    | **gfx1102** | ✅            |               |               |
| **RDNA3**    | **gfx1101** | ✅            |               |               |
| **RDNA3**    | **gfx1100** | ✅            |               |               |
| RDNA2        | gfx1036     | ✅            |               |               |
| RDNA2        | gfx1035     | ✅            |               |               |
| RDNA2        | gfx1034     | ✅            |               |               |
| RDNA2        | gfx1033     | ✅            |               |               |
| RDNA2        | gfx1032     | ✅            |               |               |
| RDNA2        | gfx1031     | ✅            |               |               |
| RDNA2        | gfx1030     | ✅            |               |               |
| RDNA1        | gfx1012     | ✅            |               |               |
| RDNA1        | gfx1011     | ✅            |               |               |
| RDNA1        | gfx1010     | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |
| GCN5.0       | gfx90c      |               |               |               |
| GCN5.0       | gfx900      | ✅            |               |               |
