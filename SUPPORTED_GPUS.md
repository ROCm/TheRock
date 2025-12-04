# ROCm Support on GPUs and APUs

> ⚠️ **Note:** This document covers the development-status of GPU support. To download official development builds for various configurations, please see the release page: [RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

TheRock represents the development branch of ROCm and this page documents the progress that each AMD GPU architecture is making towards being supported in a future released version of ROCm. The official [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) should be consulted for AMD GPU support in released ROCm software, whereas the content on this page serves as a leading indicator for what will be referenced there. Please check back as development progresses to support each GPU architecture.

Note that some fully supported GPU architectures may show a more limited state of readiness on this page because they have been qualified through a different, pre-existing release mechanism and are still in the process of being fully onboarded to TheRock.

> [!NOTE]
> For the purposes of the table below:
>
> - *Sanity-Tested* means "either in CI or some light form of manual QA has been performed".
> - *Release-Ready* means "it is supported and tested as part of our overall release process".

## ROCm on Linux

### AMD Instinct - Linux

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **CDNA4**    | **gfx950**  | ✅            |               |               |
| **CDNA3**    | **gfx942**  | ✅            | ✅            | ✅            |
| CDNA2        | gfx90a      | ✅            |               |               |
| CDNA         | gfx908      | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |

### AMD Radeon - Linux

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **RDNA4**    | **gfx1201** | ✅            | ✅            | ✅            |
| **RDNA4**    | **gfx1200** | ✅            | ✅            | ✅            |
| **RDNA3.5**  | **gfx1153** |               |               |               |
| **RDNA3.5**  | **gfx1151** | ✅            | ✅            |               |
| **RDNA3.5**  | **gfx1150** | ✅            | ✅            |               |
| **RDNA3.5**  | **gfx1152** |               |               |               |
| **RDNA3**    | **gfx1103** |               | ✅            |               |
| **RDNA3**    | **gfx1102** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1101** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1100** | ✅            | ✅            |               |
| RDNA2        | gfx1030     | ✅            | ✅            |               |
| RDNA1        | gfx1012     | ✅            |               |               |
| RDNA1        | gfx1011     | ✅            |               |               |
| RDNA1        | gfx1010     | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |

## ROCm on Windows

Check [windows_support.md](<%5Bdocs/development/windows_support.md%5D(https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md)>) on current status of development.

### AMD Radeon - Windows

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **RDNA4**    | **gfx1201** | ✅            | ✅            |               |
| **RDNA4**    | **gfx1200** | ✅            | ✅            |               |
| **RDNA3.5**  | **gfx1151** | ✅            | ✅            | ✅            |
| **RDNA3.5**  | **gfx1150** | ✅            | ✅            |               |
| **RDNA3**    | **gfx1103** |               | ✅            |               |
| **RDNA3**    | **gfx1102** | ✅            |               |               |
| **RDNA3**    | **gfx1101** | ✅            |               |               |
| **RDNA3**    | **gfx1100** | ✅            |               |               |
| RDNA2        | gfx1030     | ✅            | ✅            |               |
| RDNA1        | gfx1012     | ✅            |               |               |
| RDNA1        | gfx1011     | ✅            |               |               |
| RDNA1        | gfx1010     | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |
