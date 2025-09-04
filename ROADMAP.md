# Roadmap for Support

> [!WARNING]
> This project is still under active development and is not yet stable for
> production use.

Our goal here is to document the prioritized roadmap of target architectures we plan to test and eventually support as part of TheRock.

## Prioritized target architectures

The following is a list of prioritized roadmaps divided by OS (Linux/Windows) and architecture. Each individual section is its own roadmap and we will be in parallel trying to support at least one *new* architecture per section in parallel working top-to-bottom. Current focus areas are in __bold__. There will be exceptions from the "top-to-bottom" ordering occasionally based on test device availability.

See also the [ROCm Device Support Wishlist GitHub Discussion](https://github.com/ROCm/ROCm/discussions/4276)

> [!NOTE]
> For the purposes of the table below:
>
> - *Sanity-Tested* means "either in CI or some light form of manual QA has been performed".
> - *Release-Ready* means "it is supported and tested as part of our overall release process".

### ROCm on Linux

#### AMD Instinct - Linux

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **CDNA4**    | **gfx950**  | ✅            | ✅            | ✅            |
| **CDNA3**    | **gfx942**  | ✅            | ✅            | ✅            |
| CDNA2        | gfx90a      | ✅            |               |               |
| CDNA         | gfx908      | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |

#### AMD Radeon - Linux

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **RDNA4**    | **gfx1201** | ✅            | ✅            | ✅            |
| **RDNA4**    | **gfx1200** | ✅            | ✅            | ✅            |
| **RDNA3.5**  | **gfx1151** | ✅            |               |               |
| **RDNA3.5**  | **gfx1150** | ✅            |               |               |
| **RDNA3**    | **gfx1102** | ✅            |               |               |
| **RDNA3**    | **gfx1101** | ✅            |               |               |
| **RDNA3**    | **gfx1100** | ✅            |               |               |
| RDNA2        | gfx1036     |               |               |               |
| RDNA2        | gfx1035     |               |               |               |
| RDNA2        | gfx1032     |               |               |               |
| RDNA2        | gfx1030     |               |               |               |
| RDNA1        | gfx1012     | ✅            |               |               |
| RDNA1        | gfx1011     | ✅            |               |               |
| RDNA1        | gfx1010     | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |

### ROCm on Windows

Check [windows_support.md](docs/development/windows_support.md) on current status of development.

#### AMD Radeon - Windows

| Architecture | LLVM target | Build Passing | Sanity Tested | Release Ready |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| **RDNA4**    | **gfx1201** | ✅            |               |               |
| **RDNA4**    | **gfx1200** | ✅            |               |               |
| **RDNA3.5**  | **gfx1151** | ✅            | ✅            |               |
| **RDNA3.5**  | **gfx1150** | ✅            |               |               |
| **RDNA3**    | **gfx1102** | ✅            |               |               |
| **RDNA3**    | **gfx1101** | ✅            |               |               |
| **RDNA3**    | **gfx1100** | ✅            | ✅            |               |
| RDNA2        | gfx1036     |               |               |               |
| RDNA2        | gfx1035     |               |               |               |
| RDNA2        | gfx1032     |               |               |               |
| RDNA2        | gfx1030     |               |               |               |
| RDNA1        | gfx1012     | ✅            |               |               |
| RDNA1        | gfx1011     | ✅            |               |               |
| RDNA1        | gfx1010     | ✅            |               |               |
| GCN5.1       | gfx906      | ✅            |               |               |
