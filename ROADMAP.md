# Roadmap

[!WARNING] This project is still in active development and not intended for production use.

Our goal here is document the prioritized roadmap of target architectures we plan to test and eventually support as part of TheRock.

## Prioritized Target Architectures

The following is a list of prioritized roadmaps divided by OS (Linux/Windows) and architecture. Each individual section is its own roadmap and we will be in parallel trying to support at least one *new* architecture per section in parallel working top-to-bottom Current focus areas are in __bold__. There will be exceptions from the "top-to-bottom" ordering occassionally based on test device availability.

*Note* for the purposes of the table below Sanity-Tested = "either in CI or some light form of manual QA has been performed. "Release-Ready" will mean that its supported and tested as part of our overall release process.

### ROCm on Linux

**AMD Instinct**
Architecture | LLVM target | Sanity Tested | Release Ready
-- | -- | -- | --
**CDNA3** | **gfx942** | ✅ |
CDNA2 | gfx90a ||
CDNA | gfx908 ||
GCN5.1 | gfx906 ||
GCN5.1 | gfx900 ||

**AMD Radeon**
Architecture | LLVM target | Sanity Tested | Release Ready
-- | -- | -- | --
**RDNA3** | **gfx1100** | ✅ |
**RDNA3** | **gfx1101** ||
**RDNA3** | **gfx1102** ||
RDNA2 | gfx1030 ||
GCN5.1 | gfx906 ||

### ROCm on Windows

Check [windows_support.md](docs/development/windows_support.md) on current status of development.

**AMD Radeon**
Architecture | LLVM target | Sanity Tested | Release Ready
-- | -- | -- | --
**RDNA3** | **gfx1101** ||
**RDNA3** | **gfx1100** ||
RDNA2 | gfx1030 ||
GCN5.1 | gfx906 ||
