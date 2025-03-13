# Roadmap

[!WARNING] This project is still in active development and not intended for production use.

This page is inspired by the discussion in https://github.com/ROCm/ROCm/discussions/4276 .
Our goal here is document the prioritized roadmap of target architectures we plan to test and eventually support as part of TheRock.

## Prioritized Target Architectures

The following is a roadmap. We will endeavor to work top-to-bottom to go from Sanity Tested -> Release Ready per chipset within each section and move down while also supporting any new chipset as it comes to market. Each category is its own roadmap and we will be moving in parallel across all categories as much. Current focus areas are in __bold__. There will be exceptions from the "top-to-bottom" ordering occassionally based on test device availability.

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
RDNA3 | gfx1101 ||
RDNA2 | gfx1030 ||
GCN5.1 | gfx906 ||

### ROCm on Windows

Check [windows_support.md](docs/development/windows_support.md) on current status of development.

**AMD Instinct**
Architecture | LLVM target | Sanity Tested | Release Ready
-- | -- | -- | --
**CDNA3** | **gfx942** ||
CDNA2 | gfx90a ||
CDNA | gfx908 ||
GCN5.1 | gfx906 ||
GCN5.1 | gfx900 ||

**AMD Radeon**
Architecture | LLVM target | Sanity Tested | Release Ready
-- | -- | -- | --
RDNA3 | gfx1101 ||
RDNA3 | gfx1100 ||
RDNA2 | gfx1030 ||
GCN5.1 | gfx906 ||




