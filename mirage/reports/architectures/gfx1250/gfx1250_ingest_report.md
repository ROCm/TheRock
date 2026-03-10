# gfx1250 Architecture Ingest

## Source Resolution

- Official machine-readable source: `amdgpu_isa_rdna4.xml`
- Architecture mapping: `gfx1250` -> `AMD RDNA 4`
- LLVM processor model: `def : ProcessorModel<"gfx1250", GFX1250SpeedModel,`

## Counts

- RDNA4 XML instruction count: `1264`
- gfx950 catalog instruction count: `1242`
- Shared instruction names: `596`
- RDNA4-only instruction names vs gfx950: `668`
- gfx950-only instruction names vs RDNA4: `646`
- LLVM gfx1250 normalized symbols: `172`
- LLVM gfx1250 symbols not present verbatim in RDNA4 XML: `154`

## Focus Areas

- `vop3p` normalized symbols: `62`
  - LLVM-only naming sample: `V_PK_ADD_BF16, V_PK_ADD_MAX_I16, V_PK_ADD_MAX_U16, V_PK_ADD_MIN_I16, V_PK_ADD_MIN_U16, V_PK_FMA_BF16, V_PK_MAX3_I16, V_PK_MAX3_NUM_F16, V_PK_MAX3_U16, V_PK_MAXIMUM3_F16`
- `wmma` normalized symbols: `47`
  - LLVM-only naming sample: `TENSOR_LOAD_TO_LDS, TENSOR_STORE_FROM_LDS, V_SWMMAC_BF16F32_16X16X64_BF16_w32, V_SWMMAC_BF16_16X16X64_BF16_w32, V_SWMMAC_F16_16X16X128_BF8_BF8_w32, V_SWMMAC_F16_16X16X128_BF8_FP8_w32, V_SWMMAC_F16_16X16X128_FP8_BF8_w32, V_SWMMAC_F16_16X16X128_FP8_FP8_w32, V_SWMMAC_F16_16X16X64_F16_w32, V_SWMMAC_F32_16X16X128_BF8_BF8_w32`
- `fp8_bf8` normalized symbols: `87`
  - In RDNA4 XML sample: `V_CVT_F32_FP8, V_CVT_PK_FP8_F32, V_CVT_SR_FP8_F32`
  - LLVM-only naming sample: `V_CVT_F16_BF8, V_CVT_F16_FP8, V_CVT_PK_BF8_F16, V_CVT_PK_F16_BF8, V_CVT_PK_F16_FP8, V_CVT_PK_FP8_F16, V_CVT_SCALEF32_PK16_BF6_BF16, V_CVT_SCALEF32_PK16_BF6_F16, V_CVT_SCALEF32_PK16_BF6_F32, V_CVT_SCALEF32_PK16_FP6_BF16`
- `scale_paired` normalized symbols: `52`
  - In RDNA4 XML sample: `V_DIV_SCALE_F64`
  - LLVM-only naming sample: `V_CVT_SCALEF32_PK16_BF6_BF16, V_CVT_SCALEF32_PK16_BF6_F16, V_CVT_SCALEF32_PK16_BF6_F32, V_CVT_SCALEF32_PK16_FP6_BF16, V_CVT_SCALEF32_PK16_FP6_F16, V_CVT_SCALEF32_PK16_FP6_F32, V_CVT_SCALEF32_PK8_BF8_BF16, V_CVT_SCALEF32_PK8_BF8_F16, V_CVT_SCALEF32_PK8_BF8_F32, V_CVT_SCALEF32_PK8_FP4_BF16`

## Provenance Notes

- GPUOpen publishes the source XML at the architecture level as `AMD RDNA 4`, not as target-tagged `gfx1250` XML.
- `gfx1250` is anchored locally through LLVM `ProcessorModel<"gfx1250", ... FeatureISAVersion12_50 ...>` and the associated `gfx1250`-specific defs.
- The high-value simulator deltas visible in LLVM center on VOP3P, WMMA/SWMMAC, FP8/BF8/FP6/BF6/FP4 forms, and scale/paired operations.

## Recommended Next Slice

- Generate a reusable RDNA4 catalog in Mirage, then add a gfx12 target selector that overlays LLVM-derived `gfx1250` target deltas on top of the architecture-level RDNA4 inventory.
- Prioritize decoder/catalog plumbing for VOP3P, WMMA/SWMMAC, FP8/BF8 conversion ops, and scale/paired forms before generic long-tail RDNA4 coverage.
