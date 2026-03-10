# gfx1250 Stub Decoder Routing Report

## Summary

- Seeded instruction union: `122`
- Routed into first stub decoder layer: `70`
- Deferred for later decoder work: `52`

## Route Priority

1. `kVop3p`
2. `kMimgTensor`
3. `kVop1`
4. `kVop3Sdst`

## kVop3p

- Priority: `1`
- Description: VOP3P packed/vector forms
- Routed instruction count: `62`
- XML-backed: `0`
- LLVM-only: `62`
- Target-specific: `62`

- `V_PK_ADD_BF16`
- `V_PK_ADD_MAX_I16`
- `V_PK_ADD_MAX_U16`
- `V_PK_ADD_MIN_I16`
- `V_PK_ADD_MIN_U16`
- `V_PK_FMA_BF16`
- `V_PK_MAX3_I16`
- `V_PK_MAX3_NUM_F16`
- `V_PK_MAX3_U16`
- `V_PK_MAXIMUM3_F16`
- `V_PK_MAX_NUM_BF16`
- `V_PK_MIN3_I16`
- `V_PK_MIN3_NUM_F16`
- `V_PK_MIN3_U16`
- `V_PK_MINIMUM3_F16`
- `V_PK_MIN_NUM_BF16`
- `V_PK_MUL_BF16`
- `V_SWMMAC_BF16F32_16X16X64_BF16_w32`
- `V_SWMMAC_BF16_16X16X64_BF16_w32`
- `V_SWMMAC_F16_16X16X128_BF8_BF8_w32`
- `V_SWMMAC_F16_16X16X128_BF8_FP8_w32`
- `V_SWMMAC_F16_16X16X128_FP8_BF8_w32`
- `V_SWMMAC_F16_16X16X128_FP8_FP8_w32`
- `V_SWMMAC_F16_16X16X64_F16_w32`

## kMimgTensor

- Priority: `2`
- Description: tensor MIMG forms
- Routed instruction count: `2`
- XML-backed: `0`
- LLVM-only: `2`
- Target-specific: `2`

- `TENSOR_LOAD_TO_LDS`
- `TENSOR_STORE_FROM_LDS`

## kVop1

- Priority: `3`
- Description: VOP1 conversion forms
- Routed instruction count: `5`
- XML-backed: `1`
- LLVM-only: `4`
- Target-specific: `4`

- `V_CVT_F16_BF8`
- `V_CVT_F16_FP8`
- `V_CVT_F32_FP8`
- `V_CVT_PK_F16_BF8`
- `V_CVT_PK_F16_FP8`

## kVop3Sdst

- Priority: `4`
- Description: VOP3 SDST forms
- Routed instruction count: `1`
- XML-backed: `1`
- LLVM-only: `0`
- Target-specific: `0`

- `V_DIV_SCALE_F64`

## Deferred Seed Hints

- Deferred `kVop3` / unsupported-first-pass seeds: `52`

- `V_CVT_PK_BF8_F16`
- `V_CVT_PK_FP8_F16`
- `V_CVT_PK_FP8_F32`
- `V_CVT_SCALEF32_PK16_BF6_BF16`
- `V_CVT_SCALEF32_PK16_BF6_F16`
- `V_CVT_SCALEF32_PK16_BF6_F32`
- `V_CVT_SCALEF32_PK16_FP6_BF16`
- `V_CVT_SCALEF32_PK16_FP6_F16`
- `V_CVT_SCALEF32_PK16_FP6_F32`
- `V_CVT_SCALEF32_PK8_BF8_BF16`
- `V_CVT_SCALEF32_PK8_BF8_F16`
- `V_CVT_SCALEF32_PK8_BF8_F32`
- `V_CVT_SCALEF32_PK8_FP4_BF16`
- `V_CVT_SCALEF32_PK8_FP4_F16`
- `V_CVT_SCALEF32_PK8_FP4_F32`
- `V_CVT_SCALEF32_PK8_FP8_BF16`
- `V_CVT_SCALEF32_PK8_FP8_F16`
- `V_CVT_SCALEF32_PK8_FP8_F32`
- `V_CVT_SCALEF32_SR_PK16_BF6_BF16`
- `V_CVT_SCALEF32_SR_PK16_BF6_F16`
- `V_CVT_SCALEF32_SR_PK16_BF6_F32`
- `V_CVT_SCALEF32_SR_PK16_FP6_BF16`
- `V_CVT_SCALEF32_SR_PK16_FP6_F16`
- `V_CVT_SCALEF32_SR_PK16_FP6_F32`

## Recommended Next Slice

- Add architecture-local stub decoder entry points keyed by `StubDecoderRoute` instead of instruction name.
- Implement the first `kVop3p` selector stubs for `V_PK_ADD_BF16`, `V_PK_FMA_BF16`, and `V_WMMA_F32_16X16X4_F32_w32`.
- Add `kMimgTensor` stubs for `TENSOR_LOAD_TO_LDS` and `TENSOR_STORE_FROM_LDS`, then wire `kVop1` and `kVop3Sdst` after those are stable.
