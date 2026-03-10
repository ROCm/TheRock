# gfx1250 Decoder Seed Report

## Summary

- Seeded instruction union: `122`
- XML-backed seeded instructions: `4`
- LLVM-only seeded instructions: `118`

## vop3p

- Seeded instructions: `62`
- XML-backed: `0`
- LLVM-only: `62`
- Decode hint `kVop1`: `0`
- Decode hint `kVop3`: `0`
- Decode hint `kVop3p`: `62`
- Decode hint `kVop3Sdst`: `0`
- Decode hint `kMimgTensor`: `0`

- `V_PK_ADD_BF16`
- `V_PK_FMA_BF16`
- `V_PK_ADD_MAX_I16`
- `V_PK_ADD_MAX_U16`
- `V_PK_ADD_MIN_I16`
- `V_PK_ADD_MIN_U16`
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

## wmma

- Seeded instructions: `47`
- XML-backed: `0`
- LLVM-only: `47`
- Decode hint `kVop1`: `0`
- Decode hint `kVop3`: `0`
- Decode hint `kVop3p`: `45`
- Decode hint `kVop3Sdst`: `0`
- Decode hint `kMimgTensor`: `2`

- `V_WMMA_F32_16X16X4_F32_w32`
- `V_WMMA_BF16F32_16X16X32_BF16_w32`
- `V_SWMMAC_F32_16X16X64_F16_w32`
- `TENSOR_LOAD_TO_LDS`
- `TENSOR_STORE_FROM_LDS`
- `V_SWMMAC_BF16F32_16X16X64_BF16_w32`
- `V_SWMMAC_BF16_16X16X64_BF16_w32`
- `V_SWMMAC_F16_16X16X128_BF8_BF8_w32`
- `V_SWMMAC_F16_16X16X128_BF8_FP8_w32`
- `V_SWMMAC_F16_16X16X128_FP8_BF8_w32`
- `V_SWMMAC_F16_16X16X128_FP8_FP8_w32`
- `V_SWMMAC_F16_16X16X64_F16_w32`
- `V_SWMMAC_F32_16X16X128_BF8_BF8_w32`
- `V_SWMMAC_F32_16X16X128_BF8_FP8_w32`
- `V_SWMMAC_F32_16X16X128_FP8_BF8_w32`
- `V_SWMMAC_F32_16X16X128_FP8_FP8_w32`
- `V_SWMMAC_F32_16X16X64_BF16_w32`
- `V_SWMMAC_I32_16X16X128_IU8_w32`
- `V_WMMA_BF16_16X16X32_BF16_w32`
- `V_WMMA_F16_16X16X128_BF8_BF8_w32`
- `V_WMMA_F16_16X16X128_BF8_FP8_w32`
- `V_WMMA_F16_16X16X128_FP8_BF8_w32`
- `V_WMMA_F16_16X16X128_FP8_FP8_w32`
- `V_WMMA_F16_16X16X32_F16_w32`

## fp8_bf8

- Seeded instructions: `87`
- XML-backed: `3`
- LLVM-only: `84`
- Decode hint `kVop1`: `5`
- Decode hint `kVop3`: `52`
- Decode hint `kVop3p`: `30`
- Decode hint `kVop3Sdst`: `0`
- Decode hint `kMimgTensor`: `0`

- `V_CVT_F16_FP8`
- `V_CVT_F16_BF8`
- `V_CVT_PK_FP8_F16`
- `V_CVT_F32_FP8`
- `V_CVT_PK_BF8_F16`
- `V_CVT_PK_F16_BF8`
- `V_CVT_PK_F16_FP8`
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

## scale_paired

- Seeded instructions: `52`
- XML-backed: `1`
- LLVM-only: `51`
- Decode hint `kVop1`: `0`
- Decode hint `kVop3`: `45`
- Decode hint `kVop3p`: `6`
- Decode hint `kVop3Sdst`: `1`
- Decode hint `kMimgTensor`: `0`

- `V_WMMA_LD_SCALE_PAIRED_B32`
- `V_WMMA_LD_SCALE16_PAIRED_B64`
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
- `V_CVT_SCALEF32_SR_PK8_BF8_BF16`

## Recommended Next Slice

- Add a gfx1250-local opcode selector that starts from `DecoderSeedInfo.decode_hint` and dispatches the first decoder stubs for `kVop3p`, `kMimgTensor`, `kVop1`, and `kVop3Sdst`.
- Start with `VOP3P` packed math and `WMMA`/`SWMMAC` forms, then add `TENSOR_LOAD_TO_LDS` and `TENSOR_STORE_FROM_LDS` as the first tensor-memory stubs.
- Use the XML-backed `FP8` / `BF8` / scale forms to seed real opcode-field extraction before attempting architecture-specific long-tail instructions that exist only in LLVM today.
