# gfx1250 Stub Decoder Entrypoints

## Summary

- Public entrypoint: `DecodeStubInstruction`
- Route-keyed entrypoints:
  - `DecodeVop3pStub`
  - `DecodeMimgTensorStub`
  - `DecodeVop1Stub`
  - `DecodeVop3SdstStub`
- Current first-pass routed seed coverage: `70`
- Stub decode results now include:
  - opcode shape classification
  - execution-domain classification
  - scale/paired/tensor/accumulator flags
  - operand-layout records for first high-value seeds

## First Operand Layout Records

- `V_PK_ADD_BF16`
  - Layout: `kPkAddBf16`
  - Sources: `2`
  - Destinations: `1`

- `V_PK_FMA_BF16`
  - Layout: `kPkFmaBf16`
  - Sources: `3`
  - Destinations: `1`

- `V_WMMA_F32_16X16X4_F32_w32`
  - Layout: `kWmmaF32_16x16x4_F32W32`
  - Sources: `2`
  - Destination vectors: `1`
  - Accumulator sources: `1`

- `V_WMMA_LD_SCALE_PAIRED_B32`
  - Layout: `kWmmaLdScalePairedB32`
  - Sources: `2`
  - Destinations: `1`
  - Flags:
    - `has_scale_operand`
    - `has_paired_scale_operand`

- `TENSOR_LOAD_TO_LDS`
  - Layout: `kTensorLoadToLds`
  - Sources: `2`
  - Destinations: `0`
  - Flags:
    - `has_tensor_descriptor`
    - `touches_lds`

- `TENSOR_STORE_FROM_LDS`
  - Layout: `kTensorStoreFromLds`
  - Sources: `2`
  - Destinations: `0`
  - Flags:
    - `has_tensor_descriptor`
    - `touches_lds`
    - `is_store`

## Entrypoint Coverage

- `DecodeVop3pStub`
  - Route: `kVop3p`
  - Priority: `1`
  - Routed instructions: `62`
  - Shapes:
    - `kVop3pPackedBinary`
    - `kVop3pPackedFma`
    - `kWmmaCore`
    - `kWmmaScale`
    - `kWmmaScalePairedLoad`
    - `kSwmmacCore`
  - Representative seeds:
    - `V_PK_ADD_BF16`
    - `V_PK_FMA_BF16`
    - `V_WMMA_F32_16X16X4_F32_w32`
    - `V_WMMA_LD_SCALE_PAIRED_B32`

- `DecodeMimgTensorStub`
  - Route: `kMimgTensor`
  - Priority: `2`
  - Routed instructions: `2`
  - Shapes:
    - `kTensorLoadToLds`
    - `kTensorStoreFromLds`
  - Representative seeds:
    - `TENSOR_LOAD_TO_LDS`
    - `TENSOR_STORE_FROM_LDS`

- `DecodeVop1Stub`
  - Route: `kVop1`
  - Priority: `3`
  - Routed instructions: `5`
  - Shapes:
    - `kFp8ConvertToF16`
    - `kFp8ConvertToF32`
    - `kFp8PackedConvert`
  - Representative seeds:
    - `V_CVT_F16_FP8`
    - `V_CVT_F16_BF8`
    - `V_CVT_F32_FP8`

- `DecodeVop3SdstStub`
  - Route: `kVop3Sdst`
  - Priority: `4`
  - Routed instructions: `1`
  - Shape:
    - `kVop3SdstScale`
  - Representative seed:
    - `V_DIV_SCALE_F64`

## Deferred

- Deferred first-pass seeds are the current `kVop3` cases from the route selector, such as:
  - `V_CVT_PK_FP8_F32`
  - `V_CVT_SCALEF32_PK8_FP8_F32`
  - `V_CVT_SCALEF32_SR_PK8_FP8_F32`

## Recommended Next Slice

- Turn these first operand-layout records into route-local operand-role records for the same six seeds.
- Extend explicit operand layouts next to `V_CVT_F16_FP8`, `V_CVT_F32_FP8`, and `V_DIV_SCALE_F64`.
- After that, widen the same layout model across deferred `kVop3` FP8/scale seeds.
