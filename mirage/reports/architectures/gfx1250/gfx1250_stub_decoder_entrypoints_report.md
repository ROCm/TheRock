# gfx1250 Stub Decoder Entrypoints

## Summary

- Public entrypoint: `DecodeStubInstruction`
- Route-keyed entrypoints:
  - `DecodeVop3pStub`
  - `DecodeMimgTensorStub`
  - `DecodeVop1Stub`
  - `DecodeVop3SdstStub`
- Current first-pass routed seed coverage: `70`

## Entrypoint Coverage

- `DecodeVop3pStub`
  - Route: `kVop3p`
  - Priority: `1`
  - Routed instructions: `62`
  - Representative seeds:
    - `V_PK_ADD_BF16`
    - `V_PK_FMA_BF16`
    - `V_WMMA_F32_16X16X4_F32_w32`
    - `V_WMMA_LD_SCALE_PAIRED_B32`

- `DecodeMimgTensorStub`
  - Route: `kMimgTensor`
  - Priority: `2`
  - Routed instructions: `2`
  - Representative seeds:
    - `TENSOR_LOAD_TO_LDS`
    - `TENSOR_STORE_FROM_LDS`

- `DecodeVop1Stub`
  - Route: `kVop1`
  - Priority: `3`
  - Routed instructions: `5`
  - Representative seeds:
    - `V_CVT_F16_FP8`
    - `V_CVT_F16_BF8`
    - `V_CVT_F32_FP8`

- `DecodeVop3SdstStub`
  - Route: `kVop3Sdst`
  - Priority: `4`
  - Routed instructions: `1`
  - Representative seed:
    - `V_DIV_SCALE_F64`

## Deferred

- Deferred first-pass seeds are the current `kVop3` cases from the route selector, such as:
  - `V_CVT_PK_FP8_F32`
  - `V_CVT_SCALEF32_PK8_FP8_F32`
  - `V_CVT_SCALEF32_SR_PK8_FP8_F32`

## Recommended Next Slice

- Replace the current route-keyed stub return values with per-route opcode-shape records.
- Start `DecodeVop3pStub` on `V_PK_ADD_BF16`, `V_PK_FMA_BF16`, and `V_WMMA_F32_16X16X4_F32_w32`.
- Start `DecodeMimgTensorStub` on `TENSOR_LOAD_TO_LDS` and `TENSOR_STORE_FROM_LDS` before expanding the deferred `kVop3` set.
