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
  - operand-layout records for routed high-value seeds
  - operand-role records for routed high-value seeds
  - operand-slot records for routed high-value seeds

## Routed Seed Metadata Coverage

- `VOP3P` packed BF16 slice:
  - `V_PK_ADD_BF16`
  - `V_PK_FMA_BF16`
  - `V_PK_MUL_BF16`
  - `V_PK_MIN_NUM_BF16`
  - `V_PK_MAX_NUM_BF16`
- `VOP3P` WMMA / SWMMAC slice:
  - `V_WMMA_F32_16X16X4_F32_w32`
  - `V_WMMA_SCALE_F32_16X16X128_F8F6F4`
  - `V_WMMA_SCALE16_F32_16X16X128_F8F6F4`
  - `V_WMMA_LD_SCALE_PAIRED_B32`
  - `V_WMMA_LD_SCALE16_PAIRED_B64`
  - `V_SWMMAC_F32_16X16X128_FP8_FP8_w32`
- `MIMG tensor` slice:
  - `TENSOR_LOAD_TO_LDS`
  - `TENSOR_STORE_FROM_LDS`
- `VOP1` FP8/BF8 slice:
  - `V_CVT_F16_FP8`
  - `V_CVT_F16_BF8`
  - `V_CVT_F32_FP8`
  - `V_CVT_PK_F16_FP8`
  - `V_CVT_PK_F16_BF8`
- `VOP3_SDST` scale-assist slice:
  - `V_DIV_SCALE_F64`

## Explicit Operand Layout Records

- Packed BF16 binary:
  - `V_PK_ADD_BF16` -> `kPkAddBf16`
  - `V_PK_MUL_BF16` -> `kPkMulBf16`
  - `V_PK_MIN_NUM_BF16` -> `kPkMinNumBf16`
  - `V_PK_MAX_NUM_BF16` -> `kPkMaxNumBf16`
  - Sources: `2`
  - Destinations: `1`
- Packed BF16 ternary:
  - `V_PK_FMA_BF16` -> `kPkFmaBf16`
  - Sources: `3`
  - Destinations: `1`
- WMMA core:
  - `V_WMMA_F32_16X16X4_F32_w32` -> `kWmmaF32_16x16x4_F32W32`
  - Sources: `2`
  - Destination vectors: `1`
  - Accumulator sources: `1`
- WMMA scale:
  - `V_WMMA_SCALE_F32_16X16X128_F8F6F4` -> `kWmmaScaleF32_16x16x128_F8F6F4`
  - `V_WMMA_SCALE16_F32_16X16X128_F8F6F4` -> `kWmmaScale16F32_16x16x128_F8F6F4`
  - Sources: `3`
  - Destinations: `1`
  - Accumulator sources: `1`
  - Flags:
    - `has_scale_operand`
- WMMA scale paired loads:
  - `V_WMMA_LD_SCALE_PAIRED_B32` -> `kWmmaLdScalePairedB32`
  - `V_WMMA_LD_SCALE16_PAIRED_B64` -> `kWmmaLdScale16PairedB64`
  - Sources: `2`
  - Destinations: `1`
  - Flags:
    - `has_scale_operand`
    - `has_paired_scale_operand`
- SWMMAC:
  - `V_SWMMAC_F32_16X16X128_FP8_FP8_w32` -> `kSwmmacF32_16x16x128_Fp8Fp8W32`
  - Sources: `2`
  - Destinations: `1`
  - Accumulator sources: `1`
- Tensor routes:
  - `TENSOR_LOAD_TO_LDS` -> `kTensorLoadToLds`
  - `TENSOR_STORE_FROM_LDS` -> `kTensorStoreFromLds`
  - Flags:
    - `has_tensor_descriptor`
    - `touches_lds`
    - `is_store` for store path
- VOP1 FP8/BF8 conversions:
  - `V_CVT_F16_FP8` -> `kCvtF16Fp8`
  - `V_CVT_F16_BF8` -> `kCvtF16Bf8`
  - `V_CVT_F32_FP8` -> `kCvtF32Fp8`
  - `V_CVT_PK_F16_FP8` -> `kCvtPkF16Fp8`
  - `V_CVT_PK_F16_BF8` -> `kCvtPkF16Bf8`
- VOP3 SDST scale assist:
  - `V_DIV_SCALE_F64` -> `kVDivScaleF64`
  - Sources: `3`
  - Destinations: `2`
  - Flags:
    - `has_scale_operand`

## Operand Role Records

- Packed BF16 binary:
  - `kSource0`
  - `kSource1`
  - `kDestination`
- Packed BF16 ternary:
  - `kSource0`
  - `kSource1`
  - `kSource2`
  - `kDestination`
- WMMA / SWMMAC core:
  - `kSource0`
  - `kSource1`
  - `kAccumulator`
  - `kDestination`
- WMMA scale:
  - `kSource0`
  - `kSource1`
  - `kAccumulator`
  - `kScale`
  - `kDestination`
- WMMA scale paired loads:
  - `kSource0`
  - `kScale`
  - `kPairedScale`
  - `kDestination`
- Tensor routes:
  - load: `kTensorDescriptor`, `kTensorCoordinate`, `kLdsDestination`
  - store: `kTensorDescriptor`, `kTensorCoordinate`, `kLdsSource`
- VOP1 FP8/BF8 conversions:
  - scalar-width conversions: `kSource0`, `kDestination`
  - packed conversions: `kSource0`, `kDestination`
- VOP3 SDST scale assist:
  - `kSource0`
  - `kSource1`
  - `kScale`
  - `kDestination`

## Operand Slot Records

- Packed BF16 binary slots:
  - destination: packed vector, logical operand `0`, components `2`
  - source0: packed vector, logical operand `1`, components `2`
  - source1: packed vector, logical operand `2`, components `2`
- Packed BF16 ternary slots:
  - destination: packed vector, logical operand `0`, components `2`
  - source0: packed vector, logical operand `1`, components `2`
  - source1: packed vector, logical operand `2`, components `2`
  - source2: packed vector, logical operand `3`, components `2`
- WMMA / SWMMAC core slots:
  - destination: matrix fragment, logical operand `0`
  - source0: matrix fragment, logical operand `1`
  - source1: matrix fragment, logical operand `2`
  - accumulator: accumulator fragment, logical operand `3`
- WMMA scale slots:
  - destination: matrix fragment, logical operand `0`
  - source0: matrix fragment, logical operand `1`
  - source1: matrix fragment, logical operand `2`
  - accumulator: accumulator fragment, logical operand `3`
  - scale: scalar register, logical operand `4`
- WMMA scale paired-load slots:
  - destination: vector register, logical operand `0`
  - source0: vector register, logical operand `1`
  - scale: scalar register, logical operand `2`
  - paired scale: scalar register, logical operand `3`
- Tensor slots:
  - descriptor: tensor descriptor, logical operand `0`
  - coordinate: tensor coordinate, logical operand `1`
  - LDS load/store path: logical operand `2`
- VOP1 FP8/BF8 scalar-width slots:
  - destination: vector register, logical operand `0`
  - source0: vector register, logical operand `1`
- VOP1 FP8/BF8 packed slots:
  - destination: packed vector, logical operand `0`, components `2`
  - source0: packed vector, logical operand `1`, components `2`
- VOP3 SDST scale-assist slots:
  - vector destination: logical operand `0`, components `2`
  - scalar destination: logical operand `1`
  - source0: vector register, logical operand `2`, components `2`
  - source1: vector register, logical operand `3`, components `2`
  - scale source: vector register, logical operand `4`, components `2`

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
- `DecodeMimgTensorStub`
  - Route: `kMimgTensor`
  - Priority: `2`
  - Routed instructions: `2`
  - Shapes:
    - `kTensorLoadToLds`
    - `kTensorStoreFromLds`
- `DecodeVop1Stub`
  - Route: `kVop1`
  - Priority: `3`
  - Routed instructions: `5`
  - Shapes:
    - `kFp8ConvertToF16`
    - `kFp8ConvertToF32`
    - `kFp8PackedConvert`
- `DecodeVop3SdstStub`
  - Route: `kVop3Sdst`
  - Priority: `4`
  - Routed instructions: `1`
  - Shape:
    - `kVop3SdstScale`

## Deferred

- Deferred first-pass seeds remain the current `kVop3` cases from the route selector, such as:
  - `V_CVT_PK_FP8_F32`
  - `V_CVT_SCALEF32_PK8_FP8_F32`
  - `V_CVT_SCALEF32_SR_PK8_FP8_F32`

## Recommended Next Slice

- Keep the public stub API stable and turn the current slot records into operand-class or fragment-shape semantics for the same routed seeds.
- Widen the same slot model across more `VOP3P` routed families, especially the remaining `WMMA_SCALE*`, `SWMMAC*`, and packed BF16 variants.
- After that, carry the same metadata boundary into the currently deferred `kVop3` FP8 / scale conversion seeds.
