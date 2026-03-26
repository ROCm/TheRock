# gfx1201 Phase-0 Wave32 Boundary

This report captures the current wave32-local boundary for the `gfx1201` phase-0
execution path.

## Saturated Local Encodings

- `ENC_VOP1`: seeded `90`, executable `90`, fully executable `true`
- `ENC_VOP2`: seeded `47`, executable `47`, fully executable `true`
- `ENC_VOPC`: seeded `162`, executable `162`, fully executable `true`
- `ENC_SMEM`: seeded `28`, executable `28`, fully executable `true`
- `ENC_VGLOBAL`: seeded `65`, executable `65`, fully executable `true`

## Summary

- Phase-0 executable opcodes: `517`
- Wave size: `32`
- Remaining narrow instruction/encoding pairs outside the saturated local seed: `0`
- Recommended next frontier: `ENC_VDS`
- Safe VDS continuation under the current request: `false`
- Recommended next VDS bucket under the current request: `""`
- First unsafe VDS bucket: `append_consume`
- First unsafe VDS blocking dimension: `allocator_or_gds_semantics`
- First unsafe VDS risk rank: `0`
- First unsafe VDS ordinal range: `0..1`
- First unsafe VDS instructions: `DS_APPEND`, `DS_CONSUME`
- Remaining VDS instruction statuses: `24`
- Remaining VDS bucket order: `append_consume`, `exchange_compare_store`, `multi_address`, `bvh_stack`
- All currently seeded ENC_VOP1, ENC_VOP2, ENC_VOPC, ENC_SMEM, and ENC_VGLOBAL instruction/encoding pairs are executable on the local wave32 path.
- There are no remaining imported ENC_VOP1, ENC_VOP2, ENC_VOPC, ENC_SMEM, or ENC_VGLOBAL instruction/encoding pairs outside the current seed surface.
- ENC_SMEM is now fully bootstrapped through S_DCACHE_INV, S_PREFETCH_INST, S_PREFETCH_INST_PC_REL, S_PREFETCH_DATA, S_BUFFER_PREFETCH_DATA, S_PREFETCH_DATA_PC_REL, S_ATC_PROBE, S_ATC_PROBE_BUFFER, the full non-buffer S_LOAD_* slice, and the matching S_BUFFER_LOAD_* slice, leaving no scalar-memory seed instructions scaffolded.
- ENC_VGLOBAL is now fully bootstrapped through GLOBAL_INV, GLOBAL_WB, GLOBAL_WBINV, the plain GLOBAL_LOAD_U8/I8/U16/I16/B32/B64/B96/B128 slice, GLOBAL_LOAD_ADDTID_B32, GLOBAL_LOAD_BLOCK, GLOBAL_LOAD_TR_B64/B128, the packed GLOBAL_LOAD_D16_U8/I8/B16 and GLOBAL_LOAD_D16_HI_U8/I8/B16 slice, the plain GLOBAL_STORE_B8/B16/B32/B64/B96/B128 slice, GLOBAL_STORE_ADDTID_B32, GLOBAL_STORE_BLOCK, GLOBAL_STORE_D16_HI_B8/B16, the 32-bit integer atomic slice GLOBAL_ATOMIC_SWAP_B32, GLOBAL_ATOMIC_CMPSWAP_B32, GLOBAL_ATOMIC_ADD/SUB/SUB_CLAMP_U32, GLOBAL_ATOMIC_MIN/MAX_I32/U32, GLOBAL_ATOMIC_AND/OR/XOR_B32, GLOBAL_ATOMIC_INC_U32, GLOBAL_ATOMIC_DEC_U32, and GLOBAL_ATOMIC_COND_SUB_U32, the 64-bit integer atomic slice GLOBAL_ATOMIC_SWAP_B64, GLOBAL_ATOMIC_CMPSWAP_B64, GLOBAL_ATOMIC_ADD/SUB_U64, GLOBAL_ATOMIC_MIN/MAX_I64/U64, GLOBAL_ATOMIC_AND/OR/XOR_B64, GLOBAL_ATOMIC_INC_U64, and GLOBAL_ATOMIC_DEC_U64, the F32 atomic slice GLOBAL_ATOMIC_ADD_F32 and GLOBAL_ATOMIC_MIN/MAX_NUM_F32, the packed-half atomic pair GLOBAL_ATOMIC_PK_ADD_F16/GLOBAL_ATOMIC_PK_ADD_BF16, and GLOBAL_ATOMIC_ORDERED_ADD_B64, leaving no seeded vector-global instructions scaffolded.
- ENC_VDS now has architecture-local executable footholds through DS_NOP, the one-address non-return 32-bit LDS update slice DS_ADD_F32, DS_ADD_U32, DS_SUB_U32, DS_RSUB_U32, DS_INC_U32, DS_DEC_U32, DS_COND_SUB_U32, DS_SUB_CLAMP_U32, DS_PK_ADD_F16, DS_PK_ADD_BF16, DS_MIN_NUM_F32, DS_MAX_NUM_F32, DS_MIN_NUM_F64, DS_MAX_NUM_F64, DS_MIN_I32, DS_MIN_U32, DS_MAX_I32, DS_MAX_U32, DS_AND_B32, DS_OR_B32, DS_XOR_B32, and DS_MSKOR_B32, the matching one-address return-value 32-bit slice DS_ADD_RTN_F32, DS_ADD_RTN_U32, DS_SUB_RTN_U32, DS_RSUB_RTN_U32, DS_INC_RTN_U32, DS_DEC_RTN_U32, DS_COND_SUB_RTN_U32, DS_SUB_CLAMP_RTN_U32, DS_PK_ADD_RTN_F16, DS_PK_ADD_RTN_BF16, DS_MIN_NUM_RTN_F32, DS_MAX_NUM_RTN_F32, DS_MIN_NUM_RTN_F64, DS_MAX_NUM_RTN_F64, DS_MIN_RTN_I32, DS_MIN_RTN_U32, DS_MAX_RTN_I32, DS_MAX_RTN_U32, DS_AND_RTN_B32, DS_OR_RTN_B32, DS_XOR_RTN_B32, and DS_MSKOR_RTN_B32, the matching one-address return-value 64-bit integer LDS update slice DS_ADD_RTN_U64, DS_SUB_RTN_U64, DS_RSUB_RTN_U64, DS_INC_RTN_U64, DS_DEC_RTN_U64, DS_MIN_RTN_I64, DS_MIN_RTN_U64, DS_MAX_RTN_I64, DS_MAX_RTN_U64, DS_AND_RTN_B64, DS_OR_RTN_B64, DS_XOR_RTN_B64, and DS_MSKOR_RTN_B64, the one-address non-return 64-bit integer LDS update slice DS_ADD_U64, DS_SUB_U64, DS_RSUB_U64, DS_INC_U64, DS_DEC_U64, DS_MIN_I64, DS_MIN_U64, DS_MAX_I64, DS_MAX_U64, DS_AND_B64, DS_OR_B64, DS_XOR_B64, and DS_MSKOR_B64, the simple one-address LDS load slice DS_LOAD_B32, DS_LOAD_ADDTID_B32, DS_LOAD_B64, DS_LOAD_B96, DS_LOAD_B128, DS_LOAD_I8, DS_LOAD_U8, DS_LOAD_I16, DS_LOAD_U16, DS_LOAD_U8_D16, DS_LOAD_U8_D16_HI, DS_LOAD_I8_D16, DS_LOAD_I8_D16_HI, DS_LOAD_U16_D16, and DS_LOAD_U16_D16_HI, the matching simple one-address LDS store slice DS_STORE_B8, DS_STORE_B16, DS_STORE_B32, DS_STORE_ADDTID_B32, DS_STORE_B64, DS_STORE_B96, DS_STORE_B128, DS_STORE_B8_D16_HI, and DS_STORE_B16_D16_HI, and the single-address lane-routing utility slice DS_SWIZZLE_B32, DS_PERMUTE_B32, DS_BPERMUTE_B32, and DS_BPERMUTE_FI_B32.
- The remaining ENC_VDS tail is now explicitly bounded by append/consume allocator semantics, exchange and compare-store forms, multi-address LDS forms including stride64, and gfx1201-specific BVH stack instructions, which is the next verification-risk step before ENC_VOP3.
- There is no safe ENC_VDS continuation under the current request boundary: every remaining bucket crosses allocator-or-GDS, exchange/compare-store, multi-address, or gfx1201-specific BVH semantics.
- The boundary report now carries an exact remaining-VDS instruction-to-bucket map so the unresolved tail can be queried directly by opcode name.
- The exact unsafe-bucket escalation order is `append_consume`, then `exchange_compare_store`, then `multi_address`, then `bvh_stack`.
- The boundary report now also carries an exact next-risk step chain with first and last instruction names, cumulative remaining counts, and explicit next-bucket handoff metadata for the unresolved VDS tail.
- The first unsafe ENC_VDS bucket is now expanded inline with its blocking dimension and exact instruction list.
- The boundary report now also carries a denormalized per-op remaining-VDS status list with bucket, blocking dimension, bucket risk rank, tail ordinal, bucket ordinal, and safe flag.

## Remaining VDS Boundary

- `append_consume`: risk rank `0`, ordinal range `0..1`, `2` instructions, example `DS_APPEND`, blocking dimension `allocator_or_gds_semantics`, safe under current request `false`, covering `DS_APPEND` and `DS_CONSUME`.
- `exchange_compare_store`: risk rank `1`, ordinal range `2..8`, `7` instructions, example `DS_CONDXCHG32_RTN_B64`, blocking dimension `exchange_compare_store_semantics`, safe under current request `false`, covering `DS_CONDXCHG32_RTN_B64`, `DS_CMPSTORE_B32`, `DS_CMPSTORE_B64`, `DS_CMPSTORE_RTN_B32`, `DS_CMPSTORE_RTN_B64`, `DS_STOREXCHG_RTN_B32`, and `DS_STOREXCHG_RTN_B64`.
- `multi_address`: risk rank `2`, ordinal range `9..20`, `12` instructions, example `DS_LOAD_2ADDR_B32`, blocking dimension `multi_address_semantics`, safe under current request `false`, covering `DS_LOAD_2ADDR_B32`, `DS_LOAD_2ADDR_B64`, `DS_LOAD_2ADDR_STRIDE64_B32`, `DS_LOAD_2ADDR_STRIDE64_B64`, `DS_STOREXCHG_2ADDR_RTN_B32`, `DS_STOREXCHG_2ADDR_RTN_B64`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`, `DS_STORE_2ADDR_B32`, `DS_STORE_2ADDR_B64`, `DS_STORE_2ADDR_STRIDE64_B32`, and `DS_STORE_2ADDR_STRIDE64_B64`.
- `bvh_stack`: risk rank `3`, ordinal range `21..23`, `3` instructions, example `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, blocking dimension `gfx1201_specific_bvh_semantics`, safe under current request `false`, covering `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and `DS_BVH_STACK_PUSH8_POP2_RTN_B64`.

## Remaining VDS Next-Risk Chain

- `append_consume`: first `DS_APPEND`, last `DS_CONSUME`, `2` instructions, `24` remaining including this bucket, `22` remaining after it, next bucket `exchange_compare_store`, next blocking dimension `exchange_compare_store_semantics`, next instruction `DS_CONDXCHG32_RTN_B64`
- `exchange_compare_store`: first `DS_CONDXCHG32_RTN_B64`, last `DS_STOREXCHG_RTN_B64`, `7` instructions, `22` remaining including this bucket, `15` remaining after it, next bucket `multi_address`, next blocking dimension `multi_address_semantics`, next instruction `DS_LOAD_2ADDR_B32`
- `multi_address`: first `DS_LOAD_2ADDR_B32`, last `DS_STORE_2ADDR_STRIDE64_B64`, `12` instructions, `15` remaining including this bucket, `3` remaining after it, next bucket `bvh_stack`, next blocking dimension `gfx1201_specific_bvh_semantics`, next instruction `DS_BVH_STACK_PUSH4_POP1_RTN_B32`
- `bvh_stack`: first `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, last `DS_BVH_STACK_PUSH8_POP2_RTN_B64`, `3` instructions, `3` remaining including this bucket, `0` remaining after it, next bucket `""`, next blocking dimension `""`, next instruction `""`

## Remaining VDS Instruction Map

- `DS_APPEND` -> `append_consume`
- `DS_CONSUME` -> `append_consume`
- `DS_CONDXCHG32_RTN_B64` -> `exchange_compare_store`
- `DS_CMPSTORE_B32` -> `exchange_compare_store`
- `DS_CMPSTORE_B64` -> `exchange_compare_store`
- `DS_CMPSTORE_RTN_B32` -> `exchange_compare_store`
- `DS_CMPSTORE_RTN_B64` -> `exchange_compare_store`
- `DS_STOREXCHG_RTN_B32` -> `exchange_compare_store`
- `DS_STOREXCHG_RTN_B64` -> `exchange_compare_store`
- `DS_LOAD_2ADDR_B32` -> `multi_address`
- `DS_LOAD_2ADDR_B64` -> `multi_address`
- `DS_LOAD_2ADDR_STRIDE64_B32` -> `multi_address`
- `DS_LOAD_2ADDR_STRIDE64_B64` -> `multi_address`
- `DS_STOREXCHG_2ADDR_RTN_B32` -> `multi_address`
- `DS_STOREXCHG_2ADDR_RTN_B64` -> `multi_address`
- `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32` -> `multi_address`
- `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64` -> `multi_address`
- `DS_STORE_2ADDR_B32` -> `multi_address`
- `DS_STORE_2ADDR_B64` -> `multi_address`
- `DS_STORE_2ADDR_STRIDE64_B32` -> `multi_address`
- `DS_STORE_2ADDR_STRIDE64_B64` -> `multi_address`
- `DS_BVH_STACK_PUSH4_POP1_RTN_B32` -> `bvh_stack`
- `DS_BVH_STACK_PUSH8_POP1_RTN_B32` -> `bvh_stack`
- `DS_BVH_STACK_PUSH8_POP2_RTN_B64` -> `bvh_stack`

## Remaining VDS Instruction Statuses

- `DS_APPEND`: bucket `append_consume`, blocking dimension `allocator_or_gds_semantics`, bucket risk rank `0`, tail ordinal `0`, bucket ordinal `0`, safe under current request `false`
- `DS_CONSUME`: bucket `append_consume`, blocking dimension `allocator_or_gds_semantics`, bucket risk rank `0`, tail ordinal `1`, bucket ordinal `1`, safe under current request `false`
- `DS_CONDXCHG32_RTN_B64`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `2`, bucket ordinal `0`, safe under current request `false`
- `DS_CMPSTORE_B32`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `3`, bucket ordinal `1`, safe under current request `false`
- `DS_CMPSTORE_B64`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `4`, bucket ordinal `2`, safe under current request `false`
- `DS_CMPSTORE_RTN_B32`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `5`, bucket ordinal `3`, safe under current request `false`
- `DS_CMPSTORE_RTN_B64`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `6`, bucket ordinal `4`, safe under current request `false`
- `DS_STOREXCHG_RTN_B32`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `7`, bucket ordinal `5`, safe under current request `false`
- `DS_STOREXCHG_RTN_B64`: bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `8`, bucket ordinal `6`, safe under current request `false`
- `DS_LOAD_2ADDR_B32`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `9`, bucket ordinal `0`, safe under current request `false`
- `DS_LOAD_2ADDR_B64`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `10`, bucket ordinal `1`, safe under current request `false`
- `DS_LOAD_2ADDR_STRIDE64_B32`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `11`, bucket ordinal `2`, safe under current request `false`
- `DS_LOAD_2ADDR_STRIDE64_B64`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `12`, bucket ordinal `3`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_RTN_B32`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `13`, bucket ordinal `4`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_RTN_B64`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `14`, bucket ordinal `5`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `15`, bucket ordinal `6`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `16`, bucket ordinal `7`, safe under current request `false`
- `DS_STORE_2ADDR_B32`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `17`, bucket ordinal `8`, safe under current request `false`
- `DS_STORE_2ADDR_B64`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `18`, bucket ordinal `9`, safe under current request `false`
- `DS_STORE_2ADDR_STRIDE64_B32`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `19`, bucket ordinal `10`, safe under current request `false`
- `DS_STORE_2ADDR_STRIDE64_B64`: bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `20`, bucket ordinal `11`, safe under current request `false`
- `DS_BVH_STACK_PUSH4_POP1_RTN_B32`: bucket `bvh_stack`, blocking dimension `gfx1201_specific_bvh_semantics`, bucket risk rank `3`, tail ordinal `21`, bucket ordinal `0`, safe under current request `false`
- `DS_BVH_STACK_PUSH8_POP1_RTN_B32`: bucket `bvh_stack`, blocking dimension `gfx1201_specific_bvh_semantics`, bucket risk rank `3`, tail ordinal `22`, bucket ordinal `1`, safe under current request `false`
- `DS_BVH_STACK_PUSH8_POP2_RTN_B64`: bucket `bvh_stack`, blocking dimension `gfx1201_specific_bvh_semantics`, bucket risk rank `3`, tail ordinal `23`, bucket ordinal `2`, safe under current request `false`

## Next-Risk Encodings

- `ENC_SMEM`
- `ENC_VOP3`
- `ENC_VDS`
- `ENC_VGLOBAL`

## Suggested Frontier Order

- `ENC_SMEM`
- `ENC_VGLOBAL`
- `ENC_VDS`
- `ENC_VOP3`

## Next-Risk Encoding Status

- `ENC_SMEM`: example `S_LOAD_B32`, seeded `28`, executable `28` via `S_ATC_PROBE`, as-is `0`, decoder-rollup `3`, semantic-only `0`, gfx1201-specific `25`
- `ENC_VOP3`: example `V_ADD3_U32`, seeded `434`, executable `0` via ``, as-is `232`, decoder-rollup `91`, semantic-only `24`, gfx1201-specific `87`
- `ENC_VDS`: example `DS_ADD_U32`, seeded `123`, executable `99` via `DS_ADD_F32`, as-is `24`, decoder-rollup `38`, semantic-only `0`, gfx1201-specific `58`
- `ENC_VGLOBAL`: example `GLOBAL_LOAD_B32`, seeded `65`, executable `65` via `GLOBAL_ATOMIC_ADD_F32`, as-is `3`, decoder-rollup `0`, semantic-only `0`, gfx1201-specific `62`
