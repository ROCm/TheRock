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

- Phase-0 executable opcodes: `495`
- Wave size: `32`
- Remaining narrow instruction/encoding pairs outside the saturated local seed: `0`
- Recommended next frontier: `ENC_VDS`
- All currently seeded ENC_VOP1, ENC_VOP2, ENC_VOPC, ENC_SMEM, and ENC_VGLOBAL instruction/encoding pairs are executable on the local wave32 path.
- There are no remaining imported ENC_VOP1, ENC_VOP2, ENC_VOPC, ENC_SMEM, or ENC_VGLOBAL instruction/encoding pairs outside the current seed surface.
- ENC_SMEM is now fully bootstrapped through S_DCACHE_INV, S_PREFETCH_INST, S_PREFETCH_INST_PC_REL, S_PREFETCH_DATA, S_BUFFER_PREFETCH_DATA, S_PREFETCH_DATA_PC_REL, S_ATC_PROBE, S_ATC_PROBE_BUFFER, the full non-buffer S_LOAD_* slice, and the matching S_BUFFER_LOAD_* slice, leaving no scalar-memory seed instructions scaffolded.
- ENC_VGLOBAL is now fully bootstrapped through GLOBAL_INV, GLOBAL_WB, GLOBAL_WBINV, the plain GLOBAL_LOAD_U8/I8/U16/I16/B32/B64/B96/B128 slice, GLOBAL_LOAD_ADDTID_B32, GLOBAL_LOAD_BLOCK, GLOBAL_LOAD_TR_B64/B128, the packed GLOBAL_LOAD_D16_U8/I8/B16 and GLOBAL_LOAD_D16_HI_U8/I8/B16 slice, the plain GLOBAL_STORE_B8/B16/B32/B64/B96/B128 slice, GLOBAL_STORE_ADDTID_B32, GLOBAL_STORE_BLOCK, GLOBAL_STORE_D16_HI_B8/B16, the 32-bit integer atomic slice GLOBAL_ATOMIC_SWAP_B32, GLOBAL_ATOMIC_CMPSWAP_B32, GLOBAL_ATOMIC_ADD/SUB/SUB_CLAMP_U32, GLOBAL_ATOMIC_MIN/MAX_I32/U32, GLOBAL_ATOMIC_AND/OR/XOR_B32, GLOBAL_ATOMIC_INC_U32, GLOBAL_ATOMIC_DEC_U32, and GLOBAL_ATOMIC_COND_SUB_U32, the 64-bit integer atomic slice GLOBAL_ATOMIC_SWAP_B64, GLOBAL_ATOMIC_CMPSWAP_B64, GLOBAL_ATOMIC_ADD/SUB_U64, GLOBAL_ATOMIC_MIN/MAX_I64/U64, GLOBAL_ATOMIC_AND/OR/XOR_B64, GLOBAL_ATOMIC_INC_U64, and GLOBAL_ATOMIC_DEC_U64, the F32 atomic slice GLOBAL_ATOMIC_ADD_F32 and GLOBAL_ATOMIC_MIN/MAX_NUM_F32, the packed-half atomic pair GLOBAL_ATOMIC_PK_ADD_F16/GLOBAL_ATOMIC_PK_ADD_BF16, and GLOBAL_ATOMIC_ORDERED_ADD_B64, leaving no seeded vector-global instructions scaffolded.
- ENC_VDS now has architecture-local executable footholds through DS_NOP, the one-address non-return 32-bit LDS update slice DS_ADD_F32, DS_ADD_U32, DS_SUB_U32, DS_RSUB_U32, DS_INC_U32, DS_DEC_U32, DS_COND_SUB_U32, DS_SUB_CLAMP_U32, DS_PK_ADD_F16, DS_PK_ADD_BF16, DS_MIN_NUM_F32, DS_MAX_NUM_F32, DS_MIN_NUM_F64, DS_MAX_NUM_F64, DS_MIN_I32, DS_MIN_U32, DS_MAX_I32, DS_MAX_U32, DS_AND_B32, DS_OR_B32, and DS_XOR_B32, the matching one-address return-value slice DS_ADD_RTN_F32, DS_ADD_RTN_U32, DS_SUB_RTN_U32, DS_RSUB_RTN_U32, DS_INC_RTN_U32, DS_DEC_RTN_U32, DS_COND_SUB_RTN_U32, DS_SUB_CLAMP_RTN_U32, DS_PK_ADD_RTN_F16, DS_PK_ADD_RTN_BF16, DS_MIN_NUM_RTN_F32, DS_MAX_NUM_RTN_F32, DS_MIN_NUM_RTN_F64, DS_MAX_NUM_RTN_F64, DS_MIN_RTN_I32, DS_MIN_RTN_U32, DS_MAX_RTN_I32, DS_MAX_RTN_U32, DS_AND_RTN_B32, DS_OR_RTN_B32, and DS_XOR_RTN_B32, the one-address non-return 64-bit integer LDS update slice DS_ADD_U64, DS_SUB_U64, DS_RSUB_U64, DS_INC_U64, DS_DEC_U64, DS_MIN_I64, DS_MIN_U64, DS_MAX_I64, DS_MAX_U64, DS_AND_B64, DS_OR_B64, and DS_XOR_B64, the simple one-address LDS load slice DS_LOAD_B32, DS_LOAD_B64, DS_LOAD_B96, DS_LOAD_B128, DS_LOAD_I8, DS_LOAD_U8, DS_LOAD_I16, DS_LOAD_U16, DS_LOAD_U8_D16, DS_LOAD_U8_D16_HI, DS_LOAD_I8_D16, DS_LOAD_I8_D16_HI, DS_LOAD_U16_D16, and DS_LOAD_U16_D16_HI, and the matching simple one-address LDS store slice DS_STORE_B8, DS_STORE_B16, DS_STORE_B32, DS_STORE_B64, DS_STORE_B96, DS_STORE_B128, DS_STORE_B8_D16_HI, and DS_STORE_B16_D16_HI.
- The next coherent extension points now move further into ENC_VDS through nearby single-address utilities or the remaining low-risk single-address LDS tail, with later frontier steps moving into multi-address, exchange/compare-store, GDS, and eventually ENC_VOP3.

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
- `ENC_VDS`: example `DS_ADD_U32`, seeded `123`, executable `77` via `DS_ADD_F32`, as-is `27`, decoder-rollup `38`, semantic-only `0`, gfx1201-specific `58`
- `ENC_VGLOBAL`: example `GLOBAL_LOAD_B32`, seeded `65`, executable `65` via `GLOBAL_ATOMIC_ADD_F32`, as-is `3`, decoder-rollup `0`, semantic-only `0`, gfx1201-specific `62`
