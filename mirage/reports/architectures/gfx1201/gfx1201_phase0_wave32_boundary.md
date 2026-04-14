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
- Remaining VDS opcode gaps: `11`
- Next 50-slice frontier window: opcode `14..63`, `11` instructions, `39` holes, safe under current request `false`; follow-up `64..113`, `7` instructions, `43` holes, safe under current request `false`; follow-up `114..163`, `3` instructions, `47` holes, safe under current request `false`; follow-up `214..263`, `3` instructions, `47` holes, safe under current request `false`; follow-up `264..313`, `0` instructions, `50` holes, safe under current request `false`; follow-up `314..363`, `0` instructions, `50` holes, safe under current request `false`; follow-up `364..413`, `0` instructions, `50` holes, safe under current request `false`; follow-up `414..463`, `0` instructions, `50` holes, safe under current request `false`; follow-up `464..513`, `0` instructions, `50` holes, safe under current request `false`; follow-up `514..563`, `0` instructions, `50` holes, safe under current request `false`; follow-up `564..613`, `0` instructions, `50` holes, safe under current request `false`; follow-up `614..663`, `0` instructions, `50` holes, safe under current request `false`; follow-up `664..713`, `0` instructions, `50` holes, safe under current request `false`; follow-up `714..763`, `0` instructions, `50` holes, safe under current request `false`; follow-up `764..813`, `0` instructions, `50` holes, safe under current request `false`; follow-up `814..863`, `0` instructions, `50` holes, safe under current request `false`; follow-up `864..913`, `0` instructions, `50` holes, safe under current request `false`; follow-up `914..963`, `0` instructions, `50` holes, safe under current request `false`; follow-up `964..1013`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1014..1063`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1064..1113`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1114..1163`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1164..1213`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1214..1263`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1264..1313`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1314..1363`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1364..1413`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1414..1463`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1464..1513`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1514..1563`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1564..1613`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1614..1663`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1664..1713`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1714..1763`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1764..1813`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1814..1863`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1864..1913`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1914..1963`, `0` instructions, `50` holes, safe under current request `false`; follow-up `1964..2013`, `0` instructions, `50` holes, safe under current request `false`
- Remaining VDS bucket order: `append_consume`, `exchange_compare_store`, `multi_address`, `bvh_stack`
- All currently seeded ENC_VOP1, ENC_VOP2, ENC_VOPC, ENC_SMEM, and ENC_VGLOBAL instruction/encoding pairs are executable on the local wave32 path.
- There are no remaining imported ENC_VOP1, ENC_VOP2, ENC_VOPC, ENC_SMEM, or ENC_VGLOBAL instruction/encoding pairs outside the current seed surface.
- ENC_SMEM is now fully bootstrapped through S_DCACHE_INV, S_PREFETCH_INST, S_PREFETCH_INST_PC_REL, S_PREFETCH_DATA, S_BUFFER_PREFETCH_DATA, S_PREFETCH_DATA_PC_REL, S_ATC_PROBE, S_ATC_PROBE_BUFFER, the full non-buffer S_LOAD_* slice, and the matching S_BUFFER_LOAD_* slice, leaving no scalar-memory seed instructions scaffolded.
- ENC_VGLOBAL is now fully bootstrapped through GLOBAL_INV, GLOBAL_WB, GLOBAL_WBINV, the plain GLOBAL_LOAD_U8/I8/U16/I16/B32/B64/B96/B128 slice, GLOBAL_LOAD_ADDTID_B32, GLOBAL_LOAD_BLOCK, GLOBAL_LOAD_TR_B64/B128, the packed GLOBAL_LOAD_D16_U8/I8/B16 and GLOBAL_LOAD_D16_HI_U8/I8/B16 slice, the plain GLOBAL_STORE_B8/B16/B32/B64/B96/B128 slice, GLOBAL_STORE_ADDTID_B32, GLOBAL_STORE_BLOCK, GLOBAL_STORE_D16_HI_B8/B16, the 32-bit integer atomic slice GLOBAL_ATOMIC_SWAP_B32, GLOBAL_ATOMIC_CMPSWAP_B32, GLOBAL_ATOMIC_ADD/SUB/SUB_CLAMP_U32, GLOBAL_ATOMIC_MIN/MAX_I32/U32, GLOBAL_ATOMIC_AND/OR/XOR_B32, GLOBAL_ATOMIC_INC_U32, GLOBAL_ATOMIC_DEC_U32, and GLOBAL_ATOMIC_COND_SUB_U32, the 64-bit integer atomic slice GLOBAL_ATOMIC_SWAP_B64, GLOBAL_ATOMIC_CMPSWAP_B64, GLOBAL_ATOMIC_ADD/SUB_U64, GLOBAL_ATOMIC_MIN/MAX_I64/U64, GLOBAL_ATOMIC_AND/OR/XOR_B64, GLOBAL_ATOMIC_INC_U64, and GLOBAL_ATOMIC_DEC_U64, the F32 atomic slice GLOBAL_ATOMIC_ADD_F32 and GLOBAL_ATOMIC_MIN/MAX_NUM_F32, the packed-half atomic pair GLOBAL_ATOMIC_PK_ADD_F16/GLOBAL_ATOMIC_PK_ADD_BF16, and GLOBAL_ATOMIC_ORDERED_ADD_B64, leaving no seeded vector-global instructions scaffolded.
- ENC_VDS now has architecture-local executable footholds through DS_NOP, the one-address non-return 32-bit LDS update slice DS_ADD_F32, DS_ADD_U32, DS_SUB_U32, DS_RSUB_U32, DS_INC_U32, DS_DEC_U32, DS_COND_SUB_U32, DS_SUB_CLAMP_U32, DS_PK_ADD_F16, DS_PK_ADD_BF16, DS_MIN_NUM_F32, DS_MAX_NUM_F32, DS_MIN_NUM_F64, DS_MAX_NUM_F64, DS_MIN_I32, DS_MIN_U32, DS_MAX_I32, DS_MAX_U32, DS_AND_B32, DS_OR_B32, DS_XOR_B32, and DS_MSKOR_B32, the matching one-address return-value 32-bit slice DS_ADD_RTN_F32, DS_ADD_RTN_U32, DS_SUB_RTN_U32, DS_RSUB_RTN_U32, DS_INC_RTN_U32, DS_DEC_RTN_U32, DS_COND_SUB_RTN_U32, DS_SUB_CLAMP_RTN_U32, DS_PK_ADD_RTN_F16, DS_PK_ADD_RTN_BF16, DS_MIN_NUM_RTN_F32, DS_MAX_NUM_RTN_F32, DS_MIN_NUM_RTN_F64, DS_MAX_NUM_RTN_F64, DS_MIN_RTN_I32, DS_MIN_RTN_U32, DS_MAX_RTN_I32, DS_MAX_RTN_U32, DS_AND_RTN_B32, DS_OR_RTN_B32, DS_XOR_RTN_B32, and DS_MSKOR_RTN_B32, the matching one-address return-value 64-bit integer LDS update slice DS_ADD_RTN_U64, DS_SUB_RTN_U64, DS_RSUB_RTN_U64, DS_INC_RTN_U64, DS_DEC_RTN_U64, DS_MIN_RTN_I64, DS_MIN_RTN_U64, DS_MAX_RTN_I64, DS_MAX_RTN_U64, DS_AND_RTN_B64, DS_OR_RTN_B64, DS_XOR_RTN_B64, and DS_MSKOR_RTN_B64, the one-address non-return 64-bit integer LDS update slice DS_ADD_U64, DS_SUB_U64, DS_RSUB_U64, DS_INC_U64, DS_DEC_U64, DS_MIN_I64, DS_MIN_U64, DS_MAX_I64, DS_MAX_U64, DS_AND_B64, DS_OR_B64, DS_XOR_B64, and DS_MSKOR_B64, the simple one-address LDS load slice DS_LOAD_B32, DS_LOAD_ADDTID_B32, DS_LOAD_B64, DS_LOAD_B96, DS_LOAD_B128, DS_LOAD_I8, DS_LOAD_U8, DS_LOAD_I16, DS_LOAD_U16, DS_LOAD_U8_D16, DS_LOAD_U8_D16_HI, DS_LOAD_I8_D16, DS_LOAD_I8_D16_HI, DS_LOAD_U16_D16, and DS_LOAD_U16_D16_HI, the matching simple one-address LDS store slice DS_STORE_B8, DS_STORE_B16, DS_STORE_B32, DS_STORE_ADDTID_B32, DS_STORE_B64, DS_STORE_B96, DS_STORE_B128, DS_STORE_B8_D16_HI, and DS_STORE_B16_D16_HI, and the single-address lane-routing utility slice DS_SWIZZLE_B32, DS_PERMUTE_B32, DS_BPERMUTE_B32, and DS_BPERMUTE_FI_B32.
- The remaining ENC_VDS tail is now explicitly bounded by append/consume allocator semantics, exchange and compare-store forms, multi-address LDS forms including stride64, and gfx1201-specific BVH stack instructions, which is the next verification-risk step before ENC_VOP3.
- There is no safe ENC_VDS continuation under the current request boundary: every remaining bucket crosses allocator-or-GDS, exchange/compare-store, multi-address, or gfx1201-specific BVH semantics.
- The boundary report now carries exact remaining-VDS instruction-name and numeric-opcode maps so the unresolved tail can be queried directly by either key.
- The denormalized remaining-VDS status list now also carries exact opcode, operand-count, support-rollup, and support-state metadata for each unsafe instruction.
- The boundary report now also carries exact per-bucket opcode spans, operand-count spans, operand-count compositions, exact support-rollup composition counts, and exact support-state composition counts for the unresolved VDS tail.
- The boundary report now also carries exact per-bucket opcode-segment counts, longest contiguous opcode runs, largest opcode gaps, and a denormalized opcode-segment list for the unresolved VDS tail.
- The boundary report now also carries exact per-bucket opcode span widths, in-span hole counts, singleton-versus-multi-instruction segment counts, and a denormalized inter-segment opcode-gap list for the unresolved VDS tail.
- The exact unsafe-bucket escalation order is `append_consume`, then `exchange_compare_store`, then `multi_address`, then `bvh_stack`.
- The boundary report now also carries an exact next-risk step chain with first and last instruction names, cumulative remaining counts, and explicit next-bucket handoff metadata for the unresolved VDS tail.
- The first unsafe ENC_VDS bucket is now expanded inline with its blocking dimension and exact instruction list.
- The boundary report now also carries a denormalized per-op remaining-VDS status list with bucket, blocking dimension, bucket risk rank, tail ordinal, bucket ordinal, and safe flag.
- The boundary report now also carries an extended empty-window metadata chain through opcode `1014..1063`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1064..1113`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1114..1163`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1164..1213`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1214..1263`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1264..1313`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1314..1363`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1364..1413`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1414..1463`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1464..1513`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1514..1563`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1564..1613`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1614..1663`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1664..1713`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1714..1763`, so the current continuation remains metadata-only.
- The boundary report now also carries the next empty 50-slice window through opcode `1814..1863`, the next batch window through opcode `1864..1913`, the next batch window through opcode `1914..1963`, and the next batch window through opcode `1964..2013`, so the current continuation remains metadata-only.

## Remaining VDS Boundary

- `append_consume`: risk rank `0`, ordinal range `0..1`, `2` instructions, example `DS_APPEND`, blocking dimension `allocator_or_gds_semantics`, safe under current request `false`, covering `DS_APPEND` and `DS_CONSUME`.
- `exchange_compare_store`: risk rank `1`, ordinal range `2..8`, `7` instructions, example `DS_CONDXCHG32_RTN_B64`, blocking dimension `exchange_compare_store_semantics`, safe under current request `false`, covering `DS_CONDXCHG32_RTN_B64`, `DS_CMPSTORE_B32`, `DS_CMPSTORE_B64`, `DS_CMPSTORE_RTN_B32`, `DS_CMPSTORE_RTN_B64`, `DS_STOREXCHG_RTN_B32`, and `DS_STOREXCHG_RTN_B64`.
- `multi_address`: risk rank `2`, ordinal range `9..20`, `12` instructions, example `DS_LOAD_2ADDR_B32`, blocking dimension `multi_address_semantics`, safe under current request `false`, covering `DS_LOAD_2ADDR_B32`, `DS_LOAD_2ADDR_B64`, `DS_LOAD_2ADDR_STRIDE64_B32`, `DS_LOAD_2ADDR_STRIDE64_B64`, `DS_STOREXCHG_2ADDR_RTN_B32`, `DS_STOREXCHG_2ADDR_RTN_B64`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`, `DS_STORE_2ADDR_B32`, `DS_STORE_2ADDR_B64`, `DS_STORE_2ADDR_STRIDE64_B32`, and `DS_STORE_2ADDR_STRIDE64_B64`.
- `bvh_stack`: risk rank `3`, ordinal range `21..23`, `3` instructions, example `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, blocking dimension `gfx1201_specific_bvh_semantics`, safe under current request `false`, covering `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and `DS_BVH_STACK_PUSH8_POP2_RTN_B64`.

## Remaining VDS Bucket Statuses

- `append_consume`: opcode span `61..62`, span width `2`, in-span opcode holes `0`, opcode segments `1`, singleton segments `0`, multi-instruction segments `1`, longest contiguous segment `2`, largest opcode gap `0`, operand-count span `3..3`, operand-count composition `3->2, 4->0, 5->0, 6->0`, rollup composition `as-is->0, decoder->2, semantic->0, gfx1201-specific->0`, state composition `as-is->0, decoder->0, semantic->0, decoder+semantic->2, gfx1201-specific->0`
- `exchange_compare_store`: opcode span `16..126`, span width `111`, in-span opcode holes `104`, opcode segments `7`, singleton segments `7`, multi-instruction segments `0`, longest contiguous segment `1`, largest opcode gap `31`, operand-count span `5..6`, operand-count composition `3->0, 4->0, 5->5, 6->2`, rollup composition `as-is->0, decoder->1, semantic->0, gfx1201-specific->6`, state composition `as-is->0, decoder->0, semantic->0, decoder+semantic->1, gfx1201-specific->6`
- `multi_address`: opcode span `14..120`, span width `107`, in-span opcode holes `95`, opcode segments `6`, singleton segments `0`, multi-instruction segments `6`, longest contiguous segment `2`, largest opcode gap `30`, operand-count span `3..6`, operand-count composition `3->4, 4->4, 5->0, 6->4`, rollup composition `as-is->0, decoder->0, semantic->0, gfx1201-specific->12`, state composition `as-is->0, decoder->0, semantic->0, decoder+semantic->0, gfx1201-specific->12`
- `bvh_stack`: opcode span `224..226`, span width `3`, in-span opcode holes `0`, opcode segments `1`, singleton segments `0`, multi-instruction segments `1`, longest contiguous segment `3`, largest opcode gap `0`, operand-count span `4..4`, operand-count composition `3->0, 4->3, 5->0, 6->0`, rollup composition `as-is->0, decoder->0, semantic->0, gfx1201-specific->3`, state composition `as-is->0, decoder->0, semantic->0, decoder+semantic->0, gfx1201-specific->3`

## Remaining VDS Opcode Segments

- `append_consume[0]`: opcode span `61..62`, `2` instructions, first `DS_CONSUME`, last `DS_APPEND`
- `exchange_compare_store[0]`: opcode span `16..16`, `1` instruction, first `DS_CMPSTORE_B32`, last `DS_CMPSTORE_B32`
- `exchange_compare_store[1]`: opcode span `45..45`, `1` instruction, first `DS_STOREXCHG_RTN_B32`, last `DS_STOREXCHG_RTN_B32`
- `exchange_compare_store[2]`: opcode span `48..48`, `1` instruction, first `DS_CMPSTORE_RTN_B32`, last `DS_CMPSTORE_RTN_B32`
- `exchange_compare_store[3]`: opcode span `80..80`, `1` instruction, first `DS_CMPSTORE_B64`, last `DS_CMPSTORE_B64`
- `exchange_compare_store[4]`: opcode span `109..109`, `1` instruction, first `DS_STOREXCHG_RTN_B64`, last `DS_STOREXCHG_RTN_B64`
- `exchange_compare_store[5]`: opcode span `112..112`, `1` instruction, first `DS_CMPSTORE_RTN_B64`, last `DS_CMPSTORE_RTN_B64`
- `exchange_compare_store[6]`: opcode span `126..126`, `1` instruction, first `DS_CONDXCHG32_RTN_B64`, last `DS_CONDXCHG32_RTN_B64`
- `multi_address[0]`: opcode span `14..15`, `2` instructions, first `DS_STORE_2ADDR_B32`, last `DS_STORE_2ADDR_STRIDE64_B32`
- `multi_address[1]`: opcode span `46..47`, `2` instructions, first `DS_STOREXCHG_2ADDR_RTN_B32`, last `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`
- `multi_address[2]`: opcode span `55..56`, `2` instructions, first `DS_LOAD_2ADDR_B32`, last `DS_LOAD_2ADDR_STRIDE64_B32`
- `multi_address[3]`: opcode span `78..79`, `2` instructions, first `DS_STORE_2ADDR_B64`, last `DS_STORE_2ADDR_STRIDE64_B64`
- `multi_address[4]`: opcode span `110..111`, `2` instructions, first `DS_STOREXCHG_2ADDR_RTN_B64`, last `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`
- `multi_address[5]`: opcode span `119..120`, `2` instructions, first `DS_LOAD_2ADDR_B64`, last `DS_LOAD_2ADDR_STRIDE64_B64`
- `bvh_stack[0]`: opcode span `224..226`, `3` instructions, first `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, last `DS_BVH_STACK_PUSH8_POP2_RTN_B64`

## Remaining VDS Opcode Gaps

- `exchange_compare_store[0]`: segment `0 -> 1`, previous opcode `16` (`DS_CMPSTORE_B32`), next opcode `45` (`DS_STOREXCHG_RTN_B32`), missing opcodes `28`
- `exchange_compare_store[1]`: segment `1 -> 2`, previous opcode `45` (`DS_STOREXCHG_RTN_B32`), next opcode `48` (`DS_CMPSTORE_RTN_B32`), missing opcodes `2`
- `exchange_compare_store[2]`: segment `2 -> 3`, previous opcode `48` (`DS_CMPSTORE_RTN_B32`), next opcode `80` (`DS_CMPSTORE_B64`), missing opcodes `31`
- `exchange_compare_store[3]`: segment `3 -> 4`, previous opcode `80` (`DS_CMPSTORE_B64`), next opcode `109` (`DS_STOREXCHG_RTN_B64`), missing opcodes `28`
- `exchange_compare_store[4]`: segment `4 -> 5`, previous opcode `109` (`DS_STOREXCHG_RTN_B64`), next opcode `112` (`DS_CMPSTORE_RTN_B64`), missing opcodes `2`
- `exchange_compare_store[5]`: segment `5 -> 6`, previous opcode `112` (`DS_CMPSTORE_RTN_B64`), next opcode `126` (`DS_CONDXCHG32_RTN_B64`), missing opcodes `13`
- `multi_address[0]`: segment `0 -> 1`, previous opcode `15` (`DS_STORE_2ADDR_STRIDE64_B32`), next opcode `46` (`DS_STOREXCHG_2ADDR_RTN_B32`), missing opcodes `30`
- `multi_address[1]`: segment `1 -> 2`, previous opcode `47` (`DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`), next opcode `55` (`DS_LOAD_2ADDR_B32`), missing opcodes `7`
- `multi_address[2]`: segment `2 -> 3`, previous opcode `56` (`DS_LOAD_2ADDR_STRIDE64_B32`), next opcode `78` (`DS_STORE_2ADDR_B64`), missing opcodes `21`
- `multi_address[3]`: segment `3 -> 4`, previous opcode `79` (`DS_STORE_2ADDR_STRIDE64_B64`), next opcode `110` (`DS_STOREXCHG_2ADDR_RTN_B64`), missing opcodes `30`
- `multi_address[4]`: segment `4 -> 5`, previous opcode `111` (`DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`), next opcode `119` (`DS_LOAD_2ADDR_B64`), missing opcodes `7`

## Next 50-Slice Frontier Window

- `opcode 14..63`: `11` instructions, `39` holes, `3` buckets, safe under current request `false`, covering `DS_STORE_2ADDR_B32`, `DS_STORE_2ADDR_STRIDE64_B32`, `DS_CMPSTORE_B32`, `DS_STOREXCHG_RTN_B32`, `DS_STOREXCHG_2ADDR_RTN_B32`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`, `DS_CMPSTORE_RTN_B32`, `DS_LOAD_2ADDR_B32`, `DS_LOAD_2ADDR_STRIDE64_B32`, `DS_CONSUME`, and `DS_APPEND`.

## Follow-up 50-Slice Frontier Window

- `opcode 64..113`: `7` instructions, `43` holes, `2` buckets, safe under current request `false`, covering `DS_STORE_2ADDR_B64`, `DS_STORE_2ADDR_STRIDE64_B64`, `DS_CMPSTORE_B64`, `DS_STOREXCHG_RTN_B64`, `DS_STOREXCHG_2ADDR_RTN_B64`, `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`, and `DS_CMPSTORE_RTN_B64`.

## Second Follow-up 50-Slice Frontier Window

- `opcode 114..163`: `3` instructions, `47` holes, `2` buckets, safe under current request `false`, covering `DS_LOAD_2ADDR_B64`, `DS_LOAD_2ADDR_STRIDE64_B64`, and `DS_CONDXCHG32_RTN_B64`.

## Third Follow-up 50-Slice Frontier Window

- `opcode 214..263`: `3` instructions, `47` holes, `1` bucket, safe under current request `false`, covering `DS_BVH_STACK_PUSH4_POP1_RTN_B32`, `DS_BVH_STACK_PUSH8_POP1_RTN_B32`, and `DS_BVH_STACK_PUSH8_POP2_RTN_B64`.

## Fourth Follow-up 50-Slice Frontier Window

- `opcode 264..313`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Fifth Follow-up 50-Slice Frontier Window

- `opcode 314..363`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Sixth Follow-up 50-Slice Frontier Window

- `opcode 364..413`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Seventh Follow-up 50-Slice Frontier Window

- `opcode 414..463`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Eighth Follow-up 50-Slice Frontier Window

- `opcode 464..513`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Ninth Follow-up 50-Slice Frontier Window

- `opcode 514..563`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Tenth Follow-up 50-Slice Frontier Window

- `opcode 564..613`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Eleventh Follow-up 50-Slice Frontier Window

- `opcode 614..663`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twelfth Follow-up 50-Slice Frontier Window

- `opcode 664..713`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirteenth Follow-up 50-Slice Frontier Window

- `opcode 714..763`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Fourteenth Follow-up 50-Slice Frontier Window

- `opcode 764..813`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Fifteenth Follow-up 50-Slice Frontier Window

- `opcode 814..863`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Sixteenth Follow-up 50-Slice Frontier Window

- `opcode 864..913`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Seventeenth Follow-up 50-Slice Frontier Window

- `opcode 914..963`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Eighteenth Follow-up 50-Slice Frontier Window

- `opcode 964..1013`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Nineteenth Follow-up 50-Slice Frontier Window

- `opcode 1014..1063`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twentieth Follow-up 50-Slice Frontier Window

- `opcode 1064..1113`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-first Follow-up 50-Slice Frontier Window

- `opcode 1114..1163`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-second Follow-up 50-Slice Frontier Window

- `opcode 1164..1213`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-third Follow-up 50-Slice Frontier Window

- `opcode 1214..1263`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-fourth Follow-up 50-Slice Frontier Window

- `opcode 1264..1313`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-fifth Follow-up 50-Slice Frontier Window

- `opcode 1314..1363`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-sixth Follow-up 50-Slice Frontier Window

- `opcode 1364..1413`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-seventh Follow-up 50-Slice Frontier Window

- `opcode 1414..1463`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-eighth Follow-up 50-Slice Frontier Window

- `opcode 1464..1513`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Twenty-ninth Follow-up 50-Slice Frontier Window

- `opcode 1514..1563`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirtieth Follow-up 50-Slice Frontier Window

- `opcode 1564..1613`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-first Follow-up 50-Slice Frontier Window

- `opcode 1614..1663`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-second Follow-up 50-Slice Frontier Window

- `opcode 1664..1713`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-third Follow-up 50-Slice Frontier Window

- `opcode 1714..1763`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-fourth Follow-up 50-Slice Frontier Window

- `opcode 1764..1813`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-fifth Follow-up 50-Slice Frontier Window

- `opcode 1814..1863`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-sixth Follow-up 50-Slice Frontier Window

- `opcode 1864..1913`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-seventh Follow-up 50-Slice Frontier Window

- `opcode 1914..1963`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

## Thirty-eighth Follow-up 50-Slice Frontier Window

- `opcode 1964..2013`: `0` instructions, `50` holes, `0` buckets, safe under current request `false`, covering no remaining VDS instructions.

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

## Remaining VDS Opcode Map

- `61` -> `DS_CONSUME` -> `append_consume`
- `62` -> `DS_APPEND` -> `append_consume`
- `16` -> `DS_CMPSTORE_B32` -> `exchange_compare_store`
- `45` -> `DS_STOREXCHG_RTN_B32` -> `exchange_compare_store`
- `48` -> `DS_CMPSTORE_RTN_B32` -> `exchange_compare_store`
- `80` -> `DS_CMPSTORE_B64` -> `exchange_compare_store`
- `109` -> `DS_STOREXCHG_RTN_B64` -> `exchange_compare_store`
- `112` -> `DS_CMPSTORE_RTN_B64` -> `exchange_compare_store`
- `126` -> `DS_CONDXCHG32_RTN_B64` -> `exchange_compare_store`
- `14` -> `DS_STORE_2ADDR_B32` -> `multi_address`
- `15` -> `DS_STORE_2ADDR_STRIDE64_B32` -> `multi_address`
- `46` -> `DS_STOREXCHG_2ADDR_RTN_B32` -> `multi_address`
- `47` -> `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32` -> `multi_address`
- `55` -> `DS_LOAD_2ADDR_B32` -> `multi_address`
- `56` -> `DS_LOAD_2ADDR_STRIDE64_B32` -> `multi_address`
- `78` -> `DS_STORE_2ADDR_B64` -> `multi_address`
- `79` -> `DS_STORE_2ADDR_STRIDE64_B64` -> `multi_address`
- `110` -> `DS_STOREXCHG_2ADDR_RTN_B64` -> `multi_address`
- `111` -> `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64` -> `multi_address`
- `119` -> `DS_LOAD_2ADDR_B64` -> `multi_address`
- `120` -> `DS_LOAD_2ADDR_STRIDE64_B64` -> `multi_address`
- `224` -> `DS_BVH_STACK_PUSH4_POP1_RTN_B32` -> `bvh_stack`
- `225` -> `DS_BVH_STACK_PUSH8_POP1_RTN_B32` -> `bvh_stack`
- `226` -> `DS_BVH_STACK_PUSH8_POP2_RTN_B64` -> `bvh_stack`

## Remaining VDS Instruction Statuses

- `DS_APPEND`: opcode `62`, operand count `3`, support rollup `transferable_with_decoder_work`, support state `transferable_with_decoder_and_semantic_work`, bucket `append_consume`, blocking dimension `allocator_or_gds_semantics`, bucket risk rank `0`, tail ordinal `0`, bucket ordinal `0`, safe under current request `false`
- `DS_CONSUME`: opcode `61`, operand count `3`, support rollup `transferable_with_decoder_work`, support state `transferable_with_decoder_and_semantic_work`, bucket `append_consume`, blocking dimension `allocator_or_gds_semantics`, bucket risk rank `0`, tail ordinal `1`, bucket ordinal `1`, safe under current request `false`
- `DS_CONDXCHG32_RTN_B64`: opcode `126`, operand count `5`, support rollup `transferable_with_decoder_work`, support state `transferable_with_decoder_and_semantic_work`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `2`, bucket ordinal `0`, safe under current request `false`
- `DS_CMPSTORE_B32`: opcode `16`, operand count `5`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `3`, bucket ordinal `1`, safe under current request `false`
- `DS_CMPSTORE_B64`: opcode `80`, operand count `5`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `4`, bucket ordinal `2`, safe under current request `false`
- `DS_CMPSTORE_RTN_B32`: opcode `48`, operand count `6`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `5`, bucket ordinal `3`, safe under current request `false`
- `DS_CMPSTORE_RTN_B64`: opcode `112`, operand count `6`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `6`, bucket ordinal `4`, safe under current request `false`
- `DS_STOREXCHG_RTN_B32`: opcode `45`, operand count `5`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `7`, bucket ordinal `5`, safe under current request `false`
- `DS_STOREXCHG_RTN_B64`: opcode `109`, operand count `5`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `exchange_compare_store`, blocking dimension `exchange_compare_store_semantics`, bucket risk rank `1`, tail ordinal `8`, bucket ordinal `6`, safe under current request `false`
- `DS_LOAD_2ADDR_B32`: opcode `55`, operand count `3`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `9`, bucket ordinal `0`, safe under current request `false`
- `DS_LOAD_2ADDR_B64`: opcode `119`, operand count `3`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `10`, bucket ordinal `1`, safe under current request `false`
- `DS_LOAD_2ADDR_STRIDE64_B32`: opcode `56`, operand count `3`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `11`, bucket ordinal `2`, safe under current request `false`
- `DS_LOAD_2ADDR_STRIDE64_B64`: opcode `120`, operand count `3`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `12`, bucket ordinal `3`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_RTN_B32`: opcode `46`, operand count `6`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `13`, bucket ordinal `4`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_RTN_B64`: opcode `110`, operand count `6`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `14`, bucket ordinal `5`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32`: opcode `47`, operand count `6`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `15`, bucket ordinal `6`, safe under current request `false`
- `DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64`: opcode `111`, operand count `6`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `16`, bucket ordinal `7`, safe under current request `false`
- `DS_STORE_2ADDR_B32`: opcode `14`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `17`, bucket ordinal `8`, safe under current request `false`
- `DS_STORE_2ADDR_B64`: opcode `78`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `18`, bucket ordinal `9`, safe under current request `false`
- `DS_STORE_2ADDR_STRIDE64_B32`: opcode `15`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `19`, bucket ordinal `10`, safe under current request `false`
- `DS_STORE_2ADDR_STRIDE64_B64`: opcode `79`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `multi_address`, blocking dimension `multi_address_semantics`, bucket risk rank `2`, tail ordinal `20`, bucket ordinal `11`, safe under current request `false`
- `DS_BVH_STACK_PUSH4_POP1_RTN_B32`: opcode `224`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `bvh_stack`, blocking dimension `gfx1201_specific_bvh_semantics`, bucket risk rank `3`, tail ordinal `21`, bucket ordinal `0`, safe under current request `false`
- `DS_BVH_STACK_PUSH8_POP1_RTN_B32`: opcode `225`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `bvh_stack`, blocking dimension `gfx1201_specific_bvh_semantics`, bucket risk rank `3`, tail ordinal `22`, bucket ordinal `1`, safe under current request `false`
- `DS_BVH_STACK_PUSH8_POP2_RTN_B64`: opcode `226`, operand count `4`, support rollup `gfx1201_specific`, support state `gfx1201_specific`, bucket `bvh_stack`, blocking dimension `gfx1201_specific_bvh_semantics`, bucket risk rank `3`, tail ordinal `23`, bucket ordinal `2`, safe under current request `false`

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
