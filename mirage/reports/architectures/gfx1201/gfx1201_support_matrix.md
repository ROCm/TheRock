# gfx1201 Support Matrix

## Imported Baseline

- Target: `gfx1201`
- Architecture source: `AMD RDNA 4`
- Release date: `2025-02-19`
- Source XML: `amdgpu_isa_rdna4.xml`
- Imported instructions: `1264`
- Imported encodings: `5062`

## Rollup Categories

- `transferable_with_decoder_work` is decoder-first: the `transferable_with_decoder_and_semantic_work` subset stays in this rollup because decode is still the first local integration blocker.

### transferable_as_is

- Instruction count: `363`
- Meaning: Instruction name is already proven in gfx950 with both raw decode coverage and interpreter semantics.
- Top families: v(235), s(98), ds(27), global(3)
- Sample instructions: `DS_ADD_F32`, `DS_ADD_RTN_F32`, `DS_ADD_RTN_U32`, `DS_ADD_U32`, `DS_AND_B32`, `DS_AND_RTN_B32`, `DS_DEC_RTN_U32`, `DS_DEC_U32`, `DS_INC_RTN_U32`, `DS_INC_U32`, `DS_MAX_I32`, `DS_MAX_RTN_I32`

### transferable_with_decoder_work

- Instruction count: `197`
- Meaning: Instruction name exists in gfx950 lineage, but Mirage still needs decoder work before the local target can rely on it. This rollup is decoder-first and includes opcodes that still need both decoder and semantic work.
- Top families: v(116), ds(38), s(21), buffer(11), tbuffer(8), flat(3)
- Sample instructions: `V_CMPX_CLASS_F16`, `V_CMPX_EQ_F16`, `V_CMPX_GE_F16`, `V_CMPX_GT_F16`, `V_CMPX_LE_F16`, `V_CMPX_LG_F16`, `V_CMPX_LT_F16`, `V_CMPX_NEQ_F16`, `V_CMPX_NGE_F16`, `V_CMPX_NGT_F16`, `V_CMPX_NLE_F16`, `V_CMPX_NLG_F16`

### transferable_with_semantic_work

- Instruction count: `36`
- Meaning: Binary decode precedent exists in gfx950 coverage, but execution semantics still need to be carried over.
- Top families: v(26), s(10)
- Sample instructions: `S_GETPC_B64`, `S_MOVRELD_B32`, `S_MOVRELD_B64`, `S_MOVRELS_B32`, `S_MOVRELS_B64`, `S_RFE_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, `S_WQM_B32`, `S_WQM_B64`, `V_ASHRREV_I16`, `V_CVT_F32_BF8`

### gfx1201_specific

- Instruction count: `668`
- Meaning: Instruction name is absent from the gfx950 catalog and needs gfx1201-local handling.
- Top families: v(155), s(153), image(91), buffer(62), global(62), ds(60), flat(52), scratch(24)
- Sample instructions: `BUFFER_ATOMIC_ADD_U32`, `BUFFER_ATOMIC_ADD_U64`, `BUFFER_ATOMIC_AND_B32`, `BUFFER_ATOMIC_AND_B64`, `BUFFER_ATOMIC_CMPSWAP_B32`, `BUFFER_ATOMIC_CMPSWAP_B64`, `BUFFER_ATOMIC_COND_SUB_U32`, `BUFFER_ATOMIC_DEC_U32`, `BUFFER_ATOMIC_DEC_U64`, `BUFFER_ATOMIC_INC_U32`, `BUFFER_ATOMIC_INC_U64`, `BUFFER_ATOMIC_MAX_I32`

## Exact States

### transferable_as_is

- Instruction count: `363`
- Meaning: Both raw decode coverage and semantics already exist in gfx950.
- Sample instructions: `DS_ADD_F32`, `DS_ADD_RTN_F32`, `DS_ADD_RTN_U32`, `DS_ADD_U32`, `DS_AND_B32`, `DS_AND_RTN_B32`, `DS_DEC_RTN_U32`, `DS_DEC_U32`, `DS_INC_RTN_U32`, `DS_INC_U32`, `DS_MAX_I32`, `DS_MAX_RTN_I32`

### transferable_with_decoder_work

- Instruction count: `30`
- Meaning: Semantics exist in gfx950 coverage, but raw decode support is still missing.
- Sample instructions: `V_CMPX_CLASS_F16`, `V_CMPX_EQ_F16`, `V_CMPX_GE_F16`, `V_CMPX_GT_F16`, `V_CMPX_LE_F16`, `V_CMPX_LG_F16`, `V_CMPX_LT_F16`, `V_CMPX_NEQ_F16`, `V_CMPX_NGE_F16`, `V_CMPX_NGT_F16`, `V_CMPX_NLE_F16`, `V_CMPX_NLG_F16`

### transferable_with_semantic_work

- Instruction count: `36`
- Meaning: Raw decode support exists in gfx950 coverage, but semantics are still missing.
- Sample instructions: `S_GETPC_B64`, `S_MOVRELD_B32`, `S_MOVRELD_B64`, `S_MOVRELS_B32`, `S_MOVRELS_B64`, `S_RFE_B64`, `S_SETPC_B64`, `S_SWAPPC_B64`, `S_WQM_B32`, `S_WQM_B64`, `V_ASHRREV_I16`, `V_CVT_F32_BF8`

### transferable_with_decoder_and_semantic_work

- Instruction count: `167`
- Meaning: Instruction name exists in gfx950, but neither decode nor semantics are ready.
- Sample instructions: `BUFFER_ATOMIC_ADD_F32`, `BUFFER_ATOMIC_PK_ADD_BF16`, `BUFFER_ATOMIC_PK_ADD_F16`, `BUFFER_LOAD_FORMAT_X`, `BUFFER_LOAD_FORMAT_XY`, `BUFFER_LOAD_FORMAT_XYZ`, `BUFFER_LOAD_FORMAT_XYZW`, `BUFFER_STORE_FORMAT_X`, `BUFFER_STORE_FORMAT_XY`, `BUFFER_STORE_FORMAT_XYZ`, `BUFFER_STORE_FORMAT_XYZW`, `DS_ADD_RTN_U64`

### gfx1201_specific

- Instruction count: `668`
- Meaning: Instruction name is new relative to gfx950.
- Sample instructions: `BUFFER_ATOMIC_ADD_U32`, `BUFFER_ATOMIC_ADD_U64`, `BUFFER_ATOMIC_AND_B32`, `BUFFER_ATOMIC_AND_B64`, `BUFFER_ATOMIC_CMPSWAP_B32`, `BUFFER_ATOMIC_CMPSWAP_B64`, `BUFFER_ATOMIC_COND_SUB_U32`, `BUFFER_ATOMIC_DEC_U32`, `BUFFER_ATOMIC_DEC_U64`, `BUFFER_ATOMIC_INC_U32`, `BUFFER_ATOMIC_INC_U64`, `BUFFER_ATOMIC_MAX_I32`

