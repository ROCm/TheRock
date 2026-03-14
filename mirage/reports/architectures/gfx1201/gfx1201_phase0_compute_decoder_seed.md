# gfx1201 Phase-0 Compute Decoder Seed

## Scope

- Seeded encoding families: `ENC_SOPP`, `ENC_SOP1`, `ENC_SOP2`, `ENC_SOPC`, `ENC_SOPK`, `ENC_SMEM`, `ENC_VOP1`, `ENC_VOP2`, `ENC_VOPC`, `ENC_VOP3`, `ENC_VDS`, `ENC_VGLOBAL`
- Unique seeded instructions: `911`
- Seeded decode entries: `1200`

## Encoding Seeds

- `transferable_with_decoder_work` below is exact decoder-only work; `transferable_with_decoder_and_semantic_work` is listed separately.

### ENC_SOPP

- Example instruction: `S_ENDPGM`
- Rationale: Program control, wait/event sequencing, and branch bring-up.
- Unique instructions: `41`
- Seed entries: `41` (`default=41`, `alternate=0`)
- Support split: as-is `8`, decoder-only `0`, semantic-only `0`, decoder+semantic `14`, gfx1201-specific `19`
- Sample worklist: as-is `S_BRANCH`, `S_CBRANCH_SCC0`, `S_CBRANCH_SCC1`, `S_CBRANCH_VCCZ`, decoder-only None, semantic-only None, decoder+semantic `S_NOP`, `S_SETKILL`, `S_SETHALT`, `S_SLEEP`, gfx1201-specific `S_CLAUSE`, `S_DELAY_ALU`, `S_WAIT_ALU`, `S_WAIT_IDLE`

### ENC_SOP1

- Example instruction: `S_MOV_B32`
- Rationale: Scalar move, bit-manipulation, and exec-mask setup precedent.
- Unique instructions: `84`
- Seed entries: `84` (`default=84`, `alternate=0`)
- Support split: as-is `28`, decoder-only `0`, semantic-only `10`, decoder+semantic `0`, gfx1201-specific `46`
- Sample worklist: as-is `S_MOV_B32`, `S_MOV_B64`, `S_CMOV_B32`, `S_CMOV_B64`, decoder-only None, semantic-only `S_WQM_B32`, `S_WQM_B64`, `S_MOVRELS_B32`, `S_MOVRELS_B64`, decoder+semantic None, gfx1201-specific `S_CTZ_I32_B32`, `S_CTZ_I32_B64`, `S_CLZ_I32_U32`, `S_CLZ_I32_U64`

### ENC_SOP2

- Example instruction: `S_AND_B32`
- Rationale: Scalar binary core needed before wider control-flow work.
- Unique instructions: `72`
- Seed entries: `72` (`default=72`, `alternate=0`)
- Support split: as-is `41`, decoder-only `0`, semantic-only `0`, decoder+semantic `0`, gfx1201-specific `31`
- Sample worklist: as-is `S_ABSDIFF_I32`, `S_LSHL_B32`, `S_LSHL_B64`, `S_LSHR_B32`, decoder-only None, semantic-only None, decoder+semantic None, gfx1201-specific `S_ADD_CO_U32`, `S_SUB_CO_U32`, `S_ADD_CO_I32`, `S_SUB_CO_I32`

### ENC_SOPC

- Example instruction: `S_CMP_EQ_U32`
- Rationale: Scalar compare path for branch and predicate plumbing.
- Unique instructions: `46`
- Seed entries: `46` (`default=46`, `alternate=0`)
- Support split: as-is `18`, decoder-only `0`, semantic-only `0`, decoder+semantic `0`, gfx1201-specific `28`
- Sample worklist: as-is `S_CMP_EQ_I32`, `S_CMP_LG_I32`, `S_CMP_GT_I32`, `S_CMP_GE_I32`, decoder-only None, semantic-only None, decoder+semantic None, gfx1201-specific `S_CMP_LT_F32`, `S_CMP_LT_F16`, `S_CMP_EQ_F32`, `S_CMP_EQ_F16`

### ENC_SOPK

- Example instruction: `S_MOVK_I32`
- Rationale: Small scalar-immediate surface used by early control kernels.
- Unique instructions: `8`
- Seed entries: `8` (`default=8`, `alternate=0`)
- Support split: as-is `3`, decoder-only `0`, semantic-only `0`, decoder+semantic `3`, gfx1201-specific `2`
- Sample worklist: as-is `S_MOVK_I32`, `S_CMOVK_I32`, `S_MULK_I32`, decoder-only None, semantic-only None, decoder+semantic `S_GETREG_B32`, `S_SETREG_B32`, `S_CALL_B64`, gfx1201-specific `S_VERSION`, `S_ADDK_CO_I32`

### ENC_SMEM

- Example instruction: `S_LOAD_B32`
- Rationale: RDNA4 scalar memory is a first architecture-local blocker.
- Unique instructions: `28`
- Seed entries: `28` (`default=28`, `alternate=0`)
- Support split: as-is `0`, decoder-only `0`, semantic-only `0`, decoder+semantic `3`, gfx1201-specific `25`
- Sample worklist: as-is None, decoder-only None, semantic-only None, decoder+semantic `S_DCACHE_INV`, `S_ATC_PROBE`, `S_ATC_PROBE_BUFFER`, gfx1201-specific `S_LOAD_B32`, `S_LOAD_B64`, `S_LOAD_B128`, `S_LOAD_B256`

### ENC_VOP1

- Example instruction: `V_MOV_B32`
- Rationale: Vector move and conversion baseline reused across many programs.
- Unique instructions: `90`
- Seed entries: `90` (`default=90`, `alternate=0`)
- Support split: as-is `65`, decoder-only `0`, semantic-only `8`, decoder+semantic `0`, gfx1201-specific `17`
- Sample worklist: as-is `V_NOP`, `V_MOV_B32`, `V_READFIRSTLANE_B32`, `V_CVT_I32_F64`, decoder-only None, semantic-only `V_CVT_OFF_F32_I4`, `V_CVT_NORM_I16_F16`, `V_CVT_NORM_U16_F16`, `V_SWAP_B32`, decoder+semantic None, gfx1201-specific `V_CVT_NEAREST_I32_F32`, `V_CVT_FLOOR_I32_F32`, `V_PIPEFLUSH`, `V_MOV_B16`

### ENC_VOP2

- Example instruction: `V_ADD_F32`
- Rationale: Core vector arithmetic used by both compute and graphics paths.
- Unique instructions: `47`
- Seed entries: `47` (`default=47`, `alternate=0`)
- Support split: as-is `20`, decoder-only `0`, semantic-only `10`, decoder+semantic `0`, gfx1201-specific `17`
- Sample worklist: as-is `V_CNDMASK_B32`, `V_ADD_F64`, `V_ADD_F32`, `V_SUB_F32`, decoder-only None, semantic-only `V_SUBREV_F32`, `V_MUL_I32_I24`, `V_MUL_HI_I32_I24`, `V_MUL_U32_U24`, decoder+semantic None, gfx1201-specific `V_FMAMK_F16`, `V_FMAAK_F16`, `V_MUL_DX9_ZERO_F32`, `V_MIN_NUM_F64`
- New literal-FMA seeds: `V_FMAMK_F16` and `V_FMAAK_F16` add the first `ENC_VOP2` entries on the current phase-0 path that require a shared trailing literal dword.

### ENC_VOPC

- Example instruction: `V_CMP_EQ_F32`
- Rationale: Vector compare path for control, masking, and shader predicates.
- Unique instructions: `162`
- Seed entries: `162` (`default=162`, `alternate=0`)
- Support split: as-is `108`, decoder-only `30`, semantic-only `0`, decoder+semantic `24`, gfx1201-specific `0`
- Sample worklist: as-is `V_CMP_LT_F32`, `V_CMP_EQ_F32`, `V_CMP_LE_F32`, `V_CMP_GT_F32`, decoder-only `V_CMP_LT_F16`, `V_CMP_EQ_F16`, `V_CMP_LE_F16`, `V_CMP_GT_F16`, semantic-only None, decoder+semantic `V_CMP_LT_I16`, `V_CMP_EQ_I16`, `V_CMP_LE_I16`, `V_CMP_GT_I16`, gfx1201-specific None

### ENC_VOP3

- Example instruction: `V_ADD3_U32`
- Rationale: Largest instruction family and the main overlap with gfx950 precedent.
- Unique instructions: `434`
- Seed entries: `434` (`default=434`, `alternate=0`)
- Support split: as-is `232`, decoder-only `30`, semantic-only `24`, decoder+semantic `61`, gfx1201-specific `87`
- Sample worklist: as-is `V_NOP`, `V_MOV_B32`, `V_READFIRSTLANE_B32`, `V_CVT_I32_F64`, decoder-only `V_CMP_LT_F16`, `V_CMP_EQ_F16`, `V_CMP_LE_F16`, `V_CMP_GT_F16`, semantic-only `V_CVT_OFF_F32_I4`, `V_CVT_NORM_I16_F16`, `V_CVT_NORM_U16_F16`, `V_CVT_F32_FP8`, decoder+semantic `V_CUBEID_F32`, `V_CUBESC_F32`, `V_CUBETC_F32`, `V_CUBEMA_F32`, gfx1201-specific `V_CVT_NEAREST_I32_F32`, `V_CVT_FLOOR_I32_F32`, `V_PIPEFLUSH`, `V_MOV_B16`

### ENC_VDS

- Example instruction: `DS_ADD_U32`
- Rationale: LDS data path with a small fully-supported carry-over subset.
- Unique instructions: `123`
- Seed entries: `123` (`default=123`, `alternate=0`)
- Support split: as-is `27`, decoder-only `0`, semantic-only `0`, decoder+semantic `38`, gfx1201-specific `58`
- Sample worklist: as-is `DS_ADD_U32`, `DS_SUB_U32`, `DS_RSUB_U32`, `DS_INC_U32`, decoder-only None, semantic-only None, decoder+semantic `DS_MSKOR_B32`, `DS_MSKOR_RTN_B32`, `DS_SWIZZLE_B32`, `DS_CONSUME`, gfx1201-specific `DS_STORE_B32`, `DS_STORE_2ADDR_B32`, `DS_STORE_2ADDR_STRIDE64_B32`, `DS_CMPSTORE_B32`

### ENC_VGLOBAL

- Example instruction: `GLOBAL_LOAD_B32`
- Rationale: Global memory load/store family that replaces gfx950-specific naming.
- Unique instructions: `65`
- Seed entries: `65` (`default=65`, `alternate=0`)
- Support split: as-is `3`, decoder-only `0`, semantic-only `0`, decoder+semantic `0`, gfx1201-specific `62`
- Sample worklist: as-is `GLOBAL_ATOMIC_ADD_F32`, `GLOBAL_ATOMIC_PK_ADD_F16`, `GLOBAL_ATOMIC_PK_ADD_BF16`, decoder-only None, semantic-only None, decoder+semantic None, gfx1201-specific `GLOBAL_LOAD_U8`, `GLOBAL_LOAD_I8`, `GLOBAL_LOAD_U16`, `GLOBAL_LOAD_I16`
