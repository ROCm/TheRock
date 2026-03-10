# GFX950 Coverage Report

- Catalog instructions: 1242
- Semantic support: 510 (41.1%)
- Raw decode support: 553 (44.5% of total, 54.5% of measured)
- Raw decode measurable instructions: 1014

## Gaps

- Semantic-only coverage without measured decode: 36
- Decode-only without semantic support: 79
- Missing both semantic and decode support: 653

## Unmeasured Encoding Families

- `ENC_FLAT_SCRATCH`
- `ENC_MTBUF`
- `ENC_MUBUF`
- `ENC_VOP3P`
- `ENC_VOP3PX2`
- `SOP1_INST_LITERAL`
- `SOP2_INST_LITERAL`
- `SOPC_INST_LITERAL`
- `SOPK_INST_LITERAL`
- `VOP1_INST_LITERAL`
- `VOP1_VOP_DPP`
- `VOP1_VOP_SDWA`
- `VOP2_INST_LITERAL`
- `VOP2_VOP_DPP`
- `VOP2_VOP_SDWA`
- `VOP2_VOP_SDWA_SDST_ENC`
- `VOP3P_MFMA`
- `VOPC_INST_LITERAL`
- `VOPC_VOP_SDWA_SDST_ENC`

## Semantic-Only Sample

- `V_MIN_F64`
- `V_MAX_F64`
- `V_CMP_CLASS_F16`
- `V_CMPX_CLASS_F16`
- `V_CMP_F_F16`
- `V_CMP_LT_F16`
- `V_CMP_EQ_F16`
- `V_CMP_LE_F16`
- `V_CMP_GT_F16`
- `V_CMP_LG_F16`
- `V_CMP_GE_F16`
- `V_CMP_O_F16`
- `V_CMP_U_F16`
- `V_CMP_NGE_F16`
- `V_CMP_NLG_F16`
- `V_CMP_NGT_F16`
- `V_CMP_NLE_F16`
- `V_CMP_NEQ_F16`
- `V_CMP_NLT_F16`
- `V_CMP_TRU_F16`
- `V_CMPX_F_F16`
- `V_CMPX_LT_F16`
- `V_CMPX_EQ_F16`
- `V_CMPX_LE_F16`
- `V_CMPX_GT_F16`

## Decode-Only Sample

- `S_WQM_B32`
- `S_WQM_B64`
- `S_GETPC_B64`
- `S_SETPC_B64`
- `S_SWAPPC_B64`
- `S_RFE_B64`
- `S_MOVRELS_B32`
- `S_MOVRELS_B64`
- `S_MOVRELD_B32`
- `S_MOVRELD_B64`
- `S_CBRANCH_JOIN`
- `S_SET_GPR_IDX_IDX`
- `S_ADD_I32`
- `S_SUB_I32`
- `S_MIN_I32`
- `S_MIN_U32`
- `S_MAX_I32`
- `S_MAX_U32`
- `S_CBRANCH_G_FORK`
- `S_RFE_RESTORE_B64`
- `S_SETVSKIP`
- `S_SET_GPR_IDX_ON`
- `S_CMP_EQ_U64`
- `S_CMP_LG_U64`
- `V_NOP`

## Missing-Both Sample

- `DS_MSKOR_B32`
- `DS_WRITE2_B32`
- `DS_WRITE2ST64_B32`
- `DS_CMPST_B32`
- `DS_CMPST_F32`
- `DS_NOP`
- `DS_PK_ADD_F16`
- `DS_PK_ADD_BF16`
- `DS_WRITE_ADDTID_B32`
- `DS_ADD_RTN_U32`
- `DS_SUB_RTN_U32`
- `DS_RSUB_RTN_U32`
- `DS_INC_RTN_U32`
- `DS_DEC_RTN_U32`
- `DS_MIN_RTN_I32`
- `DS_MAX_RTN_I32`
- `DS_MIN_RTN_U32`
- `DS_MAX_RTN_U32`
- `DS_AND_RTN_B32`
- `DS_OR_RTN_B32`
- `DS_XOR_RTN_B32`
- `DS_MSKOR_RTN_B32`
- `DS_WRXCHG_RTN_B32`
- `DS_WRXCHG2_RTN_B32`
- `DS_WRXCHG2ST64_RTN_B32`
