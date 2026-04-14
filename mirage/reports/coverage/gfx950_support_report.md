# GFX950 Coverage Report

- Catalog instructions: 1242
- Semantic support: 895 (72.1%)
- Raw decode support: 946 (76.2% of total, 83.6% of measured)
- Raw decode measurable instructions: 1131

## Gaps

- Semantic-only coverage without measured decode: 0
- Decode-only without semantic support: 51
- Missing both semantic and decode support: 296

## Unmeasured Encoding Families

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

- None

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
- `S_CBRANCH_G_FORK`
- `S_RFE_RESTORE_B64`
- `S_SETVSKIP`
- `S_SET_GPR_IDX_ON`
- `V_CLREXCP`
- `V_SCREEN_PARTITION_4SE_B32`
- `V_SWAP_B32`
- `V_ACCVGPR_MOV_B32`
- `V_PRNG_B32`
- `V_PERMLANE16_SWAP_B32`
- `V_PERMLANE32_SWAP_B32`
- `V_SUBREV_F32`
- `V_FMAC_F64`

## Missing-Both Sample

- `DS_READ_B64_TR_B4`
- `DS_READ_B96_TR_B6`
- `DS_READ_B64_TR_B8`
- `DS_READ_B64_TR_B16`
- `S_CBRANCH_I_FORK`
- `S_GETREG_B32`
- `S_SETREG_B32`
- `S_SETREG_IMM32_B32`
- `S_CALL_B64`
- `S_NOP`
- `S_WAKEUP`
- `S_SETKILL`
- `S_WAITCNT`
- `S_SETHALT`
- `S_SLEEP`
- `S_SETPRIO`
- `S_SENDMSG`
- `S_SENDMSGHALT`
- `S_TRAP`
- `S_INCPERFLEVEL`
- `S_DECPERFLEVEL`
- `S_TTRACEDATA`
- `S_CBRANCH_CDBGSYS`
- `S_CBRANCH_CDBGUSER`
- `S_CBRANCH_CDBGSYS_OR_USER`
