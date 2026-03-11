# gfx1201 Phase-0 Execution Seed

This report captures the current executable `gfx1201` seed slice layered on top of the
existing phase-0 compute selector and seed catalog. The local seed surface now covers
74 executable phase-0 ops without leaving architecture-local `gfx1201` files.

## Executable opcodes

- `S_ENDPGM`
- `S_NOP`
- `S_ADD_U32`
- `S_ADD_I32`
- `S_SUB_U32`
- `S_CMP_EQ_I32`
- `S_CMP_LG_I32`
- `S_CMP_GT_I32`
- `S_CMP_EQ_U32`
- `S_CMP_LG_U32`
- `S_CMP_GE_I32`
- `S_CMP_LT_I32`
- `S_CMP_LE_I32`
- `S_CMP_GT_U32`
- `S_CMP_GE_U32`
- `S_CMP_LT_U32`
- `S_CMP_LE_U32`
- `S_BRANCH`
- `S_CBRANCH_SCC0`
- `S_CBRANCH_SCC1`
- `S_CBRANCH_VCCZ`
- `S_CBRANCH_VCCNZ`
- `S_CBRANCH_EXECZ`
- `S_CBRANCH_EXECNZ`
- `S_MOV_B32`
- `S_MOVK_I32`
- `V_MOV_B32`
- `V_CMP_EQ_I32`
- `V_CMP_NE_I32`
- `V_CMP_LT_I32`
- `V_CMP_LE_I32`
- `V_CMP_GT_I32`
- `V_CMP_GE_I32`
- `V_CMP_EQ_U32`
- `V_CMP_NE_U32`
- `V_CMP_LT_U32`
- `V_CMP_LE_U32`
- `V_CMP_GT_U32`
- `V_CMP_GE_U32`
- `V_CMPX_EQ_I32`
- `V_CMPX_NE_I32`
- `V_CMPX_LT_I32`
- `V_CMPX_LE_I32`
- `V_CMPX_GT_I32`
- `V_CMPX_GE_I32`
- `V_CMPX_EQ_U32`
- `V_CMPX_NE_U32`
- `V_CMPX_LT_U32`
- `V_CMPX_LE_U32`
- `V_CMPX_GT_U32`
- `V_CMPX_GE_U32`
- `V_NOT_B32`
- `V_BFREV_B32`
- `V_CVT_F32_UBYTE0`
- `V_CVT_F32_UBYTE1`
- `V_CVT_F32_UBYTE2`
- `V_CVT_F32_UBYTE3`
- `V_CVT_F32_I32`
- `V_CVT_F32_U32`
- `V_CVT_U32_F32`
- `V_CVT_I32_F32`
- `V_ADD_U32`
- `V_SUB_U32`
- `V_SUBREV_U32`
- `V_MIN_I32`
- `V_MAX_I32`
- `V_MIN_U32`
- `V_MAX_U32`
- `V_LSHRREV_B32`
- `V_ASHRREV_I32`
- `V_LSHLREV_B32`
- `V_AND_B32`
- `V_OR_B32`
- `V_XOR_B32`

## Covered encodings

- `ENC_SOPP`: `S_ENDPGM`, `S_NOP`, `S_BRANCH`, `S_CBRANCH_SCC0`, `S_CBRANCH_SCC1`, `S_CBRANCH_VCCZ`, `S_CBRANCH_VCCNZ`, `S_CBRANCH_EXECZ`, `S_CBRANCH_EXECNZ`
- `ENC_SOP2`: imported `S_ADD_CO_U32`, `S_ADD_CO_I32`, `S_SUB_CO_U32` normalized to `S_ADD_U32`, `S_ADD_I32`, `S_SUB_U32`
- `ENC_SOPC`: `S_CMP_EQ_I32`, `S_CMP_LG_I32`, `S_CMP_GT_I32`, `S_CMP_EQ_U32`, `S_CMP_LG_U32`, `S_CMP_GE_I32`, `S_CMP_LT_I32`, `S_CMP_LE_I32`, `S_CMP_GT_U32`, `S_CMP_GE_U32`, `S_CMP_LT_U32`, `S_CMP_LE_U32`
- `ENC_SOP1`: `S_MOV_B32`
- `ENC_SOPK`: `S_MOVK_I32`
- `ENC_VOP1`: `V_MOV_B32`, `V_NOT_B32`, `V_BFREV_B32`, `V_CVT_F32_UBYTE0`, `V_CVT_F32_UBYTE1`, `V_CVT_F32_UBYTE2`, `V_CVT_F32_UBYTE3`, `V_CVT_F32_I32`, `V_CVT_F32_U32`, `V_CVT_U32_F32`, `V_CVT_I32_F32`
- `ENC_VOP2`: imported `V_ADD_NC_U32`, `V_SUB_NC_U32`, `V_SUBREV_NC_U32` normalized to `V_ADD_U32`, `V_SUB_U32`, `V_SUBREV_U32`; plus `V_MIN_I32`, `V_MAX_I32`, `V_MIN_U32`, `V_MAX_U32`, `V_LSHRREV_B32`, `V_ASHRREV_I32`, `V_LSHLREV_B32`, `V_AND_B32`, `V_OR_B32`, `V_XOR_B32`
- `ENC_VOPC`: `V_CMP_EQ/NE/LT/LE/GT/GE_I32`, `V_CMP_EQ/NE/LT/LE/GT/GE_U32`, `V_CMPX_EQ/NE/LT/LE/GT/GE_I32`, and `V_CMPX_EQ/NE/LT/LE/GT/GE_U32`, each decoded with an implicit `VCC` destination

## Seed behavior

- Decoder now returns real `DecodedInstruction` values for the executable slice.
- Decoder now attaches shared common operand descriptors for the active seed slice, including destination/source roles, register classes, fragment shapes, and implicit `VCC` metadata for `VOPC`.
- Decoder normalizes imported arithmetic/control opcodes onto the Mirage alias surface for the new subset.
- `S_ADD_U32`, `S_ADD_I32`, and `S_SUB_U32` decode from `ENC_SOP2`.
- The full 32-bit `SOPC` compare set now decodes locally: `S_CMP_EQ/LG/GT/GE/LT/LE` across signed and unsigned `U32/I32` forms.
- `S_BRANCH`, `S_CBRANCH_SCC0`, `S_CBRANCH_SCC1`, `S_CBRANCH_VCCZ`, `S_CBRANCH_VCCNZ`, `S_CBRANCH_EXECZ`, and `S_CBRANCH_EXECNZ` decode from `ENC_SOPP` with sign-extended relative deltas.
- `S_MOV_B32` and `V_MOV_B32` accept SGPR sources, inline integer sources, and literal dwords.
- `V_NOT_B32`, `V_BFREV_B32`, and `V_CVT_F32_UBYTE0/1/2/3` decode from `ENC_VOP1` on the same source path as `V_MOV_B32`.
- `V_CVT_F32_I32`, `V_CVT_F32_U32`, `V_CVT_U32_F32`, and `V_CVT_I32_F32` continue to decode from `ENC_VOP1`.
- `V_ADD_U32`, `V_SUB_U32`, `V_SUBREV_U32`, `V_MIN/MAX_I32`, `V_MIN/MAX_U32`, `V_LSHRREV_B32`, `V_ASHRREV_I32`, `V_LSHLREV_B32`, `V_AND_B32`, `V_OR_B32`, and `V_XOR_B32` now decode from `ENC_VOP2` with the existing phase-0 selector path.
- `V_CMP_EQ/NE/LT/LE/GT/GE_I32`, `V_CMP_EQ/NE/LT/LE/GT/GE_U32`, `V_CMPX_EQ/NE/LT/LE/GT/GE_I32`, and `V_CMPX_EQ/NE/LT/LE/GT/GE_U32` now decode from `ENC_VOPC` with SGPR/VGPR/inline/literal source handling on the existing selector path.
- `S_MOVK_I32` sign-extends the imported `SIMM16` operand into the decoded `Imm32` form.
- Interpreter now compiles and executes the seed slice locally.
- Scalar compare updates `SCC` across the full local 32-bit signed/unsigned compare set.
- `S_BRANCH`, `S_CBRANCH_SCC0`, `S_CBRANCH_SCC1`, `S_CBRANCH_VCCZ`, `S_CBRANCH_VCCNZ`, `S_CBRANCH_EXECZ`, and `S_CBRANCH_EXECNZ` update `pc` on the compiled seed path using relative instruction deltas.
- Scalar add/sub updates `SCC` using the imported carry/borrow behavior.
- `V_MOV_B32` execution respects `exec_mask`.
- `V_CMP_EQ/NE/LT/LE/GT/GE_I32` and `V_CMP_EQ/NE/LT/LE/GT/GE_U32` execute lane-wise under `exec_mask` while preserving inactive-lane `vcc_mask` bits.
- `V_CMPX_EQ/NE/LT/LE/GT/GE_I32` and `V_CMPX_EQ/NE/LT/LE/GT/GE_U32` execute lane-wise under `exec_mask` and materialize the resulting active-lane compare mask into both `vcc_mask` and `exec_mask`.
- `V_NOT_B32`, `V_BFREV_B32`, and `V_CVT_F32_UBYTE0/1/2/3` execute lane-wise under `exec_mask`.
- `V_CVT_F32_I32`, `V_CVT_F32_U32`, `V_CVT_U32_F32`, and `V_CVT_I32_F32` execute lane-wise under `exec_mask` with gfx950-consistent truncation for the float-to-int subset.
- `V_ADD_U32`, `V_SUB_U32`, `V_SUBREV_U32`, `V_MIN/MAX_I32`, `V_MIN/MAX_U32`, `V_LSHRREV_B32`, `V_ASHRREV_I32`, `V_LSHLREV_B32`, `V_AND_B32`, `V_OR_B32`, and `V_XOR_B32` execute lane-wise under `exec_mask`.

## Still scaffolded

- Route-matched but non-executable phase-0 compute instructions continue to fail with
  the existing selector-aware stub errors.
- `ENC_SMEM`, `ENC_VOP3`, `ENC_VDS`, and `ENC_VGLOBAL` remain route-only in this slice.
- The remaining `ENC_VOPC` surface outside this seed is still scaffolded: floating-point, 64-bit, and 16-bit compare families.
- The next coherent extension point is the remaining `ENC_VOPC` floating-point/wider compare work or `ENC_VOP3`/half-conversion work rather than more `SOPC`/`SOPP`/`VOP1`/`VOP2` integer seed growth.
