# gfx1201 Phase-0 Execution Seed

This report captures the first executable `gfx1201` seed slice layered on top of the
existing phase-0 compute selector and seed catalog.

## Executable opcodes

- `S_ENDPGM`
- `S_NOP`
- `S_ADD_U32`
- `S_ADD_I32`
- `S_SUB_U32`
- `S_MOV_B32`
- `S_MOVK_I32`
- `V_MOV_B32`
- `V_ADD_U32`

## Covered encodings

- `ENC_SOPP`: `S_ENDPGM`, `S_NOP`
- `ENC_SOP2`: imported `S_ADD_CO_U32`, `S_ADD_CO_I32`, `S_SUB_CO_U32` normalized to `S_ADD_U32`, `S_ADD_I32`, `S_SUB_U32`
- `ENC_SOP1`: `S_MOV_B32`
- `ENC_SOPK`: `S_MOVK_I32`
- `ENC_VOP1`: `V_MOV_B32`
- `ENC_VOP2`: imported `V_ADD_NC_U32` normalized to `V_ADD_U32`

## Seed behavior

- Decoder now returns real `DecodedInstruction` values for the executable slice.
- Decoder normalizes imported arithmetic/control opcodes onto the Mirage alias surface for the new subset.
- `S_ADD_U32`, `S_ADD_I32`, and `S_SUB_U32` decode from `ENC_SOP2`.
- `S_MOV_B32` and `V_MOV_B32` accept SGPR sources, inline integer sources, and literal dwords.
- `V_ADD_U32` decodes from `ENC_VOP2` with the existing phase-0 selector path.
- `S_MOVK_I32` sign-extends the imported `SIMM16` operand into the decoded `Imm32` form.
- Interpreter now compiles and executes the seed slice locally.
- Scalar add/sub updates `SCC` using the imported carry/borrow behavior.
- `V_MOV_B32` execution respects `exec_mask`.
- `V_ADD_U32` executes lane-wise under `exec_mask`.

## Still scaffolded

- Route-matched but non-executable phase-0 compute instructions continue to fail with
  the existing selector-aware stub errors.
- `ENC_SOPC`, `ENC_SMEM`, `ENC_VOPC`, `ENC_VOP3`,
  `ENC_VDS`, and `ENC_VGLOBAL` remain route-only in this slice.
