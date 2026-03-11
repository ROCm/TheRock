# gfx1201 Phase-0 Execution Seed

This report captures the first executable `gfx1201` seed slice layered on top of the
existing phase-0 compute selector and seed catalog.

## Executable opcodes

- `S_ENDPGM`
- `S_NOP`
- `S_MOV_B32`
- `S_MOVK_I32`
- `V_MOV_B32`

## Covered encodings

- `ENC_SOPP`: `S_ENDPGM`, `S_NOP`
- `ENC_SOP1`: `S_MOV_B32`
- `ENC_SOPK`: `S_MOVK_I32`
- `ENC_VOP1`: `V_MOV_B32`

## Seed behavior

- Decoder now returns real `DecodedInstruction` values for the executable slice.
- `S_MOV_B32` and `V_MOV_B32` accept SGPR sources, inline integer sources, and literal dwords.
- `S_MOVK_I32` sign-extends the imported `SIMM16` operand into the decoded `Imm32` form.
- Interpreter now compiles and executes the seed slice locally.
- `V_MOV_B32` execution respects `exec_mask`.

## Still scaffolded

- Route-matched but non-executable phase-0 compute instructions continue to fail with
  the existing selector-aware stub errors.
- `ENC_SOP2`, `ENC_SOPC`, `ENC_SMEM`, `ENC_VOP2`, `ENC_VOPC`, `ENC_VOP3`,
  `ENC_VDS`, and `ENC_VGLOBAL` remain route-only in this slice.
