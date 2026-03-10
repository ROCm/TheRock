# gfx1201 Bring-up Plan

## Imported Baseline

- Imported target: `gfx1201`
- Architecture source: `AMD RDNA 4`
- Release date: `2025-02-19`
- Schema version: `1.1.0`
- Source XML: `amdgpu_isa_rdna4.xml`
- Imported instructions: `1264`
- Imported encodings: `5062`

## Support Buckets

### transferable_full

- Instruction count: `363`
- Meaning: Present in gfx950 with both raw decode coverage and interpreter semantics already implemented.
- Top families: v(235), s(98), ds(27), global(3)
- Top encodings: ENC_VOP3(232), VOP3_INST_LITERAL(229), VOP3_VOP_DPP16(149), VOP3_VOP_DPP8(149), ENC_VOPC(108), VOPC_INST_LITERAL(108), ENC_VOP1(65), VOP1_INST_LITERAL(63), VOPC_VOP_DPP16(54), VOPC_VOP_DPP8(54)

### transferable_decode_only

- Instruction count: `36`
- Meaning: Binary shape already appears in gfx950 coverage, but execution semantics are still missing there.
- Top families: v(26), s(10)
- Top encodings: ENC_VOP3(24), VOP3_INST_LITERAL(24), VOP3_VOP_DPP16(22), VOP3_VOP_DPP8(22), ENC_SOP1(10), ENC_VOP2(10), VOP2_INST_LITERAL(10), VOP2_VOP_DPP16(9), VOP2_VOP_DPP8(9), ENC_VOP1(8)

### transferable_semantic_only

- Instruction count: `30`
- Meaning: Execution semantics exist in gfx950 coverage, but binary decode coverage has not been wired up yet.
- Top families: v(30)
- Top encodings: VOPC_VOP_DPP16(30), VOPC_VOP_DPP8(30), ENC_VOP3(30), VOP3_INST_LITERAL(30), VOP3_VOP_DPP16(30), VOP3_VOP_DPP8(30), ENC_VOPC(30), VOPC_INST_LITERAL(30)

### known_but_unsupported

- Instruction count: `167`
- Meaning: Opcode name already exists in the gfx950 catalog, but neither decoder nor interpreter support is complete.
- Top families: v(86), ds(38), s(21), buffer(11), tbuffer(8), flat(3)
- Top encodings: ENC_VOP3(61), VOP3_INST_LITERAL(61), VOP3_VOP_DPP16(52), VOP3_VOP_DPP8(52), ENC_VDS(38), VOPC_VOP_DPP16(24), VOPC_VOP_DPP8(24), ENC_VOPC(24), VOPC_INST_LITERAL(24), ENC_VOP3P(21)

### new_vs_gfx950

- Instruction count: `668`
- Meaning: Opcode name is absent from gfx950 and needs RDNA4-local decode and semantic work.
- Top families: v(155), s(153), image(91), buffer(62), global(62), ds(60), flat(52), scratch(24)
- Top encodings: ENC_VOP3(87), VOP3_INST_LITERAL(83), ENC_VBUFFER(70), VOP3_VOP_DPP16(68), VOP3_VOP_DPP8(68), ENC_VGLOBAL(62), ENC_VDS(58), ENC_VSAMPLE(58), ENC_VFLAT(52), ENC_SOP1(46)

## Phase 0 Decoder Focus

- `ENC_SOPP` (41 instructions, example `S_ENDPGM`): Program control, wait/event sequencing, and branch bring-up.
- `ENC_SOP1` (84 instructions, example `S_MOV_B32`): Scalar move, bit-manipulation, and exec-mask setup precedent.
- `ENC_SOP2` (72 instructions, example `S_AND_B32`): Scalar binary core needed before wider control-flow work.
- `ENC_SOPC` (46 instructions, example `S_CMP_EQ_U32`): Scalar compare path for branch and predicate plumbing.
- `ENC_SOPK` (8 instructions, example `S_MOVK_I32`): Small scalar-immediate surface used by early control kernels.
- `ENC_SMEM` (28 instructions, example `S_LOAD_B32`): RDNA4 scalar memory is a first architecture-local blocker.
- `ENC_VOP1` (90 instructions, example `V_MOV_B32`): Vector move and conversion baseline reused across many programs.
- `ENC_VOP2` (45 instructions, example `V_ADD_F32`): Core vector arithmetic used by both compute and graphics paths.
- `ENC_VOPC` (162 instructions, example `V_CMP_EQ_F32`): Vector compare path for control, masking, and shader predicates.
- `ENC_VOP3` (434 instructions, example `V_ADD3_U32`): Largest instruction family and the main overlap with gfx950 precedent.
- `ENC_VDS` (123 instructions, example `DS_ADD_U32`): LDS data path with a small fully-supported carry-over subset.
- `ENC_VGLOBAL` (65 instructions, example `GLOBAL_LOAD_B32`): Global memory load/store family that replaces gfx950-specific naming.

## Phase 1 Decoder Focus

- `ENC_VBUFFER` (89 instructions, example `BUFFER_LOAD_FORMAT_X`): RDNA4 buffer resource path and typed buffer forms.
- `ENC_VFLAT` (55 instructions, example `FLAT_LOAD_B32`): Flat memory path layered after scalar/global addressing is stable.
- `ENC_VSCRATCH` (24 instructions, example `SCRATCH_LOAD_B32`): Scratch memory path is target-local and can come after flat/global.
- `ENC_VSAMPLE` (58 instructions, example `IMAGE_SAMPLE`): Graphics sampling path unique to the RDNA4 import.
- `ENC_VIMAGE` (33 instructions, example `IMAGE_LOAD`): Image load/store/atomic path separate from sampled operations.
- `ENC_VEXPORT` (1 instructions, example `EXPORT`): Graphics export path with no gfx950 precedent.
- `ENC_VINTERP` (6 instructions, example `V_INTERP_P10_F32`): Shader interpolation path that anchors graphics-local vector work.
- `ENC_VOP3P` (56 instructions, example `V_PK_ADD_F16`): Packed/vector math delta visible in the RDNA4 import.

## Carry-over Family Focus

- `v` / `transferable_full` (235 instructions, example `V_MOV_B32`): Largest fully-supported carry-over bucket from gfx950.
- `s` / `transferable_full` (98 instructions, example `S_MOV_B32`): Scalar control and ALU subset with direct gfx950 precedent.
- `ds` / `transferable_full` (27 instructions, example `DS_ADD_U32`): Small LDS subset already proven end-to-end on gfx950.
- `global` / `transferable_full` (3 instructions, example `GLOBAL_ATOMIC_ADD_F32`): Only a narrow global atomic subset currently has full precedent.
- `v` / `transferable_decode_only` (26 instructions, example `V_CVT_F32_FP8`): Binary decode precedent exists, but execution semantics still need work.
- `v` / `transferable_semantic_only` (30 instructions, example `V_CMP_LT_F16`): Interpreter precedent exists, but raw binary decode work is missing.
- `s` / `transferable_decode_only` (10 instructions, example `S_GETPC_B64`): Scalar control opcodes appear in coverage but are not executable yet.

## RDNA4 Delta Family Focus

- `s` / `new_vs_gfx950` (153 instructions, example `S_LOAD_B32`): Scalar memory and wait/barrier control differ from gfx950.
- `v` / `new_vs_gfx950` (155 instructions, example `V_INTERP_P10_F32`): Graphics-local vector forms expand the RDNA4 surface.
- `image` / `new_vs_gfx950` (91 instructions, example `IMAGE_LOAD`): Image pipeline has no direct gfx950 baseline.
- `buffer` / `new_vs_gfx950` (62 instructions, example `BUFFER_LOAD_FORMAT_X`): Buffer load/store/atomic naming and resource handling are RDNA4-local.
- `global` / `new_vs_gfx950` (62 instructions, example `GLOBAL_LOAD_B32`): Global memory family uses the RDNA4 ISA surface instead of gfx950 forms.
- `ds` / `new_vs_gfx950` (60 instructions, example `DS_LOAD_B32`): LDS load/store and atomic expansion beyond the gfx950 subset.
- `flat` / `new_vs_gfx950` (52 instructions, example `FLAT_LOAD_B32`): Flat addressing path needs RDNA4-specific decode and operand wiring.
- `scratch` / `new_vs_gfx950` (24 instructions, example `SCRATCH_LOAD_B32`): Scratch memory is absent from the current gfx950-local layout.
- `tbuffer` / `new_vs_gfx950` (8 instructions, example `TBUFFER_LOAD_FORMAT_X`): Typed buffer graphics path should layer on top of buffer support.
- `export` / `new_vs_gfx950` (1 instructions, example `EXPORT`): Graphics export path is unique to the RDNA4 import.

## Blockers

- GPUOpen publishes gfx1201 through the architecture-level RDNA4 XML, so target mapping cannot assume one XML per target.
- gfx1201 barrier/wait control differs from gfx950: the import contains `S_BARRIER_WAIT` and `S_WAIT_*`, while `S_BARRIER` is absent.
- RDNA4-only graphics and memory encodings (`ENC_VBUFFER`, `ENC_VIMAGE`, `ENC_VSAMPLE`, `ENC_VFLAT`, `ENC_VSCRATCH`, `ENC_VEXPORT`, `ENC_VINTERP`, `ENC_VOP3P`) need architecture-local plumbing before gfx950 logic can be reused.

## Notes

- This plan is derived from the imported gfx1201 inventory, the current gfx950 catalog, and the checked-in gfx950 support snapshot.
