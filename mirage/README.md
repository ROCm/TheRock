# Mirage

Mirage is a simulator-first ROCm development environment that lives inside the
TheRock monorepo but is not part of the default top-level build graph.

This subtree is intentionally split into:

- `python/` for packaging, CLI entrypoints, and orchestration glue
- `lib/sim/` for the simulator-native execution core
- `native/` for a standalone CMake target that builds the initial simulator core
- `compat/` for ROCm-facing compatibility layers such as `libhsakmt`
- `third_party/amd_gpu_isa/` for vendored machine-readable AMD ISA specs used to generate simulator instruction catalogs

The ISA layer is also split by ownership boundary:

- `lib/sim/isa/common/` for shared instruction, memory, and wave-state types
- `lib/sim/isa/gfx950/` for CDNA4-specific decoder and execution interfaces
- `native/src/isa/gfx950/` and `native/tests/isa/gfx950/` for the current
  architecture-specific implementation and tests

The legacy top-level `lib/sim/isa/gfx950_*.h` and common ISA header paths remain
as compatibility wrappers so new architecture work can land in dedicated
directories without forcing a single large include-path rewrite.

## Local Development

Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Native simulator core:

```bash
python tools/generate_gfx950_isa_catalog.py
cmake -S native -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
./build/native/mirage_gfx950_runtime_test
```

The standalone native build also vendors and builds the `simdojo` discrete
event simulation library from `third_party/simdojo/`, with a dedicated smoke
test `mirage_simdojo_smoke_test` to verify the import compiles and links
cleanly inside Mirage without pulling in other `rocjitsu` infrastructure.
Mirage now also carries a first real `simdojo` integration seam through
`lib/sim/timing/simdojo_dispatch_harness.h`: a timed dispatch source and
single-GPU executor component that use `simdojo` as the outer event shell
while preserving `SingleGpuSimulator` as the functional execution engine.
That path is covered by `mirage_simdojo_dispatch_harness_test`.

The current native target brings up a single-GPU functional simulator slice:
one virtual device, HBM-backed allocations, compute queue state, and a narrow
dispatch path for smoke-testing execution without wiring Mirage into the
top-level TheRock build. That path now includes both synthetic helper kernels
and a first real `gfx950` binary dispatch routed through the decoder and
interpreter, including optional seeded SGPR/VGPR launch state for richer
single-wave tests plus a minimal multi-wave workgroup path with shared LDS and
`S_BARRIER`. Repeated real-binary launches now flow through a compiled internal
opcode form cached by code buffer VA and write version, so hot launches bypass
both binary decode and string-based opcode dispatch.

For `gfx950`, Mirage now vendors the official AMD CDNA4 machine-readable ISA XML
and generates a compiled instruction catalog plus test fixtures from it. The
catalog currently provides instruction metadata coverage for the full ISA while
execution semantics remain intentionally narrow. The first implemented execution
slice is a real `gfx950` interpreter with both decoded-instruction and compiled
internal-opcode entrypoints for a small subset of ops: `S_MOV_B32`,
`S_ADD_{I,U}32`,
`S_MOV_B64`, `S_NOT_B32`, `S_NOT_B64`, `S_BREV_B32`, `S_BREV_B64`,
`S_BCNT{0,1}_I32_B{32,64}`, `S_{FF0,FF1}_I32_B{32,64}`,
`S_FLBIT_I32{,_B32,_B64,_I64}`, `S_CMOV_B{32,64}`, `S_CMOVK_I32`, `S_ABS_I32`,
`S_BITREPLICATE_B64_B32`, `S_QUADMASK_B{32,64}`,
`S_MOVK_I32`, `S_ADDK_I32`,
`S_ADDC_U32`, `S_SUB_{I,U}32`, `S_SUBB_U32`,
`S_{MIN,MAX}_{I,U}32`,
`S_MUL_I32`, `S_MULK_I32`,
`S_MUL_HI_U32`, `S_MUL_HI_I32`, `S_SEXT_I32_I{8,16}`,
`S_LSHL{1,2,3,4}_ADD_U32`,
`S_{LSHL,LSHR}_B32`, `S_{LSHL,LSHR}_B64`, `S_ASHR_I32`, `S_ASHR_I64`,
`S_BFM_B32`, `S_BFM_B64`,
`S_PACK_{LL,LH,HH}_B32_B16`, `S_CSELECT_B{32,64}`, `S_ABSDIFF_I32`,
`S_BFE_{U,I}{32,64}`,
`S_{AND,OR,XOR,ANDN2,ORN2,NAND,NOR,XNOR}_B32`,
`S_{AND,OR,XOR,NAND,NOR,XNOR,ANDN2,ORN2}_B64`,
`S_CMP_{EQ,LG,GT,GE,LT,LE}_{I32,U32}`, `S_CMP_{EQ,LG}_U64`, `S_CMPK_*`,
`S_BITCMP{0,1}_{B32,B64}`, `S_BRANCH`,
`S_CBRANCH_SCC0`, `S_CBRANCH_SCC1`, `S_CBRANCH_VCC{Z,NZ}`, `S_BARRIER`,
`S_CBRANCH_EXEC{Z,NZ}`,
`S_{AND,OR,XOR,ANDN1,ANDN2,ORN1,ORN2,NAND,NOR,XNOR}_SAVEEXEC_B64`,
`S_ANDN{1,2}_WREXEC_B64`,
`S_LOAD_DWORD`, `S_LOAD_DWORDX2`,
`S_STORE_DWORD`, `V_MOV_B32`, `V_MOV_B64`, `V_READFIRSTLANE_B32`,
`V_{ADD,SUB,SUBREV}_U32`, `V_{ADD,SUB,MUL,MIN,MAX}_F32`,
`V_{ADD,ADDC,SUB,SUBB,SUBREV,SUBBREV}_CO_U32`,
`V_READLANE_B32`, `V_WRITELANE_B32`, `V_MUL_LO_U32`,
`V_MUL_HI_{U32,I32}`, `V_BCNT_U32_B32`, `V_BFM_B32`,
`V_MBCNT_{LO,HI}_U32_B32`,
`V_{ADD,MUL}_F64`, `V_ADD3_U32`, `V_FMA_{F32,F64}`,
`V_LSHL_ADD_U64`, `V_{LSHL,ADD_LSHL}_U32`,
`V_{LSHL_OR,AND_OR}_B32`,
`V_OR3_B32`, `V_XAD_U32`,
`V_{LSHLREV,LSHRREV}_B64`, `V_ASHRREV_I64`,
`V_LERP_U8`, `V_PERM_B32`,
`V_BFE_{U,I32}`, `V_BFI_B32`,
`V_ALIGN{BIT,BYTE}_B32`,
`V_{MIN3,MAX3,MED3}_{I32,U32}`,
`V_SAD_{U8,HI_U8,U16,U32}`, `V_MAD_{I32_I24,U32_U24,U64_U32,I64_I32}`,
`V_CMP_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F32`,
`V_CMPX_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F32`,
`V_CMP_CLASS_F32`, `V_CMPX_CLASS_F32`,
`V_CMP_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F64`,
`V_CMPX_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F64`,
`V_CMP_CLASS_F64`, `V_CMPX_CLASS_F64`,
`V_CMP_{EQ,NE,LT,LE,GT,GE}_{I32,U32}`,
`V_CMP_{F,LT,EQ,LE,GT,NE,GE,T}_{I64,U64}`,
`V_CMPX_{F,LT,EQ,LE,GT,NE,GE,T}_{I32,U32,I64,U64}`,
`V_CNDMASK_B32`,
`V_{MIN,MAX}_{I32,U32}`, `V_{LSHLREV,LSHRREV}_B32`, `V_ASHRREV_I32`,
`V_{AND,OR,XOR}_B32`, `V_NOT_B32`, `V_BFREV_B32`, `V_FFBH_U32`,
`V_FFBL_B32`, `V_FFBH_I32`,
`V_CVT_F16_{U16,I16}`, `V_CVT_{U16,I16}_F16`,
`V_SAT_PK_U8_I16`,
`V_CVT_F32_UBYTE{0,1,2,3}`,
`V_CVT_F32_{I32,U32,F16,F64}`, `V_CVT_F64_{I32,U32}`,
`V_CVT_{I32,U32,F16,F64}_F32`, `V_CVT_{RPI,FLR}_I32_F32`,
`V_CVT_{I32,U32}_F64`,
`V_{FRACT,TRUNC,CEIL,RNDNE,FLOOR}_{F16,F32,F64}`,
`V_{RCP,SQRT,RSQ,LOG,EXP,SIN,COS}_F16`, `V_FREXP_{MANT,EXP_I16}_F16`,
`V_{EXP,LOG,RCP,RCP_IFLAG,RSQ,SQRT,SIN,COS}_F32`,
`V_{EXP,LOG}_LEGACY_F32`,
`V_FREXP_{EXP_I32,MANT}_F32`,
`V_{ADD,SUB,MUL,MIN,MAX}_F16`, `V_{ADD,MUL,MIN,MAX}_F64`,
`V_{RCP,RSQ,SQRT}_F64`, `V_FREXP_{EXP_I32,MANT}_F64`,
`V_CMP_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F16`,
`V_CMPX_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F16`,
`V_CMP_CLASS_F16`, `V_CMPX_CLASS_F16`,
`DS_NOP`, `DS_WRITE_{B8,B16,B32}`, `DS_READ_B32`,
`DS_{ADD,SUB,RSUB,INC,DEC,MIN,MAX}_{U32}`,
`DS_{MIN,MAX}_I32`, `DS_{AND,OR,XOR,MSKOR}_B32`,
`DS_CMPST_{B32,F32}`,
`DS_{ADD,MIN,MAX}_F32`,
`DS_WRITE2_{,ST64}_B32`, `DS_READ2_{,ST64}_B32`,
`DS_{ADD,SUB,RSUB,INC,DEC}_{U64}`,
`DS_{MIN,MAX}_{I64,U64,F64}`, `DS_{AND,OR,XOR,MSKOR}_B64`,
`DS_CMPST_{B64,F64}`, `DS_ADD_F64`,
`DS_WRITE_B64`, `DS_WRITE2_{,ST64}_B64`,
`DS_READ_B64`, `DS_READ2_{,ST64}_B64`,
`DS_WRITE_{B96,B128}`, `DS_READ_{B96,B128}`,
`DS_READ_{I,U}{8,16}`,
`DS_WRITE_B{8,16}_D16_HI`,
`DS_READ_{U8,I8}_D16{,_HI}`, `DS_READ_U16_D16{,_HI}`,
`DS_{ADD,SUB,RSUB,INC,DEC}_RTN_U32`,
`DS_{MIN,MAX}_RTN_{I32,U32,F32}`,
`DS_{AND,OR,XOR,MSKOR}_RTN_B32`, `DS_WRXCHG_RTN_B32`,
`DS_CMPST_RTN_{B32,F32}`, `DS_WRAP_RTN_B32`, `DS_ADD_RTN_F32`,
`DS_{ADD,SUB,RSUB,INC,DEC}_RTN_U64`,
`DS_{MIN,MAX}_RTN_{I64,U64,F64}`,
`DS_{AND,OR,XOR,MSKOR}_RTN_B64`, `DS_WRXCHG_RTN_B64`,
`DS_CMPST_RTN_{B64,F64}`, `DS_ADD_RTN_F64`,
`FLAT_LOAD_DWORD`,
`FLAT_LOAD/STORE_{U,S}{BYTE,SHORT}`, `FLAT_LOAD/STORE_DWORDX{2,3,4}`,
`FLAT_STORE_DWORD`, `GLOBAL_LOAD/STORE_{U,S}{BYTE,SHORT}`,
`GLOBAL_LOAD/STORE_DWORD`, `GLOBAL_LOAD/STORE_DWORDX{2,3,4}`, the full
`GLOBAL_ATOMIC_*` family for 32-bit integer, packed F16/BF16, F32/F64, and
integer `_X2` operations, plus `S_ENDPGM`.

The next execution layer is now present for raw instruction words as well:
Mirage can decode default `SOP1`, `SOP2`, `SOPC`, `SOPK`, `SOPP`, `VOP1`,
`VOP2`, native `ENC_VOPC` forms for `V_CMP_{EQ,NE,LT,LE,GT,GE}_{I32,U32}` and
`V_CMPX_{F,LT,EQ,LE,GT,NE,GE,T}_U32`, plus native `ENC_VOPC` forms for
`V_CMP_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F32`,
`V_CMPX_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F32`,
`V_CMP_CLASS_F32`, `V_CMPX_CLASS_F32`,
`V_CMP_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F64`,
`V_CMPX_{F,LT,EQ,LE,GT,LG,GE,O,U,NGE,NLG,NGT,NLE,NEQ,NLT,TRU}_F64`,
`V_CMP_CLASS_F64`, `V_CMPX_CLASS_F64`,
narrow `ENC_VOP3` forms for `V_MUL_LO_U32`, `V_MUL_HI_{U32,I32}`,
`V_{ADD,MUL}_F64`, `V_ADD3_U32`, `V_FMA_{F32,F64}`,
`V_CMP_{EQ,NE,LT,LE,GT,GE}_{I32,U32}`,
`V_CMPX_{F,LT,EQ,LE,GT,NE,GE,T}_{I32,U32,I64,U64}`, `V_CNDMASK_B32`,
`V_READLANE_B32`, and `V_WRITELANE_B32`, plus `VOP1` forms for `V_NOP`,
`V_CVT_F16_{U16,I16}`, `V_CVT_{U16,I16}_F16`,
`V_CVT_F32_UBYTE{0,1,2,3}`,
`V_CVT_F32_{I32,U32,F16,F64}`, `V_CVT_F64_{I32,U32}`,
`V_CVT_{I32,U32,F16,F64}_F32`, and `V_CVT_{I32,U32}_F64`, `SMEM`,
`ENC_FLAT`, and `ENC_FLAT_GLBL` forms for that same subset,
including inline integer constants, single-dword literals, signed relative
branch offsets, normalized `SOPK` immediate forms, return and no-return global
atomic decode via `SC0`, and the first 64-bit instruction decode paths for
scalar and vector memory traffic, plus DS return-atomic decode for
`DS_{ADD,SUB,RSUB,INC,DEC}_RTN_U32`,
`DS_{MIN,MAX}_RTN_{I32,U32,F32}`,
`DS_{AND,OR,XOR,MSKOR}_RTN_B32`, `DS_WRXCHG_RTN_B32`,
`DS_CMPST_RTN_{B32,F32}`, `DS_WRAP_RTN_B32`, `DS_ADD_RTN_F32`,
plus DS pair/narrow/wide access decode and non-return wide LDS update decode for
`DS_WRITE2_{,ST64}_{B32,B64}`, `DS_READ_B64`,
`DS_READ2_{,ST64}_{B32,B64}`, `DS_READ_{I,U}{8,16}`,
`DS_{ADD,SUB,RSUB,INC,DEC}_{U64}`,
`DS_{MIN,MAX}_{I64,U64,F64}`, `DS_{AND,OR,XOR,MSKOR}_B64`,
`DS_CMPST_{B64,F64}`, `DS_ADD_F64`,
plus wide DS return-atomic decode for
`DS_{ADD,SUB,RSUB,INC,DEC}_RTN_U64`,
`DS_{MIN,MAX}_RTN_{I64,U64,F64}`,
`DS_{AND,OR,XOR,MSKOR}_RTN_B64`, `DS_WRXCHG_RTN_B64`,
`DS_CMPST_RTN_{B64,F64}`, `DS_ADD_RTN_F64`,
`DS_MSKOR_B32`, `DS_CMPST_{B32,F32}`, and special scalar sources
`SRC_VCCZ`,
`SRC_EXECZ`, and `SRC_SCC`.

The scalar execution model also treats `EXEC` as the architectural pair
`s[126:127]`, so ordinary `S_*_B64` instructions that read or write that pair
update the live execution mask used by vector issue and `S_CBRANCH_EXEC{Z,NZ}`.

The native runtime test reports timing lines for both the single-wave
global-memory mix and the multi-wave workgroup barrier path. The direct
interpreter benchmark now compiles its program once before the timed loop, and
the simulator-backed path reports decode-cache hit and miss counts for compiled
program reuse. The latest metrics are written to
`build/native/mirage_gfx950_runtime_metrics.txt`. The test is informational only
and does not enforce a machine-specific performance threshold.
