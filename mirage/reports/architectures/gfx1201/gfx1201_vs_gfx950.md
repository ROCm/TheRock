# gfx1201 Ingest Report

- Official ISA source: `amdgpu_isa_rdna4.xml` from `AMD_GPU_MR_ISA_XML_2025_09_05.zip`
- Official architecture name: `AMD RDNA 4`
- LLVM processor model hits for `gfx1201`: 6
- RDNA4 / gfx1201 instruction inventory: 1264
- Current gfx950 instruction inventory: 1242
- Common instructions: 596
- gfx1201-only instructions: 668
- gfx950-only instructions: 646

## Major Family Deltas

### gfx1201-only top families

- `s`: 153
- `v`: 129
- `image`: 91
- `buffer`: 62
- `global`: 62
- `lds`: 60
- `flat`: 52
- `scratch`: 24
- `wmma`: 22
- `tbuffer`: 8

### gfx950-only top families

- `v`: 235
- `s`: 135
- `buffer`: 63
- `lds`: 61
- `global`: 58
- `flat`: 51
- `scratch`: 27
- `packed`: 8
- `tbuffer`: 8

## Encoding Family Deltas

- gfx1201-only encoding families: ENC_VBUFFER, ENC_VDS, ENC_VDSDIR, ENC_VEXPORT, ENC_VFLAT, ENC_VGLOBAL, ENC_VIMAGE, ENC_VINTERP, ENC_VSAMPLE, ENC_VSCRATCH, VOP1_VOP_DPP16, VOP1_VOP_DPP8, VOP2_VOP_DPP16, VOP2_VOP_DPP8, VOP3P_INST_LITERAL, VOP3P_VOP_DPP16, VOP3P_VOP_DPP8, VOP3_INST_LITERAL, VOP3_SDST_ENC_INST_LITERAL, VOP3_SDST_ENC_VOP_DPP16
- gfx950-only encoding families: ENC_DS, ENC_FLAT, ENC_FLAT_GLBL, ENC_FLAT_SCRATCH, ENC_MTBUF, ENC_MUBUF, ENC_VOP3PX2, VOP1_VOP_DPP, VOP1_VOP_SDWA, VOP2_VOP_DPP, VOP2_VOP_SDWA, VOP2_VOP_SDWA_SDST_ENC, VOP3P_MFMA, VOPC_VOP_SDWA_SDST_ENC

## Simulator Implications

- gfx1201 is sourced from the architecture-level RDNA4 XML, so the simulator needs a target-to-architecture mapping layer instead of assuming one XML per gpu target.
- gfx1201 inherits the generic GFX12 feature set in local LLVM, unlike `gfx1250` which has a separate `FeatureISAVersion12_50` model. That suggests a shared RDNA4 baseline with target-specific deltas layered on later.
- Compared with the current gfx950/CDNA4 baseline, gfx1201 shifts the bring-up priority toward RDNA-style graphics, buffer/image, export, and packed/vector families rather than CDNA matrix and accelerator-heavy paths.
