#include "lib/sim/isa/gfx1201/architecture_profile.h"

#include <array>

namespace mirage::sim::isa {
namespace {

// Keep these inventory-derived summaries in sync with
// tools/architecture_import/gfx1201/generate_gfx1201_bringup.py.
constexpr InstructionCatalogMetadata kImportedMetadata{
    "gfx1201",
    "AMD RDNA 4",
    "2025-02-19",
    "1.1.0",
    "amdgpu_isa_rdna4.xml",
    1264u,
    5062u,
};

constexpr std::array<Gfx1201SupportBucketSummary, 5> kSupportBuckets{{
    {Gfx1201SupportBucket::kTransferableFull,
     "transferable_full",
     363u,
     "Present in gfx950 with both raw decode coverage and interpreter "
     "semantics already implemented."},
    {Gfx1201SupportBucket::kTransferableDecodeOnly,
     "transferable_decode_only",
     36u,
     "Binary shape already appears in gfx950 coverage, but execution "
     "semantics are still missing there."},
    {Gfx1201SupportBucket::kTransferableSemanticOnly,
     "transferable_semantic_only",
     30u,
     "Execution semantics exist in gfx950 coverage, but binary decode "
     "coverage has not been wired up yet."},
    {Gfx1201SupportBucket::kKnownButUnsupported,
     "known_but_unsupported",
     167u,
     "Opcode name already exists in the gfx950 catalog, but neither decoder "
     "nor interpreter support is complete."},
    {Gfx1201SupportBucket::kNewVsGfx950,
     "new_vs_gfx950",
     668u,
     "Opcode name is absent from gfx950 and needs RDNA4-local decode and "
     "semantic work."},
}};

constexpr std::array<Gfx1201EncodingFocus, 12> kPhase0DecoderFocus{{
    {"ENC_SOPP", 41u, "S_ENDPGM",
     "Program control, wait/event sequencing, and branch bring-up."},
    {"ENC_SOP1", 84u, "S_MOV_B32",
     "Scalar move, bit-manipulation, and exec-mask setup precedent."},
    {"ENC_SOP2", 72u, "S_AND_B32",
     "Scalar binary core needed before wider control-flow work."},
    {"ENC_SOPC", 46u, "S_CMP_EQ_U32",
     "Scalar compare path for branch and predicate plumbing."},
    {"ENC_SOPK", 8u, "S_MOVK_I32",
     "Small scalar-immediate surface used by early control kernels."},
    {"ENC_SMEM", 28u, "S_LOAD_B32",
     "RDNA4 scalar memory is a first architecture-local blocker."},
    {"ENC_VOP1", 90u, "V_MOV_B32",
     "Vector move and conversion baseline reused across many programs."},
    {"ENC_VOP2", 45u, "V_ADD_F32",
     "Core vector arithmetic used by both compute and graphics paths."},
    {"ENC_VOPC", 162u, "V_CMP_EQ_F32",
     "Vector compare path for control, masking, and shader predicates."},
    {"ENC_VOP3", 434u, "V_ADD3_U32",
     "Largest instruction family and the main overlap with gfx950 "
     "precedent."},
    {"ENC_VDS", 123u, "DS_ADD_U32",
     "LDS data path with a small fully-supported carry-over subset."},
    {"ENC_VGLOBAL", 65u, "GLOBAL_LOAD_B32",
     "Global memory load/store family that replaces gfx950-specific naming."},
}};

constexpr std::array<Gfx1201EncodingFocus, 8> kPhase1DecoderFocus{{
    {"ENC_VBUFFER", 89u, "BUFFER_LOAD_FORMAT_X",
     "RDNA4 buffer resource path and typed buffer forms."},
    {"ENC_VFLAT", 55u, "FLAT_LOAD_B32",
     "Flat memory path layered after scalar/global addressing is stable."},
    {"ENC_VSCRATCH", 24u, "SCRATCH_LOAD_B32",
     "Scratch memory path is target-local and can come after flat/global."},
    {"ENC_VSAMPLE", 58u, "IMAGE_SAMPLE",
     "Graphics sampling path unique to the RDNA4 import."},
    {"ENC_VIMAGE", 33u, "IMAGE_LOAD",
     "Image load/store/atomic path separate from sampled operations."},
    {"ENC_VEXPORT", 1u, "EXPORT",
     "Graphics export path with no gfx950 precedent."},
    {"ENC_VINTERP", 6u, "V_INTERP_P10_F32",
     "Shader interpolation path that anchors graphics-local vector work."},
    {"ENC_VOP3P", 56u, "V_PK_ADD_F16",
     "Packed/vector math delta visible in the RDNA4 import."},
}};

constexpr std::array<Gfx1201FamilyFocus, 7> kCarryOverFamilyFocus{{
    {"v", Gfx1201SupportBucket::kTransferableFull, 235u, "V_MOV_B32",
     "Largest fully-supported carry-over bucket from gfx950."},
    {"s", Gfx1201SupportBucket::kTransferableFull, 98u, "S_MOV_B32",
     "Scalar control and ALU subset with direct gfx950 precedent."},
    {"ds", Gfx1201SupportBucket::kTransferableFull, 27u, "DS_ADD_U32",
     "Small LDS subset already proven end-to-end on gfx950."},
    {"global", Gfx1201SupportBucket::kTransferableFull, 3u,
     "GLOBAL_ATOMIC_ADD_F32",
     "Only a narrow global atomic subset currently has full precedent."},
    {"v", Gfx1201SupportBucket::kTransferableDecodeOnly, 26u,
     "V_CVT_F32_FP8",
     "Binary decode precedent exists, but execution semantics still need "
     "work."},
    {"v", Gfx1201SupportBucket::kTransferableSemanticOnly, 30u,
     "V_CMP_LT_F16",
     "Interpreter precedent exists, but raw binary decode work is missing."},
    {"s", Gfx1201SupportBucket::kTransferableDecodeOnly, 10u, "S_GETPC_B64",
     "Scalar control opcodes appear in coverage but are not executable yet."},
}};

constexpr std::array<Gfx1201FamilyFocus, 10> kRdna4DeltaFamilyFocus{{
    {"s", Gfx1201SupportBucket::kNewVsGfx950, 153u, "S_LOAD_B32",
     "Scalar memory and wait/barrier control differ from gfx950."},
    {"v", Gfx1201SupportBucket::kNewVsGfx950, 155u, "V_INTERP_P10_F32",
     "Graphics-local vector forms expand the RDNA4 surface."},
    {"image", Gfx1201SupportBucket::kNewVsGfx950, 91u, "IMAGE_LOAD",
     "Image pipeline has no direct gfx950 baseline."},
    {"buffer", Gfx1201SupportBucket::kNewVsGfx950, 62u,
     "BUFFER_LOAD_FORMAT_X",
     "Buffer load/store/atomic naming and resource handling are RDNA4-local."},
    {"global", Gfx1201SupportBucket::kNewVsGfx950, 62u, "GLOBAL_LOAD_B32",
     "Global memory family uses the RDNA4 ISA surface instead of gfx950 "
     "forms."},
    {"ds", Gfx1201SupportBucket::kNewVsGfx950, 60u, "DS_LOAD_B32",
     "LDS load/store and atomic expansion beyond the gfx950 subset."},
    {"flat", Gfx1201SupportBucket::kNewVsGfx950, 52u, "FLAT_LOAD_B32",
     "Flat addressing path needs RDNA4-specific decode and operand wiring."},
    {"scratch", Gfx1201SupportBucket::kNewVsGfx950, 24u, "SCRATCH_LOAD_B32",
     "Scratch memory is absent from the current gfx950-local layout."},
    {"tbuffer", Gfx1201SupportBucket::kNewVsGfx950, 8u,
     "TBUFFER_LOAD_FORMAT_X",
     "Typed buffer graphics path should layer on top of buffer support."},
    {"export", Gfx1201SupportBucket::kNewVsGfx950, 1u, "EXPORT",
     "Graphics export path is unique to the RDNA4 import."},
}};


}  // namespace

const InstructionCatalogMetadata& GetGfx1201ImportedInstructionMetadata() {
  return kImportedMetadata;
}

std::span<const Gfx1201SupportBucketSummary> GetGfx1201SupportBucketSummaries() {
  return kSupportBuckets;
}

std::span<const Gfx1201EncodingFocus> GetGfx1201Phase0DecoderFocus() {
  return kPhase0DecoderFocus;
}

std::span<const Gfx1201EncodingFocus> GetGfx1201Phase1DecoderFocus() {
  return kPhase1DecoderFocus;
}

std::span<const Gfx1201FamilyFocus> GetGfx1201CarryOverFamilyFocus() {
  return kCarryOverFamilyFocus;
}

std::span<const Gfx1201FamilyFocus> GetGfx1201Rdna4DeltaFamilyFocus() {
  return kRdna4DeltaFamilyFocus;
}

std::string_view ToString(Gfx1201SupportBucket bucket) {
  switch (bucket) {
    case Gfx1201SupportBucket::kTransferableFull:
      return "transferable_full";
    case Gfx1201SupportBucket::kTransferableDecodeOnly:
      return "transferable_decode_only";
    case Gfx1201SupportBucket::kTransferableSemanticOnly:
      return "transferable_semantic_only";
    case Gfx1201SupportBucket::kKnownButUnsupported:
      return "known_but_unsupported";
    case Gfx1201SupportBucket::kNewVsGfx950:
      return "new_vs_gfx950";
  }
  return "unknown";
}

std::string_view DescribeGfx1201BringupPhase() {
  return "Architecture-local scaffold only; phase-0 is scalar/vector core "
         "decode plus LDS/global memory, with graphics RDNA4 deltas queued "
         "behind that baseline.";
}

}  // namespace mirage::sim::isa
