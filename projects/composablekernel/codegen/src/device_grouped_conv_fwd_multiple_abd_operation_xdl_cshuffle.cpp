// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp"
#include <iostream>
#include "ck/host/stringutils.hpp"
#include "ck/host/types.hpp"
#include "ck/host/utils.hpp"
#include <cassert>

namespace ck {
namespace host {
namespace conv {

// NOTE: in CK, MNKPadding is always used for forward convolution, so didn't
// add GemmSpec function here

// function to update prologue/epilogue with user provided operation
void Operation_Conv_Fwd_Xdl_Cshuffle::update_prologue(const std::string& pro)
{
    if(!pro.empty())
    {
        this->prologue    = pro;
        this->cde_elem_op = "CDEElementOp";
    }
    else
    {
        this->prologue = "";
    }
}

void Operation_Conv_Fwd_Xdl_Cshuffle::update_epilogue(const std::string& epi)
{
    if(!epi.empty())
    {
        this->epilogue    = epi;
        this->cde_elem_op = "CDEElementOp";
    }
    else
    {
        this->epilogue = "";
    }
}

// Hard-code tuning parameters in modularized fashion, string them together into a vector of
// instances
std::vector<Operation_Conv_Fwd_Xdl_Cshuffle> Operation_Conv_Fwd_Xdl_Cshuffle::CreateOperations(
    const Problem_Conv_Fwd& prob, const std::string& prologue, const std::string& epilogue)
{
    std::vector<Operation_Conv_Fwd_Xdl_Cshuffle> result;

    std::vector<operation::TileDesc> tile_descriptions = {
        // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| NumGemmK|
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| Prefetch|
//       |      |      |      |    |    |     |     | Wave| Wave|    Stage|
//       |      |      |      |    |    |     |     |     |     |         |
  {   64,     64,   32,     32,   8,   8,   16,   16,    4,    2,        1},
  {   256,   128,   256,    32,   8,   8,   16,   16,    8,    4,        1},
  {   256,   128,   128,    32,   8,   8,   16,   16,    4,    4,        1},
  {   64,     64,   64,     32,   8,   8,   16,   16,    4,    4,        1},
  {   256,   256,   128,    32,   8,   8,   16,   16,    8,    4,        1},
  {   128,   128,   128,    32,   8,   8,   16,   16,    8,    4,        1}
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> a_block_descriptions = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1},
  {    S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1}
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> b_block_descriptions = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |
//                |               |               |              |               |               |          |
  {    S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1},
  {    S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              8,              8,         1}
        // clang-format on
    };

    std::vector<operation::CShuffleDesc> cshuffle_descriptions = {
        // clang-format off
//    CShuffle|    CShuffle|
// MXdlPerWave| NXdlPerWave|
//  PerShuffle|  PerShuffle|
//            |            |
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1}
        // clang-format on
    };

    std::vector<operation::CBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {              S<1, 16, 1, 4>,               1},
  {              S<1, 16, 1, 16>,              4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 16, 1, 4>,               1},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 16, 1, 8>,               4}
        // clang-format on
    };

    assert(tile_descriptions.size() == a_block_descriptions.size());
    assert(tile_descriptions.size() == b_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == c_block_descriptions.size());

    // Put all values together into a single operation > store into the result vector
    for(std::size_t i = 0; i < tile_descriptions.size(); i++)
    {
        Operation_Conv_Fwd_Xdl_Cshuffle x;
        x.NumDim           = prob.NumDim;
        x.tile_desc        = tile_descriptions[i];
        x.a_block_transfer = a_block_descriptions[i];
        x.b_block_transfer = b_block_descriptions[i];
        x.cshuffle         = cshuffle_descriptions[i];
        x.c_block_transfer = c_block_descriptions[i];
        x.A                = TensorDesc{prob.ADataType, prob.ALayout};
        x.B                = TensorDesc{prob.BDataType, prob.BLayout};
        x.E                = TensorDesc{prob.EDataType, prob.ELayout};
        x.Ds               = Transform(
            prob.DsLayout, prob.DsDataType, [](auto lo, auto dt) { return TensorDesc{dt, lo}; });
        x.a_elem_op   = prob.AElementOp;
        x.b_elem_op   = prob.BElementOp;
        x.cde_elem_op = prob.CDEElementOp;
        x.update_prologue(prologue);
        x.update_epilogue(epilogue);
        result.push_back(x);
    }
    return result;
}

// set up instances when not provided with a problem specification, use default operation values
std::vector<Operation_Conv_Fwd_Xdl_Cshuffle>
Operation_Conv_Fwd_Xdl_Cshuffle::CreateOperations(const std::string& prologue,
                                                  const std::string& epilogue)
{
    Problem_Conv_Fwd prob;
    return CreateOperations(prob, prologue, epilogue);
}

static const char* const CopyDevice_ConvTemplate =
    R"(
${Prologue}
${Epilogue}

using CDEElementOp = Epilogue;
using DeviceConv = ck::tensor_operation::device::CodegenDeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<${NumDim}, ${LayoutA}, ${LayoutB}, ${LayoutDs}, ${LayoutE}, ${ADataType}, ${BDataType}, ${AccDataType}, ${CShuffleDataType}, ${DsDataType}, ${EDataType}, ${AElementwiseOperation}, ${BElementwiseOperation}, ${CDEElementwiseOperation}, ${ConvSpecialization}, ${GemmSpecialization}, ${NumGemmkPrefetchStage}, ${BlockSize}, ${MPerBlock}, ${NPerBlock}, ${KPerBlock}, ${AK1}, ${BK1}, ${MPerXDL}, ${NPerXDL}, ${MXdlPerWave}, ${NXdlPerWave}, ${ABlockTransferThreadClusterLengths_AK0_M_AK1}, ${ABlockTransferThreadClusterArrangeOrder}, ${ABlockTransferSrcAccessOrder}, ${ABlockTransferSrcVectorDim}, ${ABlockTransferSrcScalarPerVector}, ${ABlockTransferDstScalarPerVector_AK1}, ${ABlockLdsExtraM}, ${BBlockTransferThreadClusterLengths_BK0_N_BK1}, ${BBlockTransferThreadClusterArrangeOrder}, ${BBlockTransferSrcAccessOrder}, ${BBlockTransferSrcVectorDim}, ${BBlockTransferSrcScalarPerVector}, ${BBlockTransferDstScalarPerVector_BK1}, ${BBlockLdsExtraN}, ${CShuffleMXdlPerWavePerShuffle}, ${CShuffleNXdlPerWavePerShuffle}, ${CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock}, ${CDEBlockTransferScalarPerVector_NPerBlock}>;

constexpr ck::index_t NumATensor = ck::tensor_operation::device::GetNumABTensors<false, ${ADataType}>();
constexpr ck::index_t NumBTensor = ck::tensor_operation::device::GetNumABTensors<false, ${BDataType}>();

extern "C" __global__ void run_${name}(
    const ${ADataType}* in_dev,
    const ${BDataType}* wei_dev,
    ${EDataType}* __restrict__ out_dev,
    ck::Array<ck::index_t, ${NumDim} + 3> in_lengths,
    ck::Array<ck::index_t, ${NumDim} + 3> in_strides,
    ck::Array<ck::index_t, ${NumDim} + 3> wei_lengths,
    ck::Array<ck::index_t, ${NumDim} + 3> wei_strides,
    ck::Array<ck::index_t, ${NumDim} + 3> out_lengths,
    ck::Array<ck::index_t, ${NumDim} + 3> out_strides,
    ck::Array<ck::index_t, ${NumDim}> conv_filter_strides,
    ck::Array<ck::index_t, ${NumDim}> conv_filter_dilations,
    ck::Array<ck::index_t, ${NumDim}> input_left_pads,
    ck::Array<ck::index_t, ${NumDim}> input_right_pads,
    const ${AElementwiseOperation} a_element_op,
    const ${BElementwiseOperation} b_element_op,
    const ${CDEElementwiseOperation} cde_element_op
){
    

    auto arg = DeviceConv::Argument(in_dev,
                                    wei_dev,
                                    ck::Array<const void*, 0>{},
                                    out_dev,
                                    in_lengths,
                                    in_strides,
                                    wei_lengths,
                                    wei_strides,
                                    ck::Array<ck::Array<ck::index_t, ${NumDim} + 3>, 0>{},
                                    ck::Array<ck::Array<ck::index_t, ${NumDim} + 3>, 0>{},
                                    out_lengths,
                                    out_strides,
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads,
                                    ${AElementwiseOperation}{},
                                    ${BElementwiseOperation}{},
                                    ${CDEElementwiseOperation}{1.0f, 1.0f});

    if(!DeviceConv::IsSupportedArgument(arg))
    {
	printf("Arguement is not supported.\n");
	return;
    };    

    constexpr ck::LoopScheduler LoopSched = ck::make_default_loop_scheduler();

    // GridwiseGemm
    using GridwiseGemm = ck::conditional_t<ck::get_warp_size() == 64,
                                           typename DeviceConv::GridwiseGemm64,
                                           typename DeviceConv::GridwiseGemm32>;
    static constexpr auto I0 = ck::Number<0>{};

    ck::tensor_operation::device::device_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                    GridwiseGemm,
                    const ${ADataType}*,
                    const ${BDataType}*,
                    typename GridwiseGemm::DsGridPointer,
                    ${EDataType},
                    ${AElementwiseOperation},
                    ${BElementwiseOperation},
                    ${CDEElementwiseOperation},
                    DeviceConv::AGridDesc_AK0_M_AK1,
                    DeviceConv::BGridDesc_BK0_N_BK1,
                    DeviceConv::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceConv::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceConv::Block2ETileMap,
		    ck::tensor_operation::device::ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, 0>,
                    ck::integral_constant<bool, true>{},
                    false,
                    false>
		    (
		     arg.p_as_grid_.At(I0),
 		     arg.p_bs_grid_.At(I0),
		     arg.p_ds_grid_,
		     arg.p_e_grid_,
		     arg.a_element_op_,
		     arg.b_element_op_,
		     arg.cde_element_op_,
		     arg.a_g_n_c_wis_lengths_[0], // Group count
                     arg.a_grid_desc_ak0_m_ak1_,
		     arg.b_grid_desc_bk0_n_bk1_,
		     arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
		     arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
		     arg.block_2_etile_map_,
		     arg.compute_ptr_offset_of_batch_
		    );
				    
}
)";

// use hardcoded instances from vector of operations to substitute values into instance template
Solution Operation_Conv_Fwd_Xdl_Cshuffle::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"name",
         std::to_string(this->tile_desc.block_size) + "_" +
             std::to_string(this->tile_desc.m_per_block) + "_" +
             std::to_string(this->tile_desc.n_per_block) + "_" +
             std::to_string(this->tile_desc.k_per_block) + "_" +
             std::to_string(this->tile_desc.ak1) + "_" + std::to_string(this->tile_desc.bk1) + "_" +
             std::to_string(this->tile_desc.m_per_XDL) + "_" +
             std::to_string(this->tile_desc.n_per_XDL) + "_" +
             std::to_string(this->tile_desc.m_Xdl_per_wave) + "_" +
             std::to_string(this->tile_desc.n_Xdl_per_wave)},
        {"NumDim", std::to_string(this->NumDim)},
        {"LayoutA", ToString(this->A.layout)},
        {"LayoutB", ToString(this->B.layout)},
        {"LayoutDs",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.layout); }))},
        {"LayoutE", ToString(this->E.layout)},
        {"ADataType", ToString(this->A.element)},
        {"BDataType", ToString(this->B.element)},
        {"AccDataType", ToString(this->acc)},
        {"ComputeDataType", ToString(this->A.element)},
        {"CShuffleDataType", ToString(this->cs_type)},
        {"DsDataType",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.element); }))},
        {"EDataType", ToString(this->E.element)},
        {"AElementwiseOperation", this->a_elem_op},
        {"BElementwiseOperation", this->b_elem_op},
        {"CDEElementwiseOperation", this->cde_elem_op},
        {"Prologue", this->prologue},
        {"Epilogue", this->epilogue},
        {"ConvSpecialization", this->conv_specialization},
        {"GemmSpecialization", this->gemm_specialization},
        {"NumGemmkPrefetchStage", std::to_string(this->tile_desc.num_gemmk_prefetch_stage)},
        {"BlockSize", std::to_string(this->tile_desc.block_size)},
        {"MPerBlock", std::to_string(this->tile_desc.m_per_block)},
        {"NPerBlock", std::to_string(this->tile_desc.n_per_block)},
        {"KPerBlock", std::to_string(this->tile_desc.k_per_block)},
        {"AK1", std::to_string(this->tile_desc.ak1)},
        {"BK1", std::to_string(this->tile_desc.bk1)},
        {"MPerXDL", std::to_string(this->tile_desc.m_per_XDL)},
        {"NPerXDL", std::to_string(this->tile_desc.n_per_XDL)},
        {"MXdlPerWave", std::to_string(this->tile_desc.m_Xdl_per_wave)},
        {"NXdlPerWave", std::to_string(this->tile_desc.n_Xdl_per_wave)},
        {"ABlockTransferThreadClusterLengths_AK0_M_AK1",
         this->a_block_transfer.thread_cluster_length},
        {"ABlockTransferThreadClusterArrangeOrder",
         this->a_block_transfer.thread_cluster_arrange_order},
        {"ABlockTransferSrcAccessOrder", this->a_block_transfer.src_access_order},
        {"ABlockTransferSrcVectorDim", std::to_string(this->a_block_transfer.src_vec_dim)},
        {"ABlockTransferSrcScalarPerVector",
         std::to_string(this->a_block_transfer.src_scalar_per_vector)},
        {"ABlockTransferDstScalarPerVector_AK1",
         std::to_string(this->a_block_transfer.dst_scalar_per_vector_k1)},
        {"ABlockLdsExtraM", std::to_string(this->a_block_transfer.lds_add_extra_dim)},
        {"BBlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b_block_transfer.thread_cluster_length},
        {"BBlockTransferThreadClusterArrangeOrder",
         this->b_block_transfer.thread_cluster_arrange_order},
        {"BBlockTransferSrcAccessOrder", this->b_block_transfer.src_access_order},
        {"BBlockTransferSrcVectorDim", std::to_string(this->b_block_transfer.src_vec_dim)},
        {"BBlockTransferSrcScalarPerVector",
         std::to_string(this->b_block_transfer.src_scalar_per_vector)},
        {"BBlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b_block_transfer.dst_scalar_per_vector_k1)},
        {"BBlockLdsExtraN", std::to_string(this->b_block_transfer.lds_add_extra_dim)},
        {"CShuffleMXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.m_Xdl_per_wave_per_shuffle)},
        {"CShuffleNXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.n_Xdl_per_wave_per_shuffle)},
        {"CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock",
         this->c_block_transfer.cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl},
        {"CDEBlockTransferScalarPerVector_NPerBlock",
         std::to_string(this->c_block_transfer.scalar_per_vector_n_wave_n_per_Xdl)},
    };

    return Solution{InterpolateString(CopyDevice_ConvTemplate, values), std::move(values)};
}

} // namespace conv
} // namespace host
} // namespace ck
