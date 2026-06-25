// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_batched_gemm_softmax_gemm/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include <cassert>

namespace ck {
namespace host {
namespace device_batched_gemm_softmax_gemm {

// calculate appropriate Gemm Specification based on input tensor dimensions
std::string GetGemmSpec(const std::size_t m,
                        const std::size_t n,
                        const std::size_t k,
                        const std::size_t n1,
                        const std::size_t m_per_block,
                        const std::size_t n_per_block,
                        const std::size_t k_per_block,
                        const std::size_t n1_per_block)
{
    std::string spec = "";
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(integer_divide_ceil(n1, n1_per_block) * n1_per_block - n1 != 0)
        spec += "O";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

// function to update prologue/epilogue with user provided operation
void Operation_Xdl_CShuffle::update_prologue(const std::string& pro)
{
    if(!prologue.empty())
    {
        this->prologue = pro;
    }
    else
    {
        this->prologue = "";
    }
}

void Operation_Xdl_CShuffle::update_epilogue(const std::string& epi)
{
    if(!epilogue.empty())
    {
        this->epilogue = epi;
    }
    else
    {
        this->epilogue = "";
    }
}

// accounts for all possible combinations of Row/Col major
static Layout ToLayout(bool Trans) { return Trans ? Layout::Column : Layout::Row; }

// Hard-code tuning parameters in modularized fashion, string them together into a vector of
// instances
std::vector<Operation_Xdl_CShuffle> Operation_Xdl_CShuffle::CreateOperations(
    const Problem& prob, const std::string& prologue, const std::string& epilogue)
{
    std::vector<Operation_Xdl_CShuffle> result;

    std::vector<operation::TileDescGemmGemm> tile_descriptions = {
        // clang-format off
//  Block| Gemm01| Gemm0| Gemm0| Gemm1| Gemm1| AK1| BK1| B1K1| MPer| NPer| Gemm0| Gemm0| Gemm1| NumGemmK|
//   Size|   MPer|  NPer|  KPer|  NPer|  KPer|    |    |     |  XDL|  XDL|  MXdl|  NXdl|  NXdl| Prefetch|
//       |  Block| Block| Block| Block| Block|    |    |     |     |     |   Per|   Per|   Per|    Stage|
//       |       |      |      |      |      |    |    |     |     |     |  Wave|  Wave|  Wave|         |
  {   256,    256,   128,    32,    64,    32,   8,   8,    2,   16,   16,     4,     8,     4,        1},
  {   256,    256,   128,    32,   128,    32,   8,   8,    2,   16,   16,     4,     8,     8,        1},
  {   256,    128,   256,    32,    64,    32,   8,   8,    2,   16,   16,     2,    16,     4,        1},
  {   256,    128,   256,    32,   128,    32,   8,   8,    2,   16,   16,     2,    16,     8,        1},
  {   256,    128,   128,    64,    64,    32,   8,   8,    2,   16,   16,     2,     8,     4,        1},
  {   256,    128,   128,    32,    64,    32,   8,   8,    2,   16,   16,     2,     8,     4,        1},
  {   256,    128,   128,    64,   128,    32,   8,   8,    2,   16,   16,     2,     8,     8,        1},
  {   256,    128,   128,    32,   128,    32,   8,   8,    2,   16,   16,     2,     8,     8,        1},
  {   256,    128,   256,    32,   128,    32,   8,   8,    2,   16,   16,     2,    16,     8,        1},
  {   256,    128,   256,    32,    64,    32,   8,   8,    2,   16,   16,     2,    16,     4,        1},
  {   256,    128,   256,    64,   128,    32,   8,   8,    2,   16,   16,     2,    16,     8,        1},
  {   256,    128,   256,    64,    64,    32,   8,   8,    2,   16,   16,     2,    16,     4,        1},
// Padded fallback kernel  
  {   256,    128,   128,    64,   128,    32,   8,   8,    2,   16,   16,     2,     8,     8,        1},
  {   256,    128,    64,    32,   128,    32,   8,   8,    2,   16,   16,     2,     4,     8,        1},
// Irregular k
  {   256,    256,   128,    48,    64,    32,   4,   4,    2,   16,   16,     4,     8,     4,        1},
  {   256,    256,   128,    48,   128,    32,   4,   4,    2,   16,   16,     4,     8,     8,        1},
  {   256,    128,   256,    48,    64,    32,   4,   4,    2,   16,   16,     2,    16,     4,        1},
  {   256,    128,   256,    48,   128,    32,   4,   4,    2,   16,   16,     2,    16,     8,        1},
  {   256,    128,   128,    48,    64,    32,   4,   4,    2,   16,   16,     2,     8,     4,        1},
  {   256,    128,   128,    48,   128,    32,   4,   4,    2,   16,   16,     2,     8,     8,        1},
        // clang-format on
    };

    const std::vector<operation::BlockTransferDesc> a_block_descriptions = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
// Padded fallback kernel
  {    S<8, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,     false},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,      true},
// Irregular k
  {    S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false},
  {    S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false},
  {    S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false},
  {    S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false},
  {    S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false},
  {    S<2,128, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,     false},
        // clang-format on
    };

    const std::vector<operation::BlockTransferDesc> b1_block_descriptions = {
        // clang-format off
//  B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockTransfer| B1BlockLds|
//    ThreadCluster|   ThreadCluster|  SrcAccessOrder|    SrcVectorDim|       SrcScalar|       DstScalar|  AddExtraN|
//  Lengths_K0_N_K1|    ArrangeOrder|                |                |       PerVector|    PerVector_K1|           |
//                 |                |                |                |                |                |           |
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
// Padded fallback kernel
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
// Irregular k
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S<16, 16, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
   {   S< 8, 32, 1>,      S<0, 2, 1>,      S<0, 2, 1>,               1,               4,               2,      false},
        // clang-format on
    };

    std::vector<operation::CShuffleDesc> cshuffle_descriptions = {
        // clang-format off
//    CShuffle|    CShuffle|
// MXdlPerWave| NXdlPerWave|
//  PerShuffle|  PerShuffle|
//            |            |
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           8},
  {          1,           4},
  {          1,           8},
  {          1,           4},
// Padded fallback kernel
  {          1,           2},
  {          1,           2},
// Irregular k
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
  {          1,           2},
        // clang-format on
    };

    std::vector<operation::CBlockTransferDesc> c_block_descriptions = {
        // clang-format off
// CBlockTransferClusterLengths|  CBlockTransfer
//         _MBlock_MWaveMPerXdl| ScalarPerVector
//         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl
//                             |                
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 16, 1,16>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 16, 1,16>,               4},
  {              S<1, 32, 1, 8>,               4},
// Padded fallback kernel
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
// Irregular k
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
        // clang-format on
    };

    assert(tile_descriptions.size() == a_block_descriptions.size());
    assert(tile_descriptions.size() == b1_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == c_block_descriptions.size());

    // Put all values together into a single operation > store into the result vector
    for(std::size_t i = 0; i < tile_descriptions.size(); i++)
    {
        Operation_Xdl_CShuffle x;
        x.tile_desc           = tile_descriptions[i];
        x.a_block_transfer    = a_block_descriptions[i];
        x.b0_block_transfer   = a_block_descriptions[i]; // b0 same as a
        x.b1_block_transfer   = b1_block_descriptions[i];
        x.cshuffle            = cshuffle_descriptions[i];
        x.c_block_transfer    = c_block_descriptions[i];
        x.A                   = TensorDesc{prob.ADataType, ToLayout(prob.TransA)};
        x.B                   = TensorDesc{prob.BDataType, ToLayout(prob.TransB)};
        x.B1                  = TensorDesc{prob.B1DataType, ToLayout(prob.TransB1)};
        x.C                   = TensorDesc{prob.CDataType, ToLayout(prob.TransC)};
        x.a_elem_op           = prob.AElementOp;
        x.b_elem_op           = prob.BElementOp;
        x.b1_elem_op          = prob.B1ElementOp;
        x.c_elem_op           = prob.CElementOp;
        x.acc_elem_op         = prob.AccElementOp;
        x.gemm_specialization = GetGemmSpec(prob.M,
                                            prob.N,
                                            prob.K,
                                            prob.O,
                                            x.tile_desc.gemm01_m_per_block,
                                            x.tile_desc.gemm0_n_per_block,
                                            x.tile_desc.gemm0_k_per_block,
                                            x.tile_desc.gemm1_n_per_block);
        x.update_prologue(prologue);
        x.update_epilogue(epilogue);
        x.mask_out_upper_triangle = prob.MaskOutUpperTriangle;
        result.push_back(x);
    }
    return result;
}

// set up instances when not provided with a problem specification, use default operation values and
// all possible layout combinations
std::vector<std::vector<Operation_Xdl_CShuffle>>
Operation_Xdl_CShuffle::CreateOperations(const std::string& prologue, const std::string& epilogue)
{
    std::vector<Problem> problems;

    Problem prob;
    prob.TransA  = false;
    prob.TransB  = true;
    prob.TransB1 = false;
    prob.TransC  = false;
    problems.push_back(prob);

    prob.MaskOutUpperTriangle = true;
    problems.push_back(prob);

    return Transform(problems,
                     [&](const Problem& p) { return CreateOperations(p, prologue, epilogue); });
}

static const char* const DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffleTemplate =
    "ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle<${LayoutA}, "
    "${LayoutB0}, ${LayoutB1}, ${LayoutC}, ${ADataType}, ${B0DataType}, ${B1DataType}, "
    "${CDataType}, ${AccDataType}, ${CShuffleDataType}, ${AElementwiseOperation}, "
    "${B0ElementwiseOperation}, ${Acc0ElementwiseOperation}, ${B1ElementwiseOperation}, "
    "${CElementwiseOperation}, ${GemmSpecialization}, ${NumGemmkPrefetchStage}, ${BlockSize}, "
    "${Gemm01MPerBlock}, ${Gemm0NPerBlock}, ${Gemm0KPerBlock}, ${Gemm1NPerBlock}, "
    "${Gemm1KPerBlock}, ${AK1}, ${BK1}, ${B1K1}, ${MPerXDL}, ${NPerXDL}, ${Gemm0MXdlPerWave}, "
    "${Gemm0NXdlPerWave}, ${Gemm1NXdlPerWave}, ${ABlockTransferThreadClusterLengths_AK0_M_AK1}, "
    "${ABlockTransferThreadClusterArrangeOrder}, ${ABlockTransferSrcAccessOrder}, "
    "${ABlockTransferSrcVectorDim}, ${ABlockTransferSrcScalarPerVector}, "
    "${ABlockTransferDstScalarPerVector_AK1}, ${ABlockLdsExtraM}, "
    "${B0BlockTransferThreadClusterLengths_BK0_N_BK1}, "
    "${B0BlockTransferThreadClusterArrangeOrder}, ${B0BlockTransferSrcAccessOrder}, "
    "${B0BlockTransferSrcVectorDim}, ${B0BlockTransferSrcScalarPerVector}, "
    "${B0BlockTransferDstScalarPerVector_BK1}, ${B0BlockLdsExtraN}, "
    "${B1BlockTransferThreadClusterLengths_BK0_N_BK1}, "
    "${B1BlockTransferThreadClusterArrangeOrder}, ${B1BlockTransferSrcAccessOrder}, "
    "${B1BlockTransferSrcVectorDim}, ${B1BlockTransferSrcScalarPerVector}, "
    "${B1BlockTransferDstScalarPerVector_BK1}, ${B1BlockLdsExtraN}, "
    "${CShuffleMXdlPerWavePerShuffle}, ${CShuffleNXdlPerWavePerShuffle}, "
    "${CBlockTransferClusterLengths_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl}, "
    "${CBlockTransferScalarPerVector_NWaveNPerXdl}, ${MaskOutUpperTriangle}>";

// use hardcoded instances from vector of operations to substitute values into instance template
Solution Operation_Xdl_CShuffle::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"name",
         std::to_string(this->tile_desc.block_size) + "_" +
             std::to_string(this->tile_desc.gemm01_m_per_block) + "_" +
             std::to_string(this->tile_desc.gemm0_n_per_block) + "_" +
             std::to_string(this->tile_desc.gemm0_k_per_block) + "_" +
             std::to_string(this->tile_desc.gemm1_n_per_block) + "_" +
             std::to_string(this->tile_desc.gemm1_k_per_block) + "_" +
             std::to_string(this->tile_desc.ak1) + "_" + std::to_string(this->tile_desc.bk1) + "_" +
             std::to_string(this->tile_desc.b1k1) + "_" +
             std::to_string(this->tile_desc.m_per_XDL) + "_" +
             std::to_string(this->tile_desc.n_per_XDL) + "_" +
             std::to_string(this->tile_desc.gemm0_m_Xdl_per_wave) + "_" +
             std::to_string(this->tile_desc.gemm0_n_Xdl_per_wave) + "_" +
             std::to_string(this->tile_desc.gemm1_n_Xdl_per_wave)},
        {"LayoutA", ToString(this->A.layout)},
        {"LayoutB0", ToString(this->B.layout)},
        {"LayoutB1", ToString(this->B1.layout)},
        {"LayoutC", ToString(this->C.layout)},
        {"ADataType", ToString(this->A.element)},
        {"B0DataType", ToString(this->B.element)},
        {"B1DataType", ToString(this->B1.element)},
        {"CDataType", ToString(this->C.element)},
        {"AccDataType", ToString(this->acc)},
        {"CShuffleDataType", ToString(this->cs_type)},
        {"AElementwiseOperation", this->a_elem_op},
        {"B0ElementwiseOperation", this->b_elem_op},
        {"Acc0ElementwiseOperation", this->acc_elem_op},
        {"B1ElementwiseOperation", this->b1_elem_op},
        {"CElementwiseOperation", this->c_elem_op},
        {"GemmSpecialization", this->gemm_specialization},
        {"NumGemmkPrefetchStage", std::to_string(this->tile_desc.num_gemmk_prefetch_stage)},
        {"BlockSize", std::to_string(this->tile_desc.block_size)},
        {"Gemm01MPerBlock", std::to_string(this->tile_desc.gemm01_m_per_block)},
        {"Gemm0NPerBlock", std::to_string(this->tile_desc.gemm0_n_per_block)},
        {"Gemm0KPerBlock", std::to_string(this->tile_desc.gemm0_k_per_block)},
        {"Gemm1NPerBlock", std::to_string(this->tile_desc.gemm1_n_per_block)},
        {"Gemm1KPerBlock", std::to_string(this->tile_desc.gemm1_k_per_block)},
        {"AK1", std::to_string(this->tile_desc.ak1)},
        {"BK1", std::to_string(this->tile_desc.bk1)},
        {"B1K1", std::to_string(this->tile_desc.b1k1)},
        {"MPerXDL", std::to_string(this->tile_desc.m_per_XDL)},
        {"NPerXDL", std::to_string(this->tile_desc.n_per_XDL)},
        {"Gemm0MXdlPerWave", std::to_string(this->tile_desc.gemm0_m_Xdl_per_wave)},
        {"Gemm0NXdlPerWave", std::to_string(this->tile_desc.gemm0_n_Xdl_per_wave)},
        {"Gemm1NXdlPerWave", std::to_string(this->tile_desc.gemm1_n_Xdl_per_wave)},
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
        {"B0BlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b0_block_transfer.thread_cluster_length},
        {"B0BlockTransferThreadClusterArrangeOrder",
         this->b0_block_transfer.thread_cluster_arrange_order},
        {"B0BlockTransferSrcAccessOrder", this->b0_block_transfer.src_access_order},
        {"B0BlockTransferSrcVectorDim", std::to_string(this->b0_block_transfer.src_vec_dim)},
        {"B0BlockTransferSrcScalarPerVector",
         std::to_string(this->b0_block_transfer.src_scalar_per_vector)},
        {"B0BlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b0_block_transfer.dst_scalar_per_vector_k1)},
        {"B0BlockLdsExtraN", std::to_string(this->b0_block_transfer.lds_add_extra_dim)},
        {"B1BlockTransferThreadClusterLengths_BK0_N_BK1",
         this->b1_block_transfer.thread_cluster_length},
        {"B1BlockTransferThreadClusterArrangeOrder",
         this->b1_block_transfer.thread_cluster_arrange_order},
        {"B1BlockTransferSrcAccessOrder", this->b1_block_transfer.src_access_order},
        {"B1BlockTransferSrcVectorDim", std::to_string(this->b1_block_transfer.src_vec_dim)},
        {"B1BlockTransferSrcScalarPerVector",
         std::to_string(this->b1_block_transfer.src_scalar_per_vector)},
        {"B1BlockTransferDstScalarPerVector_BK1",
         std::to_string(this->b1_block_transfer.dst_scalar_per_vector_k1)},
        {"B1BlockLdsExtraN", std::to_string(this->b1_block_transfer.lds_add_extra_dim)},
        {"CShuffleMXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.m_Xdl_per_wave_per_shuffle)},
        {"CShuffleNXdlPerWavePerShuffle",
         std::to_string(this->cshuffle.n_Xdl_per_wave_per_shuffle)},
        {"CBlockTransferClusterLengths_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl",
         this->c_block_transfer.cluster_lengths_m_block_m_wave_m_per_Xdl_n_block_n_wave_n_per_Xdl},
        {"CBlockTransferScalarPerVector_NWaveNPerXdl",
         std::to_string(this->c_block_transfer.scalar_per_vector_n_wave_n_per_Xdl)},
        {"MaskOutUpperTriangle", std::to_string(this->mask_out_upper_triangle)},
    };

    return Solution{InterpolateString(DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffleTemplate, values),
                    std::move(values)};
}

} // namespace device_batched_gemm_softmax_gemm
} // namespace host
} // namespace ck
