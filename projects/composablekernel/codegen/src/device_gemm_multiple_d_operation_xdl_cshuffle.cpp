// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/types.hpp"
#include "ck/host/utils.hpp"
#include <cassert>

namespace ck {
namespace host {
namespace device_gemm_multiple_d {

// calculate appropriate Gemm Specification based on input tensor dimensions
static std::string GetGemmSpec(const std::size_t m,
                               const std::size_t n,
                               const std::size_t k,
                               const std::size_t m_per_block,
                               const std::size_t n_per_block,
                               const std::size_t k_per_block)
{
    std::string spec = "";
    if(integer_divide_ceil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if(integer_divide_ceil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if(integer_divide_ceil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";
    if(spec == "")
        return "ck::tensor_operation::device::GemmSpecialization::Default";

    return "ck::tensor_operation::device::GemmSpecialization::" + spec + "Padding";
}

// function to update prologue/epilogue with user provided operation
void Operation_Xdl_CShuffle::update_prologue(const std::string& pro)
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

void Operation_Xdl_CShuffle::update_epilogue(const std::string& epi)
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

// accounts for all possible combinations of Row/Col major
static Layout ToLayout(bool Trans) { return Trans ? Layout::Column : Layout::Row; }

// clang-format off
// DeviceGemmMultipleD_Xdl_CShuffle<    Col,    Row, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,   F16, PassThrough, PassThrough, AddAddFastGelu, GemmMNKPadding,        1,    64,    16,    16,    32,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1,

// DeviceGemmMultipleD_Xdl_CShuffle<    Row,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,   F16, PassThrough, PassThrough, AddAddFastGelu, GemmMNKPadding,        1,    64,    16,    16,    32,   8,   8,   16,   16,    1,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              1,              8,         1,           1,           1,               S<1, 16, 1, 4>,               1, LoopScheduler::Default,        PipelineVersion::v1>
// clang-format on

// Hard-code tuning parameters in modularized fashion, string them together into a vector of
// instances
std::vector<Operation_Xdl_CShuffle> Operation_Xdl_CShuffle::CreateOperations(
    const Problem& prob, const std::string& prologue, const std::string& epilogue)
{
    std::vector<Operation_Xdl_CShuffle> result;

    std::vector<operation::TileDesc> tile_descriptions = {
        // clang-format off
//  Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl| NumGemmK|
//   Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per| Prefetch|
//       |      |      |      |    |    |     |     | Wave| Wave|    Stage|
//       |      |      |      |    |    |     |     |     |     |         |
  {   256,   256,   128,    32,   8,   8,   16,   16,    8,    4,        1},
  {   256,   128,   256,    32,   8,   8,   16,   16,    4,    8,        1},
  {   128,   128,   128,    32,   8,   8,   16,   16,    8,    4,        1},
  {   256,   128,   128,    32,   8,   8,   16,   16,    4,    4,        1},
  {   128,   128,    64,    32,   8,   8,   16,   16,    4,    4,        1},
  {   128,    64,   128,    32,   8,   8,   16,   16,    4,    4,        1},
  {   256,   128,    64,    32,   8,   8,   16,   16,    4,    2,        1},
  {   256,    64,   128,    32,   8,   8,   16,   16,    2,    4,        1},
//  Irregular tile
  {    64,    32,    32,    32,   8,   8,   16,   16,    2,    2,        1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> a_block_descriptions_rowmajor = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//  Irregular tile
  {    S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> a_block_descriptions_colmajor = {
        // clang-format off
//  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|
// Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1},
//  Irregular tile
  {    S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> b_block_descriptions_rowmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
  {    S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1},
  {    S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,         1},
//  Irregular tile
  {    S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,         1},
        // clang-format on
    };

    std::vector<operation::BlockTransferDesc> b_block_descriptions_colmajor = {
        // clang-format off
//  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|
//   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN|
// Lengths_K0_N_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          |
//                |               |               |               |               |               |          |
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
  {    S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1},
//  Irregular tile
  {    S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              1,              8,         1},
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
  {          1,           1},
  {          1,           1},
  {          1,           1},
  {          1,           1},
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
  {              S<1, 16, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 4>,               4},
  {              S<1, 16, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
  {              S<1, 32, 1, 8>,               4},
//  Irregular tile
  {              S<1, 16, 1, 4>,               1},
        // clang-format on
    };

    // choose correct arrangement of tuning parameters based on the layout of each tensor
    const auto a_block_descriptions =
        prob.TransA ? a_block_descriptions_colmajor : a_block_descriptions_rowmajor;
    const auto b_block_descriptions =
        prob.TransB ? b_block_descriptions_colmajor : b_block_descriptions_rowmajor;

    assert(tile_descriptions.size() == a_block_descriptions.size());
    assert(tile_descriptions.size() == b_block_descriptions.size());
    assert(tile_descriptions.size() == cshuffle_descriptions.size());
    assert(tile_descriptions.size() == c_block_descriptions.size());

    const std::vector<std::tuple<LoopScheduler, PipelineVersion>> scheduler_pipeline_descriptions =
        {
            {LoopScheduler::Default, PipelineVersion::v1},
            {LoopScheduler::Interwave, PipelineVersion::v1},
            {LoopScheduler::Default, PipelineVersion::v2},
        };
    for(auto [loop_scheduler, pipeline_version] : scheduler_pipeline_descriptions)
    {
        // Put all values together into a single operation > store into the result vector
        for(std::size_t i = 0; i < tile_descriptions.size(); i++)
        {
            Operation_Xdl_CShuffle x;
            x.tile_desc        = tile_descriptions[i];
            x.a_block_transfer = a_block_descriptions[i];
            x.b_block_transfer = b_block_descriptions[i];
            x.cshuffle         = cshuffle_descriptions[i];
            x.c_block_transfer = c_block_descriptions[i];
            x.A                = TensorDesc{prob.ADataType, ToLayout(prob.TransA)};
            x.B                = TensorDesc{prob.BDataType, ToLayout(prob.TransB)};
            x.E                = TensorDesc{prob.EDataType, ToLayout(prob.TransE)};
            x.Ds               = Transform(prob.DsTrans, prob.DsDataType, [](auto trans, auto dt) {
                return TensorDesc{dt, ToLayout(trans)};
            });
            x.a_elem_op        = prob.AElementOp;
            x.b_elem_op        = prob.BElementOp;
            x.cde_elem_op      = prob.CDEElementOp;
            x.gemm_specialization = GetGemmSpec(prob.M,
                                                prob.N,
                                                prob.K,
                                                x.tile_desc.m_per_block,
                                                x.tile_desc.n_per_block,
                                                x.tile_desc.k_per_block);
            x.loop_scheduler      = loop_scheduler;
            x.pipeline_version    = pipeline_version;
            x.update_prologue(prologue);
            x.update_epilogue(epilogue);
            result.push_back(x);
        }
    }
    return result;
}

// set up instances when not provided with a problem specification, use default operation values and
// all possible layout combinations
std::vector<std::vector<Operation_Xdl_CShuffle>>
Operation_Xdl_CShuffle::CreateOperations(const std::string& prologue, const std::string& epilogue)
{
    std::vector<Problem> problems;
    for(bool TransA : {true, false})
        for(bool TransB : {true, false})
        {
            Problem prob;
            prob.TransA = TransA;
            prob.TransB = TransB;
            problems.push_back(prob);
        }
    return Transform(problems,
                     [&](const Problem& p) { return CreateOperations(p, prologue, epilogue); });
}

static const char* const DeviceGemmMultipleD_Xdl_CShuffleTemplate =
    "ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<${LayoutA}, ${LayoutB}, "
    "${LayoutDs}, ${LayoutE}, ${ADataType}, ${BDataType}, ${AccDataType}, ${CShuffleDataType}, "
    "${DsDataType}, ${EDataType}, ${AElementwiseOperation}, ${BElementwiseOperation}, "
    "${CDEElementwiseOperation}, ${GemmSpecialization}, ${NumGemmkPrefetchStage}, ${BlockSize}, "
    "${MPerBlock}, ${NPerBlock}, ${KPerBlock}, ${AK1}, ${BK1}, ${MPerXDL}, ${NPerXDL}, "
    "${MXdlPerWave}, ${NXdlPerWave}, ${ABlockTransferThreadClusterLengths_AK0_M_AK1}, "
    "${ABlockTransferThreadClusterArrangeOrder}, ${ABlockTransferSrcAccessOrder}, "
    "${ABlockTransferSrcVectorDim}, ${ABlockTransferSrcScalarPerVector}, "
    "${ABlockTransferDstScalarPerVector_AK1}, ${ABlockLdsExtraM}, "
    "${BBlockTransferThreadClusterLengths_BK0_N_BK1}, ${BBlockTransferThreadClusterArrangeOrder}, "
    "${BBlockTransferSrcAccessOrder}, ${BBlockTransferSrcVectorDim}, "
    "${BBlockTransferSrcScalarPerVector}, ${BBlockTransferDstScalarPerVector_BK1}, "
    "${BBlockLdsExtraN}, ${CShuffleMXdlPerWavePerShuffle}, ${CShuffleNXdlPerWavePerShuffle}, "
    "${CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock}, "
    "${CDEBlockTransferScalarPerVector_NPerBlock}, ${LoopScheduler}, ${PipelineVersion}>";

// use hardcoded instances from vector of operations to substitute values into instance template
Solution Operation_Xdl_CShuffle::ToSolution() const
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
        {"LayoutA", ToString(this->A.layout)},
        {"LayoutB", ToString(this->B.layout)},
        {"LayoutDs",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.layout); }))},
        {"LayoutE", ToString(this->E.layout)},
        {"ADataType", ToString(this->A.element)},
        {"BDataType", ToString(this->B.element)},
        {"AccDataType", ToString(this->acc)},
        {"CShuffleDataType", ToString(this->cs_type)},
        {"DsDataType",
         MakeTuple(Transform(this->Ds, [](auto tensor) { return ToString(tensor.element); }))},
        {"EDataType", ToString(this->E.element)},
        {"AElementwiseOperation", this->a_elem_op},
        {"BElementwiseOperation", this->b_elem_op},
        {"CDEElementwiseOperation", this->cde_elem_op},
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
        {"LoopScheduler", ToString(this->loop_scheduler)},
        {"PipelineVersion", ToString(this->pipeline_version)},
    };

    return Solution{InterpolateString(DeviceGemmMultipleD_Xdl_CShuffleTemplate, values),
                    std::move(values)};
}

} // namespace device_gemm_multiple_d
} // namespace host
} // namespace ck
