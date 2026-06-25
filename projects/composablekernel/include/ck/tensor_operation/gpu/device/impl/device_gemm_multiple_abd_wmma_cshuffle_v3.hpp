// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/// @brief \"Universal\" GEMM operation with SplitK support and multiple D tensors.
///
/// @par Overview
///         This GEMM operation implements the following mathematical equation:
///         E{M,N} = CDE_op(A_op(As{M,K}...) * B_op(Bs{K,N}...), Ds{M,N}...)
///         Where As, Bs, Ds are input tensors and E is the output tensor. The A/B_op are
///         elementwise
//          operations that could be applied on each tensor respectively. The CDE_op is an
//          elementwise operation applied to the C and all D tensors.
///         The \"universal\" gemm comes with multiple pipelines optimized for different usage
///         scenarios. That's why it's called \"universal\". It's universal through it's design
///         and versatilty.
///
/// @note   This Kernel implementation supports SplitK algorithm. It can be configured
///         to split the dot product accumulated over the K dimension into multiple working groups.
///         The partial products of different workgroups are then reduced using the AtomicAdd
///         operation.
///
/// @tparam AsLayout    A tensors data layouts.
/// @tparam BsLayout    B tensors data layouts.
/// @tparam DsLayout    D tensors data layouts.
/// @tparam ELayout     E tensor data layout.
/// @tparam AsDataType  A tensors data types.
/// @tparam BsDataType  B tensors data types.
/// @tparam DsDataType  D tensors data types.
/// @tparam EDataType   E tensor data type.
/// @tparam AccDataType The accumulation data type related to the hardware
///                         matrix-multiplication instruction.
/// @tparam CShuffleDataType The data type used to store matrix-multiplication results into
///                          LDS memory during \"CShuffle\" data layout optimization.
/// @tparam AElementwiseOperation Elementwise operation applied to the A input tensor elements.
/// @tparam BElementwiseOperation Elementwise operation applied to the B input tensor elements.
/// @tparam CDEElementwiseOperation Elementwise operation applied to the C output tensor (after
///                                 GEMM) and D input tensors.
/// @tparam GemmSpec    Determines used "padding" version.
/// @tparam BlockSize   The number of threads within workgroup.
/// @tparam MPerBlock   The input/output data tile size in the M dimension.
/// @tparam NPerBlock   The input/output data tile size in the N dimension.
/// @tparam KPerBlock   The input data tile size in the K dimension.
/// @tparam AK1         The vector load size from global memory for A tensor.
/// @tparam BK1         The vector load size from global memory for B tensor.
/// @tparam MPerWmma    M size of Wave Matrix Multiply Accumulate (WMMA) instruction.
/// @tparam NPerWmma    N size of Wave Matrix Multiply Accumulate (WMMA) instruction.
/// @tparam MRepeat     The number of iterations in the M dimension over output tile per wavefront.
/// @tparam NRepeat     The number of iterations in the N dimension over output tile per wavefront.
/// @tparam ABlockTransferThreadClusterLengths_AK0_M_AK1 Spatial thread distribution over the input
///                                                      data. Can be interpreted as the answer
///                                                      to the question, "How many threads can be
///                                                      arranged on each input data axis?"
/// @tparam ABlockTransferThreadClusterArrangeOrder The order of thread spatial distribution over
///                                                 the input tensor dimension. Can be interpreted
///                                                 as the answer to the question: "In which
///                                                 order to spread threads through tensor axes?".
/// @tparam ABlockTransferSrcAccessOrder The order of accessing input tensor axes. Can be
///                                      interpreted as the answer to the question "Which dimension
///                                      to read first? And which next?" etc.
/// @tparam ABlockTransferSrcVectorDim   The index of axis on which we could do vectorized memory
///                                      access - the one with contiguous memory.
/// @tparam ABlockTransferSrcScalarPerVector The size of vector access instruction - the number of
///                                          elements accessed per thread per instruction.
/// @tparam ABlockTransferDstScalarPerVector_AK1 The size of vectorized store into LDS memory.
/// @tparam ABlockLdsExtraM                      Whether to use padding for LDS or not. With
///                                              universal GEMM there's no need for padding.
/// @tparam BBlockTransferThreadClusterLengths_BK0_N_BK1 Spatial thread distribution over the input
///                                                      data. Can be interpreted as the answer
///                                                      to the question: "How many threads to
///                                                      arrange on each input data axis?"
/// @tparam BBlockTransferThreadClusterArrangeOrder The order of thread spatial distribution over
///                                                 the input tensor dimension. Can be interpreted
///                                                 as the answer to the question: "In which
///                                                 order to spread threads through tensor axes?".
/// @tparam BBlockTransferSrcAccessOrder he order of accessing input tensor axes. Can be
///                                      interpreted as the answer to the question "Which dimension
///                                      to read first? And which next?" etc.
/// @tparam BBlockTransferSrcVectorDim  The index of axis on which we could do vectorized memory
///                                      access - the one with contiguous memory.
/// @tparam BBlockTransferSrcScalarPerVector The size of vector access instruction - the number of
///                                          elements accessed per thread per instruction.
/// @tparam BBlockTransferDstScalarPerVector_BK1 The size of vectorized store into LDS memory.
/// @tparam BBlockLdsExtraN                      Whether to use padding for LDS or not. With
///                                              universal GEMM there's no need for padding.
/// @tparam CShuffleMRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in M dimension.
/// @tparam CShuffleNRepeatPerShuffle   The number of matrix-multiplication instructions
///                                         results to process per wave per iteration of CShuffle
///                                         in N dimension.
/// @tparam CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock The spatial
///                                         thread distribution used for storing data into output
///                                         tensor across output data layout dimensions.
/// @tparam CDEShuffleBlockTransferScalarPerVectors The size of vectorized memory access.
///                                         Used when loading data from D tensors and storing data
///                                         to output tensor.
/// @tparam BlkGemmPipeSched    The version of blockwise-gemm pipeline scheduler (interwave or
///                             intrawave).
/// @tparam BlkGemmPipelineVer  The version of blockwise-gemm pipeline.
/// @tparam ComputeTypeA    Data type used for A input of hardware matrix-multiplication
///                         instructions.
/// @tparam ComputeTypeB    Data type used for B input of hardware matrix-multiplication
///                         instructions.
/// @tparam PermuteA            Whether the A input tensor has gridwise-gemm friendly data layout
///                             in global memory. Currently not supported!
/// @tparam PermuteB            Whether the B input tensor has gridwise-gemm friendly data layout
///                             in global memory (pre-shuffled).
template <typename AsLayout,
          typename BsLayout,
          typename DsLayout,
          typename ELayout,
          typename AsDataType,
          typename BsDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = EDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGemmMultipleABD_Wmma_CShuffleV3
    : public DeviceGemmMultipleABDSplitK<AsLayout,
                                         BsLayout,
                                         DsLayout,
                                         ELayout,
                                         AsDataType,
                                         BsDataType,
                                         DsDataType,
                                         EDataType,
                                         AElementwiseOperation,
                                         BElementwiseOperation,
                                         CDEElementwiseOperation>
{
    // Note: Pass multiple layout but then using only the first one
    // This is to replicate xdl functionality but it should be extended
    using ALayout = remove_cvref_t<tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<tuple_element_t<0, BsLayout>>;

    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        AsDataType,
        BsDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    using Argument = typename GridwiseGemm::Argument;

    using DeviceGemmCommon =
        DeviceGemm_Wmma_CShuffleV3_Common<GridwiseGemm,
                                          AsDataType,
                                          BsDataType,
                                          DsDataType,
                                          EDataType,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          BlockSize,
                                          AK1,
                                          BK1,
                                          GemmSpec,
                                          CDEShuffleBlockTransferScalarPerVectors,
                                          BlkGemmPipeSched,
                                          BlkGemmPipelineVer,
                                          ComputeTypeA,
                                          ComputeTypeB>;

    // Invoker
    using Invoker = typename DeviceGemmCommon::Invoker;

    static bool IsSupportedArgument(const Argument& arg)
    {
        return DeviceGemmCommon::IsSupportedArgument(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::array<const void*, GridwiseGemm::NumATensor> p_as,
                             std::array<const void*, GridwiseGemm::NumBTensor> p_bs,
                             std::array<const void*, GridwiseGemm::NumDTensor> p_ds,
                             void* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             std::array<ck::index_t, GridwiseGemm::NumATensor> StrideAs,
                             std::array<ck::index_t, GridwiseGemm::NumBTensor> StrideBs,
                             std::array<index_t, GridwiseGemm::NumDTensor> StrideDs,
                             index_t StrideE,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{p_as,
                        p_bs,
                        p_ds,
                        static_cast<EDataType*>(p_e),
                        M,
                        N,
                        K,
                        StrideAs,
                        StrideBs,
                        StrideDs,
                        StrideE,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, GridwiseGemm::NumATensor> p_as,
                        std::array<const void*, GridwiseGemm::NumBTensor> p_bs,
                        std::array<const void*, GridwiseGemm::NumDTensor> p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        std::array<ck::index_t, GridwiseGemm::NumATensor> StrideAs,
                        std::array<ck::index_t, GridwiseGemm::NumBTensor> StrideBs,
                        std::array<ck::index_t, GridwiseGemm::NumDTensor> StrideDs,
                        index_t StrideE,
                        index_t KBatch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) override
    {
        return std::make_unique<Argument>(p_as,
                                          p_bs,
                                          p_ds,
                                          static_cast<EDataType*>(p_e),
                                          M,
                                          N,
                                          K,
                                          StrideAs,
                                          StrideBs,
                                          StrideDs,
                                          StrideE,
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"},
            {BlockGemmPipelineVersion::v2, "v2"},
            {BlockGemmPipelineVersion::v3, "v3"},
            {BlockGemmPipelineVersion::v4, "v4"},
            {BlockGemmPipelineVersion::v5, "v5"}};

        // clang-format off
        str << "DeviceGemmMultipleABD_Wmma_CShuffleV3"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", ";
        static_for<0, GridwiseGemm::NumATensor, 1>{}([&](auto i) {
            using ALayout_ = remove_cvref_t<tuple_element_t<i.value, AsLayout>>;

            str << std::string(ALayout_::name)[0];
        });
        static_for<0, GridwiseGemm::NumBTensor, 1>{}([&](auto i) {
            using BLayout_ = remove_cvref_t<tuple_element_t<i.value, BsLayout>>;

            str << std::string(BLayout_::name)[0];
        });
        static_for<0, GridwiseGemm::NumDTensor, 1>{}([&](auto i) {
            using DLayout = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;

            str << std::string(DLayout::name)[0];
        });
        str << std::string(ELayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock << "x" << NPerBlock << "x" << KPerBlock << ", "
            << "WaveTile: "
            << MPerWmma << "x"<<NPerWmma << ", "
            << "WaveMap: "
            << MRepeat << "x" << NRepeat << ", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector << "x" << BBlockTransferSrcScalarPerVector << ", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages << ", "
            << "KPack: "
            << GridwiseGemm::KPack;
        // clang-format on

        return str.str();
    }
    REGISTER_EXTRA_PRINTING_METHODS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
