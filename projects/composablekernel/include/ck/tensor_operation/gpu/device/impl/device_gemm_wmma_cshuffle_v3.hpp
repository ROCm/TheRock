// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/// @brief \"Universal\" GEMM operation with SplitK support.
///
/// @par Overview
///         This GEMM operation implements the following mathematical equation:
///         C{M,N} = C_op(A_op(A{M,K}) * B_op(B{K,N}))
///         Where A, B are input tensors and C is the output tensor. The A/B/C_op are
///         elementwise operations applied to the A, B, and C tensors, respectively.
///         The \"universal\" gemm comes with multiple pipelines optimized for different usage
///         scenarios. That's why it's called \"universal\". It's universal through it's design
///         and versatilty.
///
/// @note   This Kernel implementation supports SplitK algorithm. It can be configured
///         to split the dot product accumulated over the K dimension into multiple working groups.
///         The partial products of different workgroups are then reduced using the AtomicAdd
///         operation.
///
/// @tparam ALayout     A tensor data layout.
/// @tparam BLayout     B tensor data layout.
/// @tparam CLayout     C tensor data layout.
/// @tparam ADataType   A tensor data type.
/// @tparam BDataType   B tensor data type.
/// @tparam CDataType   C tensor data type.
/// @tparam AccDataType The accumulation data type related to the hardware
///                         matrix-multiplication instruction.
/// @tparam CShuffleDataType The data type used to store matrix-multiplication results into
///                          LDS memory during \"CShuffle\" data layout optimization.
/// @tparam AElementwiseOperation Elementwise operation applied to the A input tensor elements.
/// @tparam BElementwiseOperation Elementwise operation applied to the B input tensor elements.
/// @tparam CElementwiseOperation Elementwise operation applied to the C output tensor
///                               (after GEMM).
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
/// @tparam CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock The spatial
///                                         thread distribution used for storing data into output
///                                         tensor across output data layout dimensions.
/// @tparam CShuffleBlockTransferScalarPerVector_NPerBlock The size of vectorized memory access.
///                                         Used when storing data to output tensor.
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
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
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
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGemm_Wmma_CShuffleV3 : public DeviceGemmV2<ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        ADataType,
                                                        BDataType,
                                                        CDataType,
                                                        AElementwiseOperation,
                                                        BElementwiseOperation,
                                                        CElementwiseOperation>
{
    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3<
        ALayout,
        BLayout,
        Tuple<>, // DsLayout
        CLayout,
        Tuple<ADataType>,
        Tuple<BDataType>,
        AccDataType,
        CShuffleDataType,
        Tuple<>, // DsDataType
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
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
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB>;

    using Argument = typename GridwiseGemm::Argument;

    using DeviceGemmCommon =
        DeviceGemm_Wmma_CShuffleV3_Common<GridwiseGemm,
                                          Tuple<ADataType>,
                                          Tuple<BDataType>,
                                          Tuple<>,
                                          CDataType,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          BlockSize,
                                          AK1,
                                          BK1,
                                          GemmSpec,
                                          Sequence<CShuffleBlockTransferScalarPerVector_NPerBlock>,
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

    index_t GetKPerBlock() override { return KPerBlock; }

    bool GetPermuteA() override { return PermuteA; }
    bool GetPermuteB() override { return PermuteB; }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation cde_element_op)
    {
        return Argument{std::array<const void*, 1>{p_a},
                        std::array<const void*, 1>{p_b},
                        std::array<const void*, 0>{}, // p_ds_grid_
                        p_c,
                        M,
                        N,
                        K,
                        std::array<index_t, 1>{StrideA},
                        std::array<index_t, 1>{StrideB},
                        std::array<index_t, 0>{}, // StrideDs_
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(std::array<const void*, 1>{p_a},
                                          std::array<const void*, 1>{p_b},
                                          std::array<const void*, 0>{}, // p_ds_grid_
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          std::array<index_t, 1>{StrideA},
                                          std::array<index_t, 1>{StrideB},
                                          std::array<index_t, 0>{}, // StrideDs_
                                          StrideC,
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
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
        str << "DeviceGemm_Wmma_CShuffleV3"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
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
