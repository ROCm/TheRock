// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_ab_scale.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename AScaleDataType,
          typename BDataType,
          typename BScaleDataType,
          typename DsDataType,
          typename CDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t ScaleBlockM, // scale block for M
          index_t ScaleBlockN, // scale block for N
          index_t ScaleBlockK, // scale block for K
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
          typename CShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          bool PermuteA                               = false,
          bool PermuteB                               = false>
struct DeviceGemmMultiD_BlockScale_Wmma_CShuffle_V3_BPreshuffle
    : public DeviceGemmMultipleD_BlockScale_BPreshuffleSplitK<ALayout,
                                                              BLayout,
                                                              DsLayout,
                                                              CLayout,
                                                              ADataType,
                                                              AScaleDataType,
                                                              BDataType,
                                                              BScaleDataType,
                                                              DsDataType,
                                                              CDataType,
                                                              ScaleBlockM,
                                                              ScaleBlockN,
                                                              ScaleBlockK,
                                                              AElementwiseOperation,
                                                              BElementwiseOperation,
                                                              CElementwiseOperation>
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    using AScaleLayout = tensor_layout::gemm::ColumnMajor;
    using BScaleLayout = BLayout;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_wmma_cshuffle_v3_ab_scale<
        ALayout,
        BLayout,
        DsLayout,
        CLayout,
        Tuple<ADataType>,
        AScaleDataType,
        Tuple<BDataType>,
        BScaleDataType,
        AccDataType,
        CShuffleDataType,
        DsDataType,
        CDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        GemmSpec,
        BlockSize,
        ScaleBlockM,
        ScaleBlockN,
        ScaleBlockK,
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
        CShuffleBlockTransferScalarPerVectors,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB,
        PermuteA,
        PermuteB,
        true,
        AScaleLayout,
        BScaleLayout>;

    using Argument = typename GridwiseGemm::Argument;
    int GetPreShuffleParameters() override { return NPerWmma; }

    using DeviceGemmCommon =
        DeviceGemm_Wmma_CShuffleV3_Common<GridwiseGemm,
                                          Tuple<ADataType>,
                                          Tuple<BDataType>,
                                          DsDataType,
                                          CDataType,
                                          MPerBlock,
                                          NPerBlock,
                                          KPerBlock,
                                          BlockSize,
                                          AK1,
                                          BK1,
                                          GemmSpec,
                                          CShuffleBlockTransferScalarPerVectors,
                                          BlkGemmPipeSched,
                                          BlkGemmPipelineVer,
                                          ComputeTypeA,
                                          ComputeTypeB,
                                          true>; // IsBPreshuffle

    // Invoker
    using Invoker = typename DeviceGemmCommon::Invoker;

    static bool IsSupportedArgument(const Argument& arg)
    {
        // with splitk the implementation doesn't work
        // when KRead % ScaleBlockK != 0, independently of K padding
        if(arg.KBatch > 1 && arg.KRead % ScaleBlockK != 0)
        {
            return false;
        }

        return DeviceGemmCommon::IsSupportedArgument(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_e,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             const std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideC,
                             const void* p_a_scale,
                             const void* p_b_scale,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation cde_element_op,
                             index_t KBatch)
    {
        index_t StrideScaleA = ck::is_same_v<AScaleLayout, tensor_layout::gemm::RowMajor>
                                   ? math::integer_divide_ceil(K, ScaleBlockK)
                                   : math::integer_divide_ceil(M, ScaleBlockM);

        index_t StrideScaleB = ck::is_same_v<BScaleLayout, ck::tensor_layout::gemm::ColumnMajor>
                                   ? math::integer_divide_ceil(K, ScaleBlockK)
                                   : math::integer_divide_ceil(N, ScaleBlockN);

        return Argument{std::array<const void*, 1>{p_a},
                        std::array<const void*, 1>{p_b},
                        p_ds,
                        static_cast<CDataType*>(p_e),
                        M,
                        N,
                        K,
                        std::array<index_t, 1>{StrideA},
                        std::array<index_t, 1>{StrideB},
                        StrideDs,
                        StrideC,
                        StrideScaleA,
                        StrideScaleB,
                        static_cast<const AScaleDataType*>(p_a_scale),
                        static_cast<const BScaleDataType*>(p_b_scale),
                        KBatch,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        std::array<const void*, NumDTensor> p_ds,
                        void* p_e,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t StrideA,
                        index_t StrideB,
                        const std::array<ck::index_t, NumDTensor> StrideDs,
                        index_t StrideC,
                        const void* p_a_scale,
                        const void* p_b_scale,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        index_t KBatch) override
    {
        index_t StrideScaleA = ck::is_same_v<AScaleLayout, tensor_layout::gemm::RowMajor>
                                   ? math::integer_divide_ceil(K, ScaleBlockK)
                                   : math::integer_divide_ceil(M, ScaleBlockM);

        index_t StrideScaleB = ck::is_same_v<BScaleLayout, ck::tensor_layout::gemm::ColumnMajor>
                                   ? math::integer_divide_ceil(K, ScaleBlockK)
                                   : math::integer_divide_ceil(N, ScaleBlockN);

        return std::make_unique<Argument>(std::array<const void*, 1>{p_a},
                                          std::array<const void*, 1>{p_b},
                                          p_ds,
                                          static_cast<CDataType*>(p_e),
                                          M,
                                          N,
                                          K,
                                          std::array<index_t, 1>{StrideA},
                                          std::array<index_t, 1>{StrideB},
                                          StrideDs,
                                          StrideC,
                                          StrideScaleA,
                                          StrideScaleB,
                                          static_cast<const AScaleDataType*>(p_a_scale),
                                          static_cast<const BScaleDataType*>(p_b_scale),
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
        str << "DeviceGemmMultiD_BlockScale_Wmma_CShuffle_V3_BPreshuffle"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerWmma<<"x"<<NPerWmma << ", "
            << "WaveMap: "
            << MRepeat<<"x" << NRepeat<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
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
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
