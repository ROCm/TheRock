// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"

namespace ck::tensor_operation::device {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename DsDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization
              ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t ABK1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsAddExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3;

} // namespace ck::tensor_operation::device

namespace ck_tile {
namespace reflect {

template <ck::index_t NDimSpatial,
          typename InLayout_,
          typename WeiLayout_,
          typename OutLayout_,
          typename DsLayout_,
          typename InDataType_,
          typename WeiDataType_,
          typename OutDataType_,
          typename AccDataType_,
          typename DsDataType_,
          typename InElementwiseOperation_,
          typename WeiElementwiseOperation_,
          typename OutElementwiseOperation_,
          ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization
              ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t ABK1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1_,
          typename ABlockTransferThreadClusterArrangeOrder_,
          typename ABlockTransferSrcAccessOrder_,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1_,
          typename BBlockTransferThreadClusterArrangeOrder_,
          typename BBlockTransferSrcAccessOrder_,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsAddExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer,
          typename ComputeTypeA_,
          typename ComputeTypeB_>
struct InstanceTraits<
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3<
        NDimSpatial,
        InLayout_,
        WeiLayout_,
        OutLayout_,
        DsLayout_,
        InDataType_,
        WeiDataType_,
        OutDataType_,
        AccDataType_,
        DsDataType_,
        InElementwiseOperation_,
        WeiElementwiseOperation_,
        OutElementwiseOperation_,
        ConvBackwardWeightSpecialization,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        ABK1,
        MPerWmma,
        NPerWmma,
        MRepeat,
        NRepeat,
        ABlockTransferThreadClusterLengths_AK0_M_AK1_,
        ABlockTransferThreadClusterArrangeOrder_,
        ABlockTransferSrcAccessOrder_,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1_,
        BBlockTransferThreadClusterArrangeOrder_,
        BBlockTransferSrcAccessOrder_,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        BBlockLdsAddExtraN,
        CShuffleMRepeatPerShuffle,
        CShuffleNRepeatPerShuffle,
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
        CShuffleBlockTransferScalarPerVector_NPerBlock,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA_,
        ComputeTypeB_>>
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3";

    static constexpr ck::index_t kNDimSpatial = NDimSpatial;

    using InLayout  = InLayout_;
    using WeiLayout = WeiLayout_;
    using OutLayout = OutLayout_;
    using DsLayout  = DsLayout_;

    using InDataType  = InDataType_;
    using WeiDataType = WeiDataType_;
    using OutDataType = OutDataType_;
    using AccDataType = AccDataType_;
    using DsDataType  = DsDataType_;

    using InElementwiseOperation  = InElementwiseOperation_;
    using WeiElementwiseOperation = WeiElementwiseOperation_;
    using OutElementwiseOperation = OutElementwiseOperation_;

    static constexpr auto kConvBackwardWeightSpecialization = ConvBackwardWeightSpecialization;

    static constexpr ck::index_t kBlockSize                 = BlockSize;
    static constexpr ck::index_t kMPerBlock                 = MPerBlock;
    static constexpr ck::index_t kNPerBlock                 = NPerBlock;
    static constexpr ck::index_t kKPerBlock                 = KPerBlock;
    static constexpr ck::index_t kABK1                      = ABK1;
    static constexpr ck::index_t kMPerWmma                  = MPerWmma;
    static constexpr ck::index_t kNPerWmma                  = NPerWmma;
    static constexpr ck::index_t kMRepeat                   = MRepeat;
    static constexpr ck::index_t kNRepeat                   = NRepeat;
    static constexpr ck::index_t kCShuffleMRepeatPerShuffle = CShuffleMRepeatPerShuffle;
    static constexpr ck::index_t kCShuffleNRepeatPerShuffle = CShuffleNRepeatPerShuffle;
    static constexpr ck::index_t kCShuffleBlockTransferScalarPerVector_NPerBlock =
        CShuffleBlockTransferScalarPerVector_NPerBlock;

    using ABlockTransferThreadClusterLengths_AK0_M_AK1 =
        ABlockTransferThreadClusterLengths_AK0_M_AK1_;
    using ABlockTransferThreadClusterArrangeOrder = ABlockTransferThreadClusterArrangeOrder_;
    using ABlockTransferSrcAccessOrder            = ABlockTransferSrcAccessOrder_;
    static constexpr ck::index_t kABlockTransferSrcVectorDim = ABlockTransferSrcVectorDim;
    static constexpr ck::index_t kABlockTransferSrcScalarPerVector =
        ABlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kABlockTransferDstScalarPerVector_AK1 =
        ABlockTransferDstScalarPerVector_AK1;
    static constexpr bool kABlockLdsAddExtraM = ABlockLdsAddExtraM;

    using BBlockTransferThreadClusterLengths_BK0_N_BK1 =
        BBlockTransferThreadClusterLengths_BK0_N_BK1_;
    using BBlockTransferThreadClusterArrangeOrder = BBlockTransferThreadClusterArrangeOrder_;
    using BBlockTransferSrcAccessOrder            = BBlockTransferSrcAccessOrder_;
    static constexpr ck::index_t kBBlockTransferSrcVectorDim = BBlockTransferSrcVectorDim;
    static constexpr ck::index_t kBBlockTransferSrcScalarPerVector =
        BBlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kBBlockTransferDstScalarPerVector_BK1 =
        BBlockTransferDstScalarPerVector_BK1;
    static constexpr bool kBBlockLdsAddExtraN = BBlockLdsAddExtraN;

    using CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock =
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_;

    static constexpr ck::BlockGemmPipelineScheduler kBlkGemmPipeSched = BlkGemmPipeSched;
    static constexpr ck::BlockGemmPipelineVersion kBlkGemmPipelineVer = BlkGemmPipelineVer;

    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3";

        // Template parameters in exact order
        oss << "<" << kNDimSpatial;                     // 1. NDimSpatial
        oss << "," << detail::layout_name<InLayout>();  // 2. InLayout
        oss << "," << detail::layout_name<WeiLayout>(); // 3. WeiLayout
        oss << "," << detail::layout_name<OutLayout>(); // 4. OutLayout
        oss << "," << detail::tuple_name<DsLayout>();   // 5. DsLayout
        oss << "," << detail::type_name<InDataType>();  // 6. InDataType
        oss << "," << detail::type_name<WeiDataType>(); // 7. WeiDataType
        oss << "," << detail::type_name<OutDataType>(); // 8. OutDataType
        oss << "," << detail::type_name<AccDataType>(); // 9. AccDataType
        oss << "," << detail::tuple_name<DsDataType>(); // 10. DsDataType
        oss << ","
            << detail::elementwise_op_name<InElementwiseOperation>(); // 11. InElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<WeiElementwiseOperation>(); // 12.
                                                                       // WeiElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<OutElementwiseOperation>(); // 13.
                                                                       // OutElementwiseOperation
        oss << ","
            << detail::conv_bwd_weight_spec_name(
                   kConvBackwardWeightSpecialization); // 14. ConvBackwardWeightSpecialization
        oss << "," << kBlockSize;                      // 15. BlockSize
        oss << "," << kMPerBlock;                      // 16. MPerBlock
        oss << "," << kNPerBlock;                      // 17. NPerBlock
        oss << "," << kKPerBlock;                      // 18. KPerBlock
        oss << "," << kABK1;                           // 19. ABK1
        oss << "," << kMPerWmma;                       // 20. MPerWmma
        oss << "," << kNPerWmma;                       // 21. NPerWmma
        oss << "," << kMRepeat;                        // 22. MRepeat
        oss << "," << kNRepeat;                        // 23. NRepeat
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterLengths_AK0_M_AK1>(); // 24.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>();      // 25.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();                 // 26.
        oss << "," << kABlockTransferSrcVectorDim;                                           // 27.
        oss << "," << kABlockTransferSrcScalarPerVector;                                     // 28.
        oss << "," << kABlockTransferDstScalarPerVector_AK1;                                 // 29.
        oss << "," << (kABlockLdsAddExtraM ? "true" : "false");                              // 30.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterLengths_BK0_N_BK1>(); // 31.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>();      // 32.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();                 // 33.
        oss << "," << kBBlockTransferSrcVectorDim;                                           // 34.
        oss << "," << kBBlockTransferSrcScalarPerVector;                                     // 35.
        oss << "," << kBBlockTransferDstScalarPerVector_BK1;                                 // 36.
        oss << "," << (kBBlockLdsAddExtraN ? "true" : "false");                              // 37.
        oss << "," << kCShuffleMRepeatPerShuffle;                                            // 38.
        oss << "," << kCShuffleNRepeatPerShuffle;                                            // 39.
        oss << ","
            << detail::sequence_name<
                   CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>(); // 40.
        oss << "," << kCShuffleBlockTransferScalarPerVector_NPerBlock;                       // 41.
        oss << "," << detail::pipeline_scheduler_name(kBlkGemmPipeSched);                    // 42.
        oss << "," << detail::pipeline_version_name(kBlkGemmPipelineVer);                    // 43.
        oss << "," << detail::type_name<ComputeTypeA>();                                     // 44.
        oss << "," << detail::type_name<ComputeTypeB>();                                     // 45.
        oss << ">";

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
