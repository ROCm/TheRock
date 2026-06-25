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
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
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
          typename ComputeTypeB,
          ck::index_t MaxTransposeTransferSrcScalarPerVector,
          ck::index_t MaxTransposeTransferDstScalarPerVector>
struct DeviceGroupedConvBwdWeight_Wmma_CShuffleV3;

} // namespace ck::tensor_operation::device

namespace ck_tile {
namespace reflect {

template <ck::index_t NDimSpatial,
          typename InLayout_,
          typename WeiLayout_,
          typename OutLayout_,
          typename InDataType_,
          typename WeiDataType_,
          typename OutDataType_,
          typename AccDataType_,
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
          typename ComputeTypeB_,
          ck::index_t MaxTransposeTransferSrcScalarPerVector,
          ck::index_t MaxTransposeTransferDstScalarPerVector>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffleV3<
    NDimSpatial,
    InLayout_,
    WeiLayout_,
    OutLayout_,
    InDataType_,
    WeiDataType_,
    OutDataType_,
    AccDataType_,
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
    ComputeTypeB_,
    MaxTransposeTransferSrcScalarPerVector,
    MaxTransposeTransferDstScalarPerVector>>
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdWeight_Wmma_CShuffleV3";

    static constexpr ck::index_t kNDimSpatial = NDimSpatial;

    using InLayout  = InLayout_;
    using WeiLayout = WeiLayout_;
    using OutLayout = OutLayout_;

    using InDataType  = InDataType_;
    using WeiDataType = WeiDataType_;
    using OutDataType = OutDataType_;
    using AccDataType = AccDataType_;

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
    static constexpr ck::index_t kMaxTransposeTransferSrcScalarPerVector =
        MaxTransposeTransferSrcScalarPerVector;
    static constexpr ck::index_t kMaxTransposeTransferDstScalarPerVector =
        MaxTransposeTransferDstScalarPerVector;

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
        oss << "DeviceGroupedConvBwdWeight_Wmma_CShuffleV3";

        // Template parameters in exact order
        oss << "<" << kNDimSpatial;                     // 1. NDimSpatial
        oss << "," << detail::layout_name<InLayout>();  // 2. InLayout
        oss << "," << detail::layout_name<WeiLayout>(); // 3. WeiLayout
        oss << "," << detail::layout_name<OutLayout>(); // 4. OutLayout
        oss << "," << detail::type_name<InDataType>();  // 5. InDataType
        oss << "," << detail::type_name<WeiDataType>(); // 6. WeiDataType
        oss << "," << detail::type_name<OutDataType>(); // 7. OutDataType
        oss << "," << detail::type_name<AccDataType>(); // 8. AccDataType
        oss << ","
            << detail::elementwise_op_name<InElementwiseOperation>(); // 9. InElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<WeiElementwiseOperation>(); // 10.
                                                                       // WeiElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<OutElementwiseOperation>(); // 11.
                                                                       // OutElementwiseOperation
        oss << ","
            << detail::conv_bwd_weight_spec_name(
                   kConvBackwardWeightSpecialization); // 12. ConvBackwardWeightSpecialization
        oss << "," << kBlockSize;                      // 13. BlockSize
        oss << "," << kMPerBlock;                      // 14. MPerBlock
        oss << "," << kNPerBlock;                      // 15. NPerBlock
        oss << "," << kKPerBlock;                      // 16. KPerBlock
        oss << "," << kABK1;                           // 17. ABK1
        oss << "," << kMPerWmma;                       // 18. MPerWmma
        oss << "," << kNPerWmma;                       // 19. NPerWmma
        oss << "," << kMRepeat;                        // 20. MRepeat
        oss << "," << kNRepeat;                        // 21. NRepeat
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterLengths_AK0_M_AK1>(); // 22.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>();      // 23.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();                 // 24.
        oss << "," << kABlockTransferSrcVectorDim;                                           // 25.
        oss << "," << kABlockTransferSrcScalarPerVector;                                     // 26.
        oss << "," << kABlockTransferDstScalarPerVector_AK1;                                 // 27.
        oss << "," << (kABlockLdsAddExtraM ? "true" : "false");                              // 28.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterLengths_BK0_N_BK1>(); // 29.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>();      // 30.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();                 // 31.
        oss << "," << kBBlockTransferSrcVectorDim;                                           // 32.
        oss << "," << kBBlockTransferSrcScalarPerVector;                                     // 33.
        oss << "," << kBBlockTransferDstScalarPerVector_BK1;                                 // 34.
        oss << "," << (kBBlockLdsAddExtraN ? "true" : "false");                              // 35.
        oss << "," << kCShuffleMRepeatPerShuffle;                                            // 36.
        oss << "," << kCShuffleNRepeatPerShuffle;                                            // 37.
        oss << ","
            << detail::sequence_name<
                   CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>(); // 38.
        oss << "," << kCShuffleBlockTransferScalarPerVector_NPerBlock;                       // 39.
        oss << "," << detail::pipeline_scheduler_name(kBlkGemmPipeSched);                    // 40.
        oss << "," << detail::pipeline_version_name(kBlkGemmPipelineVer);                    // 41.
        oss << "," << detail::type_name<ComputeTypeA>();                                     // 42.
        oss << "," << detail::type_name<ComputeTypeB>();                                     // 43.
        oss << "," << kMaxTransposeTransferSrcScalarPerVector;                               // 44.
        oss << "," << kMaxTransposeTransferDstScalarPerVector;                               // 45.
        oss << ">";

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
