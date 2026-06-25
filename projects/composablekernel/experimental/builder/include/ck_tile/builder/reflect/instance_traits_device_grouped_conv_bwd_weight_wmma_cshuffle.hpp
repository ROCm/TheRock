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
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerWMMA,
          ck::index_t NPerWMMA,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::index_t NumGemmKPrefetchStage,
          ck::LoopScheduler LoopSched,
          ck::PipelineVersion PipelineVer,
          typename ck::enable_if<NDimSpatial == 3, bool>::type>
struct DeviceGroupedConvBwdWeight_Wmma_CShuffle;

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
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerWMMA,
          ck::index_t NPerWMMA,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1_,
          typename ABlockTransferThreadClusterArrangeOrder_,
          typename ABlockTransferSrcAccessOrder_,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1_,
          typename BBlockTransferThreadClusterArrangeOrder_,
          typename BBlockTransferSrcAccessOrder_,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::index_t NumGemmKPrefetchStage,
          ck::LoopScheduler LoopSched,
          ck::PipelineVersion PipelineVer>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Wmma_CShuffle<
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
    K0PerBlock,
    K1,
    MPerWMMA,
    NPerWMMA,
    MRepeat,
    NRepeat,
    ABlockTransferThreadClusterLengths_K0_M_K1_,
    ABlockTransferThreadClusterArrangeOrder_,
    ABlockTransferSrcAccessOrder_,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_K1,
    ABlockLdsAddExtraM,
    BBlockTransferThreadClusterLengths_K0_N_K1_,
    BBlockTransferThreadClusterArrangeOrder_,
    BBlockTransferSrcAccessOrder_,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_K1,
    BBlockLdsAddExtraN,
    CShuffleMRepeatPerShuffle,
    CShuffleNRepeatPerShuffle,
    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
    CShuffleBlockTransferScalarPerVector_NPerBlock,
    NumGemmKPrefetchStage,
    LoopSched,
    PipelineVer,
    false>> // Use false to match with the default value
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdWeight_Wmma_CShuffle";

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
    static constexpr ck::index_t kK0PerBlock                = K0PerBlock;
    static constexpr ck::index_t kK1                        = K1;
    static constexpr ck::index_t kMPerWMMA                  = MPerWMMA;
    static constexpr ck::index_t kNPerWMMA                  = NPerWMMA;
    static constexpr ck::index_t kMRepeat                   = MRepeat;
    static constexpr ck::index_t kNRepeat                   = NRepeat;
    static constexpr ck::index_t kCShuffleMRepeatPerShuffle = CShuffleMRepeatPerShuffle;
    static constexpr ck::index_t kCShuffleNRepeatPerShuffle = CShuffleNRepeatPerShuffle;
    static constexpr ck::index_t kCShuffleBlockTransferScalarPerVector_NPerBlock =
        CShuffleBlockTransferScalarPerVector_NPerBlock;
    static constexpr ck::index_t kNumGemmKPrefetchStage = NumGemmKPrefetchStage;

    using ABlockTransferThreadClusterLengths_K0_M_K1 = ABlockTransferThreadClusterLengths_K0_M_K1_;
    using ABlockTransferThreadClusterArrangeOrder    = ABlockTransferThreadClusterArrangeOrder_;
    using ABlockTransferSrcAccessOrder               = ABlockTransferSrcAccessOrder_;
    static constexpr ck::index_t kABlockTransferSrcVectorDim = ABlockTransferSrcVectorDim;
    static constexpr ck::index_t kABlockTransferSrcScalarPerVector =
        ABlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kABlockTransferDstScalarPerVector_K1 =
        ABlockTransferDstScalarPerVector_K1;
    static constexpr bool kABlockLdsAddExtraM = ABlockLdsAddExtraM;

    using BBlockTransferThreadClusterLengths_K0_N_K1 = BBlockTransferThreadClusterLengths_K0_N_K1_;
    using BBlockTransferThreadClusterArrangeOrder    = BBlockTransferThreadClusterArrangeOrder_;
    using BBlockTransferSrcAccessOrder               = BBlockTransferSrcAccessOrder_;
    static constexpr ck::index_t kBBlockTransferSrcVectorDim = BBlockTransferSrcVectorDim;
    static constexpr ck::index_t kBBlockTransferSrcScalarPerVector =
        BBlockTransferSrcScalarPerVector;
    static constexpr ck::index_t kBBlockTransferDstScalarPerVector_K1 =
        BBlockTransferDstScalarPerVector_K1;
    static constexpr bool kBBlockLdsAddExtraN = BBlockLdsAddExtraN;

    using CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock =
        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_;

    static constexpr ck::LoopScheduler kLoopSched     = LoopSched;
    static constexpr ck::PipelineVersion kPipelineVer = PipelineVer;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvBwdWeight_Wmma_CShuffle";

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
        oss << "," << kK0PerBlock;                     // 16. K0PerBlock
        oss << "," << kK1;                             // 17. K1
        oss << "," << kMPerWMMA;                       // 18. MPerWMMA
        oss << "," << kNPerWMMA;                       // 19. NPerWMMA
        oss << "," << kMRepeat;                        // 20. MRepeat
        oss << "," << kNRepeat;                        // 21. NRepeat
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterLengths_K0_M_K1>(); // 22.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>();    // 23.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();               // 24.
        oss << "," << kABlockTransferSrcVectorDim;                                         // 25.
        oss << "," << kABlockTransferSrcScalarPerVector;                                   // 26.
        oss << "," << kABlockTransferDstScalarPerVector_K1;                                // 27.
        oss << "," << (kABlockLdsAddExtraM ? "true" : "false");                            // 28.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterLengths_K0_N_K1>(); // 29.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>();    // 30.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();               // 31.
        oss << "," << kBBlockTransferSrcVectorDim;                                         // 32.
        oss << "," << kBBlockTransferSrcScalarPerVector;                                   // 33.
        oss << "," << kBBlockTransferDstScalarPerVector_K1;                                // 34.
        oss << "," << (kBBlockLdsAddExtraN ? "true" : "false");                            // 35.
        oss << "," << kCShuffleMRepeatPerShuffle;                                          // 36.
        oss << "," << kCShuffleNRepeatPerShuffle;                                          // 37.
        oss << ","
            << detail::sequence_name<
                   CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>(); // 38.
        oss << "," << kCShuffleBlockTransferScalarPerVector_NPerBlock;                       // 39.
        oss << "," << kNumGemmKPrefetchStage;                                                // 40.
        oss << "," << detail::loop_scheduler_name(kLoopSched);                               // 41.
        oss << "," << detail::pipeline_version_name(kPipelineVer);                           // 42.
        oss << ">";

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
