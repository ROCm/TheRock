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
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          typename ComputeTypeA,
          typename ComputeTypeB>
struct DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle;

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
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
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
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
          ck::index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          typename ComputeTypeA_,
          typename ComputeTypeB_>
struct InstanceTraits<
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle<
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
        K0PerBlock,
        K1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
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
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        ComputeTypeA_,
        ComputeTypeB_>>
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle";

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

    static constexpr ck::index_t kBlockSize                     = BlockSize;
    static constexpr ck::index_t kMPerBlock                     = MPerBlock;
    static constexpr ck::index_t kNPerBlock                     = NPerBlock;
    static constexpr ck::index_t kK0PerBlock                    = K0PerBlock;
    static constexpr ck::index_t kK1                            = K1;
    static constexpr ck::index_t kMPerXDL                       = MPerXDL;
    static constexpr ck::index_t kNPerXDL                       = NPerXDL;
    static constexpr ck::index_t kMXdlPerWave                   = MXdlPerWave;
    static constexpr ck::index_t kNXdlPerWave                   = NXdlPerWave;
    static constexpr ck::index_t kCShuffleMXdlPerWavePerShuffle = CShuffleMXdlPerWavePerShuffle;
    static constexpr ck::index_t kCShuffleNXdlPerWavePerShuffle = CShuffleNXdlPerWavePerShuffle;
    static constexpr ck::index_t kCBlockTransferScalarPerVector_NWaveNPerXdl =
        CBlockTransferScalarPerVector_NWaveNPerXdl;

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

    using CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_;

    using ComputeTypeA = ComputeTypeA_;
    using ComputeTypeB = ComputeTypeB_;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle";

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
        oss << "," << kK0PerBlock;                     // 18. K0PerBlock
        oss << "," << kK1;                             // 19. K1
        oss << "," << kMPerXDL;                        // 20. MPerXDL
        oss << "," << kNPerXDL;                        // 21. NPerXDL
        oss << "," << kMXdlPerWave;                    // 22. MXdlPerWave
        oss << "," << kNXdlPerWave;                    // 23. NXdlPerWave
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterLengths_K0_M_K1>(); // 24.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>();    // 25.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();               // 26.
        oss << "," << kABlockTransferSrcVectorDim;                                         // 27.
        oss << "," << kABlockTransferSrcScalarPerVector;                                   // 28.
        oss << "," << kABlockTransferDstScalarPerVector_K1;                                // 29.
        oss << "," << (kABlockLdsAddExtraM ? "true" : "false");                            // 30.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterLengths_K0_N_K1>(); // 31.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>();    // 32.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();               // 33.
        oss << "," << kBBlockTransferSrcVectorDim;                                         // 34.
        oss << "," << kBBlockTransferSrcScalarPerVector;                                   // 35.
        oss << "," << kBBlockTransferDstScalarPerVector_K1;                                // 36.
        oss << "," << (kBBlockLdsAddExtraN ? "true" : "false");                            // 37.
        oss << "," << kCShuffleMXdlPerWavePerShuffle;                                      // 38.
        oss << "," << kCShuffleNXdlPerWavePerShuffle;                                      // 39.
        oss << ","
            << detail::sequence_name<
                   CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>(); // 40.
        oss << "," << kCBlockTransferScalarPerVector_NWaveNPerXdl;                    // 41.
        oss << "," << detail::type_name<ComputeTypeA>();                              // 42.
        oss << "," << detail::type_name<ComputeTypeB>();                              // 43.
        oss << ">";

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
