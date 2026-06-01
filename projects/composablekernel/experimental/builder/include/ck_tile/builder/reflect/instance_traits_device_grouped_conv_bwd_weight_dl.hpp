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
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t M1PerThread,
          ck::index_t N1PerThread,
          ck::index_t KPerThread,
          typename M1N1ThreadClusterM1Xs,
          typename M1N1ThreadClusterN1Xs,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
          typename CThreadTransferSrcDstAccessOrder,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceGroupedConvBwdWeight_Dl;

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
          ck::index_t M1PerThread,
          ck::index_t N1PerThread,
          ck::index_t KPerThread,
          typename M1N1ThreadClusterM1Xs_,
          typename M1N1ThreadClusterN1Xs_,
          typename ABlockTransferThreadSliceLengths_K0_M0_M1_K1_,
          typename ABlockTransferThreadClusterLengths_K0_M0_M1_K1_,
          typename ABlockTransferThreadClusterArrangeOrder_,
          typename ABlockTransferSrcAccessOrder_,
          typename ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1_,
          typename ABlockTransferSrcVectorTensorContiguousDimOrder_,
          typename ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1_,
          typename BBlockTransferThreadSliceLengths_K0_N0_N1_K1_,
          typename BBlockTransferThreadClusterLengths_K0_N0_N1_K1_,
          typename BBlockTransferThreadClusterArrangeOrder_,
          typename BBlockTransferSrcAccessOrder_,
          typename BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1_,
          typename BBlockTransferSrcVectorTensorContiguousDimOrder_,
          typename BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1_,
          typename CThreadTransferSrcDstAccessOrder_,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Dl<
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
    M1PerThread,
    N1PerThread,
    KPerThread,
    M1N1ThreadClusterM1Xs_,
    M1N1ThreadClusterN1Xs_,
    ABlockTransferThreadSliceLengths_K0_M0_M1_K1_,
    ABlockTransferThreadClusterLengths_K0_M0_M1_K1_,
    ABlockTransferThreadClusterArrangeOrder_,
    ABlockTransferSrcAccessOrder_,
    ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1_,
    ABlockTransferSrcVectorTensorContiguousDimOrder_,
    ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1_,
    BBlockTransferThreadSliceLengths_K0_N0_N1_K1_,
    BBlockTransferThreadClusterLengths_K0_N0_N1_K1_,
    BBlockTransferThreadClusterArrangeOrder_,
    BBlockTransferSrcAccessOrder_,
    BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1_,
    BBlockTransferSrcVectorTensorContiguousDimOrder_,
    BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1_,
    CThreadTransferSrcDstAccessOrder_,
    CThreadTransferSrcDstVectorDim,
    CThreadTransferDstScalarPerVector>>
{
    static constexpr auto kTensorOpName = "DeviceGroupedConvBwdWeight_Dl";

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

    static constexpr ck::index_t kBlockSize   = BlockSize;
    static constexpr ck::index_t kMPerBlock   = MPerBlock;
    static constexpr ck::index_t kNPerBlock   = NPerBlock;
    static constexpr ck::index_t kK0PerBlock  = K0PerBlock;
    static constexpr ck::index_t kK1          = K1;
    static constexpr ck::index_t kM1PerThread = M1PerThread;
    static constexpr ck::index_t kN1PerThread = N1PerThread;
    static constexpr ck::index_t kKPerThread  = KPerThread;

    using M1N1ThreadClusterM1Xs = M1N1ThreadClusterM1Xs_;
    using M1N1ThreadClusterN1Xs = M1N1ThreadClusterN1Xs_;

    using ABlockTransferThreadSliceLengths_K0_M0_M1_K1 =
        ABlockTransferThreadSliceLengths_K0_M0_M1_K1_;
    using ABlockTransferThreadClusterLengths_K0_M0_M1_K1 =
        ABlockTransferThreadClusterLengths_K0_M0_M1_K1_;
    using ABlockTransferThreadClusterArrangeOrder = ABlockTransferThreadClusterArrangeOrder_;
    using ABlockTransferSrcAccessOrder            = ABlockTransferSrcAccessOrder_;
    using ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 =
        ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1_;
    using ABlockTransferSrcVectorTensorContiguousDimOrder =
        ABlockTransferSrcVectorTensorContiguousDimOrder_;
    using ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 =
        ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1_;

    using BBlockTransferThreadSliceLengths_K0_N0_N1_K1 =
        BBlockTransferThreadSliceLengths_K0_N0_N1_K1_;
    using BBlockTransferThreadClusterLengths_K0_N0_N1_K1 =
        BBlockTransferThreadClusterLengths_K0_N0_N1_K1_;
    using BBlockTransferThreadClusterArrangeOrder = BBlockTransferThreadClusterArrangeOrder_;
    using BBlockTransferSrcAccessOrder            = BBlockTransferSrcAccessOrder_;
    using BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 =
        BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1_;
    using BBlockTransferSrcVectorTensorContiguousDimOrder =
        BBlockTransferSrcVectorTensorContiguousDimOrder_;
    using BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 =
        BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1_;

    using CThreadTransferSrcDstAccessOrder = CThreadTransferSrcDstAccessOrder_;
    static constexpr ck::index_t kCThreadTransferSrcDstVectorDim = CThreadTransferSrcDstVectorDim;
    static constexpr ck::index_t kCThreadTransferDstScalarPerVector =
        CThreadTransferDstScalarPerVector;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvBwdWeight_Dl";

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
        oss << "," << kM1PerThread;                    // 18. M1PerThread
        oss << "," << kN1PerThread;                    // 19. N1PerThread
        oss << "," << kKPerThread;                     // 20. KPerThread
        oss << "," << detail::sequence_name<M1N1ThreadClusterM1Xs>();                        // 21.
        oss << "," << detail::sequence_name<M1N1ThreadClusterN1Xs>();                        // 22.
        oss << "," << detail::sequence_name<ABlockTransferThreadSliceLengths_K0_M0_M1_K1>(); // 23.
        oss << ","
            << detail::sequence_name<ABlockTransferThreadClusterLengths_K0_M0_M1_K1>(); // 24.
        oss << "," << detail::sequence_name<ABlockTransferThreadClusterArrangeOrder>(); // 25.
        oss << "," << detail::sequence_name<ABlockTransferSrcAccessOrder>();            // 26.
        oss << ","
            << detail::sequence_name<ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1>(); // 27.
        oss << ","
            << detail::sequence_name<ABlockTransferSrcVectorTensorContiguousDimOrder>(); // 28.
        oss << ","
            << detail::sequence_name<ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1>();    // 29.
        oss << "," << detail::sequence_name<BBlockTransferThreadSliceLengths_K0_N0_N1_K1>(); // 30.
        oss << ","
            << detail::sequence_name<BBlockTransferThreadClusterLengths_K0_N0_N1_K1>(); // 31.
        oss << "," << detail::sequence_name<BBlockTransferThreadClusterArrangeOrder>(); // 32.
        oss << "," << detail::sequence_name<BBlockTransferSrcAccessOrder>();            // 33.
        oss << ","
            << detail::sequence_name<BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1>(); // 34.
        oss << ","
            << detail::sequence_name<BBlockTransferSrcVectorTensorContiguousDimOrder>(); // 35.
        oss << ","
            << detail::sequence_name<BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1>(); // 36.
        oss << "," << detail::sequence_name<CThreadTransferSrcDstAccessOrder>();          // 37.
        oss << "," << kCThreadTransferSrcDstVectorDim;                                    // 38.
        oss << "," << kCThreadTransferDstScalarPerVector;                                 // 39.
        oss << ">";

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
