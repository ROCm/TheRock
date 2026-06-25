// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// InstanceTraits specialization for DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
//
// CRITICAL MAINTENANCE NOTE:
// This InstanceTraits file MUST be kept strictly in sync with the device implementation header:
//   ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp
// "In sync" means that the template parameter order, names, and types in the declaration below
// MUST EXACTLY MATCH those in the device implementation. If these diverge, you may encounter
// compilation errors, subtle template instantiation mismatches, or silent runtime bugs that are
// difficult to diagnose. Always update both files together and review changes carefully.

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"

// Forward declaration to avoid circular dependency.
// This file will be included by the device implementation header, so we cannot include
// the implementation header here. We only need the template signature to pattern-match
// on template parameters - we don't need any implementation details.

namespace ck::tensor_operation::device {

template <index_t NDimSpatial,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
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
struct DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {

// Specialization for DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
template <ck::index_t NDimSpatial,
          typename ADataType_,
          typename BDataType_,
          typename DsDataType_,
          typename EDataType_,
          typename AccDataType_,
          typename ALayout_,
          typename BLayout_,
          typename DsLayout_,
          typename ELayout_,
          typename AElementwiseOperation_,
          typename BElementwiseOperation_,
          typename CDEElementwiseOperation_,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
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
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
    NDimSpatial,
    ADataType_,
    BDataType_,
    DsDataType_,
    EDataType_,
    AccDataType_,
    ALayout_,
    BLayout_,
    DsLayout_,
    ELayout_,
    AElementwiseOperation_,
    BElementwiseOperation_,
    CDEElementwiseOperation_,
    ConvForwardSpecialization,
    GemmSpec,
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
    // Spatial dimension
    static constexpr int kSpatialDim = NDimSpatial;

    // Data types
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using DsDataType  = DsDataType_;
    using EDataType   = EDataType_;
    using AccDataType = AccDataType_;

    // Layout types
    using ALayout  = ALayout_;
    using BLayout  = BLayout_;
    using DsLayout = DsLayout_;
    using ELayout  = ELayout_;

    // Element-wise operations
    using AElementwiseOperation   = AElementwiseOperation_;
    using BElementwiseOperation   = BElementwiseOperation_;
    using CDEElementwiseOperation = CDEElementwiseOperation_;

    // Specialization
    static constexpr ck::tensor_operation::device::ConvolutionForwardSpecialization
        kConvForwardSpecialization = ConvForwardSpecialization;
    static constexpr ck::tensor_operation::device::GemmSpecialization kGemmSpecialization =
        GemmSpec;

    // Block configuration
    static constexpr int kBlockSize  = BlockSize;
    static constexpr int kMPerBlock  = MPerBlock;
    static constexpr int kNPerBlock  = NPerBlock;
    static constexpr int kK0PerBlock = K0PerBlock;

    // Tuning parameters
    static constexpr int kK1          = K1;
    static constexpr int kM1PerThread = M1PerThread;
    static constexpr int kN1PerThread = N1PerThread;
    static constexpr int kKPerThread  = KPerThread;

    // Thread cluster configurations
    using M1N1ThreadClusterM1Xs = M1N1ThreadClusterM1Xs_;
    using M1N1ThreadClusterN1Xs = M1N1ThreadClusterN1Xs_;

    // A block transfer parameters
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

    // B block transfer parameters
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

    // C thread transfer parameters
    using CThreadTransferSrcDstAccessOrder                  = CThreadTransferSrcDstAccessOrder_;
    static constexpr int kCThreadTransferSrcDstVectorDim    = CThreadTransferSrcDstVectorDim;
    static constexpr int kCThreadTransferDstScalarPerVector = CThreadTransferDstScalarPerVector;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK";

        // Template parameters in exact order matching the device implementation
        oss << "<" << kSpatialDim;                      // 1. NDimSpatial
        oss << "," << detail::type_name<ADataType>();   // 2. ADataType
        oss << "," << detail::type_name<BDataType>();   // 3. BDataType
        oss << "," << detail::tuple_name<DsDataType>(); // 4. DsDataType
        oss << "," << detail::type_name<EDataType>();   // 5. EDataType
        oss << "," << detail::type_name<AccDataType>(); // 6. AccDataType
        oss << "," << detail::layout_name<ALayout>();   // 7. ALayout
        oss << "," << detail::layout_name<BLayout>();   // 8. BLayout
        oss << "," << detail::tuple_name<DsLayout>();   // 9. DsLayout
        oss << "," << detail::layout_name<ELayout>();   // 10. ELayout
        oss << ","
            << detail::elementwise_op_name<AElementwiseOperation>(); // 11. AElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<BElementwiseOperation>(); // 12. BElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<CDEElementwiseOperation>(); // 13.
                                                                       // CDEElementwiseOperation
        oss << ","
            << detail::conv_fwd_spec_name(
                   kConvForwardSpecialization);                    // 14. ConvForwardSpecialization
        oss << "," << detail::gemm_spec_name(kGemmSpecialization); // 15. GemmSpec
        oss << "," << kBlockSize;                                  // 16. BlockSize
        oss << "," << kMPerBlock;                                  // 17. MPerBlock
        oss << "," << kNPerBlock;                                  // 18. NPerBlock
        oss << "," << kK0PerBlock;                                 // 19. K0PerBlock
        oss << "," << kK1;                                         // 20. K1
        oss << "," << kM1PerThread;                                // 21. M1PerThread
        oss << "," << kN1PerThread;                                // 22. N1PerThread
        oss << "," << kKPerThread;                                 // 23. KPerThread
        oss << "," << detail::sequence_name<M1N1ThreadClusterM1Xs>(); // 24. M1N1ThreadClusterM1Xs
        oss << "," << detail::sequence_name<M1N1ThreadClusterN1Xs>(); // 25. M1N1ThreadClusterN1Xs
        oss << ","
            << detail::sequence_name<
                   ABlockTransferThreadSliceLengths_K0_M0_M1_K1>(); // 26.
                                                                    // ABlockTransferThreadSliceLengths_K0_M0_M1_K1
        oss << ","
            << detail::sequence_name<
                   ABlockTransferThreadClusterLengths_K0_M0_M1_K1>(); // 27.
                                                                      // ABlockTransferThreadClusterLengths_K0_M0_M1_K1
        oss << ","
            << detail::sequence_name<
                   ABlockTransferThreadClusterArrangeOrder>(); // 28.
                                                               // ABlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::sequence_name<
                   ABlockTransferSrcAccessOrder>(); // 29. ABlockTransferSrcAccessOrder
        oss << ","
            << detail::sequence_name<
                   ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1>(); // 30.
                                                                        // ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1
        oss << ","
            << detail::sequence_name<
                   ABlockTransferSrcVectorTensorContiguousDimOrder>(); // 31.
                                                                       // ABlockTransferSrcVectorTensorContiguousDimOrder
        oss << ","
            << detail::sequence_name<
                   ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1>(); // 32.
                                                                        // ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1
        oss << ","
            << detail::sequence_name<
                   BBlockTransferThreadSliceLengths_K0_N0_N1_K1>(); // 33.
                                                                    // BBlockTransferThreadSliceLengths_K0_N0_N1_K1
        oss << ","
            << detail::sequence_name<
                   BBlockTransferThreadClusterLengths_K0_N0_N1_K1>(); // 34.
                                                                      // BBlockTransferThreadClusterLengths_K0_N0_N1_K1
        oss << ","
            << detail::sequence_name<
                   BBlockTransferThreadClusterArrangeOrder>(); // 35.
                                                               // BBlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::sequence_name<
                   BBlockTransferSrcAccessOrder>(); // 36. BBlockTransferSrcAccessOrder
        oss << ","
            << detail::sequence_name<
                   BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1>(); // 37.
                                                                        // BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1
        oss << ","
            << detail::sequence_name<
                   BBlockTransferSrcVectorTensorContiguousDimOrder>(); // 38.
                                                                       // BBlockTransferSrcVectorTensorContiguousDimOrder
        oss << ","
            << detail::sequence_name<
                   BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1>(); // 39.
                                                                        // BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1
        oss << ","
            << detail::sequence_name<
                   CThreadTransferSrcDstAccessOrder>();   // 40. CThreadTransferSrcDstAccessOrder
        oss << "," << kCThreadTransferSrcDstVectorDim;    // 41. CThreadTransferSrcDstVectorDim
        oss << "," << kCThreadTransferDstScalarPerVector; // 42. CThreadTransferDstScalarPerVector
        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
