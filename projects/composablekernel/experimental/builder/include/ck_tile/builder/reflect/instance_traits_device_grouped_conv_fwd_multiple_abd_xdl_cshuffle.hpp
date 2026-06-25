// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// InstanceTraits specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
//
// CRITICAL MAINTENANCE NOTE:
// This InstanceTraits file MUST be kept strictly in sync with the device implementation header:
//   ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp
// "In sync" means that the template parameter order, names, and types in the declaration below
// MUST EXACTLY MATCH those in the device implementation. If these diverge, you may encounter
// compilation errors, subtle template instantiation mismatches, or silent runtime bugs that are
// difficult to diagnose. Always update both files together and review changes carefully.

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"

// Forward declaration to avoid circular dependency.
namespace ck::tensor_operation::device {

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          typename AComputeDataType,
          typename BComputeDataType,
          ck::LoopScheduler LoopSched,
          ck::index_t NumGroupsToMerge>
struct DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {

// Specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
template <ck::index_t NDimSpatial,
          typename ALayout_,
          typename BLayout_,
          typename DsLayout_,
          typename ELayout_,
          typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CShuffleDataType_,
          typename DsDataType_,
          typename EDataType_,
          typename AElementwiseOperation_,
          typename BElementwiseOperation_,
          typename CDEElementwiseOperation_,
          ck::tensor_operation::device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::tensor_operation::device::GemmSpecialization GemmSpec,
          ck::index_t NumGemmKPrefetchStage,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder_,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          ck::index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder_,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          ck::index_t BBlockLdsExtraN,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEBlockTransferScalarPerVector_NPerBlock,
          typename AComputeDataType_,
          typename BComputeDataType_,
          ck::LoopScheduler LoopSched,
          ck::index_t NumGroupsToMerge>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial,
    ALayout_,
    BLayout_,
    DsLayout_,
    ELayout_,
    ADataType_,
    BDataType_,
    AccDataType_,
    CShuffleDataType_,
    DsDataType_,
    EDataType_,
    AElementwiseOperation_,
    BElementwiseOperation_,
    CDEElementwiseOperation_,
    ConvForwardSpecialization,
    GemmSpec,
    NumGemmKPrefetchStage,
    BlockSize,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    AK1,
    BK1,
    MPerXDL,
    NPerXDL,
    MXdlPerWave,
    NXdlPerWave,
    ABlockTransferThreadClusterLengths_AK0_M_AK1,
    ABlockTransferThreadClusterArrangeOrder,
    ABlockTransferSrcAccessOrder_,
    ABlockTransferSrcVectorDim,
    ABlockTransferSrcScalarPerVector,
    ABlockTransferDstScalarPerVector_AK1,
    ABlockLdsExtraM,
    BBlockTransferThreadClusterLengths_BK0_N_BK1,
    BBlockTransferThreadClusterArrangeOrder,
    BBlockTransferSrcAccessOrder_,
    BBlockTransferSrcVectorDim,
    BBlockTransferSrcScalarPerVector,
    BBlockTransferDstScalarPerVector_BK1,
    BBlockLdsExtraN,
    CShuffleMXdlPerWavePerShuffle,
    CShuffleNXdlPerWavePerShuffle,
    CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CDEBlockTransferScalarPerVector_NPerBlock,
    AComputeDataType_,
    BComputeDataType_,
    LoopSched,
    NumGroupsToMerge>>
{
    // Spatial dimension
    static constexpr int kSpatialDim = NDimSpatial;

    // Layout types
    using ALayout  = ALayout_;
    using BLayout  = BLayout_;
    using DsLayout = DsLayout_;
    using ELayout  = ELayout_;

    // Data types
    using ADataType        = ADataType_;
    using BDataType        = BDataType_;
    using AccDataType      = AccDataType_;
    using CShuffleDataType = CShuffleDataType_;
    using DsDataType       = DsDataType_;
    using EDataType        = EDataType_;

    // Element-wise operations
    using AElementwiseOperation   = AElementwiseOperation_;
    using BElementwiseOperation   = BElementwiseOperation_;
    using CDEElementwiseOperation = CDEElementwiseOperation_;

    // Specialization
    static constexpr ck::tensor_operation::device::ConvolutionForwardSpecialization
        kConvForwardSpecialization = ConvForwardSpecialization;
    static constexpr ck::tensor_operation::device::GemmSpecialization kGemmSpecialization =
        GemmSpec;

    // Prefetch stage
    static constexpr int kNumGemmKPrefetchStage = NumGemmKPrefetchStage;

    // Block configuration
    static constexpr int kBlockSize = BlockSize;
    static constexpr int kMPerBlock = MPerBlock;
    static constexpr int kNPerBlock = NPerBlock;
    static constexpr int kKPerBlock = KPerBlock;

    // Tuning parameters
    static constexpr int kAK1         = AK1;
    static constexpr int kBK1         = BK1;
    static constexpr int kMPerXDL     = MPerXDL;
    static constexpr int kNPerXDL     = NPerXDL;
    static constexpr int kMXdlPerWave = MXdlPerWave;
    static constexpr int kNXdlPerWave = NXdlPerWave;

    // A block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kAThreadClusterLengths =
        detail::SequenceToArray<ABlockTransferThreadClusterLengths_AK0_M_AK1>::value;
    static constexpr auto kAThreadClusterArrangeOrder =
        detail::SequenceToArray<ABlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kABlockTransferSrcAccessOrder =
        detail::SequenceToArray<ABlockTransferSrcAccessOrder_>::value;
    static constexpr int kABlockTransferSrcVectorDim         = ABlockTransferSrcVectorDim;
    static constexpr int kABlockTransferSrcScalarPerVector   = ABlockTransferSrcScalarPerVector;
    static constexpr int kABlockTransferDstScalarPerVectorK1 = ABlockTransferDstScalarPerVector_AK1;
    static constexpr int kABlockLdsExtraM                    = ABlockLdsExtraM;

    // B block transfer thread cluster dimensions (converted to std::array)
    static constexpr auto kBThreadClusterLengths =
        detail::SequenceToArray<BBlockTransferThreadClusterLengths_BK0_N_BK1>::value;
    static constexpr auto kBThreadClusterArrangeOrder =
        detail::SequenceToArray<BBlockTransferThreadClusterArrangeOrder>::value;
    static constexpr auto kBBlockTransferSrcAccessOrder =
        detail::SequenceToArray<BBlockTransferSrcAccessOrder_>::value;
    static constexpr int kBBlockTransferSrcVectorDim         = BBlockTransferSrcVectorDim;
    static constexpr int kBBlockTransferSrcScalarPerVector   = BBlockTransferSrcScalarPerVector;
    static constexpr int kBBlockTransferDstScalarPerVectorK1 = BBlockTransferDstScalarPerVector_BK1;
    static constexpr int kBBlockLdsExtraN                    = BBlockLdsExtraN;

    // C shuffle parameters (converted to std::array)
    static constexpr int kCShuffleMXdlPerWavePerShuffle = CShuffleMXdlPerWavePerShuffle;
    static constexpr int kCShuffleNXdlPerWavePerShuffle = CShuffleNXdlPerWavePerShuffle;
    static constexpr auto kCThreadClusterLengths        = detail::SequenceToArray<
               CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>::value;
    static constexpr int kCBlockTransferScalarPerVector = CDEBlockTransferScalarPerVector_NPerBlock;

    // Compute data types
    using AComputeDataType = AComputeDataType_;
    using BComputeDataType = BComputeDataType_;

    // Loop scheduler
    static constexpr ck::LoopScheduler kLoopScheduler = LoopSched;

    // Groups to merge
    static constexpr int kNumGroupsToMerge = NumGroupsToMerge;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle";

        // Template parameters in exact order matching InstanceTraits member order
        oss << "<" << kSpatialDim;                                  // 1. NDimSpatial
        oss << "," << detail::layout_name<ALayout>();               // 2. ALayout
        oss << "," << detail::layout_name<BLayout>();               // 3. BLayout
        oss << "," << detail::tuple_name<DsLayout>();               // 4. DsLayout
        oss << "," << detail::layout_name<ELayout>();               // 5. ELayout
        oss << "," << detail::type_or_type_tuple_name<ADataType>(); // 6. ADataType
        oss << "," << detail::type_or_type_tuple_name<BDataType>(); // 7. BDataType
        oss << "," << detail::type_name<AccDataType>();             // 8. AccDataType
        oss << "," << detail::type_name<CShuffleDataType>();        // 9. CShuffleDataType
        oss << "," << detail::tuple_name<DsDataType>();             // 10. DsDataType
        oss << "," << detail::type_name<EDataType>();               // 11. EDataType
        oss << ","
            << detail::elementwise_op_name<AElementwiseOperation>(); // 12. AElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<BElementwiseOperation>(); // 13. BElementwiseOperation
        oss << ","
            << detail::elementwise_op_name<CDEElementwiseOperation>(); // 14.
                                                                       // CDEElementwiseOperation
        oss << ","
            << detail::conv_fwd_spec_name(
                   kConvForwardSpecialization);                    // 15. ConvForwardSpecialization
        oss << "," << detail::gemm_spec_name(kGemmSpecialization); // 16. GemmSpec
        oss << "," << kNumGemmKPrefetchStage;                      // 17. NumGemmKPrefetchStage
        oss << "," << kBlockSize;                                  // 18. BlockSize
        oss << "," << kMPerBlock;                                  // 19. MPerBlock
        oss << "," << kNPerBlock;                                  // 20. NPerBlock
        oss << "," << kKPerBlock;                                  // 21. KPerBlock
        oss << "," << kAK1;                                        // 22. AK1
        oss << "," << kBK1;                                        // 23. BK1
        oss << "," << kMPerXDL;                                    // 24. MPerXDL
        oss << "," << kNPerXDL;                                    // 25. NPerXDL
        oss << "," << kMXdlPerWave;                                // 26. MXdlPerWave
        oss << "," << kNXdlPerWave;                                // 27. NXdlPerWave
        oss << ","
            << detail::array_to_string(
                   kAThreadClusterLengths); // 28. ABlockTransferThreadClusterLengths
        oss << ","
            << detail::array_to_string(
                   kAThreadClusterArrangeOrder); // 29. ABlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::array_to_string(
                   kABlockTransferSrcAccessOrder);       // 30. ABlockTransferSrcAccessOrder
        oss << "," << kABlockTransferSrcVectorDim;       // 31. ABlockTransferSrcVectorDim
        oss << "," << kABlockTransferSrcScalarPerVector; // 32. ABlockTransferSrcScalarPerVector
        oss << ","
            << kABlockTransferDstScalarPerVectorK1; // 33. ABlockTransferDstScalarPerVector_AK1
        oss << "," << (kABlockLdsExtraM ? "true" : "false"); // 34. ABlockLdsExtraM
        oss << ","
            << detail::array_to_string(
                   kBThreadClusterLengths); // 35. BBlockTransferThreadClusterLengths
        oss << ","
            << detail::array_to_string(
                   kBThreadClusterArrangeOrder); // 36. BBlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::array_to_string(
                   kBBlockTransferSrcAccessOrder);       // 37. BBlockTransferSrcAccessOrder
        oss << "," << kBBlockTransferSrcVectorDim;       // 38. BBlockTransferSrcVectorDim
        oss << "," << kBBlockTransferSrcScalarPerVector; // 39. BBlockTransferSrcScalarPerVector
        oss << ","
            << kBBlockTransferDstScalarPerVectorK1; // 40. BBlockTransferDstScalarPerVector_BK1
        oss << "," << (kBBlockLdsExtraN ? "true" : "false"); // 41. BBlockLdsExtraN
        oss << "," << kCShuffleMXdlPerWavePerShuffle;        // 42. CShuffleMXdlPerWavePerShuffle
        oss << "," << kCShuffleNXdlPerWavePerShuffle;        // 43. CShuffleNXdlPerWavePerShuffle
        oss << ","
            << detail::array_to_string(
                   kCThreadClusterLengths); // 44. CDEBlockTransferClusterLengths
        oss << ","
            << kCBlockTransferScalarPerVector; // 45. CDEBlockTransferScalarPerVector_NPerBlock
        oss << "," << detail::type_name<AComputeDataType>();       // 46. AComputeDataType
        oss << "," << detail::type_name<BComputeDataType>();       // 47. BComputeDataType
        oss << "," << detail::loop_scheduler_name(kLoopScheduler); // 48. LoopSched
        oss << "," << kNumGroupsToMerge;                           // 49. NumGroupsToMerge
        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
