// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// InstanceTraits specialization for DeviceGroupedConvFwdMultipleD_Wmma_CShuffle
//
// CRITICAL MAINTENANCE NOTE:
// This InstanceTraits file MUST be kept strictly in sync with the device implementation header:
//   ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp
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

// Forward declare types from ck namespace that are used in the template parameters
namespace ck {
enum struct PipelineVersion;
enum struct LoopScheduler;
} // namespace ck

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
          ck::index_t K1,
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
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched,
          ck::PipelineVersion PipelineVer>
struct DeviceGroupedConvFwdMultipleD_Wmma_CShuffle;

} // namespace ck::tensor_operation::device

namespace ck_tile::reflect {

// Specialization for DeviceGroupedConvFwdMultipleD_Wmma_CShuffle
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
          ck::index_t K1,
          ck::index_t MPerWmma,
          ck::index_t NPerWmma,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder_,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder_,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          ck::index_t CShuffleMRepeatPerShuffle,
          ck::index_t CShuffleNRepeatPerShuffle,
          typename CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CDEShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::LoopScheduler LoopSched,
          ck::PipelineVersion PipelineVer>
struct InstanceTraits<ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
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
    K1,
    MPerWmma,
    NPerWmma,
    MRepeat,
    NRepeat,
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
    CShuffleMRepeatPerShuffle,
    CShuffleNRepeatPerShuffle,
    CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
    CDEShuffleBlockTransferScalarPerVector_NPerBlock,
    LoopSched,
    PipelineVer>>
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
    static constexpr int kK1       = K1;
    static constexpr int kMPerWmma = MPerWmma;
    static constexpr int kNPerWmma = NPerWmma;
    static constexpr int kMRepeat  = MRepeat;
    static constexpr int kNRepeat  = NRepeat;

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
    static constexpr bool kABlockLdsExtraM                   = ABlockLdsExtraM;

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
    static constexpr bool kBBlockLdsExtraN                   = BBlockLdsExtraN;

    // C shuffle parameters (converted to std::array)
    static constexpr int kCShuffleMRepeatPerShuffle = CShuffleMRepeatPerShuffle;
    static constexpr int kCShuffleNRepeatPerShuffle = CShuffleNRepeatPerShuffle;
    static constexpr auto kCDEThreadClusterLengths  = detail::SequenceToArray<
         CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock>::value;
    static constexpr int kCDEBlockTransferScalarPerVector =
        CDEShuffleBlockTransferScalarPerVector_NPerBlock;

    // Pipeline configuration
    static constexpr ck::LoopScheduler kLoopScheduler     = LoopSched;
    static constexpr ck::PipelineVersion kPipelineVersion = PipelineVer;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "DeviceGroupedConvFwdMultipleD_Wmma_CShuffle";

        // Template parameters in exact order matching InstanceTraits member order
        oss << "<" << kSpatialDim;                           // 1. NDimSpatial
        oss << "," << detail::layout_name<ALayout>();        // 2. ALayout
        oss << "," << detail::layout_name<BLayout>();        // 3. BLayout
        oss << "," << detail::tuple_name<DsLayout>();        // 4. DsLayout
        oss << "," << detail::layout_name<ELayout>();        // 5. ELayout
        oss << "," << detail::type_name<ADataType>();        // 6. ADataType
        oss << "," << detail::type_name<BDataType>();        // 7. BDataType
        oss << "," << detail::type_name<AccDataType>();      // 8. AccDataType
        oss << "," << detail::type_name<CShuffleDataType>(); // 9. CShuffleDataType
        oss << "," << detail::tuple_name<DsDataType>();      // 10. DsDataType
        oss << "," << detail::type_name<EDataType>();        // 11. EDataType
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
        oss << "," << kK1;                                         // 22. K1
        oss << "," << kMPerWmma;                                   // 23. MPerWmma
        oss << "," << kNPerWmma;                                   // 24. NPerWmma
        oss << "," << kMRepeat;                                    // 25. MRepeat
        oss << "," << kNRepeat;                                    // 26. NRepeat
        oss << ","
            << detail::array_to_string(
                   kAThreadClusterLengths); // 27. ABlockTransferThreadClusterLengths
        oss << ","
            << detail::array_to_string(
                   kAThreadClusterArrangeOrder); // 28. ABlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::array_to_string(
                   kABlockTransferSrcAccessOrder);       // 29. ABlockTransferSrcAccessOrder
        oss << "," << kABlockTransferSrcVectorDim;       // 30. ABlockTransferSrcVectorDim
        oss << "," << kABlockTransferSrcScalarPerVector; // 31. ABlockTransferSrcScalarPerVector
        oss << ","
            << kABlockTransferDstScalarPerVectorK1; // 32. ABlockTransferDstScalarPerVector_AK1
        oss << "," << (kABlockLdsExtraM ? "true" : "false"); // 33. ABlockLdsExtraM
        oss << ","
            << detail::array_to_string(
                   kBThreadClusterLengths); // 34. BBlockTransferThreadClusterLengths
        oss << ","
            << detail::array_to_string(
                   kBThreadClusterArrangeOrder); // 35. BBlockTransferThreadClusterArrangeOrder
        oss << ","
            << detail::array_to_string(
                   kBBlockTransferSrcAccessOrder);       // 36. BBlockTransferSrcAccessOrder
        oss << "," << kBBlockTransferSrcVectorDim;       // 37. BBlockTransferSrcVectorDim
        oss << "," << kBBlockTransferSrcScalarPerVector; // 38. BBlockTransferSrcScalarPerVector
        oss << ","
            << kBBlockTransferDstScalarPerVectorK1; // 39. BBlockTransferDstScalarPerVector_BK1
        oss << "," << (kBBlockLdsExtraN ? "true" : "false"); // 40. BBlockLdsExtraN
        oss << "," << kCShuffleMRepeatPerShuffle;            // 41. CShuffleMRepeatPerShuffle
        oss << "," << kCShuffleNRepeatPerShuffle;            // 42. CShuffleNRepeatPerShuffle
        oss << ","
            << detail::array_to_string(
                   kCDEThreadClusterLengths); // 43. CDEShuffleBlockTransferClusterLengths
        oss << ","
            << kCDEBlockTransferScalarPerVector; // 44.
                                                 // CDEShuffleBlockTransferScalarPerVector_NPerBlock
        oss << "," << detail::loop_scheduler_name(kLoopScheduler);     // 45. LoopSched
        oss << "," << detail::pipeline_version_name(kPipelineVersion); // 46. PipelineVer
        oss << ">";

        return oss.str();
    }
};

} // namespace ck_tile::reflect
