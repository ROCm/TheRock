// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// InstanceTraits specialization for GroupedConvolutionForwardKernel
//
// CRITICAL MAINTENANCE NOTE:
// This InstanceTraits file MUST be kept strictly in sync with the device implementation header:
//   ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp
// "In sync" means that the template parameter order, names, and types in the declaration below
// MUST EXACTLY MATCH those in the device implementation. If these diverge, you may encounter
// compilation errors, subtle template instantiation mismatches, or silent runtime bugs that are
// difficult to diagnose. Always update both files together and review changes carefully.

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"

// Forward declaration to avoid circular dependency.
namespace ck_tile {

template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionForwardKernel;

} // namespace ck_tile

namespace ck_tile {
namespace reflect {

// Specialization for GroupedConvolutionForwardKernel
template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct InstanceTraits<ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType_,
                                                               TilePartitioner_,
                                                               GemmPipeline_,
                                                               EpiloguePipeline_>>
{
    // CK Tile Conv Traits
    // Spatial dimension
    static constexpr int kSpatialDim = GroupedConvTraitsType_::NDimSpatial;
    // Specialization
    static constexpr ck_tile::ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType_::ConvSpecialization;
    // DataType types
    using InLayout  = typename GroupedConvTraitsType_::InLayout;
    using WeiLayout = typename GroupedConvTraitsType_::WeiLayout;
    using DsLayout  = typename GroupedConvTraitsType_::DsLayout;
    using OutLayout = typename GroupedConvTraitsType_::OutLayout;
    // Vector size
    static constexpr int kVectorSizeA = GroupedConvTraitsType_::VectorSizeA;
    static constexpr int kVectorSizeB = GroupedConvTraitsType_::VectorSizeB;
    static constexpr int kVectorSizeC = GroupedConvTraitsType_::VectorSizeC;
    // Num Groups To Merge
    static constexpr int kNumGroupsToMerge = GroupedConvTraitsType_::NumGroupsToMerge;
    // Split image (large tensors)
    static constexpr bool kEnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
    // Explicit GEMM
    static constexpr int kExplicitGemm = GroupedConvTraitsType_::ExplicitGemm;

    // TilePartitioner
    // Block configuration
    static constexpr int kMPerBlock = TilePartitioner_::MPerBlock;
    static constexpr int kNPerBlock = TilePartitioner_::NPerBlock;
    static constexpr int kKPerBlock = TilePartitioner_::KPerBlock;

    static constexpr int kMWarp = TilePartitioner_::BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr int kNWarp = TilePartitioner_::BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr int kKWarp = TilePartitioner_::BlockGemmShape::BlockWarps::at(number<2>{});

    static constexpr int kMWarpTile = TilePartitioner_::BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int kNWarpTile = TilePartitioner_::BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr int kKWarpTile = TilePartitioner_::BlockGemmShape::WarpTile::at(number<2>{});

    // Data types
    using ADataType = typename GemmPipeline_::ADataType;
    using BDataType = typename GemmPipeline_::BDataType;
    // Gemm Pipeline
    using GemmPipeline                                                 = GemmPipeline_;
    static constexpr ck_tile::GemmPipelineScheduler kPipelineScheduler = GemmPipeline_::Scheduler;
    static constexpr bool kDoubleSmemBuffer = GemmPipeline_::DoubleSmemBuffer;
    static constexpr int kNumWaveGroups     = GemmPipeline_::NumWaveGroups;

    // Epilogue Pipeline
    using AccDataType             = typename EpiloguePipeline_::AccDataType;
    using EDataType               = typename EpiloguePipeline_::ODataType;
    using DsDataType              = typename EpiloguePipeline_::DsDataType;
    using CDEElementwiseOperation = typename EpiloguePipeline_::CDElementwise;

    // Static member function to generate instance string
    static std::string instance_string()
    {
        std::ostringstream oss;

        // Kernel type name
        oss << "GroupedConvolutionForwardKernel";

        // Template parameters in exact order matching InstanceTraits member order
        oss << "<" << kSpatialDim; // 1. NDimSpatial
        oss << ","
            << ck_tile::getConvSpecializationString(ConvSpecialization);   // 2. ConvSpecialization
        oss << "," << detail::layout_name<InLayout>();                     // 3. InLayout
        oss << "," << detail::layout_name<WeiLayout>();                    // 4. WeiLayout
        oss << "," << detail::tuple_name<DsLayout>();                      // 5. DsLayout
        oss << "," << detail::layout_name<OutLayout>();                    // 6. OutLayout
        oss << "," << kVectorSizeA;                                        // 7. VectorSizeA
        oss << "," << kVectorSizeB;                                        // 8. VectorSizeB
        oss << "," << kVectorSizeC;                                        // 9. VectorSizeC
        oss << "," << kNumGroupsToMerge;                                   // 10. NumGroupsToMerge
        oss << "," << kEnableSplitImage;                                   // 11. EnableSplitImage
        oss << "," << kExplicitGemm;                                       // 12. ExplicitGemm
        oss << "," << kMPerBlock;                                          // 13. MPerBlock
        oss << "," << kNPerBlock;                                          // 14. NPerBlock
        oss << "," << kKPerBlock;                                          // 15. KPerBlock
        oss << "," << kMWarp;                                              // 16. MWarp
        oss << "," << kNWarp;                                              // 17. NWarp
        oss << "," << kKWarp;                                              // 18. KWarp
        oss << "," << kMWarpTile;                                          // 19. MWarpTile
        oss << "," << kNWarpTile;                                          // 20. NWarpTile
        oss << "," << kKWarpTile;                                          // 21. KWarpTile
        oss << "," << detail::type_name<ADataType>();                      // 22. ADataType
        oss << "," << detail::type_name<BDataType>();                      // 23. BDataType
        oss << "," << GemmPipeline::GetPipelineName();                     // 24. BlkGemmPipelineVer
        oss << "," << detail::pipeline_scheduler_name(kPipelineScheduler); // 25. BlkGemmPipeSched
        oss << "," << kDoubleSmemBuffer;                                   // 26. DoubleSmemBuffer
        oss << "," << kNumWaveGroups;                                      // 27. NumWaveGroups
        oss << "," << detail::type_name<AccDataType>();                    // 28. AccDataType
        oss << "," << detail::type_name<EDataType>();                      // 29. EDataType
        oss << "," << detail::tuple_name<DsDataType>();                    // 30. DsDataType
        oss << ","
            << detail::elementwise_op_name<CDEElementwiseOperation>(); // 31.
                                                                       // CDEElementwiseOperation
        oss << ">";

        return oss.str();
    }
};

} // namespace reflect
} // namespace ck_tile
