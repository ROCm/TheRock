// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// InstanceTraits specializations for Reference convolution kernels
//
// This file provides compile-time reflection for all three reference kernel directions
// (Forward, Backward Data, Backward Weight) using a shared base to reduce duplication.

#pragma once

#include "instance_traits.hpp"
#include "instance_traits_util.hpp"
#include "ck_tile/builder/factory/reference_factory.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_layout.hpp"
#include <sstream>

namespace ck_tile::reflect {

namespace internal {

// Common traits shared by all reference implementations
template <auto SIGNATURE>
struct ReferenceCommonTraits
{
    // Spatial dimension
    static constexpr int kSpatialDim = SIGNATURE.spatial_dim;

    // Layouts - map from enum to type using existing helper
    using InLayout =
        typename builder::factory::internal::LayoutToCK<SIGNATURE.input.config.layout>::type;
    using WeiLayout =
        typename builder::factory::internal::LayoutToCK<SIGNATURE.weight.config.layout>::type;
    using OutLayout =
        typename builder::factory::internal::LayoutToCK<SIGNATURE.output.config.layout>::type;

    // Data types - extract from factory's type helper
    using Types       = builder::factory::internal::FwdConvTensorDataTypes<SIGNATURE>;
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using EDataType   = typename Types::EDataType;
    using AccDataType = float; // Reference uses float accumulation

    // Elementwise operations - reference only supports PassThrough
    using AElementwiseOperation   = ck_tile::element_wise::PassThrough;
    using BElementwiseOperation   = ck_tile::element_wise::PassThrough;
    using CDEElementwiseOperation = ck_tile::element_wise::PassThrough;

    // Reference has no block/tile configuration (simple kernel)
    // These are set to 0 to indicate "not applicable"
    static constexpr int kBlockSize = 0;
    static constexpr int kMPerBlock = 0;
    static constexpr int kNPerBlock = 0;
    static constexpr int kKPerBlock = 0;
};

} // namespace internal

// ============================================================================
// InstanceTraits specialization for Reference Forward Convolution
// ============================================================================
template <typename Instance>
    requires(
        std::is_same_v<std::remove_const_t<decltype(Instance::kAlgorithm.specialization)>,
                       builder::ConvAlgorithmSpecialization> &&
        (Instance::kAlgorithm.specialization == builder::ConvAlgorithmSpecialization::REFERENCE) &&
        builder::ConvDirectionIsForward<Instance::kSignature>)
struct InstanceTraits<Instance> : internal::ReferenceCommonTraits<Instance::kSignature>
{
    using Base = internal::ReferenceCommonTraits<Instance::kSignature>;

    // Bring base class members into scope
    using Base::kBlockSize;
    using Base::kKPerBlock;
    using Base::kMPerBlock;
    using Base::kNPerBlock;
    using Base::kSpatialDim;
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::AElementwiseOperation;
    using typename Base::BDataType;
    using typename Base::BElementwiseOperation;
    using typename Base::CDEElementwiseOperation;
    using typename Base::EDataType;
    using typename Base::InLayout;
    using typename Base::OutLayout;
    using typename Base::WeiLayout;

    static constexpr builder::ConvDirection direction = builder::ConvDirection::FORWARD;

    static std::string instance_string()
    {
        std::ostringstream oss;
        oss << "GPU_Reference_Forward_" << kSpatialDim << "D";
        oss << "_" << detail::type_name<ADataType>();
        oss << "_" << detail::layout_name<InLayout>();
        oss << "_" << detail::layout_name<WeiLayout>();
        oss << "_" << detail::layout_name<OutLayout>();
        return oss.str();
    }
};

// ============================================================================
// InstanceTraits specialization for Reference Backward Data Convolution
// ============================================================================
template <typename Instance>
    requires(
        std::is_same_v<std::remove_const_t<decltype(Instance::kAlgorithm.specialization)>,
                       builder::ConvAlgorithmSpecialization> &&
        (Instance::kAlgorithm.specialization == builder::ConvAlgorithmSpecialization::REFERENCE) &&
        builder::ConvDirectionIsBackwardData<Instance::kSignature>)
struct InstanceTraits<Instance> : internal::ReferenceCommonTraits<Instance::kSignature>
{
    using Base = internal::ReferenceCommonTraits<Instance::kSignature>;

    // Bring base class members into scope
    using Base::kBlockSize;
    using Base::kKPerBlock;
    using Base::kMPerBlock;
    using Base::kNPerBlock;
    using Base::kSpatialDim;
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::AElementwiseOperation;
    using typename Base::BDataType;
    using typename Base::BElementwiseOperation;
    using typename Base::CDEElementwiseOperation;
    using typename Base::EDataType;
    using typename Base::InLayout;
    using typename Base::OutLayout;
    using typename Base::WeiLayout;

    static constexpr builder::ConvDirection direction = builder::ConvDirection::BACKWARD_DATA;

    static std::string instance_string()
    {
        std::ostringstream oss;
        oss << "GPU_Reference_BackwardData_" << kSpatialDim << "D";
        oss << "_" << detail::type_name<ADataType>();
        oss << "_" << detail::layout_name<InLayout>();
        oss << "_" << detail::layout_name<WeiLayout>();
        oss << "_" << detail::layout_name<OutLayout>();
        return oss.str();
    }
};

// ============================================================================
// InstanceTraits specialization for Reference Backward Weight Convolution
// ============================================================================
template <typename Instance>
    requires(
        std::is_same_v<std::remove_const_t<decltype(Instance::kAlgorithm.specialization)>,
                       builder::ConvAlgorithmSpecialization> &&
        (Instance::kAlgorithm.specialization == builder::ConvAlgorithmSpecialization::REFERENCE) &&
        builder::ConvDirectionIsBackwardWeight<Instance::kSignature>)
struct InstanceTraits<Instance> : internal::ReferenceCommonTraits<Instance::kSignature>
{
    using Base = internal::ReferenceCommonTraits<Instance::kSignature>;

    // Bring base class members into scope
    using Base::kBlockSize;
    using Base::kKPerBlock;
    using Base::kMPerBlock;
    using Base::kNPerBlock;
    using Base::kSpatialDim;
    using typename Base::AccDataType;
    using typename Base::ADataType;
    using typename Base::AElementwiseOperation;
    using typename Base::BDataType;
    using typename Base::BElementwiseOperation;
    using typename Base::CDEElementwiseOperation;
    using typename Base::EDataType;
    using typename Base::InLayout;
    using typename Base::OutLayout;
    using typename Base::WeiLayout;

    static constexpr builder::ConvDirection direction = builder::ConvDirection::BACKWARD_WEIGHT;

    static std::string instance_string()
    {
        std::ostringstream oss;
        oss << "GPU_Reference_BackwardWeight_" << kSpatialDim << "D";
        oss << "_" << detail::type_name<ADataType>();
        oss << "_" << detail::layout_name<InLayout>();
        oss << "_" << detail::layout_name<WeiLayout>();
        oss << "_" << detail::layout_name<OutLayout>();
        return oss.str();
    }
};

} // namespace ck_tile::reflect
