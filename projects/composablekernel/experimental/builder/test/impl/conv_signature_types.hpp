// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <variant>
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::test {

using namespace ck_tile::builder;

struct TensorConfig
{
    TensorLayout layout;
    // Optional data types, override the type defined in the signature if provided.
    DataType data_type{DataType::UNDEFINED_DATA_TYPE};
    DataType compute_type{DataType::UNDEFINED_DATA_TYPE};
};

template <TensorConfig... Configs>
struct TensorOperation
{
    ElementwiseOperation elementwise_operation{ElementwiseOperation::PASS_THROUGH};
    std::array<TensorConfig, sizeof...(Configs)> auxiliary_operand_configs{Configs...};

    // Add builder to add auxiliary tensor configs
    template <auto... AuxiliaryConfigs>
    constexpr auto with_auxiliary_operand_configs() const
    {
        return TensorOperation<Configs..., TensorConfig{AuxiliaryConfigs}...>{
            .elementwise_operation = this->elementwise_operation};
    }
};

template <typename Op = TensorOperation<>>
struct ConvolutionTensor
{
    TensorConfig config;
    Op operation{};
};

template <typename InputTensor  = ConvolutionTensor<>,
          typename WeightTensor = ConvolutionTensor<>,
          typename OutputTensor = ConvolutionTensor<>>
struct ConvSignature
{
    int spatial_dim;
    ConvDirection direction;
    DataType data_type;
    DataType accumulation_data_type;
    InputTensor input;
    WeightTensor weight;
    OutputTensor output;
};

} // namespace ck_tile::builder::test
