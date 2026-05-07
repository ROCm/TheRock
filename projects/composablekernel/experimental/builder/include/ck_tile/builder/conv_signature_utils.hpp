// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {
/**********************************************
 * constexpr helper functions for optional parameters
 **********************************************/

template <auto Sig>
concept ProvidesElementwiseOperation = requires { Sig.elementwise_operation; };

template <auto Sig>
concept ProvidesDataType = requires { Sig.data_type; };

template <auto ConvTensor>
concept ConvTensorHasOp = requires { ConvTensor.operation; };

template <auto Sig>
concept ProvidesConvolutionDirection = requires { Sig.direction; };

// returns elementwise operation for input tensor
// will defalut to signature's generic type if provided
// otherwise, default to PASS_THROUGH
template <auto Sig>
    requires ValidConvSignature<Sig>
constexpr auto getInputElementwiseOperation()
{
    if constexpr(ConvTensorHasOp<Sig.input>)
    {
        return Sig.input.operation.elementwise_operation;
    }
    else if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

// returns elementwise operation for weight tensor
// will defalut to signature's generic type if provided
// otherwise, default to PASS_THROUGH
template <auto Sig>
    requires ValidConvSignature<Sig>
constexpr auto getWeightElementwiseOperation()
{
    if constexpr(ConvTensorHasOp<Sig.weight>)
    {
        return Sig.weight.operation.elementwise_operation;
    }
    else if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

// returns elementwise operation for output tensor
// will defalut to signature's generic type if provided
// otherwise, default to PASS_THROUGH
template <auto Sig>
    requires ValidConvSignature<Sig>
constexpr auto getOutputElementwiseOperation()
{
    if constexpr(ConvTensorHasOp<Sig.output>)
    {
        return Sig.output.operation.elementwise_operation;
    }
    else if constexpr(ProvidesElementwiseOperation<Sig>)
    {
        return Sig.elementwise_operation;
    }
    else
    {
        return ElementwiseOperation::PASS_THROUGH;
    }
}

// returns convolution direction for signature. Will default to FORWARD if not provided by signature
template <auto Sig>
    requires ValidConvSignature<Sig>
constexpr auto getConvDirection()
{
    if constexpr(ProvidesConvolutionDirection<Sig>)
    {
        return Sig.direction;
    }
    else
    {
        return ConvDirection::FORWARD;
    }
}

// generic helper that returns data_type if provided and UNDEFINED otherwise
// can be used on both signature and TensorConfigDescriptor objects
template <auto TensorConfigOrSig>
constexpr auto getDataType()
{
    if constexpr(ProvidesDataType<TensorConfigOrSig>)
    {
        return TensorConfigOrSig.data_type;
    }
    else
    {
        return DataType::UNDEFINED_DATA_TYPE;
    }
}

// return data type of input tensor
template <auto Sig>
    requires ValidConvSignature<Sig>
consteval auto getInputDataType()
{
    constexpr auto tensorDataType    = getDataType<Sig.input.config>();
    constexpr auto universalDataType = getDataType<Sig>();
    if constexpr(tensorDataType != DataType::UNDEFINED_DATA_TYPE)
    {
        return tensorDataType;
    }
    else
    {
        return universalDataType;
    }
}

template <auto Sig>
    requires ValidConvSignature<Sig>
consteval auto getWeightDataType()
{
    constexpr auto tensorDataType    = getDataType<Sig.weight.config>();
    constexpr auto universalDataType = getDataType<Sig>();
    if constexpr(tensorDataType != DataType::UNDEFINED_DATA_TYPE)
    {
        return tensorDataType;
    }
    else
    {
        return universalDataType;
    }
}

template <auto Sig>
    requires ValidConvSignature<Sig>
consteval auto getOutputDataType()
{
    constexpr auto tensorDataType    = getDataType<Sig.output.config>();
    constexpr auto universalDataType = getDataType<Sig>();
    if constexpr(tensorDataType != DataType::UNDEFINED_DATA_TYPE)
    {
        return tensorDataType;
    }
    else
    {
        return universalDataType;
    }
}

// returns data type if and only if all tensors have the same type.
// Otherwise, return DataType::UNDEFINED_DATA_TYPE
template <auto Sig>
    requires ValidConvSignature<Sig>
consteval auto getDataTypeIfCommon()
{

    auto inputDataType  = getInputDataType<Sig>();
    auto weightDataType = getWeightDataType<Sig>();
    auto outputDataType = getOutputDataType<Sig>();

    if(inputDataType == weightDataType && inputDataType == outputDataType)
    {
        return inputDataType;
    }
    else
    {
        return DataType::UNDEFINED_DATA_TYPE;
    }
}
} // namespace ck_tile::builder
