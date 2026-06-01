// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/builder_utils.hpp"

namespace ck_tile::builder::factory::internal {

// Type mappings from builder convolution data type to CK Tile tensor types.
template <DataType T>
struct TileConvTensorTypes
{
    // This will trigger if a specialization for the given DataType is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(UnsupportedEnumValue<T>) == 0,
                  "Internal error. Unsupported data type for convolution factory.");
};

template <>
struct TileConvTensorTypes<DataType::FP16>
{
    using ADataType        = ck_tile::half_t;
    using AComputeType     = ck_tile::half_t;
    using BDataType        = ck_tile::half_t;
    using BComputeType     = ck_tile::half_t;
    using CShuffleDataType = ck_tile::half_t;
    using DsDataTypes      = ck_tile::tuple<>;
    using AccDataType      = float;
    using EDataType        = ck_tile::half_t;
};

template <>
struct TileConvTensorTypes<DataType::BF16>
{
    using ADataType        = ck_tile::bf16_t;
    using AComputeType     = ck_tile::bf16_t;
    using BDataType        = ck_tile::bf16_t;
    using BComputeType     = ck_tile::bf16_t;
    using CShuffleDataType = ck_tile::bf16_t;
    using DsDataTypes      = ck_tile::tuple<>;
    using AccDataType      = float;
    using EDataType        = ck_tile::bf16_t;
};

template <>
struct TileConvTensorTypes<DataType::FP32>
{
    using ADataType        = float;
    using AComputeType     = float;
    using BDataType        = float;
    using BComputeType     = float;
    using CShuffleDataType = float;
    using DsDataTypes      = ck_tile::tuple<>;
    using AccDataType      = float;
    using EDataType        = float;
};

template <>
struct TileConvTensorTypes<DataType::I8>
{
    using ADataType        = int8_t;
    using AComputeType     = int8_t;
    using BDataType        = int8_t;
    using BComputeType     = int8_t;
    using CShuffleDataType = int8_t;
    using DsDataTypes      = ck_tile::tuple<>;
    using AccDataType      = int32_t;
    using EDataType        = int8_t;
};

template <>
struct TileConvTensorTypes<DataType::FP8>
{
    using ADataType        = ck_tile::fp8_t;
    using AComputeType     = ck_tile::fp8_t;
    using BDataType        = ck_tile::fp8_t;
    using BComputeType     = ck_tile::fp8_t;
    using CShuffleDataType = ck_tile::fp8_t;
    using DsDataTypes      = ck_tile::tuple<>;
    using AccDataType      = float;
    using EDataType        = ck_tile::fp8_t;
};

} // namespace ck_tile::builder::factory::internal
