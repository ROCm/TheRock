// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include "ck_tile/builder/types.hpp"

/// This file implements various backend-independent traits for
/// CK-Builder types.

namespace ck_tile::builder::test {

/// @brief Query the size of a data type in memory.
///
/// This function computes the size of a variant of `DataType` in memory.
/// This is more complicated than it seems. For most types, this is just
/// the size of the equivalent C++-type, but for sub-byte type we have to
/// represent each byte by multiple values, for example. For now, we only
/// care about types which consist of an integral number of bytes, though.
///
/// @note The details of this function are likely going to change with the
/// support of sub-byte types.
///
/// @param data_type The type to query the in-memory size of.
/// @returns The number of bytes that an element of this data type requires
///   in memory.
constexpr size_t data_type_sizeof(DataType data_type)
{
    switch(data_type)
    {
    case DataType::UNDEFINED_DATA_TYPE: return 0;
    case DataType::FP32: return 4;
    case DataType::FP32_FP32: return 8;
    case DataType::FP16: return 2;
    case DataType::FP16_FP16: return 4;
    case DataType::BF16: return 2;
    case DataType::BF16_BF16: return 4;
    case DataType::FP8: return 1;
    case DataType::BF8: return 1;
    case DataType::FP64: return 8;
    case DataType::I32: return 4;
    case DataType::I8: return 1;
    case DataType::I8_I8: return 2;
    case DataType::U8: return 1;
    }
    return 0; // Default case to ensure all control paths return a value
}

} // namespace ck_tile::builder::test
