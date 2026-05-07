// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <variant>
#include "ck_tile/core/arch/arch.hpp"

auto string_to_datatype(const std::string& datatype)
{
    using PrecVariant = std::variant<ck_tile::half_t, ck_tile::bf16_t, float>;

    if(datatype == "fp16")
    {
        return PrecVariant{ck_tile::half_t{}};
    }
    else if(datatype == "bf16")
    {
        return PrecVariant{ck_tile::bf16_t{}};
    }
    else if(datatype == "fp32")
    {
        return PrecVariant{float{}};
    }
    else
    {
        throw std::runtime_error("Unsupported data type: " + datatype);
    }
};
