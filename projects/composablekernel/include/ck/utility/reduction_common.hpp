// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/reduction_enums.hpp"

namespace ck {

struct float_equal_one
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(1.0f) and x >= static_cast<T>(1.0f);
    };
};

struct float_equal_zero
{
    template <class T>
    __host__ __device__ inline bool operator()(T x)
    {
        return x <= static_cast<T>(0.0f) and x >= static_cast<T>(0.0f);
    };
};

} // namespace ck
