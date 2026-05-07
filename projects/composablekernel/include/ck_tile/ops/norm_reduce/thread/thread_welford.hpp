// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename T, bool kFastFDiv = false>
CK_TILE_DEVICE void welford_update(T& mean, T& var, T x, int count, bool_constant<kFastFDiv> = {})
{
    // TODO: check nan? maybe no
    T delta = x - mean;
    if(kFastFDiv && std::is_same_v<T, float>)
    {
        mean += delta * __builtin_amdgcn_rcpf(count);
    }
    else
    {
        mean += delta / count;
    }
    T delta2 = x - mean;
    var += delta * delta2;
}

template <typename T, bool kFastFDiv = false>
CK_TILE_DEVICE static void welford_merge(T& mean_a,
                                         T& var_a,
                                         int& count_a,
                                         T mean_b,
                                         T var_b,
                                         int count_b,
                                         bool_constant<kFastFDiv> = {})
{
    int count  = count_a + count_b;
    T count_   = type_convert<T>(count);
    T count_a_ = type_convert<T>(count_a);
    T count_b_ = type_convert<T>(count_b);
    T count_b_over_count;
    if(kFastFDiv && std::is_same_v<T, float>)
    {
        count_b_over_count =
            count == 0 ? type_convert<T>(0) : count_b_ * __builtin_amdgcn_rcpf(count_);
    }
    else
    {
        count_b_over_count = count == 0 ? type_convert<T>(0) : count_b_ / count_;
    }

    T delta = mean_b - mean_a;
    mean_a += delta * count_b_over_count;
    var_a += var_b + delta * delta * count_a_ * count_b_over_count;
    count_a = count;
}

} // namespace ck_tile
