// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCReg
// Default policy class should not be templated, put template on member functions instead
struct PracticeGemmWarpPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        constexpr index_t kMWarp = 4;
        constexpr index_t kNWarp = 1;

        if constexpr(std::is_same_v<typename Problem::ADataType, half_t> &&
                     std::is_same_v<typename Problem::BDataType, half_t> &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            return make_tuple(
                WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution{}, kMWarp, kNWarp);
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
