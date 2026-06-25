// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

// Default policy for BlockGemmASmemBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmASmemBSmemCRegV1DefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
#if defined(__gfx950__)
        constexpr bool is_a_load_tr = std::is_same_v<remove_cvref_t<typename Problem::ALayout>,
                                                     tensor_layout::gemm::ColumnMajor>;
        constexpr bool is_b_load_tr = std::is_same_v<remove_cvref_t<typename Problem::BLayout>,
                                                     tensor_layout::gemm::RowMajor>;
#else
        constexpr bool is_a_load_tr = false;
        constexpr bool is_b_load_tr = false;
#endif
        constexpr auto wg_attr_num_access = (is_a_load_tr || is_b_load_tr)
                                                ? WGAttrNumAccessEnum::Double
                                                : WGAttrNumAccessEnum::Single;

        if constexpr(((std::is_same_v<typename Problem::ADataType, half_t> &&
                       std::is_same_v<typename Problem::BDataType, half_t>) ||
                      (std::is_same_v<typename Problem::ADataType, bf16_t> &&
                       std::is_same_v<typename Problem::BDataType, bf16_t>)) &&
                     std::is_same_v<typename Problem::CDataType, float>)
        {
            if constexpr(get_warp_size() == 64)
            {
                using WG = WarpGemmDispatcher<typename Problem::ADataType,
                                              typename Problem::BDataType,
                                              typename Problem::CDataType,
                                              32,
                                              32,
                                              16,
                                              true,
                                              false,
                                              false,
                                              wg_attr_num_access>;
                return make_tuple(WG{}, 4, 1);
            }
            else
            {
                using WG = WarpGemmDispatcher<typename Problem::ADataType,
                                              typename Problem::BDataType,
                                              typename Problem::CDataType,
                                              16,
                                              16,
                                              16,
                                              true,
                                              false,
                                              false,
                                              wg_attr_num_access>;
                return make_tuple(WG{}, 4, 1);
            }
        }
        else
        {
            static_assert(false, "Unsupported data type configuration for GEMM warp execution.");
        }
    }
};

} // namespace ck_tile
