// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include <cmath>
#include <vector>

namespace ck_tile {

enum struct PositionEncodingEnum
{
    NO    = 0,
    ALIBI = 1,
};

/*
VERTICAL:
   [0] 1  2  3  4  5
   [0] 1  2  3  4  5
   [0] 1  2  3  4  5
   [0] 1  2  3  4  5

TOP_LEFT(but negative):
   [0] 1  2  3  4  5
    1 [0] 1  2  3  4
    2  1 [0] 1  2  3
    3  2  1 [0] 1  2

FROM_BOTTOM_RIGHT(but negative):
    2  1 [0] 1  2  3
    3  2  1 [0] 1  2
    4  3  2  1 [0] 1
    5  4  3  2  1 [0]
*/

enum struct AlibiMode
{
    VERTICAL          = 0,
    FROM_TOP_LEFT     = 1, // keep sync with mask enum
    FROM_BOTTOM_RIGHT = 2,
};

template <typename DataType, bool RowMajor = true, unsigned LogMaxSadOprndSize = 16>
struct Alibi
{
    static_assert(1 <= LogMaxSadOprndSize && LogMaxSadOprndSize <= 32,
                  "for LogMaxSadOprndSize <= 16, we use SAD uint16_t, otherwise, use SAD uint32_t");

    // RowMajor here means if pixel within the same thread are along the row, or col
    // this may impact the performance of update(), while the result are the same.
    // e.g. fwd prefer use RowMajor=true, bwd some cases prefer use RowMajor=false
    CK_TILE_HOST_DEVICE Alibi(DataType slope_,
                              index_t y_total_,
                              index_t x_total_,
                              AlibiMode mode_ = AlibiMode::VERTICAL)
    {
        slope = mode_ == AlibiMode::VERTICAL ? slope_ : -slope_;

        shift_left_up = [&]() {
            if(RowMajor)
            {
                return mode_ == AlibiMode::FROM_BOTTOM_RIGHT ? max(y_total_ - x_total_, 0) : 0;
            }
            else
            {
                return mode_ == AlibiMode::FROM_BOTTOM_RIGHT ? max(x_total_ - y_total_, 0) : 0;
            }
        }();
        shift_right_down = [&]() {
            if(RowMajor)
            {
                return mode_ == AlibiMode::FROM_BOTTOM_RIGHT ? max(x_total_ - y_total_, 0) : 0;
            }
            else
            {
                return mode_ == AlibiMode::FROM_BOTTOM_RIGHT ? max(y_total_ - x_total_, 0) : 0;
            }
        }();
        mode = mode_;
    }

    CK_TILE_HOST uint32_t sad(uint32_t x, uint32_t y, uint32_t acc) { return sad_u32(x, y, acc); }

    CK_TILE_DEVICE uint32_t sad(uint32_t x, uint32_t y, uint32_t acc)
    {
        if constexpr(LogMaxSadOprndSize <= 16)
        {
            return sad_u16(
                static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint16_t>(acc));
        }

        return sad_u32(x, y, acc);
    }

    CK_TILE_HOST_DEVICE void update(DataType& pixel, index_t row_idx, index_t col_idx)
    {
        if constexpr(RowMajor)
        {
            // at least 3 instructions per row
            index_t current_zero_point =
                mode == AlibiMode::VERTICAL ? shift_right_down : row_idx + shift_right_down;

            // for every threads, most of the pixels are along the row, below operation should be
            // the main hot spot.
            auto position = type_convert<DataType>(sad(bit_cast<uint32_t>(current_zero_point),
                                                       bit_cast<uint32_t>(col_idx + shift_left_up),
                                                       0));
            pixel += slope * position;
        }
        else
        {
            // at least 3 instructions per col;
            index_t current_zero_point = mode == AlibiMode::VERTICAL
                                             ? row_idx + col_idx + shift_right_down
                                             : col_idx + shift_right_down;

            // for every threads, most of the pixels are along the col, below operation should be
            // the main hot spot.
            auto position = type_convert<DataType>(sad(bit_cast<uint32_t>(current_zero_point),
                                                       bit_cast<uint32_t>(row_idx + shift_left_up),
                                                       0));
            pixel += slope * position;
        }
    }

    DataType slope;           // float?
    index_t shift_left_up;    // always possitive
    index_t shift_right_down; // always possitive
    AlibiMode mode;
};

template <typename DataType>
struct EmptyPositionEncoding
{
    CK_TILE_HOST_DEVICE void update(DataType& /*pixel*/, index_t /*row_idx*/, index_t /*col_idx*/)
    {
    }
};

//
// can convert from the FA style left/right to our generic coordinate
// if left_size < 0 && right_size = 0, it is normal causal mask
// local is left_size >=0 or right_size >=0
template <typename DataType, bool RowMajor = true, unsigned LogMaxSadOprndSize = 16>
CK_TILE_HOST_DEVICE auto make_alibi_from_lr_mask(DataType slope,
                                                 index_t window_left_size,
                                                 index_t window_right_size,
                                                 index_t y_total,
                                                 index_t x_total,
                                                 GenericAttentionMaskEnum mask_enum)
{
    // assume mask_enum will never be NO_MASK, since if we do not have mask, it's
    // totally OK to use constexpr
    bool is_causal = window_left_size < 0 && window_right_size == 0;
    AlibiMode alibi_mode =
        is_causal ? AlibiMode::VERTICAL
                  : static_cast<AlibiMode>(mask_enum) /*either top-left or bottom-right*/;
    return Alibi<DataType, RowMajor, LogMaxSadOprndSize>{slope, y_total, x_total, alibi_mode};
}

// https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
// Do we need a device version?
template <typename DataType>
CK_TILE_HOST std::vector<DataType> get_alibi_slopes(ck_tile::index_t nheads)
{
    auto get_slopes_power_of_2 = [](ck_tile::index_t n) {
        float start = std::powf(
            static_cast<float>(2),
            -std::powf(static_cast<float>(2), -static_cast<float>((integer_log2_floor(n) - 3))));

        std::vector<DataType> rtn;
        for(auto i = 0; i < n; i++)
        {
            rtn.push_back(static_cast<DataType>(start * std::powf(start, i)));
        }
        return rtn;
    };
    if(is_power_of_two_integer(nheads))
    {
        // power of 2 calculation
        return get_slopes_power_of_2(nheads);
    }
    else
    {
        ck_tile::index_t closest_power_of_2 = 1 << integer_log2_floor(nheads);
        auto v0                             = get_slopes_power_of_2(closest_power_of_2);
        auto v1                             = get_slopes_power_of_2(closest_power_of_2 * 2);
        auto v1_sliced                      = [&](auto vec, ck_tile::index_t rem) {
            std::vector<DataType> sliced;
            for(ck_tile::index_t i = 0; i < static_cast<ck_tile::index_t>(vec.size()); i++)
            {
                if(i % 2 == 0)
                    sliced.push_back(vec[i]);
            }
            std::vector<DataType> sliced_2(sliced.begin(), sliced.begin() + rem);
            return sliced_2;
        }(v1, nheads - closest_power_of_2);
        v0.insert(v0.end(), v1_sliced.begin(), v1_sliced.end());
        return v0;
    }
}
} // namespace ck_tile
