// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"

#ifndef TEST_ALIBI_VERBOSE
#define TEST_ALIBI_VERBOSE 0
#endif

template <typename DataType>
struct attention_score
{
    ck_tile::index_t rows, cols;
    std::vector<DataType> pixels;

    attention_score(ck_tile::index_t rows_,
                    ck_tile::index_t cols_,
                    DataType init_v_ = static_cast<DataType>(0))
        : rows(rows_), cols(cols_), pixels(rows_ * cols_, init_v_)
    {
    }

    auto& operator()(ck_tile::index_t i_row, ck_tile::index_t i_col)
    {
        return pixels[i_row * cols + i_col];
    }

    void print()
    {
        for(auto i_row = 0; i_row < rows; i_row++)
        {
            for(auto i_col = 0; i_col < cols; i_col++)
            {
                std::cout << pixels[i_row * cols + i_col] << " ";
            }
            std::cout << std::endl;
        }
    }
};

template <bool RowMajor, typename DataType>
void alibi_traverse_with_slope(attention_score<DataType>& score,
                               DataType slope,
                               ck_tile::AlibiMode mode = ck_tile::AlibiMode::VERTICAL)
{
    using Alibi = ck_tile::Alibi<DataType, RowMajor>;
    auto alibi  = Alibi{slope, score.rows, score.cols, mode};

    for(ck_tile::index_t i_row = 0; i_row < score.rows; i_row++)
    {
        for(ck_tile::index_t i_col = 0; i_col < score.cols; i_col++)
        {
            alibi.update(score(i_row, i_col), i_row, i_col);
        }
    }
}

std::string alibi_mode_to_str(ck_tile::AlibiMode mode)
{
    if(mode == ck_tile::AlibiMode::VERTICAL)
        return std::string("alibi_verti");
    else if(mode == ck_tile::AlibiMode::FROM_TOP_LEFT)
        return std::string("alibi_top-l");
    else if(mode == ck_tile::AlibiMode::FROM_BOTTOM_RIGHT)
        return std::string("alibi_bot-r");
    return "";
}

template <bool RowMajor, typename DataType>
bool test_alibi_traverse_with_slope(ck_tile::index_t rows,
                                    ck_tile::index_t cols,
                                    DataType slope,
                                    ck_tile::AlibiMode mode,
                                    const std::vector<DataType>& expected)
{
    attention_score<DataType> score{rows, cols};
    alibi_traverse_with_slope<RowMajor, DataType>(score, slope, mode);

    bool is_match = std::equal(score.pixels.begin(), score.pixels.end(), expected.begin());
#if TEST_ALIBI_VERBOSE
    std::cout << "---------" << alibi_mode_to_str(mode) << ", " << rows << "x" << cols << "("
              << (RowMajor ? "row_major" : "col_major") << ")"
              << (is_match ? ", valie:y" : ", valid:n") << std::endl;
    score.print();
#endif
    return is_match;
}

template <typename DataType>
bool test_alibi_slope_generation(ck_tile::index_t nheads, const std::vector<DataType>& expected)
{
    auto slopes = ck_tile::get_alibi_slopes<DataType>(nheads);

    bool is_match = std::equal(slopes.begin(),
                               slopes.end(),
                               expected.begin(),
                               expected.end(),
                               [](const DataType& lhs, const DataType& rhs) {
                                   constexpr float rtol = 1e-6;
                                   auto error           = std::abs(lhs - rhs);
                                   return error < rtol * std::abs(rhs);
                               });
#if TEST_ALIBI_VERBOSE
    std::cout << "-------------------- slopes " << nheads << ", " << (is_match ? "y" : "n")
              << std::endl;
    for(ck_tile::index_t i = 0; i < nheads; i++)
    {
        std::cout << slopes[i] << " ";
    }
    std::cout << std::endl;
#endif
    return is_match;
}

int main()
{
    using dtype = int32_t;
    dtype slope = static_cast<dtype>(1);

    bool rtn = true;

    // clang-format off
    rtn &= test_alibi_traverse_with_slope<true, dtype>(4, 6, slope, ck_tile::AlibiMode::VERTICAL,          {0, 1, 2, 3, 4, 5,
                                                                                                            0, 1, 2, 3, 4, 5,
                                                                                                            0, 1, 2, 3, 4, 5,
                                                                                                            0, 1, 2, 3, 4, 5});

    rtn &= test_alibi_traverse_with_slope<true, dtype>(4, 6, slope, ck_tile::AlibiMode::FROM_TOP_LEFT,     { 0, -1, -2, -3, -4, -5,
                                                                                                            -1,  0, -1, -2, -3, -4,
                                                                                                            -2, -1,  0, -1, -2, -3,
                                                                                                            -3, -2, -1,  0, -1, -2});

    rtn &= test_alibi_traverse_with_slope<true, dtype>(6, 4, slope, ck_tile::AlibiMode::FROM_TOP_LEFT,     { 0, -1, -2, -3,
                                                                                                            -1,  0, -1, -2,
                                                                                                            -2, -1,  0, -1,
                                                                                                            -3, -2, -1,  0,
                                                                                                            -4, -3, -2, -1,
                                                                                                            -5, -4, -3, -2});

    rtn &= test_alibi_traverse_with_slope<true, dtype>(3, 3, slope, ck_tile::AlibiMode::FROM_TOP_LEFT,     { 0, -1, -2,
                                                                                                            -1,  0, -1,
                                                                                                            -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<true, dtype>(4, 6, slope, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT, {-2, -1,  0, -1, -2, -3,
                                                                                                            -3, -2, -1,  0, -1, -2,
                                                                                                            -4, -3, -2, -1,  0, -1,
                                                                                                            -5, -4, -3, -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<true, dtype>(6, 4, slope, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT, {-2, -3, -4, -5,
                                                                                                            -1, -2, -3, -4,
                                                                                                             0, -1, -2, -3,
                                                                                                            -1,  0, -1, -2,
                                                                                                            -2, -1,  0, -1,
                                                                                                            -3, -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<true, dtype>(3, 3, slope, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT, { 0, -1, -2,
                                                                                                            -1,  0, -1,
                                                                                                            -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(4, 6, slope, ck_tile::AlibiMode::VERTICAL,         {0, 1, 2, 3, 4, 5,
                                                                                                            0, 1, 2, 3, 4, 5,
                                                                                                            0, 1, 2, 3, 4, 5,
                                                                                                            0, 1, 2, 3, 4, 5});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(4, 6, slope, ck_tile::AlibiMode::FROM_TOP_LEFT,    { 0, -1, -2, -3, -4, -5,
                                                                                                            -1,  0, -1, -2, -3, -4,
                                                                                                            -2, -1,  0, -1, -2, -3,
                                                                                                            -3, -2, -1,  0, -1, -2});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(6, 4, slope, ck_tile::AlibiMode::FROM_TOP_LEFT,    { 0, -1, -2, -3,
                                                                                                            -1,  0, -1, -2,
                                                                                                            -2, -1,  0, -1,
                                                                                                            -3, -2, -1,  0,
                                                                                                            -4, -3, -2, -1,
                                                                                                            -5, -4, -3, -2});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(3, 3, slope, ck_tile::AlibiMode::FROM_TOP_LEFT,    { 0, -1, -2,
                                                                                                            -1,  0, -1,
                                                                                                            -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(4, 6, slope, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT, {-2, -1,  0, -1, -2, -3,
                                                                                                             -3, -2, -1,  0, -1, -2,
                                                                                                             -4, -3, -2, -1,  0, -1,
                                                                                                             -5, -4, -3, -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(6, 4, slope, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT, {-2, -3, -4, -5,
                                                                                                             -1, -2, -3, -4,
                                                                                                              0, -1, -2, -3,
                                                                                                             -1,  0, -1, -2,
                                                                                                             -2, -1,  0, -1,
                                                                                                             -3, -2, -1,  0});

    rtn &= test_alibi_traverse_with_slope<false, dtype>(3, 3, slope, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT, { 0, -1, -2,
                                                                                                             -1,  0, -1,
                                                                                                             -2, -1,  0});

    rtn &= test_alibi_slope_generation<float>(8, {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625});
    rtn &= test_alibi_slope_generation<float>(16, {0.7071067811865476, 0.5, 0.35355339059327384, 0.25000000000000006, 0.17677669529663692,
                                                   0.12500000000000006, 0.08838834764831849, 0.06250000000000004, 0.044194173824159244,
                                                   0.03125000000000002, 0.022097086912079626, 0.01562500000000001, 0.011048543456039816,
                                                   0.007812500000000007, 0.005524271728019908, 0.003906250000000004});
    rtn &= test_alibi_slope_generation<float>(1, {0.00390625});
    rtn &= test_alibi_slope_generation<float>(5, {0.25, 0.0625, 0.015625, 0.00390625, 0.5});
    rtn &= test_alibi_slope_generation<float>(6, {0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125});
    rtn &= test_alibi_slope_generation<float>(7, {0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125, 0.03125});
    rtn &= test_alibi_slope_generation<float>(9, {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.7071067811865476});
    // clang-format on
    return rtn ? 0 : -1;
}
