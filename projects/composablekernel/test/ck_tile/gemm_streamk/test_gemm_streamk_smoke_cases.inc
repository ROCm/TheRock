// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, StreamK_EdgeCase)
{
    ck_tile::index_t M = 256;
    ck_tile::index_t N = 256;
    ck_tile::index_t K = 256;

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_DPOnly)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // For DP only, we ensure that the number of tiles is a multiple of the number of CUs. This
    // assumes tile sizes are large enough such that occupancy is 1.
    ck_tile::index_t M = M_Tile * num_cu;
    ck_tile::index_t N = N_Tile;
    ck_tile::index_t K = K_Tile;

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    // For SK only, we have 4 macro tiles in C. But, we need to make sure there is enough work along
    // the K dimension to avoid falling into the edge case. Thus, we always have at least num_cu
    // macro tiles in the K dimension. This assumes tile sizes are large enough such that occupancy
    // is 1.
    ck_tile::index_t M = M_Tile * 2;
    ck_tile::index_t N = N_Tile * 2;
    ck_tile::index_t K = K_Tile * num_cu;

    this->Run(M, N, K);
}
