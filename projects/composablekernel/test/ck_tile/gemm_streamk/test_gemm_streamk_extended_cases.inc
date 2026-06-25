// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, StreamK_DP2TSK)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;

    // For DP 2-Tile SK, there are 2 important terms:
    // Term 1: (M_Tile * num_cu * 2) - This ensures we have at least 2 cycles that will fully
    // saturate all CUs. This assumes tile sizes are large enough such that occupancy is 1.
    // Term 2: (M_Tile * 2) - This ensures we have 1 cycle that does not fully saturate all CUs
    // (i.e., we will have remainder tiles). This guarantees we have 1 full tile cycle plus
    // remainder tiles for the 2 Tile SK portion; the rest of the tiles will fully saturate all CUs
    // for the DP portion.
    ck_tile::index_t M = (M_Tile * num_cu * 2) + (M_Tile * 2);
    ck_tile::index_t N = N_Tile;
    ck_tile::index_t K = 2048;

    this->Run(M, N, K);
}
