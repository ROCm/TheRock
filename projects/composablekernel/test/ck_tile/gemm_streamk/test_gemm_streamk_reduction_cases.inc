// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly_OneTile_Tree)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    ck_tile::index_t M = M_Tile;
    ck_tile::index_t N = N_Tile;
    ck_tile::index_t K = K_Tile * num_cu;

    this->Run(M, N, K, ck_tile::StreamKReductionStrategy::TreeReduction);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly_OneTile)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    ck_tile::index_t M = M_Tile;
    ck_tile::index_t N = N_Tile;
    ck_tile::index_t K = K_Tile * num_cu;

    this->Run(M, N, K, ck_tile::StreamKReductionStrategy::Reduction);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly_4Tiles_Tree)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    ck_tile::index_t M = M_Tile * 4;
    ck_tile::index_t N = N_Tile;
    ck_tile::index_t K = K_Tile * num_cu + (25 * K_Tile);

    this->Run(M, N, K, ck_tile::StreamKReductionStrategy::TreeReduction);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly_4Tiles_Reduction)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    ck_tile::index_t M = M_Tile * 4;
    ck_tile::index_t N = N_Tile;
    ck_tile::index_t K = K_Tile * num_cu + (25 * K_Tile);

    this->Run(M, N, K, ck_tile::StreamKReductionStrategy::Reduction);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly_21Tiles_Tree)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    ck_tile::index_t M = M_Tile * 3;
    ck_tile::index_t N = N_Tile * 7;
    ck_tile::index_t K = K_Tile * num_cu + (30 * K_Tile);

    this->Run(M, N, K, ck_tile::StreamKReductionStrategy::TreeReduction);
}

TYPED_TEST(TEST_SUITE_NAME, StreamK_SKOnly_21Tiles)
{
    const ck_tile::index_t num_cu     = get_cu_count();
    constexpr ck_tile::index_t M_Tile = std::tuple_element_t<7, TypeParam>::value;
    constexpr ck_tile::index_t N_Tile = std::tuple_element_t<8, TypeParam>::value;
    constexpr ck_tile::index_t K_Tile = std::tuple_element_t<9, TypeParam>::value;

    ck_tile::index_t M = M_Tile * 3;
    ck_tile::index_t N = N_Tile * 7;
    ck_tile::index_t K = K_Tile * num_cu + (30 * K_Tile);

    this->Run(M, N, K, ck_tile::StreamKReductionStrategy::Reduction);
}
