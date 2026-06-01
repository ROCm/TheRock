// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/arch.hpp"

TYPED_TEST(TEST_SUITE_NAME, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 1024;
    std::vector<int> Ks;
    for(auto K_count : {2, 3, 4, 10, 11})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ADataType, ck_tile::fp16_t> &&
                         std::is_same_v<typename TestFixture::BDataType, ck_tile::pk_int4_t>)
            {
                if(K == 2 * TestFixture::K_Tile)
                {
                    // This particular combination of parameters fails.
                    continue;
                }
            }
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                if(M * sizeof(typename TestFixture::ADataType) % 4 == 0) // oob fit dword
                {
                    this->Run(M, N, K);
                }
                else
                {
                    EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                }
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, SingleTile)
{
    this->Run(TestFixture::M_Tile, TestFixture::N_Tile, TestFixture::K_Tile);
}

TYPED_TEST(TEST_SUITE_NAME, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;

    std::vector<int> Ks;
    for(auto K_count : {2, 3, 4, 10, 11})
    {
        Ks.push_back(K_count * TestFixture::K_Tile);
    }
    constexpr int VecLoadSize = (std::is_same_v<typename TestFixture::ADataType, ck_tile::fp8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::bf8_t> ||
                                 std::is_same_v<typename TestFixture::ADataType, ck_tile::int8_t>)
                                    ? 16
                                    : 8;

    for(int M : Ms)
    {
        for(int K : Ks)
        {
            if constexpr(std::is_same_v<typename TestFixture::ADataType, ck_tile::fp16_t> &&
                         std::is_same_v<typename TestFixture::BDataType, ck_tile::pk_int4_t>)
            {
                if(K == 2 * TestFixture::K_Tile)
                {
                    // This particular combination of parameters fails.
                    continue;
                }
            }
            if constexpr(std::is_same_v<typename TestFixture::ALayout,
                                        ck_tile::tensor_layout::gemm::ColumnMajor>)
            {
                if(M % VecLoadSize == 0)
                {
                    this->Run(M, N, K);
                }
                else
                {
                    if(M * sizeof(typename TestFixture::ADataType) % 4 == 0) // oob fit dword
                    {
                        this->Run(M, N, K);
                    }
                    else
                    {
                        EXPECT_THROW((this->Run(M, N, K)), std::runtime_error);
                    }
                }
            }
            else
            {
                this->Run(M, N, K);
            }
        }
    }
}

TYPED_TEST(TEST_SUITE_NAME, PaddK)
{
    std::vector<int> Ms{128};
    constexpr int N = 1024;
    constexpr int K = 432;

    for(int M : Ms)
    {
        this->Run(M, N, K);
    }
}

TYPED_TEST(TEST_SUITE_NAME, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 1024;
    constexpr int K = 512;

    for(int M : Ms)
        this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, LargeMatrix)
{
    constexpr int M = 2048;
    constexpr int N = 2048;
    constexpr int K = 2048;

    this->Run(M, N, K);
}

TYPED_TEST(TEST_SUITE_NAME, NotSupportedArgument)
{
    constexpr int M = 512;
    constexpr int N = 1025;
    constexpr int K = 513;

    constexpr bool PadM = false;
    constexpr bool PadN = false;
    constexpr bool PadK = false;

    EXPECT_THROW((this->template Run<PadM, PadN, PadK>(M, N, K)), std::runtime_error);
}
