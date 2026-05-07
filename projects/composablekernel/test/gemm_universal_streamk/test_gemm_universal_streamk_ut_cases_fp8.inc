// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestGemmUniversal_Streamk_FP8_MK_KN, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_Streamk_FP8_MK_NK, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_Streamk_FP8_MK_KN, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 512;
    constexpr int K = 437;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_Streamk_FP8_MK_NK, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 512;
    constexpr int K = 437;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_Streamk_FP8_MK_KN, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 512;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_Streamk_FP8_MK_NK, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 512;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}
