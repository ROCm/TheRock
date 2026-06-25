// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestGemmUniversal_FP16_MK_KN, SmallM)
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

TYPED_TEST(TestGemmUniversal_FP16_MK_NK, SmallM)
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

TYPED_TEST(TestGemmUniversal_FP16_KM_KN, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_KM_NK, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_MK_KN, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideA = K;
    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_FP16_MK_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC);
}

TYPED_TEST(TestGemmUniversal_FP16_KM_KN, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_KM_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 320;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_MK_KN, PaddK)
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

TYPED_TEST(TestGemmUniversal_FP16_MK_NK, PaddK)
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

TYPED_TEST(TestGemmUniversal_FP16_KM_KN, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 512;
    constexpr int K = 437;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_KM_NK, PaddK)
{
    std::vector<int> Ms{127};
    constexpr int N = 512;
    constexpr int K = 437;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_MK_KN, Regular)
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

TYPED_TEST(TestGemmUniversal_FP16_MK_NK, Regular)
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

TYPED_TEST(TestGemmUniversal_FP16_KM_KN, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 512;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}

TYPED_TEST(TestGemmUniversal_FP16_KM_NK, Regular)
{
    std::vector<int> Ms{512};
    constexpr int N = 512;
    constexpr int K = 512;

    constexpr int StrideB = N;
    constexpr int StrideC = N;

    for(int M : Ms)
    {
        int StrideA = M;
        this->Run(M, N, K, StrideA, StrideB, StrideC);
    }
}
