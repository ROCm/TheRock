// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestBatchedGemmBScale_MK_NK, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 256;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    constexpr int NBatches = 10;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC, NBatches);
}

TYPED_TEST(TestBatchedGemmBScale_MK_NK, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 512;
    constexpr int K = 768;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    constexpr int NBatches = 7;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC, NBatches);
}

TYPED_TEST(TestBatchedGemmBScale_MK_NK, Regular)
{
    std::vector<int> Ms{512, 1024};
    constexpr int N = 512;
    constexpr int K = 1024;

    constexpr int StrideA = K;
    constexpr int StrideB = K;
    constexpr int StrideC = N;

    constexpr int NBatches = 3;

    for(int M : Ms)
        this->Run(M, N, K, StrideA, StrideB, StrideC, NBatches);
}
