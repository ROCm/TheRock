// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestGemmQuantization, SmallM)
{
    std::vector<int> Ms{1, 2, 3, 4, 5, 6};
    constexpr int N = 512;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run({{M, N, K}});
}

TYPED_TEST(TestGemmQuantization, MidLargeM)
{
    std::vector<int> Ms{127, 255, 312, 799, 1573};
    constexpr int N = 1024;
    constexpr int K = 320;

    for(int M : Ms)
        this->Run({{M, N, K}});
}

TYPED_TEST(TestGemmQuantization, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    for(int M : Ms)
        this->Run({{M, N, K}});
}

TYPED_TEST(TestGemmQuantization, Regular)
{
    constexpr int M = 512;
    constexpr int N = 512;
    std::vector<int> Ks{512};

    for(int K : Ks)
        this->Run({{M, N, K}});
}
