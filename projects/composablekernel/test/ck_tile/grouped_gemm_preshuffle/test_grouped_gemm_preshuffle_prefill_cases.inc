// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Test with prefill config struct
TYPED_TEST(TestCkTileGroupedGemmPreshuffle, PrefillVariant)
{
    const int group_count = 4;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    for(int i = 0; i < group_count; i++)
    {

        Ms.push_back(256 + 128 * i);
        Ns.push_back(256 + 128 * i);
        Ks.push_back(128 * (i + 1));

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

TYPED_TEST(TestCkTileGroupedGemmPreshuffle, VariedDimensions)
{
    const int group_count = 6;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;

    std::vector<std::tuple<int, int, int>> test_cases = {{64, 128, 256},
                                                         {128, 256, 512},
                                                         {256, 512, 1024},
                                                         {512, 256, 128},
                                                         {128, 128, 128},
                                                         {64, 512, 256}};

    for(int i = 0; i < group_count; i++)
    {
        auto [M, N, K] = test_cases[i];
        Ms.push_back(M);
        Ns.push_back(N);
        Ks.push_back(K);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}
