// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// kPadK is not needed for these k values
TYPED_TEST(TestCkTileGroupedGemmPreshuffle, kPadKFalse)
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
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 256 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}

// kPadK is needed to be true for these k values
TYPED_TEST(TestCkTileGroupedGemmPreshuffle, kPadKTrue)
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
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 128 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Cs.push_back(Ns[i]);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, kbatch, group_count);
}
