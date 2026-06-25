// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TestCkTileGroupedGemmMultiD, K256)
{
    const int group_count = 7;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Es;
    std::vector<int> stride_D0;
    std::vector<int> stride_D1;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 256 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Es.push_back(Ns[i]);
        stride_D0.push_back(Ns[i]);
        stride_D1.push_back(Ns[i]);
    }

    this->Run(
        Ms, Ns, Ks, stride_As, stride_Bs, stride_Es, stride_D0, stride_D1, kbatch, group_count);
}

TYPED_TEST(TestCkTileGroupedGemmMultiD, K128)
{
    const int group_count = 5;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Es;
    std::vector<int> stride_D0;
    std::vector<int> stride_D1;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 128 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Es.push_back(Ns[i]);
        stride_D0.push_back(Ns[i]);
        stride_D1.push_back(Ns[i]);
    }

    this->Run(
        Ms, Ns, Ks, stride_As, stride_Bs, stride_Es, stride_D0, stride_D1, kbatch, group_count);
}

TYPED_TEST(TestCkTileGroupedGemmMultiD, LargeMNK_8Groups)
{
    const int group_count = 8;
    const int kbatch      = 1;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Es;
    std::vector<int> stride_D0;
    std::vector<int> stride_D1;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(512 + 256 * i);
        Ns.push_back(512 + 256 * i);
        Ks.push_back(768 + 256 * i);

        stride_As.push_back(Ks[i]);
        stride_Bs.push_back(Ks[i]);
        stride_Es.push_back(Ns[i]);
        stride_D0.push_back(Ns[i]);
        stride_D1.push_back(Ns[i]);
    }

    this->Run(
        Ms, Ns, Ks, stride_As, stride_Bs, stride_Es, stride_D0, stride_D1, kbatch, group_count);
}
