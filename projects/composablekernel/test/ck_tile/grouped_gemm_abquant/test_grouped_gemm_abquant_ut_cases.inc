// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TYPED_TEST(TEST_CLASS_NAME, Basic)
{
    const int group_count = 6;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;
    std::vector<int> stride_AQs;
    std::vector<int> stride_BQs;
    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * i);
        Ns.push_back(256 + 512 * i);
        Ks.push_back(512 + 128 * i);

        stride_As.push_back(0);
        stride_Bs.push_back(0);
        stride_Cs.push_back(0);
        stride_AQs.push_back(0);
        stride_BQs.push_back(0);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, stride_AQs, stride_BQs, group_count);
}

// No Hot Loop Test Case, this is to test the correctness of the kernel when there is no hot loop
// Using 256x256x128 to match the test kernel's tile size (M_Tile=128, N_Tile=128, K_Tile=128)
TYPED_TEST(TEST_CLASS_NAME, SmallUniform)
{
    const int group_count = 2;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;
    std::vector<int> stride_AQs;
    std::vector<int> stride_BQs;
    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256);
        Ns.push_back(256);
        Ks.push_back(256);

        stride_As.push_back(0);
        stride_Bs.push_back(0);
        stride_Cs.push_back(0);
        stride_AQs.push_back(0);
        stride_BQs.push_back(0);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, stride_AQs, stride_BQs, group_count);
}

TYPED_TEST(TEST_CLASS_NAME, OddTail)
{
    const int group_count = 2;
    std::vector<int> Ms;
    std::vector<int> Ns;
    std::vector<int> Ks;
    std::vector<int> stride_As;
    std::vector<int> stride_Bs;
    std::vector<int> stride_Cs;
    std::vector<int> stride_AQs;
    std::vector<int> stride_BQs;
    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256);
        Ns.push_back(256);
        Ks.push_back(128);

        stride_As.push_back(0);
        stride_Bs.push_back(0);
        stride_Cs.push_back(0);
        stride_AQs.push_back(0);
        stride_BQs.push_back(0);
    }

    this->Run(Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs, stride_AQs, stride_BQs, group_count);
}
