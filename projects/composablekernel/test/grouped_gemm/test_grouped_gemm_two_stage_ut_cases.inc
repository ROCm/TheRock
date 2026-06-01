// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

TEST_P(RRR_BF16_BF16_BF16, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_BF16_BF16_BF16, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RRR_BF16_I8_BF16, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), N);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}

TEST_P(RCR_BF16_I8_BF16, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);
    const std::vector<int> StrideAs(Ms.size(), K);
    const std::vector<int> StrideBs(Ms.size(), K);
    const std::vector<int> StrideCs(Ms.size(), N);

    this->Run(Ms, Ns, Ks, StrideAs, StrideBs, StrideCs, this->GetParam());
}
