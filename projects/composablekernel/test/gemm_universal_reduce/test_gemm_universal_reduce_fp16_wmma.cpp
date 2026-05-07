// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>

#include "gtest/gtest.h"
#include "profiler/profile_gemm_universal_reduce_impl.hpp"

TEST(GemmUniversalReduce, FP16)
{
    using Row = ck::tensor_layout::gemm::RowMajor;

    int M      = 512;
    int N      = 256;
    int K      = 128;
    int KBatch = 1;

    bool pass = true;

    pass = pass && ck::profiler::profile_gemm_universal_reduce_impl<ck::half_t,
                                                                    ck::half_t,
                                                                    ck::Tuple<>,
                                                                    float,
                                                                    ck::half_t,
                                                                    Row,
                                                                    Row,
                                                                    ck::Tuple<>,
                                                                    Row>(
                       true, 1, false, true, M, N, K, K, N, N, KBatch, 1, 10);
    EXPECT_TRUE(pass);
}
