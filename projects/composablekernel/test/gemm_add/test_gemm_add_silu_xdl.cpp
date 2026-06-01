// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_add_silu_impl.hpp"
#include "test_gemm_common.hpp"

template <typename Tuple>
class TestGemmAddSilu : public TestGemmD0Common<Tuple>
{
    using ProfileCall = typename TestGemmD0Common<Tuple>::ProfileCall;

    ProfileCall GetImpl() override
    {
        return ck::profiler::profile_gemm_add_silu_impl<
            typename TestGemmD0Common<Tuple>::ADataType,
            typename TestGemmD0Common<Tuple>::BDataType,
            typename TestGemmD0Common<Tuple>::AccDataType,
            typename TestGemmD0Common<Tuple>::D0DataType,
            typename TestGemmD0Common<Tuple>::EDataType,
            typename TestGemmD0Common<Tuple>::ALayout,
            typename TestGemmD0Common<Tuple>::BLayout,
            typename TestGemmD0Common<Tuple>::D0Layout,
            typename TestGemmD0Common<Tuple>::ELayout>;
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, I8, F32, F16, F16, Row, Row, Row, Row>,
                                     std::tuple<BF16, I8, F32, BF16, BF16, Row, Row, Row, Row>>;

TYPED_TEST_SUITE(TestGemmAddSilu, KernelTypes);
TYPED_TEST(TestGemmAddSilu, Test_BF16FP16_INT8) { this->Run(); }
