// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "test_gemm_common.hpp"
#include "profiler/profile_gemm_add_multiply_impl.hpp"

template <typename Tuple>
class TestGemmAddMultiply : public TestGemmD0D1Common<Tuple>
{
    using ProfileCall = typename TestGemmD0D1Common<Tuple>::ProfileCall;

    ProfileCall GetImpl() override
    {
        return ck::profiler::profile_gemm_add_multiply_impl<
            typename TestGemmD0D1Common<Tuple>::ADataType,
            typename TestGemmD0D1Common<Tuple>::BDataType,
            typename TestGemmD0D1Common<Tuple>::AccDataType,
            typename TestGemmD0D1Common<Tuple>::D0DataType,
            typename TestGemmD0D1Common<Tuple>::D1DataType,
            typename TestGemmD0D1Common<Tuple>::EDataType,
            typename TestGemmD0D1Common<Tuple>::ALayout,
            typename TestGemmD0D1Common<Tuple>::BLayout,
            typename TestGemmD0D1Common<Tuple>::D0Layout,
            typename TestGemmD0D1Common<Tuple>::D1Layout,
            typename TestGemmD0D1Common<Tuple>::ELayout>;
    }
};

using KernelTypes =
    ::testing::Types<std::tuple<F16, F16, F32, F16, F16, F16, Row, Col, Row, Row, Row>,
                     std::tuple<F16, F16, F32, F16, F16, F16, Row, Row, Row, Row, Row>,
                     std::tuple<F16, F16, F32, F16, F16, F16, Col, Col, Row, Row, Row>,
                     std::tuple<F16, F16, F32, F16, F16, F16, Col, Row, Row, Row, Row>>;

TYPED_TEST_SUITE(TestGemmAddMultiply, KernelTypes);
// Due to F16 shuffle data type tests has to run with limited K size. Change instances to FP32?
TYPED_TEST(TestGemmAddMultiply, Test) { this->Run({{16, 32, 64}, {2048, 1024, 256}}); }
