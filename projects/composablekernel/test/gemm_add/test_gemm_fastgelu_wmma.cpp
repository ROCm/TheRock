// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_fastgelu_impl.hpp"
#include "test_gemm_common.hpp"

template <typename Tuple>
class TestGemmFastgelu : public TestGemmCommon<Tuple>
{
    using ProfileCall = typename TestGemmCommon<Tuple>::ProfileCall;

    ProfileCall GetImpl() override
    {
        return ck::profiler::profile_gemm_fastgelu_impl<typename TestGemmCommon<Tuple>::ADataType,
                                                        typename TestGemmCommon<Tuple>::BDataType,
                                                        typename TestGemmCommon<Tuple>::AccDataType,
                                                        typename TestGemmCommon<Tuple>::EDataType,
                                                        typename TestGemmCommon<Tuple>::ALayout,
                                                        typename TestGemmCommon<Tuple>::BLayout,
                                                        typename TestGemmCommon<Tuple>::ELayout>;
    }
};

using KernelTypes = ::testing::Types<std::tuple<F16, F16, F32, F16, Row, Row, Row>,
                                     std::tuple<F16, F16, F32, F16, Row, Col, Row>,
                                     std::tuple<F16, F16, F32, F16, Col, Row, Row>,
                                     std::tuple<F16, F16, F32, F16, Col, Col, Row>>;

TYPED_TEST_SUITE(TestGemmFastgelu, KernelTypes);
TYPED_TEST(TestGemmFastgelu, Test_BF16FP16) { this->Run(); }
