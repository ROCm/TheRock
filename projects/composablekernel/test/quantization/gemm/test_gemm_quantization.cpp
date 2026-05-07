// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_quantization_impl.hpp"
#include "test_gemm_quantization_util.hpp"

using I8  = int8_t;
using I32 = int32_t;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestGemmQuantization : public ck::test::TestGemmQuantizationCommon<Tuple>
{
    protected:
    using ProfileCall = bool (*const)(int, int, bool, bool, int, int, int, int, int, int, float);

    ProfileCall GetImpl() override
    {
        return &ck::profiler::profile_gemm_quantization_impl<
            typename ck::test::TestGemmQuantizationCommon<Tuple>::ADataType,
            typename ck::test::TestGemmQuantizationCommon<Tuple>::BDataType,
            typename ck::test::TestGemmQuantizationCommon<Tuple>::AccDataType,
            typename ck::test::TestGemmQuantizationCommon<Tuple>::EDataType,
            typename ck::test::TestGemmQuantizationCommon<Tuple>::ALayout,
            typename ck::test::TestGemmQuantizationCommon<Tuple>::BLayout,
            typename ck::test::TestGemmQuantizationCommon<Tuple>::ELayout>;
    }
};

using KernelTypes = ::testing::Types<std::tuple<I8, I8, I32, I8, Row, Row, Row>,
                                     std::tuple<I8, I8, I32, I8, Row, Col, Row>,
                                     std::tuple<I8, I8, I32, I8, Col, Row, Row>,
                                     std::tuple<I8, I8, I32, I8, Col, Col, Row>>;

TYPED_TEST_SUITE(TestGemmQuantization, KernelTypes);

#include "test_gemm_quantization_ut_cases.inc"
