// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
using Row  = ck::tensor_layout::gemm::RowMajor;
using Col  = ck::tensor_layout::gemm::ColumnMajor;

using RRR_F16_F16_F16 = ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Row, Row, F16, F16, F16>>;
using RCR_F16_F16_F16 = ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Col, Row, F16, F16, F16>>;
using RRR_F16_F16_F16_LargeK =
    ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Row, Row, F16, F16, F16>>;
using RCR_F16_F16_F16_LargeK =
    ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Col, Row, F16, F16, F16>>;
using RRR_BF16_BF16_BF16 =
    ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Row, Row, BF16, BF16, BF16>>;
using RCR_BF16_BF16_BF16 =
    ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Col, Row, BF16, BF16, BF16>>;
using RRR_BF16_I8_BF16 =
    ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Row, Row, BF16, I8, BF16>>;
using RCR_BF16_I8_BF16 =
    ck::test::TestGroupedGemmTwoStage<std::tuple<Row, Col, Row, BF16, I8, BF16>>;

const std::vector<int> KBATCH{1, 2, 3, 5, 8};

INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_MK_KN,
                         RRR_F16_F16_F16,
                         testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_MK_NK,
                         RCR_F16_F16_F16,
                         testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_MK_KN_BF16,
                         RRR_BF16_BF16_BF16,
                         testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_MK_NK_BF16,
                         RCR_BF16_BF16_BF16,
                         testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_MK_KN_BF16_INT8,
                         RRR_BF16_I8_BF16,
                         testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_MK_NK_BF16_INT8,
                         RCR_BF16_I8_BF16,
                         testing::ValuesIn(KBATCH));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_LargeK_MK_KN,
                         RRR_F16_F16_F16_LargeK,
                         testing::Values(32, 64));
INSTANTIATE_TEST_SUITE_P(TestGroupedGemmTwoStage_splitk_LargeK_MK_NK,
                         RCR_F16_F16_F16_LargeK,
                         testing::Values(32, 64));

#include "test_grouped_gemm_ut_cases.inc"
#include "test_grouped_gemm_two_stage_ut_cases.inc"
