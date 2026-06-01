// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "gtest/gtest.h"
#include "test_grouped_gemm_util.hpp"

ck::index_t param_mask     = 0xffffff;
ck::index_t instance_index = -1;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F8   = ck::f8_t;
using I8   = int8_t;

using AElementOp   = ck::tensor_operation::element_wise::PassThrough;
using BElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CDEElementOp = ck::tensor_operation::element_wise::FastGelu;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename Tuple>
class TestGroupedGemm : public ck::test::TestGroupedGemm<Tuple, true>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<     Row, Row, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>,
    std::tuple<     Row, Col, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>,
    std::tuple<     Col, Row, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>,
    std::tuple<     Col, Col, Row, F16, F16, F16, AElementOp, BElementOp, CDEElementOp>
>;
// clang-format on

TYPED_TEST_SUITE(TestGroupedGemm, KernelTypes);

#include "test_grouped_gemm_ut_cases.inc"
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 3)
    {
        param_mask     = strtol(argv[1], nullptr, 0);
        instance_index = atoi(argv[2]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1,2: param_mask instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
