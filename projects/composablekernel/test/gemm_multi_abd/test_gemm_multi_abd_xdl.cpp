// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/ck.hpp"
#include "profiler/profile_gemm_multi_abd_impl.hpp"
#include "test_gemm_common.hpp"

namespace ck {
namespace test {

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using I8   = int8_t;
using BF16 = ck::bhalf_t;

using PassThrough         = ck::tensor_operation::element_wise::PassThrough;
using Multiply            = ck::tensor_operation::element_wise::Multiply;
using Add                 = ck::tensor_operation::element_wise::Add;
using MultiplyAdd         = ck::tensor_operation::element_wise::MultiplyAdd;
using FastGelu            = ck::tensor_operation::element_wise::FastGelu;
using AddFastGelu         = ck::tensor_operation::element_wise::AddFastGelu;
using MultiplyAddFastGelu = ck::tensor_operation::element_wise::MultiplyAddFastGelu;
using MultiplyFastGelu    = ck::tensor_operation::element_wise::MultiplyFastGelu;

using KernelTypesABD = ::testing::Types<std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row, Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   Add>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Col, Col>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   Add>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row, Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   AddFastGelu>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Col, Col>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   AddFastGelu>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row, Row>,
                                                   ck::Tuple<>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   FastGelu>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Col, Col>,
                                                   ck::Tuple<>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   FastGelu>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row, Row>,
                                                   ck::Tuple<>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   PassThrough>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Col, Col>,
                                                   ck::Tuple<>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8, BF16>,
                                                   ck::Tuple<>,
                                                   BF16,
                                                   PassThrough,
                                                   Multiply,
                                                   PassThrough>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<Row, Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8>,
                                                   ck::Tuple<BF16, BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   PassThrough,
                                                   MultiplyAddFastGelu>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<Row, Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8>,
                                                   ck::Tuple<BF16, BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   PassThrough,
                                                   MultiplyAdd>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8>,
                                                   ck::Tuple<BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   PassThrough,
                                                   MultiplyFastGelu>,
                                        std::tuple<ck::Tuple<Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<Row>,
                                                   ck::Tuple<BF16>,
                                                   ck::Tuple<I8>,
                                                   ck::Tuple<BF16>,
                                                   BF16,
                                                   PassThrough,
                                                   PassThrough,
                                                   Multiply>>;

TYPED_TEST_SUITE(TestGemmCommon, KernelTypesABD);
TYPED_TEST(TestGemmCommon, Test_BF16I8BF16) { this->Run(); }

} // namespace test
} // namespace ck
