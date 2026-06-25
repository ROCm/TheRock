// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>

#include "gtest/gtest.h"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "test_gemm_universal_streamk_util.hpp"

ck::index_t param_mask     = 0xffff;
ck::index_t instance_index = -1;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace {

template <typename X, typename Y>
struct tuple_concat;

template <typename... Xs, typename... Ys>
struct tuple_concat<std::tuple<Xs...>, std::tuple<Ys...>>
{
    using type = std::tuple<Xs..., Ys...>;
};

} // namespace

template <typename Tuple>
class TestGemmUniversal_Streamk_BF16_MK_KN
    : public ck::test::TestGemmUniversal_Streamk<
          typename tuple_concat<std::tuple<Row, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmUniversal_Streamk_BF16_MK_NK
    : public ck::test::TestGemmUniversal_Streamk<
          typename tuple_concat<std::tuple<Row, Col>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmUniversal_Streamk_BF16_KM_KN
    : public ck::test::TestGemmUniversal_Streamk<
          typename tuple_concat<std::tuple<Col, Row>, Tuple>::type>
{
};

template <typename Tuple>
class TestGemmUniversal_Streamk_BF16_KM_NK
    : public ck::test::TestGemmUniversal_Streamk<
          typename tuple_concat<std::tuple<Col, Col>, Tuple>::type>
{
};

// clang-format off
using KernelTypes_MK_KN = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;
using KernelTypes_MK_NK = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType

    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;

using KernelTypes_KM_NK = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;

using KernelTypes_KM_KN = ::testing::Types<
    //         ADataType, BDataType, ComputeDataType, CDataType
    std::tuple<     BF16,      BF16,            BF16,    BF16>
    >;

// clang-format on

TYPED_TEST_SUITE(TestGemmUniversal_Streamk_BF16_MK_KN, KernelTypes_MK_KN);
TYPED_TEST_SUITE(TestGemmUniversal_Streamk_BF16_MK_NK, KernelTypes_MK_NK);
TYPED_TEST_SUITE(TestGemmUniversal_Streamk_BF16_KM_KN, KernelTypes_KM_KN);
TYPED_TEST_SUITE(TestGemmUniversal_Streamk_BF16_KM_NK, KernelTypes_KM_NK);

#include "test_gemm_universal_streamk_ut_cases_bf16.inc"
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
