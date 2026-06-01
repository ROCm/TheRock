// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <gtest/gtest.h>

#include "profiler/profile_batched_gemm_impl.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp"

static ck::index_t instance_index = -1;

namespace {
using F16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using Empty_Tuple = ck::Tuple<>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

template <typename Tuple>
class TestBatchedGemmMultiD : public ::testing::Test
{
    protected:
    using ALayout = std::tuple_element_t<0, Tuple>;
    using BLayout = std::tuple_element_t<1, Tuple>;
    using CLayout = std::tuple_element_t<2, Tuple>;

    static constexpr int M          = 512;
    static constexpr int N          = 256;
    static constexpr int K          = 128;
    static constexpr int BatchCount = 3;

    template <typename DataType>
    void Run()
    {
        using namespace ck::tensor_operation::device;

        const bool pass =
            ck::profiler::profile_batched_gemm_impl<DataType,
                                                    DataType,
                                                    DataType,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough,
                                                    DeviceBatchedGemmMultiD<ALayout,
                                                                            BLayout,
                                                                            Empty_Tuple,
                                                                            CLayout,
                                                                            DataType,
                                                                            DataType,
                                                                            Empty_Tuple,
                                                                            DataType,
                                                                            PassThrough,
                                                                            PassThrough,
                                                                            PassThrough>>(
                true,  // do_verification
                1,     // init_method
                false, // do_log
                false, // time_kernel,
                M,
                N,
                K,
                std::is_same_v<ALayout, Row> ? K : M, // strideA
                std::is_same_v<BLayout, Row> ? N : K, // strideB
                std::is_same_v<CLayout, Row> ? N : M, // strideC
                // BatchStrideA BatchStrideB, BatchStrideC
                M * K,
                K * N,
                M * N,
                BatchCount,
                instance_index);
        EXPECT_TRUE(pass);
    }
};

using KernelTypes = ::testing::Types<std::tuple<Row, Row, Row>,
                                     std::tuple<Row, Col, Row>,
                                     std::tuple<Col, Row, Row>,
                                     std::tuple<Col, Col, Row>>;
} // namespace

TYPED_TEST_SUITE(TestBatchedGemmMultiD, KernelTypes);
#ifdef __fp16
TYPED_TEST(TestBatchedGemmMultiD, f16) { this->template Run<F16>(); }
#endif
#ifdef CK_ENABLE_INT8
TYPED_TEST(TestBatchedGemmMultiD, int8) { this->template Run<int8_t>(); }
#endif
int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 2)
    {
        instance_index = atoi(argv[1]);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1: instance_index(-1 means all)" << std::endl;
    }
    return RUN_ALL_TESTS();
}
